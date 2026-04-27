import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


class ACFQLARAgent(flax.struct.PyTreeNode):
    """Action-chunked FQL with Flow-Anchored Reweighted BC.

    This is intentionally based on ACFQL, not the NACFQL noised-critic line:
    it keeps the chunked FQL critic/actor stack and only adds FlowAR's
    sample-local BC weighting mechanism.
    """

    rng: Any
    network: Any
    critic_loss_ema: jnp.ndarray
    var_tq_ema: jnp.ndarray
    config: Any = nonpytree_field()

    def _flat_actions(self, batch):
        if self.config['action_chunking']:
            return jnp.reshape(batch['actions'], (batch['actions'].shape[0], -1))
        return batch['actions'][..., 0, :]

    def _full_action_dim(self):
        if self.config['action_chunking']:
            return self.config['action_dim'] * self.config['horizon_length']
        return self.config['action_dim']

    def critic_loss(self, batch, grad_params, rng):
        """Compute the ACFQL critic loss and target-Q variance for gating."""
        batch_actions = self._flat_actions(batch)

        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(
            batch['next_observations'][..., -1, :], rng=sample_rng
        )

        next_qs = self.network.select('target_critic')(
            batch['next_observations'][..., -1, :], actions=next_actions
        )
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'][..., -1] + (
            self.config['discount'] ** self.config['horizon_length']
        ) * batch['masks'][..., -1] * next_q

        q = self.network.select('critic')(
            batch['observations'], actions=batch_actions, params=grad_params
        )
        critic_loss = (
            jnp.square(q - target_q) * batch['valid'][..., -1]
        ).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'var_target_q': jnp.var(target_q) + 1e-8,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    @staticmethod
    def _ess_targeted_weights(a_norm, ess_target, n_iters=14):
        """Bisect log(tau) so normalized softmax weights hit target ESS."""
        b = a_norm.shape[0]

        def ess_over_b(log_tau):
            tau = jnp.exp(log_tau)
            z = a_norm / tau
            z = z - jnp.max(z)
            w = jnp.exp(z)
            s1 = jnp.sum(w)
            s2 = jnp.sum(w * w)
            return (s1 * s1) / (b * s2 + 1e-12)

        log_lo = jnp.log(1e-3)
        log_hi = jnp.log(1e3)

        def body(_, state):
            lo, hi = state
            mid = 0.5 * (lo + hi)
            ess_mid = ess_over_b(mid)
            new_lo = jnp.where(ess_mid < ess_target, mid, lo)
            new_hi = jnp.where(ess_mid < ess_target, hi, mid)
            return new_lo, new_hi

        lo, hi = jax.lax.fori_loop(0, n_iters, body, (log_lo, log_hi))
        log_tau = 0.5 * (lo + hi)
        tau = jnp.exp(log_tau)

        z = a_norm / tau
        z = z - jnp.max(z)
        w = jnp.exp(z)
        w_norm = w / (jnp.mean(w) + 1e-12)
        ess_achieved = ess_over_b(log_tau)
        return w_norm, tau, ess_achieved

    def _roll_flow_from(self, observations, x_t, t_start, n_steps):
        """Integrate actor_bc_flow from (x_t, t_start) to t = 1."""
        if self.config['encoder'] is not None:
            obs_enc = self.network.select('actor_bc_flow_encoder')(observations)
        else:
            obs_enc = observations

        x = x_t
        dt = (1.0 - t_start) / float(n_steps)
        for i in range(n_steps):
            tau = t_start + i * dt
            vels = self.network.select('actor_bc_flow')(
                obs_enc, x, tau, is_encoded=True
            )
            x = x + vels * dt
        return jnp.clip(x, -1, 1)

    def actor_loss(self, batch, grad_params, rng):
        """FlowAR-weighted ACFQL actor loss."""
        batch_actions = self._flat_actions(batch)
        batch_size, full_action_dim = batch_actions.shape

        # ----- Branch A: FlowAR advantage via self-denoising ---------------
        rng, eps_adv_rng, t_adv_rng = jax.random.split(rng, 3)
        eps_adv = jax.random.normal(eps_adv_rng, (batch_size, full_action_dim))
        t_adv = jax.random.uniform(
            t_adv_rng,
            (batch_size, 1),
            minval=self.config['adv_t_lo'],
            maxval=self.config['adv_t_hi'],
        )
        x_t_adv = (1.0 - t_adv) * eps_adv + t_adv * batch_actions
        a_prime = self._roll_flow_from(
            batch['observations'],
            x_t_adv,
            t_adv,
            n_steps=self.config['adv_flow_steps'],
        )
        a_prime = jax.lax.stop_gradient(a_prime)

        q_a_all = self.network.select('target_critic')(
            batch['observations'], actions=batch_actions
        )
        q_aprime_all = self.network.select('target_critic')(
            batch['observations'], actions=a_prime
        )
        if self.config['q_agg'] == 'min':
            q_a = q_a_all.min(axis=0)
            q_aprime = q_aprime_all.min(axis=0)
        else:
            q_a = q_a_all.mean(axis=0)
            q_aprime = q_aprime_all.mean(axis=0)

        delta = jax.lax.stop_gradient(q_a - q_aprime)
        d_med = jnp.median(delta)
        beta_mad = 1.4826 * jnp.median(jnp.abs(delta - d_med))
        beta_mad = jnp.maximum(beta_mad, 1e-6)
        d_norm = (delta - d_med) / beta_mad

        w_exp, tau_star, ess_achieved = self._ess_targeted_weights(
            d_norm, jnp.asarray(self.config['ess_target'])
        )

        critic_valid = (self.critic_loss_ema >= 0) & (self.var_tq_ema > 0)
        r2_critic = jnp.where(
            critic_valid,
            1.0 - self.critic_loss_ema / jnp.maximum(self.var_tq_ema, 1e-8),
            jnp.asarray(-1.0),
        )
        gate_critic = jax.nn.sigmoid(
            (r2_critic - self.config['r2_critic_target'])
            / self.config['gate_kappa_critic']
        )
        gate_critic = jnp.where(critic_valid, gate_critic, jnp.asarray(0.0))

        online = jnp.asarray(
            batch.get('online', jnp.asarray(0.0)), dtype=jnp.float32
        )
        offline_weighting = jnp.asarray(
            0.0 if self.config['weighted_bc_online_only'] else 1.0,
            dtype=jnp.float32,
        )
        weighting_active = jnp.maximum(online, offline_weighting)
        gate_c = gate_critic * weighting_active
        bc_weights = 1.0 + gate_c * (w_exp - 1.0)
        bc_weights = jax.lax.stop_gradient(bc_weights)

        # ----- Branch B: independent K-time CFM BC loss --------------------
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        k = self.config['n_actor_time_samples']
        x_0 = jax.random.normal(x_rng, (batch_size, k, full_action_dim))
        x_1 = batch_actions[:, None, :]
        t = jax.random.uniform(t_rng, (batch_size, k, 1))
        x_t = (1.0 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        obs_tail = batch['observations'].shape[1:]
        obs_exp = jnp.broadcast_to(
            batch['observations'][:, None], (batch_size, k) + obs_tail
        )
        obs_flat = obs_exp.reshape((batch_size * k,) + obs_tail)
        x_t_flat = x_t.reshape(batch_size * k, full_action_dim)
        t_flat = t.reshape(batch_size * k, 1)
        pred_flat = self.network.select('actor_bc_flow')(
            obs_flat, x_t_flat, t_flat, params=grad_params
        )
        pred = pred_flat.reshape(batch_size, k, full_action_dim)

        if self.config['action_chunking']:
            per_step_sq = jnp.reshape(
                (pred - vel) ** 2,
                (
                    batch_size,
                    k,
                    self.config['horizon_length'],
                    self.config['action_dim'],
                ),
            ) * batch['valid'][:, None, :, None]
            per_time_bc = jnp.mean(per_step_sq, axis=(2, 3))
        else:
            per_time_bc = jnp.mean((pred - vel) ** 2, axis=-1)
        bc_flow_loss = jnp.mean(bc_weights[:, None] * per_time_bc)

        if self.config['actor_type'] == 'distill-ddpg':
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, full_action_dim))
            target_flow_actions = self.compute_flow_actions(
                batch['observations'], noises=noises
            )
            actor_actions = self.network.select('actor_onestep_flow')(
                batch['observations'], noises, params=grad_params
            )
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

            actor_actions_clip = jnp.clip(actor_actions, -1, 1)
            qs = self.network.select('critic')(
                batch['observations'], actions=actor_actions_clip
            )
            q = jnp.mean(qs, axis=0)
            q_loss = -q.mean()
            if self.config['normalize_q_loss']:
                lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
                q_loss = lam * q_loss
        else:
            distill_loss = jnp.zeros(())
            q_loss = jnp.zeros(())
            q = jnp.zeros(())

        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        dist_a_aprime = jnp.linalg.norm(batch_actions - a_prime, axis=-1)
        frac_a_better = jnp.mean((delta > 0).astype(jnp.float32))

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'flowar/delta_mean': delta.mean(),
            'flowar/delta_std': delta.std(),
            'flowar/dist_a_aprime_mean': dist_a_aprime.mean(),
            'flowar/dist_a_aprime_p50': jnp.median(dist_a_aprime),
            'flowar/frac_a_better': frac_a_better,
            'flowar/q_a_mean': q_a.mean(),
            'flowar/q_aprime_mean': q_aprime.mean(),
            'flowar/t_adv_mean': t_adv.mean(),
            'beta_mad': beta_mad,
            'tau_star': tau_star,
            'ess_achieved': ess_achieved,
            'gate_critic': gate_critic,
            'gate_c': gate_c,
            'online_phase': online,
            'weighting_active': weighting_active,
            'r2_critic': r2_critic,
            'bc_weight_mean': bc_weights.mean(),
            'bc_weight_std': bc_weights.std(),
            'bc_weight_max': bc_weights.max(),
            'bc_weight_min': bc_weights.min(),
            'n_actor_time_samples': jnp.asarray(k, dtype=jnp.float32),
            't_mean': t.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')

        decay = agent.config['gate_ema_decay']
        uninitialised = agent.critic_loss_ema < 0

        cur_critic_loss = info['critic/critic_loss']
        cur_var_tq = info['critic/var_target_q']
        new_critic_ema = jnp.where(
            uninitialised,
            cur_critic_loss,
            decay * agent.critic_loss_ema + (1 - decay) * cur_critic_loss,
        )
        new_var_tq_ema = jnp.where(
            uninitialised,
            cur_var_tq,
            decay * agent.var_tq_ema + (1 - decay) * cur_var_tq,
        )

        info['gate/critic_loss_ema'] = new_critic_ema
        info['gate/var_tq_ema'] = new_var_tq_ema
        info['gate/r2_critic_ema'] = (
            1.0 - new_critic_ema / jnp.maximum(new_var_tq_ema, 1e-8)
        )

        return agent.replace(
            network=new_network,
            critic_loss_ema=new_critic_ema,
            var_tq_ema=new_var_tq_ema,
            rng=new_rng,
        ), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_actions(self, observations, rng=None):
        full_action_dim = self._full_action_dim()

        if self.config['actor_type'] == 'distill-ddpg':
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],
                    full_action_dim,
                ),
            )
            actions = self.network.select('actor_onestep_flow')(observations, noises)
            actions = jnp.clip(actions, -1, 1)

        elif self.config['actor_type'] == 'best-of-n':
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],
                    self.config['actor_num_samples'],
                    full_action_dim,
                ),
            )
            observations = jnp.repeat(
                observations[..., None, :],
                self.config['actor_num_samples'],
                axis=-2,
            )
            actions = self.compute_flow_actions(observations, noises)
            actions = jnp.clip(actions, -1, 1)
            if self.config['q_agg'] == 'mean':
                q = self.network.select('critic')(observations, actions).mean(axis=0)
            else:
                q = self.network.select('critic')(observations, actions).min(axis=0)
            indices = jnp.argmax(q, axis=-1)

            bshape = indices.shape
            indices = indices.reshape(-1)
            bsize = len(indices)
            actions = jnp.reshape(
                actions, (-1, self.config['actor_num_samples'], full_action_dim)
            )[jnp.arange(bsize), indices, :].reshape(bshape + (full_action_dim,))

        return actions

    @jax.jit
    def compute_flow_actions(self, observations, noises):
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(
                observations, actions, t, is_encoded=True
            )
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config['action_chunking']:
            full_actions = jnp.concatenate(
                [ex_actions] * config['horizon_length'], axis=-1
            )
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
        )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
            use_fourier_features=config['use_fourier_features'],
            fourier_feature_dim=config['fourier_feature_dim'],
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, full_actions)),
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
        )
        if encoders.get('actor_bc_flow') is not None:
            network_info['actor_bc_flow_encoder'] = (
                encoders.get('actor_bc_flow'), (ex_observations,)
            )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config['weight_decay'] > 0.:
            network_tx = optax.adamw(
                learning_rate=config['lr'], weight_decay=config['weight_decay']
            )
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        sentinel = jnp.asarray(-1.0, dtype=jnp.float32)
        return cls(
            rng,
            network=network,
            critic_loss_ema=sentinel,
            var_tq_ema=sentinel,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='acfql_ar',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            q_agg='mean',
            alpha=100.0,
            num_qs=2,
            flow_steps=10,
            normalize_q_loss=False,
            encoder=ml_collections.config_dict.placeholder(str),
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=True,
            actor_type='distill-ddpg',
            actor_num_samples=32,
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
            # ---- BC flow CFM samples per chunk action ----
            n_actor_time_samples=1,
            # ---- FlowAR advantage path ----
            adv_t_lo=0.4,
            adv_t_hi=0.7,
            adv_flow_steps=3,
            # ---- ESS-targeted reweighting ----
            ess_target=0.7,
            weighted_bc_online_only=True,
            # ---- Critic R^2 reliability gate ----
            r2_critic_target=0.5,
            gate_kappa_critic=0.05,
            gate_ema_decay=0.999,
        )
    )
    return config
