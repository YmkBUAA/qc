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


class NACFQL3Agent(flax.struct.PyTreeNode):
    """NACFQL v3 — NACFQL v2 + offline-phase kill switch.

    Motivation: in empirical runs the v2 actor weighting
        bc_weights = 1 + c_gate * trust * (w_exp - 1)
    reshapes the BC flow loss from the first gradient step onwards. Since
    the R^2 gates saturate quickly (gate_qn * gate_critic -> ~0.99 within
    ~10k steps), offline training runs with a noisy posterior-informed
    reweighting from very early on, which lagged the pure-ACFQL baseline
    on cube-double-play-task3/task5 (off-end deltas -0.08 to -0.24).

    v3 keeps Q_n training on during the offline phase (so the anchor head
    warms up) but forces `bc_weights = 1.0` offline by multiplying c_gate
    by a phase indicator `online ∈ {0, 1}` supplied per batch. At the
    online boundary the weighting activates automatically.

    The actor BC flow loss uses K sampled flow times per data action. Each
    (x_t, t) point receives its own Q_n residual, ESS-targeted weight, trust
    value, and per-time loss contribution.

    Phase flag contract: caller places `batch['online']` as a scalar
    jnp.float32 (0.0 offline, 1.0 online). If the key is absent, the
    agent defaults to `online=1.0` — matching NACFQL v2 behaviour so
    existing call sites that don't set the flag are unaffected.

    Noised-critic and value-anchor training details from v2 are retained.
    """

    rng: Any
    network: Any
    noised_network: Any
    noised_loss_ema: jnp.ndarray
    var_q_ema: jnp.ndarray
    critic_loss_ema: jnp.ndarray
    var_tq_ema: jnp.ndarray
    config: Any = nonpytree_field()

    # ------------------------------------------------------------------
    # Shape helpers.
    # ------------------------------------------------------------------
    def _flat_actions(self, batch):
        if self.config['action_chunking']:
            return jnp.reshape(batch['actions'], (batch['actions'].shape[0], -1))
        return batch['actions'][..., 0, :]

    def _full_action_dim(self):
        if self.config['action_chunking']:
            return self.config['action_dim'] * self.config['horizon_length']
        return self.config['action_dim']

    # ------------------------------------------------------------------
    # Critic loss — identical to NACFQL v2 / ACFQL.
    # ------------------------------------------------------------------
    def critic_loss(self, batch, grad_params, rng):
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

        target_q = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config['horizon_length']) * \
            batch['masks'][..., -1] * next_q

        q = self.network.select('critic')(
            batch['observations'], actions=batch_actions, params=grad_params
        )
        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        var_target_q = jnp.var(target_q) + 1e-8

        return critic_loss, {
            'critic_loss': critic_loss,
            'var_target_q': var_target_q,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    # ------------------------------------------------------------------
    # Value loss — V(s) = E_{a ~ pi}[Q_tgt(s, a)] via MC.
    # ------------------------------------------------------------------
    def value_loss(self, batch, grad_params, rng):
        obs = batch['observations']
        batch_size = obs.shape[0]
        obs_tail = obs.shape[1:]
        full_action_dim = self._full_action_dim()
        n = self.config['n_v_samples']

        noises = jax.random.normal(rng, (batch_size, n, full_action_dim))
        obs_exp = jnp.broadcast_to(obs[:, None], (batch_size, n) + obs_tail)
        obs_flat = obs_exp.reshape((batch_size * n,) + obs_tail)
        noises_flat = noises.reshape(batch_size * n, full_action_dim)

        if self.config['actor_type'] == 'distill-ddpg':
            a_samples = self.network.select('actor_onestep_flow')(obs_flat, noises_flat)
        else:
            a_samples = self.compute_flow_actions(obs_flat, noises_flat)
        a_samples = jnp.clip(a_samples, -1, 1)
        a_samples = jax.lax.stop_gradient(a_samples)

        q_samples_flat = self.network.select('target_critic')(
            obs_flat, actions=a_samples
        )
        ensemble = q_samples_flat.shape[0]
        q_samples = q_samples_flat.reshape(ensemble, batch_size, n)
        v_target = jax.lax.stop_gradient(q_samples.mean(axis=(0, 2)))

        v_pred = self.network.select('value')(obs, params=grad_params)

        loss = jnp.mean(((v_pred - v_target) ** 2) * batch['valid'][..., -1])
        return loss, {
            'value_loss': loss,
            'v_pred_mean': v_pred.mean(),
            'v_target_mean': v_target.mean(),
            'v_target_std': v_target.std(),
        }

    # ------------------------------------------------------------------
    # ESS-targeted temperature (unchanged).
    # ------------------------------------------------------------------
    @staticmethod
    def _ess_targeted_weights(a_norm, ess_target, n_iters=14):
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
            return (new_lo, new_hi)

        lo, hi = jax.lax.fori_loop(0, n_iters, body, (log_lo, log_hi))
        log_tau = 0.5 * (lo + hi)
        tau = jnp.exp(log_tau)

        z = a_norm / tau
        z = z - jnp.max(z)
        w = jnp.exp(z)
        w_norm = w / (jnp.mean(w) + 1e-12)
        ess_achieved = ess_over_b(log_tau)
        return w_norm, tau, ess_achieved

    # ------------------------------------------------------------------
    # Actor loss — v2 actor reweighting gated by `online` phase flag.
    # offline (online=0) -> c_gate forced to 0 -> bc_weights == 1 -> pure
    # chunked ACFQL BC flow loss. online (online=1) -> K-way per-time weighting.
    # ------------------------------------------------------------------
    def actor_loss(self, batch, grad_params, rng):
        batch_actions = self._flat_actions(batch)
        batch_size, full_action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        k = self.config['n_actor_time_samples']

        x_0 = jax.random.normal(x_rng, (batch_size, k, full_action_dim))
        x_1 = batch_actions
        x_1_exp = x_1[:, None, :]
        t = jax.random.uniform(t_rng, (batch_size, k, 1))
        x_t = (1 - t) * x_0 + t * x_1_exp
        vel = x_1_exp - x_0

        original_qs = self.network.select('critic')(
            batch['observations'], actions=batch_actions
        )
        original_q = jax.lax.stop_gradient(original_qs.mean(axis=0))

        obs_tail = batch['observations'].shape[1:]
        obs_exp = jnp.broadcast_to(
            batch['observations'][:, None], (batch_size, k) + obs_tail
        )
        x_t_with_t = jnp.concatenate([x_t, t], axis=-1)
        if self.config['encoder'] is not None:
            obs_flat = obs_exp.reshape((batch_size * k,) + obs_tail)
            actions_flat = x_t_with_t.reshape((batch_size * k, -1))
            noised_qs_flat = self.noised_network.select('noised_critic')(
                obs_flat,
                actions=jax.lax.stop_gradient(actions_flat),
            )
            noised_qs = noised_qs_flat.reshape(
                (noised_qs_flat.shape[0], batch_size, k)
            )
        else:
            noised_qs = self.noised_network.select('noised_critic')(
                obs_exp, actions=jax.lax.stop_gradient(x_t_with_t)
            )
        noised_q = jax.lax.stop_gradient(noised_qs.mean(axis=0))
        noised_q_disagree = jax.lax.stop_gradient(
            jnp.abs(noised_qs[0] - noised_qs[1])
        )

        a_local = original_q[:, None] - noised_q
        a_local_flat = a_local.reshape(-1)

        a_med = jnp.median(a_local_flat)
        beta_mad = 1.4826 * jnp.median(jnp.abs(a_local_flat - a_med))
        beta_mad = jnp.maximum(beta_mad, 1e-6)
        a_norm = (a_local_flat - a_med) / beta_mad

        w_exp_flat, tau_star, ess_achieved = self._ess_targeted_weights(
            a_norm, jnp.asarray(self.config['ess_target'])
        )
        w_exp = w_exp_flat.reshape(batch_size, k)

        delta_med = jnp.maximum(jnp.median(noised_q_disagree.reshape(-1)), 1e-6)
        trust = jnp.exp(-noised_q_disagree / delta_med)

        ema_valid = (self.noised_loss_ema >= 0) & (self.var_q_ema > 0)
        r2_qn = jnp.where(
            ema_valid,
            1.0 - self.noised_loss_ema / jnp.maximum(self.var_q_ema, 1e-8),
            jnp.asarray(-1.0),
        )
        gate_qn = jax.nn.sigmoid(
            (r2_qn - self.config['r2_target']) / self.config['gate_kappa']
        )
        gate_qn = jnp.where(ema_valid, gate_qn, jnp.asarray(0.0))

        critic_valid = (self.critic_loss_ema >= 0) & (self.var_tq_ema > 0)
        r2_critic = jnp.where(
            critic_valid,
            1.0 - self.critic_loss_ema / jnp.maximum(self.var_tq_ema, 1e-8),
            jnp.asarray(-1.0),
        )
        gate_critic = jax.nn.sigmoid(
            (r2_critic - self.config['r2_critic_target']) / self.config['gate_kappa_critic']
        )
        gate_critic = jnp.where(critic_valid, gate_critic, jnp.asarray(0.0))

        # Phase indicator: 0.0 during offline, 1.0 during online. Missing
        # key -> 1.0 (v2 behaviour) so external callers aren't broken.
        online = jnp.asarray(
            batch.get('online', jnp.asarray(1.0)), dtype=jnp.float32
        )
        c_gate = gate_qn * gate_critic * online
        bc_weights = 1.0 + c_gate * trust * (w_exp - 1.0)
        bc_weights = jax.lax.stop_gradient(bc_weights)

        pred = self.network.select('actor_bc_flow')(
            obs_exp, x_t, t, params=grad_params
        )

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
        bc_flow_loss = jnp.mean(bc_weights * per_time_bc)

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

        frac_suboptimal = jnp.mean(original_q[:, None] < noised_q)

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'frac_suboptimal': frac_suboptimal,
            'beta_mad': beta_mad,
            'tau_star': tau_star,
            'ess_achieved': ess_achieved,
            'gate_qn': gate_qn,
            'gate_critic': gate_critic,
            'gate_c': c_gate,
            'online_phase': online,
            'r2_qn': r2_qn,
            'r2_critic': r2_critic,
            'trust_mean': trust.mean(),
            'bc_weight_mean': bc_weights.mean(),
            'bc_weight_max': bc_weights.max(),
            'bc_weight_min': bc_weights.min(),
            'n_actor_time_samples': jnp.asarray(k, dtype=jnp.float32),
            't_mean': t.mean(),
        }

    # ------------------------------------------------------------------
    # Noised critic loss — Recipe 1 + Recipe 6 (unchanged from v2).
    # Q_n keeps training during offline so it is warm at the online switch.
    # ------------------------------------------------------------------
    def noised_critic_loss(self, batch, grad_params, rng):
        batch_size = batch['observations'].shape[0]
        full_action_dim = self._full_action_dim()
        K = self.config['n_action_samples']

        rng, a_noise_rng, x0_rng, t_rng, xa_rng = jax.random.split(rng, 5)

        a_noises = jax.random.normal(
            a_noise_rng, (batch_size, K, full_action_dim)
        )
        obs_tail = batch['observations'].shape[1:]
        obs_exp = jnp.broadcast_to(
            batch['observations'][:, None], (batch_size, K) + obs_tail
        )
        obs_flat = obs_exp.reshape((batch_size * K,) + obs_tail)
        a_noises_flat = a_noises.reshape(batch_size * K, full_action_dim)

        a_k_flat = self.compute_flow_actions(obs_flat, a_noises_flat)
        a_k_flat = jnp.clip(a_k_flat, -1, 1)
        a_k_flat = jax.lax.stop_gradient(a_k_flat)
        a_k = a_k_flat.reshape(batch_size, K, full_action_dim)

        q_tgt_flat = self.network.select('target_critic')(
            obs_flat, actions=a_k_flat
        )
        if self.config['q_agg'] == 'min':
            q_tgt_flat = q_tgt_flat.min(axis=0)
        else:
            q_tgt_flat = q_tgt_flat.mean(axis=0)
        q_targets = jax.lax.stop_gradient(
            q_tgt_flat.reshape(batch_size, K)
        )

        if self.config['t_sampling'] == 'beta':
            t = jax.random.beta(
                t_rng,
                self.config['t_beta_a'],
                self.config['t_beta_b'],
                shape=(batch_size, K, 1),
            )
        else:
            t = jax.random.uniform(t_rng, (batch_size, K, 1))

        x_0 = jax.random.normal(x0_rng, (batch_size, K, full_action_dim))
        x_t = (1 - t) * x_0 + t * a_k
        x_t_with_t = jnp.concatenate([x_t, t], axis=-1)

        if self.config['encoder'] is not None:
            noised_qs_flat = self.noised_network.select('noised_critic')(
                obs_flat,
                actions=x_t_with_t.reshape((batch_size * K, -1)),
                params=grad_params,
            )
            noised_qs = noised_qs_flat.reshape(
                (noised_qs_flat.shape[0], batch_size, K)
            )
        else:
            noised_qs = self.noised_network.select('noised_critic')(
                obs_exp, actions=x_t_with_t, params=grad_params
            )
        noised_q = noised_qs.mean(axis=0)

        anchor_sq = (noised_q - q_targets) ** 2
        valid_mask = batch['valid'][..., -1][:, None]
        anchor_loss = (anchor_sq * valid_mask).mean()

        x_0_v = jax.random.normal(xa_rng, (batch_size, full_action_dim))
        t_zero = jnp.zeros((batch_size, 1))
        xt_with_t_zero = jnp.concatenate([x_0_v, t_zero], axis=-1)
        qn_t0_all = self.noised_network.select('noised_critic')(
            batch['observations'], actions=xt_with_t_zero, params=grad_params
        )
        qn_t0 = qn_t0_all.mean(axis=0)
        v_for_anchor = jax.lax.stop_gradient(
            self.network.select('value')(batch['observations'])
        )
        v_anchor_sq = (qn_t0 - v_for_anchor) ** 2
        v_anchor_loss = (v_anchor_sq * batch['valid'][..., -1]).mean()

        total = (
            self.config['lambda_anchor'] * anchor_loss
            + self.config['lambda_v_anchor'] * v_anchor_loss
        )

        var_q = jnp.var(q_targets) + 1e-8

        flat_t = t.reshape(-1)
        flat_sq = anchor_sq.reshape(-1)

        def _bucket_r2(lo, hi):
            mask = ((flat_t >= lo) & (flat_t < hi)).astype(jnp.float32)
            mse_bin = (flat_sq * mask).sum() / jnp.maximum(mask.sum(), 1.0)
            return 1.0 - mse_bin / var_q

        info = {
            'noised_critic_loss': total,
            'anchor_loss': anchor_loss,
            'v_anchor_loss': v_anchor_loss,
            'var_q': var_q,
            'noised_q_mean': noised_q.mean(),
            'noised_q_std': noised_q.std(),
            'qn_t0_mean': qn_t0.mean(),
            'q_targets_mean': q_targets.mean(),
            'q_targets_std': q_targets.std(),
            'v_at_anchor_mean': v_for_anchor.mean(),
            't_mean': t.mean(),
            'r2_t_0.00_0.25': _bucket_r2(0.00, 0.25),
            'r2_t_0.25_0.50': _bucket_r2(0.25, 0.50),
            'r2_t_0.50_0.75': _bucket_r2(0.50, 0.75),
            'r2_t_0.75_1.00': _bucket_r2(0.75, 1.00 + 1e-6),
        }
        return total, info

    # ------------------------------------------------------------------
    # Combined main-network loss.
    # ------------------------------------------------------------------
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng, value_rng = jax.random.split(rng, 4)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        v_loss, v_info = self.value_loss(batch, grad_params, value_rng)
        for k, v in v_info.items():
            info[f'value/{k}'] = v

        loss = critic_loss + actor_loss + self.config['value_loss_weight'] * v_loss
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
        fql_rng, noised_rng = jax.random.split(rng)

        def fql_loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=fql_rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=fql_loss_fn)
        agent.target_update(new_network, 'critic')

        def noised_loss_fn(grad_params):
            return agent.noised_critic_loss(batch, grad_params, rng=noised_rng)

        new_noised_network, noised_info = agent.noised_network.apply_loss_fn(
            loss_fn=noised_loss_fn
        )
        for k, v in noised_info.items():
            info[f'noised_critic/{k}'] = v

        decay = agent.config['gate_ema_decay']
        uninitialised = agent.noised_loss_ema < 0

        cur_noised_loss = noised_info['noised_critic_loss']
        cur_var_q = noised_info['var_q']
        new_noised_ema = jnp.where(
            uninitialised, cur_noised_loss,
            decay * agent.noised_loss_ema + (1 - decay) * cur_noised_loss,
        )
        new_var_q_ema = jnp.where(
            uninitialised, cur_var_q,
            decay * agent.var_q_ema + (1 - decay) * cur_var_q,
        )

        cur_critic_loss = info['critic/critic_loss']
        cur_var_tq = info['critic/var_target_q']
        new_critic_ema = jnp.where(
            uninitialised, cur_critic_loss,
            decay * agent.critic_loss_ema + (1 - decay) * cur_critic_loss,
        )
        new_var_tq_ema = jnp.where(
            uninitialised, cur_var_tq,
            decay * agent.var_tq_ema + (1 - decay) * cur_var_tq,
        )

        info['gate/noised_loss_ema'] = new_noised_ema
        info['gate/var_q_ema'] = new_var_q_ema
        info['gate/r2_qn_ema'] = 1.0 - new_noised_ema / jnp.maximum(new_var_q_ema, 1e-8)
        info['gate/critic_loss_ema'] = new_critic_ema
        info['gate/var_tq_ema'] = new_var_tq_ema
        info['gate/r2_critic_ema'] = 1.0 - new_critic_ema / jnp.maximum(new_var_tq_ema, 1e-8)

        return agent.replace(
            network=new_network,
            noised_network=new_noised_network,
            noised_loss_ema=new_noised_ema,
            var_q_ema=new_var_q_ema,
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

    # ------------------------------------------------------------------
    # Inference — unchanged from v2.
    # ------------------------------------------------------------------
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
                observations[..., None, :], self.config['actor_num_samples'], axis=-2
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

    # ------------------------------------------------------------------
    # Factory.
    # ------------------------------------------------------------------
    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng, noised_init_rng = jax.random.split(rng, 3)

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
            encoders['value'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()
            encoders['noised_critic'] = encoder_module()

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
        )
        value_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('value'),
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
            value=(value_def, (ex_observations,)),
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

        ex_noised_actions_with_t = jnp.concatenate([full_actions, ex_times], axis=-1)
        noised_critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('noised_critic'),
        )
        noised_network_def = ModuleDict({'noised_critic': noised_critic_def})
        noised_network_tx = optax.adam(learning_rate=config['lr'])
        noised_network_params = noised_network_def.init(
            noised_init_rng,
            noised_critic=(ex_observations, ex_noised_actions_with_t),
        )['params']
        noised_network = TrainState.create(
            noised_network_def, noised_network_params, tx=noised_network_tx
        )

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        sentinel = jnp.asarray(-1.0, dtype=jnp.float32)
        return cls(
            rng,
            network=network,
            noised_network=noised_network,
            noised_loss_ema=sentinel,
            var_q_ema=sentinel,
            critic_loss_ema=sentinel,
            var_tq_ema=sentinel,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='nacfql_3',
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
            # ---- Recipe 1: multi-sample action-diversity anchor ----
            n_action_samples=4,
            lambda_anchor=1.0,
            # ---- V head and V-anchor on Q_n at t=0 ----
            n_v_samples=8,
            lambda_v_anchor=1.0,
            value_loss_weight=1.0,
            # ---- Recipe 6: Beta-distributed flow time ----
            t_sampling='beta',
            t_beta_a=2.0,
            t_beta_b=2.0,
            # ---- K-way ESS-targeted BC weighting ----
            n_actor_time_samples=1,
            ess_target=0.7,
            # ---- Dual R^2 reliability gate ----
            r2_target=0.75,
            gate_kappa=0.05,
            r2_critic_target=0.5,
            gate_kappa_critic=0.05,
            gate_ema_decay=0.999,
        )
    )
    return config
