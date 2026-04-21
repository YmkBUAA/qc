# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research code for the NeurIPS 2025 paper "Reinforcement Learning with Action Chunking" (Q-chunking / QC). RL on a *temporally extended (action-chunked) action space* with an expressive behavior constraint to leverage prior data for improved exploration and online sample efficiency. Built on top of [FQL](https://github.com/seohongpark/fql); `rlpd_networks/` and `rlpd_distributions/` are taken directly from [RLPD](https://github.com/ikostrikov/rlpd).

## Environment & install

```bash
pip install -r requirements.txt
```

Uses JAX + Flax (GPU via `jax-cuda12-plugin`), `ogbench`, `mujoco`, `gymnasium`, `wandb`. MuJoCo rendering expects `MUJOCO_GL=egl`. Training auto-forwards `CUDA_VISIBLE_DEVICES` to `EGL_DEVICE_ID` / `MUJOCO_EGL_DEVICE_ID` (see `main*.py` top). Results and checkpoints are saved under `exp/<project>/<run_group>/<env_name>/<exp_name>/` and logged to Weights & Biases (`project='qc'`).

There is no test suite, linter, or build step — this is a research codebase run via CLI commands.

## Datasets

- Robomimic: expected at `~/.robomimic/{lift,can,square}/mh/low_dim_v15.hdf5` (Multi-Human datasets from robomimic.github.io).
- OGBench `cube-quadruple`: use the 100M offline dataset; pass `--ogbench_dataset_dir=<path>` (see README for the `wget` command). When a dataset dir is given, shards are rotated through every `--dataset_replace_interval` steps.

## Running experiments

Two entry points, selected by algorithm family:

- **`main.py`** — offline-to-online (QC, FQL, BFN and their variants). Default agent config is `agents/acfql.py`.
- **`main_online.py`** — pure online from scratch with an offline prior (RLPD, RLPD-AC, QC-RLPD). Default agent config is `agents/acrlpd.py`.

Canonical commands are in `README.md`. Key command-line patterns (flags live on `main*.py`; agent hyperparameters live under the `--agent.*` namespace via `ml_collections.config_flags`):

```bash
# QC (offline→online with flow-matching BC + chunked critic)
MUJOCO_GL=egl python main.py --run_group=reproduce \
  --agent.actor_type=best-of-n --agent.actor_num_samples=32 \
  --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5

# QC-RLPD (online RLPD with chunking + behavior cloning)
MUJOCO_GL=egl python main_online.py --env_name=... --horizon_length=5 --agent.bc_alpha=0.01
```

For the `scene` and `puzzle-3x3` OGBench domains, pass `--sparse=True`. `--horizon_length=1` disables chunking (i.e. standard 1-step actions); `--agent.action_chunking=False` switches chunked agents to n-step-return mode instead.

To swap agents entirely, pass `--agent=agents/acrlpd.py` (or vice versa).

## Architecture

### Training loops
`main.py` runs `offline_steps` of pure offline updates followed by `online_steps` of online rollouts + updates, sharing a single agent. The offline → online handoff seeds a `ReplayBuffer` from the offline `Dataset` (`utils/datasets.py`). Online rollouts use an **action queue**: the agent samples a whole chunk, pushes it into `action_queue`, and pops one action per env step; the queue is flushed on episode end. Both loops call `agent.update` (or `batch_update` with `--utd_ratio`) on a **sequence batch** sampled via `Dataset.sample_sequence(..., sequence_length=horizon_length, discount=...)` — the dataset is responsible for returning time-aligned chunks plus a `valid` mask and the precomputed discounted reward for the chunk.

`main_online.py` is the same pattern without the offline phase; the offline dataset is mixed in via the replay buffer so the RLPD-style agent can sample from both.

### Agents (`agents/`)
Both agents extend `flax.struct.PyTreeNode` and store an `rng`, a `ModuleDict`-based `TrainState` network, and a frozen `config`. `get_config()` in each file defines the hyperparameter schema; `cls.create(seed, ex_obs, ex_actions, config)` builds networks and returns an agent. Common pattern: `critic_loss` / `actor_loss` take `(batch, grad_params, rng)` and are combined in `total_loss`, then `update(batch)` returns `(new_agent, info)`.

- `acfql.py` — **Flow Q-learning with action chunking.** The actor is a flow-matching velocity field (`ActorVectorField`) trained with a BC flow loss; the critic sees the flattened chunk (`horizon_length * action_dim`). `actor_type` selects the policy head: `distill-ddpg` (one-step distillation student) or `best-of-n` (sample `actor_num_samples` chunks from the BC flow and pick the argmax-Q). `action_chunking=False` collapses to 1-step FQL with an n-step return.
- `acrlpd.py` — **SAC / RLPD with chunking.** Standard SAC temperature + Tanh-Normal actor (from `rlpd_distributions/`), critic ensemble (`rlpd_networks/Ensemble`) over the flattened chunk. `bc_alpha > 0` adds a behavior-cloning penalty (QC-RLPD). When `action_chunking=False` the chunk length acts as an n-step return horizon.
- `agents/__init__.py` registers agents in the `agents` dict keyed by `agent_name` (`acfql`, `acrlpd`).

### Environments (`envs/`)
`env_utils.make_env_and_datasets(env_name)` dispatches to OGBench, D4RL, or robomimic via name heuristics. `is_robomimic_env` / `is_ogbench_env` shape reward post-processing in the training loops — notably:
- robomimic: rewards are shifted by `-1.0` both offline (in `process_train_dataset`) and online (in the step loop).
- D4RL antmaze (`diverse`/`play`/`umaze`): online rewards shifted by `-1.0`.
- `--sparse=True`: rewards remapped to `-1.0 / 0.0`.

### Networks and utilities
- `utils/networks.py` — `ActorVectorField`, `Value`, MLPs; `utils/encoders.py` exposes visual encoders (`impala_small` etc.) selected via `--agent.encoder`.
- `utils/flax_utils.py` — `ModuleDict`, `TrainState`, `save_agent`, `nonpytree_field`.
- `utils/datasets.py` — `Dataset` (immutable FrozenDict-backed) and `ReplayBuffer` with `sample_sequence` that returns chunked samples plus `valid` masks and the chunk's discounted return. Adding a transition advances a ring buffer; `create_from_initial_dataset` seeds it.
- `rlpd_networks/`, `rlpd_distributions/` — imported verbatim from RLPD; don't modify unless you mean to diverge from that codebase.

### Evaluation & logging
`evaluation.evaluate` runs `num_eval_episodes` with a fresh action queue per episode (the full chunk is executed). Logging flows through `log_utils.LoggingHelper` → per-prefix `CsvLogger` + wandb; prefixes are `eval`, `env`, `offline_agent`, `online_agent`. Flags are dumped to `flags.json` and the wandb run URL to `token.tk` in the save dir.

## Conventions and gotchas

- **Sequence-first batches.** Most tensors have shape `(batch, horizon_length, ...)`. Agent code frequently flattens to `(batch, horizon_length * action_dim)` for critics and reshapes back for BC losses; respect the `batch["valid"]` mask on the chunk time axis when changing losses.
- **`config['horizon_length']` is injected from the CLI** in `main*.py` before `agent.create`; don't rely on the default in `get_config()`.
- **Offline `n`-step vs chunking.** Setting `--horizon_length=5 --agent.action_chunking=False` gives n-step return with 1-step actions; `--horizon_length=1` gives vanilla 1-step. These are distinct baselines in the paper.
- **Checkpoints.** `save_agent` only fires when `--save_interval > 0`; default is off. Evaluation runs at `--eval_interval` and once at the end of each phase.
- **wandb is required**: `setup_wandb` is called unconditionally; set `WANDB_MODE=offline` if running without network.
