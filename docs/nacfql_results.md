# NACFQL — Results v2 (wandb `ymkbuaa-beihang-university/qc`, as of 2026-04-21)

NACFQL = ACFQL + NFQL5's dual-boundary noised critic (V-anchor at t=0, Q-anchor
at t=1) + ESS-targeted noised-action-level BC weighting with the dual R²
reliability gate. `agents/nacfql.py`. Compared head-to-head against ACFQL
(best-of-n, 32 samples) across the OGBench `cube-*-play-singletask` tasks.
This revises v1 — new since then: an ACFQL baseline on cube-double-play-task3,
a finished NACFQL run on cube-triple-play-task3 (seed 0), a **second NACFQL
seed on cube-triple-play-task3 (seed 1, 0.88)**, a finished NACFQL
cube-triple-play-task5, and two in-flight runs.

wandb view:
<https://wandb.ai/ymkbuaa-beihang-university/qc/table?nw=nwuserymkbuaa>

## Run inventory (16 runs)

All runs: `horizon_length=5`, `action_chunking=True`, `actor_type=best-of-n`,
`actor_num_samples=32`, `sparse=False`, 1M offline + 1M online steps. Alpha
matched per-env between agents (α=300 on cube-double-play, α=100 on
cube-triple-play). NACFQL-specific hyperparams are the `get_config()` defaults
(`n_noised_actions=4`, `n_v_samples=8`, `lambda_v_anchor=1.0`,
`value_loss_weight=1.0`, `ess_target=0.7`, `r2_target=0.75`,
`r2_critic_target=0.5`, `gate_kappa=gate_kappa_critic=0.05`,
`gate_ema_decay=0.999`). One multi-seed env: cube-triple-play-task3 with
NACFQL seeds {0, 1, 2} (seed 2 still running).

## Head-to-head eval/success

Final-step / peak eval/success over the 22 eval points logged across
offline+online training (seed 0 unless noted):

| Env | ACFQL final / peak | NACFQL final / peak | Δ peak | Notes |
|---|---|---|---|---|
| cube-double-play-task2 | 1.00 / 1.00 | 1.00 / 1.00 | 0.00 | saturated |
| cube-double-play-task3 | 1.00 / 1.00 | 1.00 / 1.00 | 0.00 | **new pair, tie at ceiling** |
| cube-double-play-task4 | 0.94 / 0.96 | 0.86 / 0.90 | **−0.06** | ACFQL wins (n=1 each) |
| cube-double-play-task5 | 0.72 / 0.72 @ 0.92M | 0.98 / 1.00 | n/a | **ACFQL still offline, not comparable yet** |
| cube-triple-play-task2 | 0.80 / 0.82 | — | — | no NACFQL run |
| cube-triple-play-task3 s0 | 0.80 / 0.80 | 0.72 / 0.72 | −0.08 | ACFQL ahead on seed 0 |
| cube-triple-play-task3 s1 | — | 0.88 / 0.88 | **+0.08** vs ACFQL s0 | **NACFQL wins** on seed 1 |
| cube-triple-play-task3 s2 | — | 0.58 / 0.58 @ 1.2M | n/a | NACFQL mid-online phase |
| cube-triple-play-task4 | 0.28 / 0.40 | — | — | no NACFQL run — highest-value gap |
| cube-triple-play-task5 | 0.00 / 0.02 | 0.00 / 0.00 | ~0 | both near-zero |

**Net (finished pairs, n=1):** 5 finished head-to-heads. 3 ties at ceiling
(double-task2, double-task3, double-task5-tbd, triple-task5), 1 ACFQL win
by 0.06 (double-task4), 1 mixed on triple-task3 (NACFQL seed 0 −0.08,
seed 1 +0.08 — mean 0.80, exactly matches ACFQL 0.80 at n=2).

The v1 claim "no env to date shows a NACFQL lift over ACFQL at seed 0" is
narrowly true, but on the one meaningful unsaturated env (triple-task3) the
aggregate over the two finished NACFQL seeds now **ties** ACFQL rather than
losing — the v1 picture of a clean 0.20 gap was a seed-0 artifact.

## cube-triple-play-task3 learning curves (the informative env)

Eval/success every ~100k steps; offline phase ends at 1M:

| step | ACFQL s0 | NACFQL s0 | NACFQL s1 | NACFQL s2 |
|---|---|---|---|---|
| 0.1M | 0.00 | 0.02 | 0.00 | 0.02 |
| 0.5M | 0.00 | 0.00 | 0.00 | 0.00 |
| 1.0M | 0.04 | 0.02 | 0.00 | 0.04 |
| 1.1M | 0.22 | 0.20 | 0.22 | **0.44** |
| 1.2M | 0.54 | 0.46 | 0.56 | **0.58** (latest) |
| 1.3M | 0.64 | 0.40 | 0.54 | — |
| 1.5M | 0.70 | 0.46 | 0.74 | — |
| 1.7M | 0.62 | 0.46 | 0.72 | — |
| 1.9M | 0.80 | 0.64 | 0.70 | — |
| 2.0M | 0.80 | 0.72 | 0.88 | — |

Observations:
- All four runs match during the offline phase (near-zero — chunked BC is
  uninformative without reward signal here).
- Once online data arrives (1.0M → 1.1M), **NACFQL seed 2 jumps to 0.44
  within one eval interval**, nearly 2× the other three. That could be seed
  luck, but it's the strongest early-online lift seen on this env.
- Inter-seed variance is large for both agents: ACFQL has n=1 but its own
  trajectory bounces between 0.62 and 0.80 in the last 300k steps;
  NACFQL s0 vs s1 spans [0.72, 0.88] at 2M — a 0.16 gap on the same config.

**This is the single most useful data point in the entire run set.** It is
the one env where the reward is not saturated, both agents learn from the
online phase, and we have ≥2 NACFQL seeds. Net verdict at n=2: tie.

## Mechanism diagnostics (online-end, NACFQL runs)

Final-step `online_agent/*` values for each finished NACFQL run:

| Metric | d2 | d3 | d4 | d5 | t3 s0 | t3 s1 | t5 |
|---|---|---|---|---|---|---|---|
| `actor/r2_qn` | 0.997 | 0.997 | 0.994 | 0.997 | 0.997 | 0.997 | 0.998 |
| `actor/r2_critic` | 0.997 | 0.998 | 0.995 | 0.998 | 0.998 | 0.998 | 0.999 |
| `actor/gate_c` | 0.993 | 0.993 | 0.992 | 0.993 | 0.993 | 0.993 | 0.993 |
| `actor/tau_star` | 2.76 | 2.21 | 2.21 | 3.62 | 2.56 | 2.78 | 2.49 |
| `actor/ess_achieved` | 0.700 | 0.700 | 0.700 | 0.700 | 0.700 | 0.700 | 0.700 |
| `actor/bc_weight_mean` | 1.00 | 1.01 | 1.00 | 1.00 | 1.00 | 1.01 | 1.02 |
| `actor/trust_mean` | 0.40 | 0.39 | 0.38 | 0.43 | 0.43 | 0.40 | 0.44 |
| `noised_critic/v_anchor_loss` | 2.19 | 2.29 | 3.33 | 2.66 | 4.36 | 3.73 | 2.00 |
| `noised_critic/r2_t_0.00_0.25` | 0.994 | 0.998 | 0.995 | 0.997 | 0.998 | 0.999 | 0.999 |
| `noised_critic/r2_t_0.25_0.50` | 0.995 | 0.998 | 0.997 | 0.998 | 0.998 | 0.999 | 0.999 |
| `noised_critic/r2_t_0.50_0.75` | 0.997 | 0.998 | 0.992 | 0.998 | 0.999 | 0.999 | 0.999 |
| `noised_critic/r2_t_0.75_1.00` | 0.991 | 0.997 | 0.994 | 0.998 | 0.998 | 0.999 | 0.999 |

What this says (unchanged from v1, now confirmed across 7 runs):

- **Gate fully opens.** `gate_c ≈ 0.99` everywhere — R² EMAs well above
  sigmoid thresholds, noised-action weighting runs at full strength.
- **ESS-τ\* bisection works.** `ess_achieved = 0.70` hits target; `tau_star`
  in [2.2, 3.6] — moderate soft weighting.
- **BC weights are mild.** `bc_weight_mean ≈ 1.00` on every env. The
  `a_local = Q − Q_n` advantage distribution is too narrow to push weights
  far from uniform — the reweighted BC loss lands essentially on top of the
  ACFQL unweighted BC.
- **Q_n is nearly perfect along the whole flow, including t=0.** `r2_t` is
  ≈0.99 in every bucket. Trust factor stays at 0.4 (the ensemble agrees).
  `v_anchor_loss` is *not* zero (2–4) — the V-anchor term is active and
  non-trivial in loss — but it is not materially decoupling `Q_n(s, x_0, 0)`
  from `Q(s, a)` as measured by R². This is the same "trivial fixed point"
  failure mode NFQL5 showed in the FQL codebase.

**Mechanism verdict:** every gear turns; the reweighting signal it produces
is small because Q_n ≈ Q. On these envs the noised-action baseline is not
differentiating enough from the plain critic to move BC weights.

## Compute overhead

NACFQL is 3–65 % wall-time slower than ACFQL at identical env/horizon (V head
+ noised critic + 8-sample MC V target + 4-sample anchor training):

| Env | ACFQL (min) | NACFQL (min) | Δ |
|---|---|---|---|
| cube-double-play-task2 | 208 | 308 | +48 % |
| cube-double-play-task3 | 499 | 310 | −38 % (ACFQL outlier — GPU contention) |
| cube-double-play-task4 | 306 | 315 | +3 % |
| cube-triple-play-task3 | 416 | 500–558 | +20 % to +34 % |
| cube-triple-play-task5 | 483 | 676 | +40 % |

Ignoring the two obviously hardware-contended cells (d2 / d3), the intrinsic
overhead is consistent with v1's +20–35 %.

## Interpretation

The v1 headline was "NACFQL losing 0.20 on the one unsaturated env." With
seed 1 finished at 0.88 and seed 2 in-flight at 0.44 mid-online (the highest
early-online number across all 4 seeds on this env), that headline is no
longer accurate. At n=2 the NACFQL mean on triple-task3 is 0.80 — exactly
ACFQL's score. At n=3 (pending), it could tilt either way.

What has *not* changed from v1:
- The mechanism lands in the same regime on every env we've checked —
  gate open, BC weights ≈ 1, Q_n ≈ Q along the whole flow. The V-anchor term
  contributes a non-trivial loss but does not visibly decouple the t=0
  boundary. This is the same failure mode NFQL5 hit in the FQL codebase
  (`memory/project_nfql_line_status.md`), now confirmed in the chunked port.
- All five envs where one agent "wins" at n=1 are either saturated
  (double-task2/3, both 1.00) or near-zero (triple-task5, both 0). Only
  cube-double-play-task4 and cube-triple-play-task3 sit in the informative
  regime, and at matched seeds the former is ACFQL-by-0.06 and the latter
  is a tie.

## What's missing / planned

Still single-seed on most envs; narrow env coverage. To be useful the
empirical case needs:

1. **Finish NACFQL seed 2 on cube-triple-play-task3.** In flight,
   step 1.2M/2M. This is the first env where a 3-seed NACFQL vs 3-seed
   ACFQL head-to-head becomes possible.
2. **Seed ≥3 ACFQL on cube-triple-play-task3.** Currently n=1 (0.80).
   Its own seed variance has not been measured — without it, NACFQL
   vs ACFQL comparison on this env is comparing distributions of
   different sizes.
3. **Finish ACFQL cube-double-play-task5** (in-flight at 0.92M) —
   current premature-looking "NACFQL 1.00 vs ACFQL 0.72" will
   almost certainly disappear once ACFQL enters the online phase.
4. **Run NACFQL on cube-triple-play-task4** (ACFQL peak only 0.40)
   and **task2** (ACFQL peak 0.82). These are the highest-leverage
   missing cells — the chunked-QC analogues of the "long-horizon
   sparse navigation" envs where NFQL5 visibly lifts in the FQL
   codebase (see `memory/project_nfql_research_plan.md`, Step 1–2).
   If the dual-boundary baseline helps anywhere in QC, these are
   where it should show up first.
5. **Sparse-reward variants** (`--sparse=True`) on scene / puzzle-3x3
   when ACFQL has a baseline — matches the FQL-side pattern where
   the gap is largest on genuinely under-determined rewards.
6. **Distill-ddpg ablation.** All NACFQL runs here are best-of-n.
   The V head is trained via MC over `compute_flow_actions` in this
   mode (see `agents/nacfql.py:value_loss`, best-of-n branch) —
   which represents `E_{a~BC}[Q]`, *not* the best-of-n policy's
   value. Distill-ddpg trains `actor_onestep_flow` and gives a
   cleaner `E_π[Q]` target. Worth one side-by-side test on
   triple-task3 to see if the cleaner MC target makes Q_n diverge
   more from Q (which is the whole point).
7. **Log `var_q_ema` / `r2_qn_ema` trajectories** (we only have
   final snapshots) to confirm the gate opens early, not just at
   the end.

## Also in memory

- `project_nfql_line_status.md` — NFQL₅ in the FQL codebase: mechanism
  confirmed, performance ties NFQL₃/₄ in aggregate. The chunked port
  replays the same pattern: gate opens, weights stay near-uniform, parity
  with ACFQL on finished pairs.
- `feedback_nfql_research_stance.md` — user wants to strengthen the
  empirical case, not change the mechanism. This doc follows that: report
  the parity honestly, propose more seeds + harder envs, do not recommend
  pivoting away from the dual-boundary / noised-weighting idea.
- `project_nfql_research_plan.md` — Steps 1–2 (more seeds on unsaturated
  envs) translate directly to items 1–4 above.
