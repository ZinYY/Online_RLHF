# Maintenance review

Source snapshot: eca88b8d76460eb9dba76cd6ec5bd6eb1eeaef00.
Paper-repository maintenance baseline: a5a813106fb6501d32a7d7c9ed2752e36901673e.
This baseline is not an established historical fork-base relative to OpenRLHF.

## Fletcher-Reeves correction

The conjugate-gradient solvers in rm_trainer_head_hvp.py,
rm_active_trainer_head_hvp.py and rm_trainer_hvp.py now retain the previous
squared residual and compute beta = new / old. Previously both operands used
the updated residual, making beta equal to one whenever its denominator was
nonzero.

The CPU regressions exercise the real solvers through autograd on SPD systems.
Historical measurements on the four-dimensional logistic fixture
(seed 7, cg_damping=0.2, mixing=0) show a four-step relative residual changing
from 0.231158 to approximately 4.493e-6, with post-fix relative solution error
7.672e-6. A separate rotated quadratic-system regression checks convergence
with conjugate directions. These are fixture-specific CPU results, not evidence
of improved training performance. Standard CG convergence assumes an SPD
operator; an arbitrary training Hessian need not satisfy that assumption.

## Explicit mixing control

The five direct entry points accept --cg_mixing_weight as a float with default
None: train_rm_head_hvp.py, train_rm_hvp.py, active_train_rm_head_hvp.py,
online_train_rm_head_hvp.py and online_train_rm_hvp.py.

After parsing, an explicit None check falls back to damping; zero is preserved.
For max_iter > 1, the solver returns:

```text
x = w * grad + (1 - w) * x_cg
```

A one-step budget does not mix. Missing/None and explicit damping are compared
on the same corrected solver; equality does not claim bitwise equivalence with
the pre-fix algorithm. The default mixing rule is retained, but the beta fix
changes multi-step output. A coefficient of 0.8 is not a guarantee that 80% of
the final vector norm comes from the gradient.

CLI tests execute the actual argparse constructor, argument definitions, parsing
and fallback extracted from source. For damping=0.37, all five entries cover
omitted, 0.0, 0.3 and 1.0 weights. Trainer tests cover missing/None fallback,
explicit zero, intermediate blending and single-step independence from weight.
Historical reverse checks caught both a removed flag and an incorrect or
fallback. These are parameter-block and numerical tests, not complete CLI
startup or end-to-end training tests.

The parsed namespace flows through get_strategy(args), strategy.args and trainer
self.args. In particular, online_train_rm_hvp.py currently imports
rm_trainer_head_hvp.RewardModelTrainer; this import is preserved. Pipeline
wrappers do not forward the new flag. Zero mixing only controls output when
CG is called; it does not bypass the head trainer's schedule gate.

## CPU maintenance and score compatibility

The suite covers HVP, CG, schedules, selection/conversion helpers, score aliases
and isolation behavior. Modules are loaded by file path behind controlled package
shells. A fail-fast DeepSpeed sentinel rejects unexpected training API access;
only the full-model GatheredParameters gate uses an empty context. Gradients,
HVPs and CG outputs are neither mocked nor replaced with copied algorithms.
Tests preserve Python and torch CPU random state.

CI runs CPU tests separately from pinned Ruff lint/format checks, with dependency
checks and version summaries. Optional hooks select Python files under tests/.
See the [environment record](environment.md) for historical local results,
verified source-commit CI and the checks performed during preparation.

The scoring dispatcher accepts apo as an alias for uncertainty_score. This is
an API expansion, not strictly zero behavior change. The squared embedding
distance is named one_step_embedding_l2_score; one_step_fisher_score remains
a compatibility alias without a runtime deprecation warning. Pipeline score
choices still reject apo. Scoring numerics are unchanged by these naming changes.

## Known numerical and integration limits

- residual_tol compares the squared residual, so 1e-10 corresponds to a residual
  norm threshold of 1e-5. The existing stopping semantics are documented and
  tested, not changed.
- The fixed +1e-8 in alpha's denominator is scale-sensitive. In the recorded
  float64 one-parameter quadratic experiment, theta=0,
  loss=0.5*theta.square().sum()-1e-6*theta.sum(), cg_damping=0, mixing=0 and
  residual_tol=1e-30, the exact solution is -1e-6. All three solvers returned
  approximately -9.99900010e-11, -5.99251181e-10 and -5.42337651e-9 after
  1, 3 and 10 steps. The same scale sensitivity existed before the beta fix.
  This is not a new accuracy contract or proof of training harm; epsilon and
  tolerances remain unchanged.
- HVP regularization uses the constructor-time cg_damping. Schedule formulas,
  time normalization and HVP wiring are separate issues. For growth rate > 1,
  normalized time uses total_steps / damping_growth_rate rather than total_T.
  Head trainers gate HVP on use_hvp and current_damping < 1; the full-model
  trainer uses the scheduled value for logging without the same gate. Cosine
  can fall below one again and re-enter HVP. No schedule or gate was changed.
- Selection retains the fixed fourth-best index, global Python randomness and
  set-based conversion ordering. No new ordering or randomness guarantee is made.

Changes to these numerical or integration contracts require separate decisions.
Historical experiment impact remains unquantified; full training was not
validated by this maintenance work.
