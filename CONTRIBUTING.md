# Contributing

## CPU and lint checks

The CPU suite exercises real reward-model numerics through isolated file-path
imports. Separate tests execute the five HVP CLI argparse blocks without loading
models or starting training. See the [validation record](docs/maintenance/environment.md)
and [maintenance review](docs/maintenance/review.md) for evidence and limitations.

From the repository root, create dedicated environments outside the checkout.
Do not install the training requirements or use conda base for these checks.

```bash
python3.12 -m venv ../online-rlhf-cpu-venv
../online-rlhf-cpu-venv/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
../online-rlhf-cpu-venv/bin/python -m pip install -r requirements/cpu-test.txt
../online-rlhf-cpu-venv/bin/python -m pip check
../online-rlhf-cpu-venv/bin/python -m pytest tests/cpu -q -ra --strict-markers
python3.12 -m venv ../online-rlhf-lint-venv
../online-rlhf-lint-venv/bin/python -m pip install -r requirements/lint.txt
../online-rlhf-lint-venv/bin/python -m pip check
../online-rlhf-lint-venv/bin/ruff check tests/
../online-rlhf-lint-venv/bin/ruff format --check tests/
```

Pins cover maintenance tools and direct CPU dependencies, not the entire training
stack or all transitive dependencies. Keep formatting limited to tests/.

## Optional hooks

The local system hooks select Python files under tests/. To use them, activate
the lint environment or put its bin directory first on PATH. An absolute
pre-commit executable path alone does not select the Ruff used by system hooks.

```bash
source ../online-rlhf-lint-venv/bin/activate
python -m pip install pre-commit
ruff --version
pre-commit install
pre-commit run --all-files
```

Ruff should match requirements/lint.txt. The lint hook can apply fixes; inspect
the resulting diff. Hook installation is optional.

## Interpreting failures and compatibility

The recorded suite has no failures, skips or xfails. Diagnose the environment,
loaded implementation, numerical path and source changes before attributing a
regression to a beta revert. Do not weaken tolerances without evidence or hide
unexpected failures with xfail. Reserve strict xfail for an agreed, reproducible
known defect; investigate unexpected passes.

All five direct HVP CLIs accept --cg_mixing_weight. Missing/None values retain
damping-based mixing; explicit zero remains zero. The beta fix still changes
multi-step numerical output. A zero weight does not bypass the head trainer's
schedule gate, and pipeline wrappers do not forward the new option.

The apo score key is an additional accepted API value; pipeline choices still
reject it. one_step_fisher_score remains a compatibility alias for
one_step_embedding_l2_score, without a runtime deprecation warning.
Changes to numerical contracts, schedules or training defaults need separate
discussion and evidence; existing behavior and limitations are summarized in
the maintenance review.
