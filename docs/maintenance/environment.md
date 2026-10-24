# CPU validation environment

These checks cover isolated CPU behavior, not the authors' full training
environment. Reproduction commands are in [CONTRIBUTING](../../CONTRIBUTING.md).

## Recorded environment

| Component | Historical local version |
| --- | --- |
| Platform | Linux x86_64 |
| Python | CPython 3.12.3 |
| PyTorch | 2.5.1+cpu |
| NumPy | 2.5.3 |
| pytest | 9.1.1 |
| tqdm | 4.70.0 |
| Ruff (separate environment) | 0.16.7 |

Install PyTorch from the CPU wheel index before requirements/cpu-test.txt.
The dependency files pin maintenance tools and direct CPU dependencies; they
do not lock the complete training stack or all transitive dependencies.

## Historical local evidence

The recorded 2026-09-18 local run reported **113 passed in 0.71s**, with no
failures, skips or xfails, using:

```bash
python -m pytest tests/cpu -q -ra --strict-markers
ruff check tests/
ruff format --check tests/
python -m pip check
```

These commands were run in the dedicated CPU/lint environments. All exited 0;
Ruff reported all checks passed and 13 files already formatted, and pip check
reported no broken requirements in either environment. These are historical
results preserved in the development snapshot, not tests rerun during material
preparation.

The 113 figure counts parameterized test instances, not independent behaviors
or project-wide coverage. It includes 20 CLI parameter-block instances and
30 trainer CG instances. CLI parameter-block tests are not full module startup
or training integration tests.

## Verified source-commit CI

The GitHub API was checked on 2026-10-23 for
[cpu-checks run 35412445760](https://github.com/Gary-CF/Online_RLHF/actions/runs/35412445760).
Its head SHA is **eca88b8d76460eb9dba76cd6ec5bd6eb1eeaef00** and its state is
completed/success.

| Job | Result | Start (UTC, 2026-09-19) | End (UTC) |
| --- | --- | --- | --- |
| Lint new maintenance code | success | 01:22:25 | 01:22:32 |
| CPU pytest (isolated from training stack) | success | 01:22:26 | 01:22:56 |

The run was created at 01:22:23 UTC and updated at 01:22:57 UTC. This verifies
CI for the source commit, not a future PR commit or these edited documents.
The API job status does not independently establish the historical local
test duration or count.

## Material-preparation checks

Preparation checked the source/base commit relationship, current remote refs,
the delivery manifest, relative document links, dependency-version claims,
CI SHA and per-file byte equality against the source commit. Only
CONTRIBUTING.md and these two maintenance documents were edited in the external
delivery files; code, tests, CI, dependencies, configuration and README were
preserved byte-for-byte. No CPU tests, Ruff checks or pip checks were rerun
during preparation.

## Not verified

GPU training, distributed/DeepSpeed ZeRO execution, FlashAttention, complete CLI
startup, README training commands, training-performance gains and complete
training-environment reproducibility are not established by this evidence.
The isolation adapter uses controlled package shells and a fail-fast DeepSpeed
sentinel; only the full-model GatheredParameters context is replaced for CPU
tests. Gradients, HVPs and CG outputs come from the real implementations.
