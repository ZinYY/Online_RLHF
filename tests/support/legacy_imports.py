"""Isolated import adapter for the legacy OpenRLHF modules under test.

The production package aggregators (``openrlhf.models.__init__``,
``openrlhf.utils.__init__``) pull in the heavy training stack (transformers,
deepspeed, ...), which is intentionally NOT installed for the CPU tests.
This adapter loads the real production modules by file path from this
checkout, behind controlled package shells, so the code under test is the
checked-out implementation, not a copy.

Boundary rules:
- Adapter-owned namespaces: ``openrlhf`` and everything under ``openrlhf.*``,
  plus the explicitly replaced ``deepspeed``. Only these names are saved,
  isolated and restored; third-party modules imported inside the context
  (torch, numpy, ...) are left alone.
- Pre-existing ``openrlhf.*`` entries are removed from ``sys.modules`` on
  entry so they cannot shadow this checkout's modules, and are restored on
  exit.
- ``deepspeed`` is replaced by a fail-fast sentinel with normal module
  metadata (``__file__``/``__spec__``): metadata reads succeed, any real
  training API access raises immediately.
- All owned names are restored on normal AND exceptional exit.
- This is an isolation mechanism for unit tests, NOT a normal package
  import path and NOT a GPU integration test.
"""

import contextlib
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OPENRLHF_DIR = REPO_ROOT / "openrlhf"

_OWNED_PREFIXES = ("openrlhf", "deepspeed")


def _is_owned(name):
    return any(
        name == prefix or name.startswith(prefix + ".") for prefix in _OWNED_PREFIXES
    )


_TRAINER_FILES = {
    "rm": "trainer/rm_trainer_head_hvp.py",
    "rm_active": "trainer/rm_active_trainer_head_hvp.py",
    "rm_hvp": "trainer/rm_trainer_hvp.py",
}

_MISSING = object()


class _FailFastModule(types.ModuleType):
    """Module sentinel: importing succeeds, metadata reads succeed, any real
    training API use fails immediately (never returns fake results)."""

    def __init__(self, name):
        super().__init__(name)
        # Normal module metadata so stdlib tooling (inspect, importlib)
        # can read it without touching the training stack.
        self.__file__ = None
        self.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)

    def __getattr__(self, item):
        raise RuntimeError(
            f"'{self.__name__}.{item}' was touched by a CPU test. The real "
            f"'{self.__name__}' training stack is not installed; tests must not "
            "route around this with mocks or fake computations."
        )


def _package_shell(name, path):
    module = types.ModuleType(name)
    module.__file__ = str(path / "__init__.py")
    module.__path__ = [str(path)]
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    module.__spec__.submodule_search_locations = module.__path__
    return module


def _module_from_checkout(dotted_name, relpath):
    path = (OPENRLHF_DIR / relpath).resolve()
    # Path-relationship check BEFORE execution: the target must be a real
    # file inside this checkout's openrlhf/ tree.
    if not path.is_file() or not path.is_relative_to(OPENRLHF_DIR.resolve()):
        raise RuntimeError(f"legacy module path escapes this checkout: {path}")
    spec = importlib.util.spec_from_file_location(dotted_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Post-execution verification: the module really came from the expected file.
    if module.__file__ is None or Path(module.__file__).resolve() != path:
        raise RuntimeError(
            f"legacy module {dotted_name} resolved to {module.__file__}, "
            f"expected {path}"
        )
    sys.modules[dotted_name] = module
    return module


@contextlib.contextmanager
def isolated_legacy_imports():
    """Isolate the adapter-owned namespaces for the duration of the block.

    Entry: original objects of every owned name present in ``sys.modules``
    are saved; pre-existing ``openrlhf*`` entries are removed so this
    checkout's modules are the ones actually loaded; ``deepspeed`` is
    replaced by the fail-fast sentinel.

    Exit (normal or exceptional): every owned name introduced by the block
    is dropped, then the original objects are put back. Third-party modules
    imported inside the block are NOT touched.
    """
    saved = {name: module for name, module in sys.modules.items() if _is_owned(name)}
    try:
        for name in list(saved):
            del sys.modules[name]
        sys.modules["deepspeed"] = _FailFastModule("deepspeed")
        sys.modules["openrlhf"] = _package_shell("openrlhf", OPENRLHF_DIR)
        sys.modules["openrlhf.models"] = _package_shell(
            "openrlhf.models", OPENRLHF_DIR / "models"
        )
        sys.modules["openrlhf.utils"] = _package_shell(
            "openrlhf.utils", OPENRLHF_DIR / "utils"
        )
        sys.modules["openrlhf.trainer"] = _package_shell(
            "openrlhf.trainer", OPENRLHF_DIR / "trainer"
        )
        yield
    finally:
        for name in [name for name in sys.modules if _is_owned(name)]:
            del sys.modules[name]
        sys.modules.update(saved)


def load_legacy_loss():
    """Load the real ``openrlhf/models/loss.py`` and expose it on the shell."""
    loss = _module_from_checkout("openrlhf.models.loss", "models/loss.py")
    models_shell = sys.modules["openrlhf.models"]
    models_shell.PairWiseLoss = loss.PairWiseLoss
    models_shell.LogExpLoss = loss.LogExpLoss
    return loss


def load_trainer(kind):
    """Load one of the legacy head-HVP reward-model trainer modules."""
    if kind not in _TRAINER_FILES:
        raise ValueError(f"unknown trainer kind: {kind!r}")
    load_legacy_loss()
    return _module_from_checkout(
        f"openrlhf.trainer.{Path(_TRAINER_FILES[kind]).stem}", _TRAINER_FILES[kind]
    )


def load_convert_to_dataset():
    """Load the real ``openrlhf/utils/convert_to_dataset.py``."""
    return _module_from_checkout(
        "openrlhf.utils.convert_to_dataset", "utils/convert_to_dataset.py"
    )


def load_rm_score_selection():
    """Load the real ``openrlhf/utils/rm_score_selection.py``."""
    return _module_from_checkout(
        "openrlhf.utils.rm_score_selection", "utils/rm_score_selection.py"
    )


def make_trainer_shell(trainer_module, **state):
    """Build a trainer instance without running its constructor.

    Only the attributes touched by ``hessian_vector_product``,
    ``conjugate_gradient_solver`` and ``get_current_damping`` are injected.
    """
    defaults = {
        "cg_damping": 0.1,
        "damping_strategy": "",
        "damping_growth_rate": 5.0,
        "total_steps": 0,
        "total_T": 1,
        "args": types.SimpleNamespace(damping=0.0),
    }
    defaults.update(state)
    trainer = trainer_module.RewardModelTrainer.__new__(
        trainer_module.RewardModelTrainer
    )
    for key, value in defaults.items():
        setattr(trainer, key, value)
    return trainer


@contextlib.contextmanager
def deepspeed_zero_gather_passthrough():
    """Scoped test-side pass-through for ``deepspeed.zero.GatheredParameters``.

    Only the FULL-MODEL trainer's production HVP wraps the real autograd math
    in this parameter-gathering gate (a ZeRO sharding helper). Inside this
    context the gate is a no-op: no computation is mocked and no fake result
    is returned — on real multi-GPU ZeRO this call would gather sharded
    parameters, which does not exist on the CPU fixture. Every other
    deepspeed attribute access still fails fast.
    """
    sentinel = sys.modules["deepspeed"]
    previous = sentinel.__dict__.get("zero", _MISSING)

    class _Zero:
        @staticmethod
        @contextlib.contextmanager
        def GatheredParameters(params, modifier_rank=0):
            yield

    sentinel.zero = _Zero()
    try:
        yield
    finally:
        if previous is _MISSING:
            sentinel.__dict__.pop("zero", None)
        else:
            sentinel.zero = previous


def trainer_hvp_gate(kind):
    """Context manager under which the given trainer kind's HVP may run.

    The full-model trainer production code calls deepspeed's
    GatheredParameters gate; the head trainers do not.
    """
    if kind == "rm_hvp":
        return deepspeed_zero_gather_passthrough()
    return contextlib.nullcontext()
