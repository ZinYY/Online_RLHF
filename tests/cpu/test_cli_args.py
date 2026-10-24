"""Five real CLI parameter blocks; no module startup or training integration."""

import sys
from pathlib import Path

import pytest

from tests.support.cli_args import parse_cli_block

CLI_DIR = Path(__file__).resolve().parents[2] / "openrlhf" / "cli"
CLI_FILES = [
    "train_rm_head_hvp.py",
    "train_rm_hvp.py",
    "active_train_rm_head_hvp.py",
    "online_train_rm_head_hvp.py",
    "online_train_rm_hvp.py",
]


@pytest.mark.parametrize("filename", CLI_FILES)
@pytest.mark.parametrize(
    ("weight", "expected"),
    [(None, 0.37), ("0.0", 0.0), ("0.3", 0.3), ("1.0", 1.0)],
    ids=["omitted", "zero", "intermediate", "one"],
)
def test_cli_mixing_weight(filename, weight, expected, monkeypatch):
    argv = [filename, "--damping", "0.37"]
    if weight is not None:
        argv.extend(["--cg_mixing_weight", weight])
    monkeypatch.setattr(sys, "argv", argv)
    parser, args = parse_cli_block(CLI_DIR / filename)
    assert parser.get_default("cg_mixing_weight") is None
    assert args.damping == 0.37
    assert isinstance(args.cg_mixing_weight, float)
    assert args.cg_mixing_weight == expected
