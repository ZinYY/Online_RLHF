"""End-to-end characterization tests for ``convert_to_preference_dataset``.

Scope: the real ``openrlhf/utils/convert_to_dataset.py`` conversion pipeline
with tiny JSONL fixtures in a tmp directory. The four index-selection helpers
are covered separately in test_pair_selection.py.

Boundary: ``common_inputs`` is a ``set`` intersection and the conversion
loop iterates it directly, so OUTPUT ROW ORDER IS NON-DETERMINISTIC. Tests
therefore assert sets/counts/scores keyed by prompt, never row order.
"""

import json

import pytest


def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def _eval_row(input_text, outputs, rewards):
    return {"input": input_text, "outputs": outputs, "rewards": rewards}


def test_conversion_produces_valid_preference_pairs(
    legacy_convert_to_dataset, tmp_path
):
    eval_file = tmp_path / "eval.jsonl"
    true_file = tmp_path / "true.jsonl"
    out_file = tmp_path / "out.jsonl"
    # Two common inputs; eval rewards select best/worst; true rewards agree
    # with the eval ordering (no swap expected).
    _write_jsonl(
        eval_file,
        [
            _eval_row("prompt A", ["a0", "a1", "a2"], [0.2, 0.9, 0.5]),
            _eval_row("prompt B", ["b0", "b1"], [0.4, 0.6]),
            _eval_row("eval only", ["x0", "x1"], [0.1, 0.2]),  # not in true file
        ],
    )
    _write_jsonl(
        true_file,
        [
            _eval_row("prompt A", ["a0", "a1", "a2"], [1.0, 9.0, 5.0]),
            _eval_row("prompt B", ["b0", "b1"], [4.0, 6.0]),
            _eval_row("true only", ["y0", "y1"], [0.3, 0.7]),  # not in eval file
        ],
    )

    legacy_convert_to_dataset.convert_to_preference_dataset(
        str(eval_file), str(true_file), str(out_file)
    )

    rows = _read_jsonl(out_file)
    assert len(rows) == 2  # the non-common inputs are excluded
    by_prompt = {row["prompt"]: row for row in rows}
    assert set(by_prompt) == {"prompt A", "prompt B"}

    row_a = by_prompt["prompt A"]
    # best/worst on eval rewards: a1 chosen, a0 rejected; true scores agree.
    assert row_a["chosen"][-1]["content"] == "a1"
    assert row_a["rejected"][-1]["content"] == "a0"
    assert row_a["chosen_score"] == 9.0
    assert row_a["rejected_score"] == 1.0
    for row in rows:
        assert row["chosen_score"] >= row["rejected_score"]
        assert row["chosen"][0]["role"] == "user"
        assert row["chosen"][1]["role"] == "assistant"
        assert row["rejected"][1]["role"] == "assistant"


def test_conversion_swaps_when_true_scores_disagree(
    legacy_convert_to_dataset, tmp_path, capsys
):
    eval_file = tmp_path / "eval.jsonl"
    true_file = tmp_path / "true.jsonl"
    out_file = tmp_path / "out.jsonl"
    # Eval rewards rank o1 best and o0 worst, but TRUE scores are opposite:
    # the pipeline must swap so chosen_score >= rejected_score.
    _write_jsonl(eval_file, [_eval_row("prompt S", ["o0", "o1"], [0.1, 0.9])])
    _write_jsonl(true_file, [_eval_row("prompt S", ["o0", "o1"], [8.0, 2.0])])

    legacy_convert_to_dataset.convert_to_preference_dataset(
        str(eval_file), str(true_file), str(out_file)
    )

    (row,) = _read_jsonl(out_file)
    assert row["chosen"][-1]["content"] == "o0"  # swapped
    assert row["rejected"][-1]["content"] == "o1"
    assert row["chosen_score"] == 8.0
    assert row["rejected_score"] == 2.0
    assert "Swapped 1 pairs" in capsys.readouterr().out


def test_conversion_skips_outputs_missing_from_true_file(
    legacy_convert_to_dataset, tmp_path, capsys
):
    eval_file = tmp_path / "eval.jsonl"
    true_file = tmp_path / "true.jsonl"
    out_file = tmp_path / "out.jsonl"
    # "gone" is selected by eval rewards but absent from the true outputs:
    # the pipeline warns and skips this input entirely.
    _write_jsonl(eval_file, [_eval_row("prompt M", ["keep", "gone"], [0.9, 0.1])])
    _write_jsonl(true_file, [_eval_row("prompt M", ["keep"], [5.0])])

    legacy_convert_to_dataset.convert_to_preference_dataset(
        str(eval_file), str(true_file), str(out_file)
    )

    assert _read_jsonl(out_file) == []
    assert "Warning: Output not found" in capsys.readouterr().out


def test_conversion_rejects_unknown_strategy(legacy_convert_to_dataset, tmp_path):
    eval_file = tmp_path / "eval.jsonl"
    true_file = tmp_path / "true.jsonl"
    out_file = tmp_path / "out.jsonl"
    _write_jsonl(eval_file, [_eval_row("p", ["a", "b"], [0.1, 0.2])])
    _write_jsonl(true_file, [_eval_row("p", ["a", "b"], [1.0, 2.0])])

    with pytest.raises(ValueError, match="Unknown strategy"):
        legacy_convert_to_dataset.convert_to_preference_dataset(
            str(eval_file), str(true_file), str(out_file), strategy="not_a_strategy"
        )
