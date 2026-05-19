import json

import pytest

from niah.analyze import compare_runs
from niah.data import write_json, write_jsonl


def test_compare_refuses_different_dataset_hashes(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    write_json(left / "manifest.json", {"dataset_sha256": "aaa"})
    write_json(right / "manifest.json", {"dataset_sha256": "bbb"})
    write_jsonl(left / "predictions.jsonl", [{"example_id": "a", "model_label": "A", "target_length": 1, "correct": True, "failure_type": "correct"}])
    write_jsonl(right / "predictions.jsonl", [{"example_id": "a", "model_label": "B", "target_length": 1, "correct": True, "failure_type": "correct"}])

    with pytest.raises(ValueError):
        compare_runs([left, right], tmp_path / "out")


def test_compare_writes_outputs_for_matching_hashes(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    for run_dir, label in [(left, "A"), (right, "B")]:
        write_json(run_dir / "manifest.json", {"dataset_sha256": "aaa"})
        write_jsonl(
            run_dir / "predictions.jsonl",
            [
                {
                    "example_id": "ex1",
                    "model_label": label,
                    "target_length": 1024,
                    "correct": True,
                    "failure_type": "correct",
                    "needle_position_fraction": 0.5,
                }
            ],
        )

    outputs = compare_runs([left, right], tmp_path / "out")
    assert len(outputs["accuracy_table"]) == 2
    assert (tmp_path / "out" / "accuracy_table.csv").exists()
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["dataset_sha256"] == "aaa"
