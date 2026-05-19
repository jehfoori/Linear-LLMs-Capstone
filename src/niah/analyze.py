from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .data import file_sha256, read_jsonl, write_json
from .metrics import accuracy_summary, failure_summary, position_summary


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_run_predictions(run_dir: str | Path) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    jsonl_path = run_dir / "predictions.jsonl"
    if jsonl_path.exists():
        return read_jsonl(jsonl_path)
    csv_path = run_dir / "predictions.csv"
    if csv_path.exists():
        return _coerce_prediction_rows(read_csv(csv_path))
    raise FileNotFoundError(f"No predictions.jsonl or predictions.csv in {run_dir}")


def analyze_run(run_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    run_dir = Path(run_dir)
    rows = load_run_predictions(run_dir)
    summary = accuracy_summary(rows)
    failures = failure_summary(rows)
    positions = position_summary(rows)

    write_csv(run_dir / "summary.csv", summary)
    write_csv(run_dir / "failure_summary.csv", failures)
    write_csv(run_dir / "position_summary.csv", positions)

    return {
        "summary": summary,
        "failure_summary": failures,
        "position_summary": positions,
    }


def compare_runs(run_dirs: list[str | Path], out_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    manifests = []
    dataset_hashes = set()
    for run_dir in map(Path, run_dirs):
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        manifests.append({"run_dir": str(run_dir), "manifest": manifest})
        if manifest.get("dataset_sha256"):
            dataset_hashes.add(manifest["dataset_sha256"])

        rows = load_run_predictions(run_dir)
        all_rows.extend(rows)

    if len(dataset_hashes) > 1:
        raise ValueError(f"Refusing to compare runs with different dataset hashes: {sorted(dataset_hashes)}")

    accuracy = accuracy_summary(all_rows)
    failures = failure_summary(all_rows)
    positions = position_summary(all_rows)

    write_csv(out_dir / "accuracy_table.csv", accuracy)
    write_csv(out_dir / "failure_summary.csv", failures)
    write_csv(out_dir / "position_summary.csv", positions)
    write_json(
        out_dir / "manifest.json",
        {
            "run_dirs": [str(path) for path in run_dirs],
            "dataset_sha256": next(iter(dataset_hashes), None),
            "source_manifests": manifests,
        },
    )
    return {
        "accuracy_table": accuracy,
        "failure_summary": failures,
        "position_summary": positions,
    }


def write_predictions_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    flattened = []
    for row in rows:
        clean = dict(row)
        for key, value in list(clean.items()):
            if isinstance(value, (dict, list)):
                clean[key] = json.dumps(value, sort_keys=True)
        flattened.append(clean)
    write_csv(path, flattened)


def dataset_manifest(dataset_path: str | Path) -> dict[str, Any]:
    rows = read_jsonl(dataset_path)
    return {
        "dataset_path": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "num_examples": len(rows),
        "target_lengths": sorted({row.get("target_length") for row in rows}),
        "tasks": sorted({row.get("task") for row in rows}),
    }


def _coerce_prediction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        if "correct" in row:
            row["correct"] = str(row["correct"]).lower() in {"true", "1", "yes"}
        for key in ["target_length", "input_tokens", "new_tokens"]:
            if key in row and row[key] not in ("", None):
                row[key] = int(float(row[key]))
        for key in ["elapsed_sec", "peak_mem_gib", "needle_position_fraction"]:
            if key in row and row[key] not in ("", None):
                row[key] = float(row[key])
    return rows
