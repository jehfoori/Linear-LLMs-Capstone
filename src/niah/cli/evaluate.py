from __future__ import annotations

import argparse
import json
from pathlib import Path

from niah.analyze import dataset_manifest, write_predictions_csv
from niah.data import load_config, read_jsonl, write_json
from niah.env import collect_environment
from niah.models import build_runner
from niah.scoring import score_example


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model inference on a generated NIAH dataset.")
    parser.add_argument("--dataset", required=True, help="Input dataset JSONL.")
    parser.add_argument("--model-config", required=True, help="Model YAML/JSON config.")
    parser.add_argument("--out", required=True, help="Output run directory.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N selected examples.")
    parser.add_argument("--example-ids", nargs="+", default=None, help="Evaluate only these example IDs, in dataset order.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_dataset = read_jsonl(args.dataset)
    dataset = select_examples(full_dataset, example_ids=args.example_ids, limit=args.limit)
    model_config = load_config(args.model_config)
    environment = collect_environment()
    write_json(out_dir / "environment.json", environment)

    manifest = {
        **dataset_manifest(args.dataset),
        "model_config_path": args.model_config,
        "model_config": model_config,
        "evaluation_subset": {
            "requested_example_ids": args.example_ids,
            "limit": args.limit,
            "num_selected_examples": len(dataset),
            "num_total_examples": len(full_dataset),
        },
        "generation": {
            "max_new_tokens": model_config.get("max_new_tokens", 8),
            "do_sample": model_config.get("do_sample", False),
            "use_cache": model_config.get("use_cache", True),
            "decode_strategy": model_config.get("decode_strategy"),
        },
        "scoring": {"method": "first_number_exact_match"},
    }
    write_json(out_dir / "manifest.json", manifest)

    runner = build_runner(model_config)
    try:
        load_report = runner.load()
    finally:
        write_json(out_dir / "model_load_report.json", runner.load_report.to_dict())

    predictions: list[dict] = []
    predictions_jsonl = out_dir / "predictions.jsonl"
    with predictions_jsonl.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(dataset, start=1):
            generation = runner.generate(example["prompt"])
            score = score_example(generation.generated_text, example)
            row = {
                "example_id": example["example_id"],
                "model_id": model_config.get("model_id"),
                "model_label": model_config.get("label", model_config.get("model_id")),
                "task": example.get("task"),
                "target_length": example["target_length"],
                "seed": example.get("seed"),
                "key": example.get("key"),
                "answer": str(example["answer"]),
                "needle_position_fraction": example.get("needle_position_fraction"),
                "target_record_index": example.get("target_record_index"),
                "num_distractors": example.get("num_distractors"),
                "generated_text": generation.generated_text,
                "pred_number": score["pred_number"],
                "correct": score["correct"],
                "failure_type": score["failure_type"],
                "input_tokens": generation.input_tokens,
                "new_tokens": generation.new_tokens,
                "elapsed_sec": generation.elapsed_sec,
                "peak_mem_gib": generation.peak_mem_gib,
            }
            predictions.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            print(f"[{index}/{len(dataset)}] {row['example_id']} correct={row['correct']} pred={row['pred_number']}")

    write_predictions_csv(out_dir / "predictions.csv", predictions)
    print(f"Wrote run outputs to {out_dir}")


def select_examples(
    dataset: list[dict],
    *,
    example_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[dict]:
    selected = dataset
    if example_ids:
        requested = set(example_ids)
        selected = [row for row in selected if row.get("example_id") in requested]
        found = {row.get("example_id") for row in selected}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"Example IDs not found in dataset: {missing}")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        selected = selected[:limit]
    if not selected:
        raise ValueError("No examples selected for evaluation.")
    return selected


if __name__ == "__main__":
    main()
