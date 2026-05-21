from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from niah.analyze import analyze_run, dataset_manifest, write_predictions_csv
from niah.data import load_config, read_jsonl, write_json, write_jsonl
from niah.env import collect_environment
from niah.models import build_runner
from niah.scoring import score_example


VARIANTS = {
    "baseline": (
        "End of document.\n"
        "Repeat the matching passkey record from the document above.\n"
        "PASSKEY_RECORD[{key}] ="
    ),
    "value_only": (
        "End of document.\n"
        "Only output the numeric value for PASSKEY_RECORD[{key}].\n"
        "Answer:"
    ),
    "strict_value_only": (
        "End of document.\n"
        "Return only the 7-digit number. Do not repeat any record key.\n"
        "PASSKEY_RECORD[{key}] ="
    ),
    "unknown_sentinel": (
        "End of document.\n"
        "Return either the matching 7-digit number or ZXQ_UNKNOWN if you cannot determine it.\n"
        "Do not repeat any record key.\n"
        "PASSKEY_RECORD[{key}] ="
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small prompt-format probe on existing NIAH examples.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--target-length", type=int, default=2560)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    source_rows = [
        row
        for row in read_jsonl(args.dataset)
        if int(row.get("target_length", -1)) == args.target_length
    ][: args.limit]
    if not source_rows:
        raise ValueError(f"No examples found for target_length={args.target_length}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    model_config = load_config(args.model_config)

    for variant_name, query_template in VARIANTS.items():
        variant_rows = [
            rewrite_query(row, variant_name=variant_name, query_template=query_template)
            for row in source_rows
        ]
        dataset_path = out_root / f"{variant_name}.jsonl"
        write_jsonl(dataset_path, variant_rows)
        write_json(str(dataset_path) + ".manifest.json", {"variant": variant_name, "num_examples": len(variant_rows)})

        run_dir = out_root / variant_name
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "environment.json", collect_environment())
        write_json(
            run_dir / "manifest.json",
            {
                **dataset_manifest(dataset_path),
                "source_dataset_path": args.dataset,
                "prompt_variant": variant_name,
                "query_template": query_template,
                "model_config_path": args.model_config,
                "model_config": model_config,
                "generation": {
                    "max_new_tokens": model_config.get("max_new_tokens", 8),
                    "decode_strategy": model_config.get("decode_strategy"),
                },
                "scoring": {"method": "first_number_exact_match"},
            },
        )

        runner = build_runner(model_config)
        try:
            runner.load()
        finally:
            write_json(run_dir / "model_load_report.json", runner.load_report.to_dict())

        predictions = []
        with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for index, example in enumerate(variant_rows, start=1):
                generation = runner.generate(example["prompt"])
                score = score_example(generation.generated_text, example)
                row = {
                    "example_id": example["example_id"],
                    "source_example_id": example.get("source_example_id"),
                    "prompt_variant": variant_name,
                    "model_id": model_config.get("model_id"),
                    "model_label": f"{model_config.get('label', model_config.get('model_id'))} [{variant_name}]",
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
                print(
                    f"[{variant_name} {index}/{len(variant_rows)}] "
                    f"{row['source_example_id']} correct={row['correct']} pred={row['pred_number']}",
                    flush=True,
                )

        write_predictions_csv(run_dir / "predictions.csv", predictions)
        analyze_run(run_dir)


def rewrite_query(row: dict[str, Any], *, variant_name: str, query_template: str) -> dict[str, Any]:
    query_pattern = re.compile(
        r"\n\nEnd of document\.\n"
        r"Repeat the matching passkey record from the document above\.\n"
        r"PASSKEY_RECORD\[[^\]]+\] =$"
    )
    replacement = "\n\n" + query_template.format(key=row["key"])
    prompt, count = query_pattern.subn(replacement, row["prompt"])
    if count != 1:
        raise ValueError(f"Could not rewrite query for example {row.get('example_id')!r}")

    rewritten = dict(row)
    rewritten["source_example_id"] = row["example_id"]
    rewritten["example_id"] = f"{row['example_id']}__{variant_name}"
    rewritten["prompt_variant"] = variant_name
    rewritten["query_template"] = query_template
    rewritten["prompt"] = prompt
    return rewritten


if __name__ == "__main__":
    main()
