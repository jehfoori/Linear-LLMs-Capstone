from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from niah.data import generate_dataset, load_config, write_json, write_jsonl
from niah.models import build_runner


DEFAULT_MODEL_CONFIGS = [
    "configs/model_state_spaces_transformerpp_2_7b.yaml",
    "configs/model_state_spaces_mamba2_2_7b.yaml",
    "configs/model_state_spaces_mamba2attn_2_7b.yaml",
]

DEFAULT_DATASET_CONFIGS = [
    "configs/dataset_state_spaces_probe_single_n10_l2048_8192.yaml",
    "configs/dataset_state_spaces_probe_multikey_d20_n10_l2048_8192.yaml",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score direct numeric answer likelihoods for State Spaces 2.7B NIAH examples."
    )
    parser.add_argument("--out", default="results/state_spaces_numeric_likelihood_diagnostic")
    parser.add_argument("--model-configs", nargs="+", default=DEFAULT_MODEL_CONFIGS)
    parser.add_argument("--dataset-configs", nargs="+", default=DEFAULT_DATASET_CONFIGS)
    parser.add_argument("--tokenizer-id", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--num-single-negatives", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "datasets").mkdir(parents=True, exist_ok=True)
    (out_dir / "model_load_reports").mkdir(parents=True, exist_ok=True)

    datasets = load_or_generate_datasets(
        dataset_config_paths=args.dataset_configs,
        tokenizer_id=args.tokenizer_id,
        out_dir=out_dir / "datasets",
    )

    rows: list[dict[str, Any]] = []
    for model_config_path in args.model_configs:
        model_config = load_config(model_config_path)
        runner = build_runner(model_config)
        load_report = runner.load()
        write_json(
            out_dir / "model_load_reports" / f"{safe_name(model_config.get('label', model_config['model_id']))}.json",
            load_report.to_dict(),
        )

        for dataset_name, examples in datasets.items():
            for example in examples:
                row = run_numeric_likelihood(
                    runner=runner,
                    model_config=model_config,
                    dataset_name=dataset_name,
                    example=example,
                    num_single_negatives=args.num_single_negatives,
                )
                rows.append(row)
                print(
                    f"{row['model_label']} {row['task']} {row['target_length']} "
                    f"{row['example_id']} correct={row['correct']} "
                    f"pred={row['pred_number']} answer={row['answer']} "
                    f"rank={row['correct_rank']} margin={row['margin_vs_runner_up']}",
                    flush=True,
                )

        unload_runner(runner)

    write_rows(out_dir / "predictions.csv", rows)
    write_jsonl(out_dir / "predictions.jsonl", rows)
    write_summary(out_dir / "summary.csv", rows)
    write_json(
        out_dir / "manifest.json",
        {
            "model_configs": args.model_configs,
            "dataset_configs": args.dataset_configs,
            "tokenizer_id": args.tokenizer_id,
            "num_single_negatives": args.num_single_negatives,
            "mode": "numeric_loglikelihood",
            "notes": [
                "Scores direct numeric answer strings after an Answer: prefix.",
                "Uses average token log-likelihood so candidates with different tokenizations are comparable.",
                "Single-needle examples use synthetic numeric negatives.",
                "Multi-key examples rank the target against all record values in the document.",
            ],
        },
    )


def load_or_generate_datasets(
    *,
    dataset_config_paths: list[str],
    tokenizer_id: str,
    out_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    datasets: dict[str, list[dict[str, Any]]] = {}
    for config_path in dataset_config_paths:
        config = load_config(config_path)
        config["tokenizer_id"] = tokenizer_id
        rows = generate_dataset(config)
        name = Path(config_path).stem
        datasets[name] = rows
        write_jsonl(out_dir / f"{name}.jsonl", rows)
        write_json(out_dir / f"{name}.manifest.json", {"config": config, "num_examples": len(rows)})
    return datasets


def run_numeric_likelihood(
    *,
    runner: Any,
    model_config: dict[str, Any],
    dataset_name: str,
    example: dict[str, Any],
    num_single_negatives: int,
) -> dict[str, Any]:
    torch = runner.torch
    model = runner.model
    tokenizer = runner.tokenizer
    if torch is None or model is None or tokenizer is None:
        raise RuntimeError("Runner must expose torch, model, and tokenizer after load().")

    candidates = build_candidates(example, num_single_negatives)
    prompt = build_answer_prompt(example)
    scored_candidates = []
    start = time.time()
    for candidate in candidates:
        score = score_continuation(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            continuation=" " + candidate["answer"],
            device=model_config.get("device", "cuda"),
        )
        scored_candidates.append({**candidate, **score})

    elapsed_sec = time.time() - start
    ranked = sorted(scored_candidates, key=lambda candidate: candidate["avg_logprob"], reverse=True)
    predicted = ranked[0]
    correct = bool(predicted["is_target"])
    correct_index = next(index for index, candidate in enumerate(ranked, start=1) if candidate["is_target"])
    correct_candidate = next(candidate for candidate in ranked if candidate["is_target"])
    runner_up = ranked[1] if len(ranked) > 1 else None
    peak_mem_gib = None
    if torch.cuda.is_available():
        peak_mem_gib = torch.cuda.max_memory_allocated() / 1024**3

    return {
        "model_id": model_config.get("model_id"),
        "model_label": model_config.get("label", model_config.get("model_id")),
        "dataset_name": dataset_name,
        "mode": "numeric_loglikelihood",
        "task": task_label(example),
        "example_id": example["example_id"],
        "target_length": example["target_length"],
        "input_tokens": count_input_tokens(tokenizer, prompt),
        "key": example.get("key"),
        "answer": str(example["answer"]),
        "pred_number": predicted["answer"],
        "correct": correct,
        "failure_type": "correct" if correct else classify_failure(predicted, example),
        "correct_rank": correct_index,
        "candidate_count": len(candidates),
        "correct_avg_logprob": correct_candidate["avg_logprob"],
        "best_avg_logprob": predicted["avg_logprob"],
        "runner_up_avg_logprob": runner_up["avg_logprob"] if runner_up else None,
        "margin_vs_runner_up": (
            predicted["avg_logprob"] - runner_up["avg_logprob"] if runner_up else None
        ),
        "correct_margin_vs_best": correct_candidate["avg_logprob"] - predicted["avg_logprob"],
        "choices_json": json.dumps(scored_candidates, sort_keys=True),
        "elapsed_sec": elapsed_sec,
        "peak_mem_gib": peak_mem_gib,
    }


def build_answer_prompt(example: dict[str, Any]) -> str:
    document = document_prefix(example)
    return (
        document
        + "\n\nEnd of document.\n"
        + "Question: What is the 7-digit value assigned to "
        + f"PASSKEY_RECORD[{example['key']}] in the document above?\n"
        + "Answer:"
    )


def document_prefix(example: dict[str, Any]) -> str:
    marker = "\n\nEnd of document.\n"
    return example["prompt"].split(marker, 1)[0]


def build_candidates(example: dict[str, Any], num_single_negatives: int) -> list[dict[str, Any]]:
    records = list(example.get("records", []))
    if len(records) > 1:
        answers = [(str(record["answer"]), bool(record.get("is_target", False))) for record in records]
    else:
        target = str(example["answer"])
        negatives = make_numeric_negatives(example, num_single_negatives)
        answers = [(target, True)] + [(negative, False) for negative in negatives]

    rng = random.Random(int(example.get("seed", 0)) + 731337)
    rng.shuffle(answers)
    return [{"answer": answer, "is_target": is_target} for answer, is_target in answers]


def make_numeric_negatives(example: dict[str, Any], count: int) -> list[str]:
    target = str(example["answer"])
    used = {target}
    rng = random.Random(int(example.get("seed", 0)) + 991337)
    negatives: list[str] = []
    while len(negatives) < count:
        candidate = str(rng.randint(1000000, 9999999))
        if candidate not in used:
            used.add(candidate)
            negatives.append(candidate)
    return negatives


def score_continuation(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    prompt: str,
    continuation: str,
    device: str,
) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    continuation_ids = tokenizer(continuation, add_special_tokens=False).input_ids
    input_ids = torch.tensor([prompt_ids + continuation_ids], device=device)
    prompt_len = len(prompt_ids)
    continuation_len = len(continuation_ids)

    with torch.inference_mode():
        logits = model(input_ids).logits
        selected_logits = logits[:, prompt_len - 1 : prompt_len + continuation_len - 1, :]
        log_probs = torch.log_softmax(selected_logits.float(), dim=-1)
        target_ids = torch.tensor(continuation_ids, device=device).view(1, -1, 1)
        token_logprobs = log_probs.gather(dim=-1, index=target_ids).squeeze(0).squeeze(-1)

    total_logprob = float(token_logprobs.sum().detach().cpu())
    avg_logprob = float(token_logprobs.mean().detach().cpu())
    return {
        "total_logprob": total_logprob,
        "avg_logprob": avg_logprob,
        "continuation_tokens": continuation_len,
    }


def classify_failure(predicted: dict[str, Any], example: dict[str, Any]) -> str:
    pred = str(predicted["answer"])
    distractors = {str(record["answer"]) for record in example.get("records", []) if not record.get("is_target", False)}
    if pred in distractors:
        return "distractor_value"
    return "not_in_document"


def task_label(example: dict[str, Any]) -> str:
    task = example.get("task")
    if task == "passkey_single":
        return "single_needle"
    if task == "passkey_distractors":
        return "multi_key"
    return str(task)


def count_input_tokens(tokenizer: Any, prompt: str) -> int:
    return len(tokenizer(prompt, add_special_tokens=False).input_ids)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    failures: dict[tuple[str, str, int], Counter[str]] = defaultdict(Counter)
    ranks: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for row in rows:
        key = (row["model_label"], row["task"], int(row["target_length"]))
        grouped[key].append(row)
        failures[key][str(row["failure_type"])] += 1
        ranks[key].append(int(row["correct_rank"]))

    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "model_label",
            "task",
            "target_length",
            "correct",
            "n",
            "accuracy_pct",
            "mean_correct_rank",
            "failures",
            "avg_elapsed_sec",
            "avg_peak_mem_gib",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(grouped):
            group = grouped[key]
            correct = sum(1 for row in group if row["correct"])
            peak_values = [float(row["peak_mem_gib"]) for row in group if row["peak_mem_gib"] not in {None, ""}]
            writer.writerow(
                {
                    "model_label": key[0],
                    "task": key[1],
                    "target_length": key[2],
                    "correct": correct,
                    "n": len(group),
                    "accuracy_pct": round(100 * correct / len(group), 2),
                    "mean_correct_rank": sum(ranks[key]) / len(ranks[key]),
                    "failures": json.dumps(dict(failures[key]), sort_keys=True),
                    "avg_elapsed_sec": sum(float(row["elapsed_sec"]) for row in group) / len(group),
                    "avg_peak_mem_gib": (sum(peak_values) / len(peak_values)) if peak_values else None,
                }
            )


def unload_runner(runner: Any) -> None:
    if getattr(runner, "model", None) is not None:
        del runner.model
        runner.model = None
    if getattr(runner, "tokenizer", None) is not None:
        del runner.tokenizer
        runner.tokenizer = None
    torch = getattr(runner, "torch", None)
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_name(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


if __name__ == "__main__":
    main()
