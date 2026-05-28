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

from niah.data import (
    file_sha256,
    generate_dataset,
    load_config,
    read_jsonl,
    write_json,
    write_jsonl,
)
from niah.models import build_runner


MODEL_CONFIGS = [
    "configs/model_state_spaces_mamba_2_8b.yaml",
    "configs/model_state_spaces_mamba2_2_7b.yaml",
    "configs/model_state_spaces_mamba2attn_2_7b.yaml",
]
DATASET_CONFIG = "configs/dataset_state_spaces_mamba_family_primary_n30_l2048_8192.yaml"
DATASET_PATH = "datasets/state_spaces_mamba_family_primary_n30_l2048_8192_neox_tok.jsonl"
SMOKE_EXAMPLE_IDS = [
    "passkey_distractors_d0_2048_0000",
    "passkey_distractors_d5_2048_0000",
    "passkey_distractors_d10_2048_0000",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score numeric answer likelihoods for the Mamba-family primary matrix.")
    parser.add_argument("--out", default="results/state_spaces_mamba_family_likelihood")
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--dataset-config", default=DATASET_CONFIG)
    parser.add_argument("--model-configs", nargs="+", default=MODEL_CONFIGS)
    parser.add_argument("--tokenizer-id", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--num-single-negatives", type=int, default=4)
    parser.add_argument(
        "--prompt-style",
        choices=["answer", "passkey_completion"],
        default="answer",
        help="Prompt ending used before candidate numeric continuations.",
    )
    parser.add_argument("--example-ids", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_load_reports").mkdir(parents=True, exist_ok=True)

    dataset_path = ensure_dataset(args.dataset, args.dataset_config, args.tokenizer_id)
    examples = select_examples(read_jsonl(dataset_path), example_ids=args.example_ids, limit=args.limit)

    rows: list[dict[str, Any]] = []
    for model_config_path in args.model_configs:
        model_config = load_config(model_config_path)
        runner = build_runner(model_config)
        load_report = runner.load()
        write_json(out_dir / "model_load_reports" / f"{safe_name(runner.label)}.json", load_report.to_dict())

        for index, example in enumerate(examples, start=1):
            row = score_example_likelihood(
                runner=runner,
                model_config=model_config,
                model_config_path=model_config_path,
                dataset_path=dataset_path,
                example=example,
                num_single_negatives=args.num_single_negatives,
                prompt_style=args.prompt_style,
            )
            rows.append(row)
            print(
                f"[{index}/{len(examples)}] {row['model_label']} {row['example_id']} "
                f"d={row['num_distractors']} len={row['target_length']} "
                f"correct={row['correct']} pred={row['pred_number']} rank={row['correct_rank']}",
                flush=True,
            )

        unload_runner(runner)

    write_rows(out_dir / "predictions.csv", rows)
    write_jsonl(out_dir / "predictions.jsonl", rows)
    write_summary(out_dir / "summary.csv", rows)
    write_json(
        out_dir / "manifest.json",
        {
            "mode": "numeric_loglikelihood",
            "dataset_path": str(dataset_path),
            "dataset_sha256": file_sha256(dataset_path),
            "num_total_examples": len(read_jsonl(dataset_path)),
            "num_selected_examples": len(examples),
            "example_ids": args.example_ids,
            "limit": args.limit,
            "model_configs": args.model_configs,
            "tokenizer_id": args.tokenizer_id,
            "num_single_negatives": args.num_single_negatives,
            "prompt_style": args.prompt_style,
            "notes": [
                "Scores candidate numeric continuations after the selected prompt ending.",
                "For distractor examples, candidates are all record values in the document.",
                "For zero-distractor examples, candidates are the target value plus synthetic numeric negatives.",
                "Candidates are ranked by average token log-likelihood.",
            ],
        },
    )


def ensure_dataset(dataset_path: str, dataset_config_path: str, tokenizer_id: str) -> Path:
    path = Path(dataset_path)
    if path.exists():
        return path

    config = load_config(dataset_config_path)
    config["tokenizer_id"] = tokenizer_id
    rows = generate_dataset(config)
    write_jsonl(path, rows)
    write_json(str(path) + ".manifest.json", {"config": config, "num_examples": len(rows)})
    return path


def select_examples(
    examples: list[dict[str, Any]],
    *,
    example_ids: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = examples
    if example_ids:
        requested = set(example_ids)
        selected = [example for example in selected if example.get("example_id") in requested]
        found = {example.get("example_id") for example in selected}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"Example IDs not found in dataset: {missing}")
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No examples selected.")
    return selected


def score_example_likelihood(
    *,
    runner: Any,
    model_config: dict[str, Any],
    model_config_path: str,
    dataset_path: Path,
    example: dict[str, Any],
    num_single_negatives: int,
    prompt_style: str,
) -> dict[str, Any]:
    torch = runner.torch
    model = runner.model
    tokenizer = runner.tokenizer
    if torch is None or model is None or tokenizer is None:
        raise RuntimeError("Runner must expose torch, model, and tokenizer after load().")

    prompt = build_likelihood_prompt(example, prompt_style=prompt_style)
    candidates = build_candidates(example, num_single_negatives)
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
    correct_candidate = next(candidate for candidate in ranked if candidate["is_target"])
    correct_rank = next(index for index, candidate in enumerate(ranked, start=1) if candidate["is_target"])
    runner_up = ranked[1] if len(ranked) > 1 else None
    correct = bool(predicted["is_target"])
    peak_mem_gib = None
    if torch.cuda.is_available():
        peak_mem_gib = torch.cuda.max_memory_allocated() / 1024**3

    return {
        "model_id": model_config.get("model_id"),
        "model_label": model_config.get("label", model_config.get("model_id")),
        "model_config_path": model_config_path,
        "dataset_path": str(dataset_path),
        "mode": "numeric_loglikelihood",
        "prompt_style": prompt_style,
        "task": example.get("task"),
        "example_id": example["example_id"],
        "target_length": example["target_length"],
        "num_distractors": int(example.get("num_distractors", 0)),
        "input_tokens": count_input_tokens(tokenizer, prompt),
        "key": example.get("key"),
        "answer": str(example["answer"]),
        "pred_number": predicted["answer"],
        "correct": correct,
        "failure_type": "correct" if correct else classify_failure(predicted, example),
        "correct_rank": correct_rank,
        "candidate_count": len(candidates),
        "correct_avg_logprob": correct_candidate["avg_logprob"],
        "best_avg_logprob": predicted["avg_logprob"],
        "runner_up_avg_logprob": runner_up["avg_logprob"] if runner_up else None,
        "margin_vs_runner_up": predicted["avg_logprob"] - runner_up["avg_logprob"] if runner_up else None,
        "correct_margin_vs_best": correct_candidate["avg_logprob"] - predicted["avg_logprob"],
        "choices_json": json.dumps(scored_candidates, sort_keys=True),
        "elapsed_sec": elapsed_sec,
        "peak_mem_gib": peak_mem_gib,
    }


def build_likelihood_prompt(example: dict[str, Any], *, prompt_style: str) -> str:
    if prompt_style == "answer":
        return build_answer_prompt(example)
    if prompt_style == "passkey_completion":
        return build_passkey_completion_prompt(example)
    raise ValueError(f"Unknown prompt style: {prompt_style}")


def build_answer_prompt(example: dict[str, Any]) -> str:
    document = example["prompt"].split("\n\nEnd of document.\n", 1)[0]
    return (
        document
        + "\n\nEnd of document.\n"
        + "Question: What is the 7-digit value assigned to "
        + f"PASSKEY_RECORD[{example['key']}] in the document above?\n"
        + "Answer:"
    )


def build_passkey_completion_prompt(example: dict[str, Any]) -> str:
    prompt = example["prompt"]
    needle = f"PASSKEY_RECORD[{example['key']}] ="
    if prompt.rstrip().endswith(needle):
        return prompt

    document = prompt.split("\n\nEnd of document.\n", 1)[0]
    return (
        document
        + "\n\nEnd of document.\n"
        + "Repeat the matching passkey record from the document above.\n"
        + needle
    )


def build_candidates(example: dict[str, Any], num_single_negatives: int) -> list[dict[str, Any]]:
    records = list(example.get("records", []))
    if len(records) > 1:
        answers = [(str(record["answer"]), bool(record.get("is_target", False))) for record in records]
    else:
        target = str(example["answer"])
        answers = [(target, True)] + [(negative, False) for negative in make_numeric_negatives(example, num_single_negatives)]

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
    del input_ids, logits, selected_logits, log_probs, target_ids, token_logprobs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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


def count_input_tokens(tokenizer: Any, prompt: str) -> int:
    return len(tokenizer(prompt, add_special_tokens=False).input_ids)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    failures: dict[tuple[str, int, int], Counter[str]] = defaultdict(Counter)
    ranks: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for row in rows:
        key = (row["model_label"], int(row["target_length"]), int(row["num_distractors"]))
        grouped[key].append(row)
        failures[key][str(row["failure_type"])] += 1
        ranks[key].append(int(row["correct_rank"]))

    fieldnames = [
        "model_label",
        "target_length",
        "num_distractors",
        "correct",
        "n",
        "accuracy_pct",
        "mean_correct_rank",
        "failures",
        "avg_elapsed_sec",
        "avg_peak_mem_gib",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(grouped):
            group = grouped[key]
            correct = sum(1 for row in group if row["correct"])
            peak_values = [float(row["peak_mem_gib"]) for row in group if row["peak_mem_gib"] not in {None, ""}]
            writer.writerow(
                {
                    "model_label": key[0],
                    "target_length": key[1],
                    "num_distractors": key[2],
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
