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
from niah.scoring import score_example


DEFAULT_MODEL_CONFIGS = [
    "configs/model_state_spaces_transformerpp_2_7b.yaml",
    "configs/model_state_spaces_mamba2_2_7b.yaml",
    "configs/model_state_spaces_mamba2attn_2_7b.yaml",
]

DEFAULT_DATASET_CONFIGS = [
    "configs/dataset_state_spaces_probe_single_n10_l2048_8192.yaml",
    "configs/dataset_state_spaces_probe_multikey_d20_n10_l2048_8192.yaml",
]

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a base-model-friendly State Spaces diagnostic with RULER-like "
            "answer prefixes and multiple-choice log-likelihood ranking."
        )
    )
    parser.add_argument("--out", default="results/state_spaces_ruler_diagnostic")
    parser.add_argument("--model-configs", nargs="+", default=DEFAULT_MODEL_CONFIGS)
    parser.add_argument("--dataset-configs", nargs="+", default=DEFAULT_DATASET_CONFIGS)
    parser.add_argument("--tokenizer-id", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--num-single-negatives", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_load_reports").mkdir(parents=True, exist_ok=True)

    datasets = load_or_generate_datasets(
        dataset_config_paths=args.dataset_configs,
        tokenizer_id=args.tokenizer_id,
        out_dir=out_dir / "datasets",
    )
    all_rows: list[dict[str, Any]] = []

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
                if not args.skip_generation:
                    all_rows.append(
                        run_answer_prefix_generation(
                            runner=runner,
                            model_config=model_config,
                            dataset_name=dataset_name,
                            example=example,
                            max_new_tokens=args.max_new_tokens,
                        )
                    )
                all_rows.append(
                    run_choice_ranking(
                        runner=runner,
                        model_config=model_config,
                        dataset_name=dataset_name,
                        example=example,
                        num_single_negatives=args.num_single_negatives,
                    )
                )

        unload_runner(runner)

    write_rows(out_dir / "predictions.csv", all_rows)
    write_jsonl(out_dir / "predictions.jsonl", all_rows)
    write_summary(out_dir / "summary.csv", all_rows)
    write_json(
        out_dir / "manifest.json",
        {
            "model_configs": args.model_configs,
            "dataset_configs": args.dataset_configs,
            "tokenizer_id": args.tokenizer_id,
            "num_single_negatives": args.num_single_negatives,
            "max_new_tokens": args.max_new_tokens,
            "modes": ["answer_prefix_generate", "choice_loglikelihood"],
            "notes": [
                "choice_loglikelihood scores letter choices with one forward pass per example.",
                "single-needle examples use synthetic numeric negatives because they contain only one record.",
                "multi-key examples rank the target against all record values from the document.",
            ],
        },
    )


def load_or_generate_datasets(
    *,
    dataset_config_paths: list[str],
    tokenizer_id: str,
    out_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
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


def run_answer_prefix_generation(
    *,
    runner: Any,
    model_config: dict[str, Any],
    dataset_name: str,
    example: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt = build_numeric_answer_prompt(example)
    eval_example = {**example, "prompt": prompt}

    original_max_new_tokens = getattr(runner, "max_new_tokens", None)
    if original_max_new_tokens is not None:
        runner.max_new_tokens = max_new_tokens
    generation = runner.generate(prompt)
    if original_max_new_tokens is not None:
        runner.max_new_tokens = original_max_new_tokens

    score = score_example(generation.generated_text, eval_example)
    return base_row(
        model_config=model_config,
        dataset_name=dataset_name,
        example=example,
        mode="answer_prefix_generate",
        input_tokens=generation.input_tokens,
        elapsed_sec=generation.elapsed_sec,
        peak_mem_gib=generation.peak_mem_gib,
        correct=score["correct"],
        failure_type=score["failure_type"],
        pred_number=score["pred_number"],
        generated_text=generation.generated_text,
        max_new_tokens=max_new_tokens,
    )


def run_choice_ranking(
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
    if len(candidates) > len(LETTERS):
        raise ValueError(f"Too many candidates for letter choices: {len(candidates)}")

    prompt, correct_label = build_choice_prompt(example, candidates)
    label_scores = score_next_token_choices(
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        labels=[candidate["label"] for candidate in candidates],
        device=model_config.get("device", "cuda"),
    )
    ranked = sorted(candidates, key=lambda candidate: label_scores[candidate["label"]], reverse=True)
    predicted = ranked[0]
    correct = predicted["label"] == correct_label
    correct_score = label_scores[correct_label]
    best_score = label_scores[predicted["label"]]
    runner_up_score = label_scores[ranked[1]["label"]] if len(ranked) > 1 else None

    return base_row(
        model_config=model_config,
        dataset_name=dataset_name,
        example=example,
        mode="choice_loglikelihood",
        input_tokens=count_input_tokens(tokenizer, prompt),
        elapsed_sec=label_scores["_elapsed_sec"],
        peak_mem_gib=label_scores.get("_peak_mem_gib"),
        correct=correct,
        failure_type="correct" if correct else classify_choice_failure(predicted, example),
        pred_number=predicted["answer"],
        generated_text="",
        correct_label=correct_label,
        pred_label=predicted["label"],
        correct_score=correct_score,
        best_score=best_score,
        runner_up_score=runner_up_score,
        margin_vs_runner_up=(best_score - runner_up_score) if runner_up_score is not None else None,
        candidate_count=len(candidates),
        choices_json=json.dumps(candidates, sort_keys=True),
    )


def build_numeric_answer_prompt(example: dict[str, Any]) -> str:
    document = document_prefix(example)
    return (
        document
        + "\n\nEnd of document.\n"
        + "Question: What is the 7-digit value assigned to "
        + f"PASSKEY_RECORD[{example['key']}] in the document above?\n"
        + "Answer:"
    )


def build_choice_prompt(example: dict[str, Any], candidates: list[dict[str, str]]) -> tuple[str, str]:
    document = document_prefix(example)
    choices = "\n".join(f"{candidate['label']}. {candidate['answer']}" for candidate in candidates)
    correct_label = next(candidate["label"] for candidate in candidates if candidate["is_target"])
    prompt = (
        document
        + "\n\nEnd of document.\n"
        + "Choices:\n"
        + choices
        + "\n\nQuestion: Which choice gives the value assigned to "
        + f"PASSKEY_RECORD[{example['key']}] in the document above?\n"
        + "Answer:"
    )
    return prompt, correct_label


def document_prefix(example: dict[str, Any]) -> str:
    marker = "\n\nEnd of document.\n"
    return example["prompt"].split(marker, 1)[0]


def build_candidates(example: dict[str, Any], num_single_negatives: int) -> list[dict[str, str]]:
    records = list(example.get("records", []))
    if len(records) > 1:
        answers = [(str(record["answer"]), bool(record.get("is_target", False))) for record in records]
    else:
        target = str(example["answer"])
        negatives = make_numeric_negatives(example, num_single_negatives)
        answers = [(target, True)] + [(negative, False) for negative in negatives]

    rng = random.Random(int(example.get("seed", 0)) + 701337)
    rng.shuffle(answers)
    return [
        {"label": LETTERS[index], "answer": answer, "is_target": is_target}
        for index, (answer, is_target) in enumerate(answers)
    ]


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


def score_next_token_choices(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    prompt: str,
    labels: list[str],
    device: str,
) -> dict[str, float]:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    label_token_ids = {}
    for label in labels:
        ids = tokenizer(" " + label, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"Choice label {label!r} is not a single token when prefixed with a space: {ids}")
        label_token_ids[label] = ids[0]

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    start = time.time()
    with torch.inference_mode():
        logits = model(input_ids).logits[:, -1, :]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
    elapsed_sec = time.time() - start

    scores = {label: float(log_probs[0, token_id].detach().cpu()) for label, token_id in label_token_ids.items()}
    scores["_elapsed_sec"] = elapsed_sec
    if torch.cuda.is_available():
        scores["_peak_mem_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    return scores


def classify_choice_failure(predicted: dict[str, Any], example: dict[str, Any]) -> str:
    pred = str(predicted["answer"])
    distractors = {str(record["answer"]) for record in example.get("records", []) if not record.get("is_target", False)}
    if pred in distractors:
        return "distractor_value"
    return "not_in_document"


def count_input_tokens(tokenizer: Any, prompt: str) -> int:
    return len(tokenizer(prompt, add_special_tokens=False).input_ids)


def base_row(
    *,
    model_config: dict[str, Any],
    dataset_name: str,
    example: dict[str, Any],
    mode: str,
    input_tokens: int,
    elapsed_sec: float,
    peak_mem_gib: float | None,
    correct: bool,
    failure_type: str,
    pred_number: str | None,
    generated_text: str,
    max_new_tokens: int | None = None,
    correct_label: str | None = None,
    pred_label: str | None = None,
    correct_score: float | None = None,
    best_score: float | None = None,
    runner_up_score: float | None = None,
    margin_vs_runner_up: float | None = None,
    candidate_count: int | None = None,
    choices_json: str | None = None,
) -> dict[str, Any]:
    task = example.get("task")
    task_label = "single_needle" if task == "passkey_single" else "multi_key" if task == "passkey_distractors" else task
    return {
        "model_id": model_config.get("model_id"),
        "model_label": model_config.get("label", model_config.get("model_id")),
        "dataset_name": dataset_name,
        "mode": mode,
        "task": task_label,
        "example_id": example["example_id"],
        "target_length": example["target_length"],
        "input_tokens": input_tokens,
        "key": example.get("key"),
        "answer": str(example["answer"]),
        "pred_number": pred_number,
        "correct": correct,
        "failure_type": failure_type,
        "generated_text": generated_text,
        "max_new_tokens": max_new_tokens,
        "correct_label": correct_label,
        "pred_label": pred_label,
        "correct_score": correct_score,
        "best_score": best_score,
        "runner_up_score": runner_up_score,
        "margin_vs_runner_up": margin_vs_runner_up,
        "candidate_count": candidate_count,
        "choices_json": choices_json,
        "elapsed_sec": elapsed_sec,
        "peak_mem_gib": peak_mem_gib,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    failures: dict[tuple[str, str, str, int], Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = (row["model_label"], row["mode"], row["task"], int(row["target_length"]))
        grouped[key].append(row)
        failures[key][str(row["failure_type"])] += 1

    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "model_label",
            "mode",
            "task",
            "target_length",
            "correct",
            "n",
            "accuracy_pct",
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
                    "mode": key[1],
                    "task": key[2],
                    "target_length": key[3],
                    "correct": correct,
                    "n": len(group),
                    "accuracy_pct": round(100 * correct / len(group), 2),
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
