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

from niah.data import generate_single_example, load_config, write_json, write_jsonl
from niah.models import build_runner
from niah.scoring import score_example
from transformers import AutoTokenizer


DEFAULT_MODEL_CONFIGS = [
    "configs/model_state_spaces_transformerpp_2_7b.yaml",
    "configs/model_state_spaces_mamba2_2_7b.yaml",
]

TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
QUERY_LENGTHS = [4096, 8192]
POSITION_LENGTHS = [4096, 8192]
POSITIONS = [0.10, 0.25, 0.50, 0.75, 0.90]


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose query-format and needle-position effects.")
    parser.add_argument("--out", default="results/state_spaces_query_position_diagnostic")
    parser.add_argument("--model-configs", nargs="+", default=DEFAULT_MODEL_CONFIGS)
    parser.add_argument("--query-n", type=int, default=10)
    parser.add_argument("--position-n", type=int, default=5)
    parser.add_argument("--num-negatives", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_load_reports").mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    def count_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    examples = build_examples(count_tokens=count_tokens, query_n=args.query_n, position_n=args.position_n)
    write_jsonl(out_dir / "diagnostic_examples.jsonl", examples)

    rows: list[dict[str, Any]] = []
    for model_config_path in args.model_configs:
        model_config = load_config(model_config_path)
        runner = build_runner(model_config)
        load_report = runner.load()
        write_json(
            out_dir / "model_load_reports" / f"{safe_name(model_config.get('label', model_config['model_id']))}.json",
            load_report.to_dict(),
        )

        for example in examples:
            for variant in variants_for_example(example):
                prompt = build_prompt(example, variant)
                rows.append(
                    run_generation(
                        runner=runner,
                        model_config=model_config,
                        example=example,
                        variant=variant,
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                    )
                )
                rows.append(
                    run_numeric_likelihood(
                        runner=runner,
                        model_config=model_config,
                        example=example,
                        variant=variant,
                        prompt=prompt,
                        num_negatives=args.num_negatives,
                    )
                )
                latest = rows[-1]
                print(
                    f"{latest['model_label']} {latest['diagnostic']} {latest['variant']} "
                    f"len={latest['target_length']} pos={latest['needle_position_fraction']} "
                    f"likelihood_correct={latest['correct']} rank={latest.get('correct_rank')}",
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
            "tokenizer_id": TOKENIZER_ID,
            "query_lengths": QUERY_LENGTHS,
            "position_lengths": POSITION_LENGTHS,
            "positions": POSITIONS,
            "query_n": args.query_n,
            "position_n": args.position_n,
            "num_negatives": args.num_negatives,
            "max_new_tokens": args.max_new_tokens,
            "notes": [
                "query_format examples compare baseline, front_instruction, front_end_instruction, and question_first prompts.",
                "position_sweep examples use baseline prompt and fixed needle positions.",
                "Each prompt is evaluated with free generation and direct numeric likelihood.",
            ],
        },
    )


def build_examples(*, count_tokens, query_n: int, position_n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for length in QUERY_LENGTHS:
        for index in range(query_n):
            rows.append(
                {
                    **generate_single_example(
                        example_id=f"query_format_{length}_{index:04d}",
                        target_length=length,
                        seed=992024 + length * 1000 + index,
                        count_length=count_tokens,
                        length_metric=f"tokens:{TOKENIZER_ID}",
                    ),
                    "diagnostic": "query_format",
                }
            )
    for length in POSITION_LENGTHS:
        for position in POSITIONS:
            for index in range(position_n):
                rows.append(
                    {
                        **generate_single_example(
                            example_id=f"position_sweep_{length}_{int(position * 100):02d}_{index:04d}",
                            target_length=length,
                            seed=993024 + length * 1000 + int(position * 100) * 100 + index,
                            position_fraction=position,
                            count_length=count_tokens,
                            length_metric=f"tokens:{TOKENIZER_ID}",
                        ),
                        "diagnostic": "position_sweep",
                    }
                )
    return rows


def variants_for_example(example: dict[str, Any]) -> list[str]:
    if example["diagnostic"] == "query_format":
        return ["baseline", "front_instruction", "front_end_instruction", "question_first"]
    return ["baseline"]


def build_prompt(example: dict[str, Any], variant: str) -> str:
    document = document_prefix(example)
    key = example["key"]
    if variant == "baseline":
        return (
            document
            + "\n\nEnd of document.\n"
            + "Repeat the matching passkey record from the document above.\n"
            + f"PASSKEY_RECORD[{key}] ="
        )
    if variant == "front_instruction":
        return (
            "Task: Remember the passkey records in the document and answer the query after the document.\n"
            + "Only the number assigned to the requested key is correct.\n\n"
            + document
            + "\n\nEnd of document.\n"
            + f"PASSKEY_RECORD[{key}] ="
        )
    if variant == "front_end_instruction":
        return (
            "Task: Remember the passkey records in the document and answer the query after the document.\n"
            + "Only the number assigned to the requested key is correct.\n\n"
            + document
            + "\n\nEnd of document.\n"
            + "Use the document above. Write only the 7-digit value for the requested key.\n"
            + f"PASSKEY_RECORD[{key}] ="
        )
    if variant == "question_first":
        return (
            f"Question: What is the 7-digit value assigned to PASSKEY_RECORD[{key}]?\n"
            + "Read the document, then answer this exact question at the end.\n\n"
            + document
            + "\n\nEnd of document.\n"
            + "Answer:"
        )
    raise ValueError(f"Unknown prompt variant: {variant}")


def document_prefix(example: dict[str, Any]) -> str:
    return example["prompt"].split("\n\nEnd of document.\n", 1)[0]


def run_generation(
    *,
    runner: Any,
    model_config: dict[str, Any],
    example: dict[str, Any],
    variant: str,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    original_max_new_tokens = getattr(runner, "max_new_tokens", None)
    if original_max_new_tokens is not None:
        runner.max_new_tokens = max_new_tokens
    generation = runner.generate(prompt)
    if original_max_new_tokens is not None:
        runner.max_new_tokens = original_max_new_tokens

    eval_example = {**example, "prompt": prompt}
    score = score_example(generation.generated_text, eval_example)
    return base_row(
        runner=runner,
        model_config=model_config,
        example=example,
        variant=variant,
        mode="generate",
        input_tokens=generation.input_tokens,
        elapsed_sec=generation.elapsed_sec,
        peak_mem_gib=generation.peak_mem_gib,
        correct=score["correct"],
        failure_type=score["failure_type"],
        pred_number=score["pred_number"],
        generated_text=generation.generated_text,
    )


def run_numeric_likelihood(
    *,
    runner: Any,
    model_config: dict[str, Any],
    example: dict[str, Any],
    variant: str,
    prompt: str,
    num_negatives: int,
) -> dict[str, Any]:
    torch = runner.torch
    model = runner.model
    tokenizer = runner.tokenizer
    if torch is None or model is None or tokenizer is None:
        raise RuntimeError("Runner must expose torch, model, and tokenizer after load().")

    candidates = build_candidates(example, num_negatives)
    start = time.time()
    scored = []
    for candidate in candidates:
        score = score_continuation(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            prompt=likelihood_prompt(prompt, variant),
            continuation=" " + candidate["answer"],
            device=model_config.get("device", "cuda"),
        )
        scored.append({**candidate, **score})

    ranked = sorted(scored, key=lambda row: row["avg_logprob"], reverse=True)
    predicted = ranked[0]
    correct_candidate = next(row for row in ranked if row["is_target"])
    correct_rank = next(index for index, row in enumerate(ranked, start=1) if row["is_target"])
    runner_up = ranked[1] if len(ranked) > 1 else None
    peak_mem_gib = None
    if torch.cuda.is_available():
        peak_mem_gib = torch.cuda.max_memory_allocated() / 1024**3

    return base_row(
        runner=runner,
        model_config=model_config,
        example=example,
        variant=variant,
        mode="numeric_loglikelihood",
        input_tokens=count_input_tokens(tokenizer, prompt),
        elapsed_sec=time.time() - start,
        peak_mem_gib=peak_mem_gib,
        correct=bool(predicted["is_target"]),
        failure_type="correct" if predicted["is_target"] else "not_in_document",
        pred_number=predicted["answer"],
        generated_text="",
        correct_rank=correct_rank,
        candidate_count=len(candidates),
        correct_avg_logprob=correct_candidate["avg_logprob"],
        best_avg_logprob=predicted["avg_logprob"],
        runner_up_avg_logprob=runner_up["avg_logprob"] if runner_up else None,
        margin_vs_runner_up=(
            predicted["avg_logprob"] - runner_up["avg_logprob"] if runner_up else None
        ),
        correct_margin_vs_best=correct_candidate["avg_logprob"] - predicted["avg_logprob"],
        choices_json=json.dumps(scored, sort_keys=True),
    )


def likelihood_prompt(prompt: str, variant: str) -> str:
    if variant in {"question_first", "front_end_instruction"}:
        return prompt
    marker = "\n\nEnd of document.\n"
    if marker in prompt:
        document = prompt.split(marker, 1)[0]
        return (
            document
            + marker
            + "Question: What is the 7-digit value assigned to the requested passkey record?\n"
            + "Answer:"
        )
    return prompt


def build_candidates(example: dict[str, Any], num_negatives: int) -> list[dict[str, Any]]:
    target = str(example["answer"])
    used = {target}
    rng = random.Random(int(example["seed"]) + 994337)
    candidates = [{"answer": target, "is_target": True}]
    while len(candidates) < num_negatives + 1:
        value = str(rng.randint(1000000, 9999999))
        if value not in used:
            used.add(value)
            candidates.append({"answer": value, "is_target": False})
    rng.shuffle(candidates)
    return candidates


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

    return {
        "total_logprob": float(token_logprobs.sum().detach().cpu()),
        "avg_logprob": float(token_logprobs.mean().detach().cpu()),
        "continuation_tokens": continuation_len,
    }


def base_row(
    *,
    runner: Any,
    model_config: dict[str, Any],
    example: dict[str, Any],
    variant: str,
    mode: str,
    input_tokens: int,
    elapsed_sec: float,
    peak_mem_gib: float | None,
    correct: bool,
    failure_type: str,
    pred_number: str | None,
    generated_text: str,
    correct_rank: int | None = None,
    candidate_count: int | None = None,
    correct_avg_logprob: float | None = None,
    best_avg_logprob: float | None = None,
    runner_up_avg_logprob: float | None = None,
    margin_vs_runner_up: float | None = None,
    correct_margin_vs_best: float | None = None,
    choices_json: str | None = None,
) -> dict[str, Any]:
    return {
        "model_id": model_config.get("model_id"),
        "model_label": model_config.get("label", model_config.get("model_id")),
        "diagnostic": example["diagnostic"],
        "variant": variant,
        "mode": mode,
        "example_id": example["example_id"],
        "target_length": example["target_length"],
        "needle_position_fraction": example["needle_position_fraction"],
        "input_tokens": input_tokens,
        "key": example["key"],
        "answer": str(example["answer"]),
        "pred_number": pred_number,
        "correct": correct,
        "failure_type": failure_type,
        "generated_text": generated_text,
        "correct_rank": correct_rank,
        "candidate_count": candidate_count,
        "correct_avg_logprob": correct_avg_logprob,
        "best_avg_logprob": best_avg_logprob,
        "runner_up_avg_logprob": runner_up_avg_logprob,
        "margin_vs_runner_up": margin_vs_runner_up,
        "correct_margin_vs_best": correct_margin_vs_best,
        "choices_json": choices_json,
        "elapsed_sec": elapsed_sec,
        "peak_mem_gib": peak_mem_gib,
    }


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
    grouped: dict[tuple[str, str, str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    failures: dict[tuple[str, str, str, str, int, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = (
            row["model_label"],
            row["diagnostic"],
            row["variant"],
            row["mode"],
            int(row["target_length"]),
            str(row["needle_position_fraction"]),
        )
        grouped[key].append(row)
        failures[key][str(row["failure_type"])] += 1

    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "model_label",
            "diagnostic",
            "variant",
            "mode",
            "target_length",
            "needle_position_fraction",
            "correct",
            "n",
            "accuracy_pct",
            "mean_correct_rank",
            "failures",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(grouped):
            group = grouped[key]
            correct = sum(1 for row in group if row["correct"])
            ranks = [int(row["correct_rank"]) for row in group if row["correct_rank"] not in {None, ""}]
            writer.writerow(
                {
                    "model_label": key[0],
                    "diagnostic": key[1],
                    "variant": key[2],
                    "mode": key[3],
                    "target_length": key[4],
                    "needle_position_fraction": key[5],
                    "correct": correct,
                    "n": len(group),
                    "accuracy_pct": round(100 * correct / len(group), 2),
                    "mean_correct_rank": (sum(ranks) / len(ranks)) if ranks else None,
                    "failures": json.dumps(dict(failures[key]), sort_keys=True),
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
