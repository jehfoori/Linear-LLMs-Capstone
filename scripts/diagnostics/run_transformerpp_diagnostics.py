from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from niah.data import generate_single_example, write_json, write_jsonl
from niah.scoring import score_example
from transformers import AutoTokenizer


MODEL_ID = "state-spaces/transformerpp-2.7b"
MODEL_REVISION = "15a431b71c40c284138c379d07d4008a28fea397"
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
TOKENIZER_REVISION = "c292233c833e336628618a88a648727eb3dff0a7"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run targeted Transformer++ long-context diagnostics.")
    parser.add_argument("--out", default="results/state_spaces_transformerpp_diagnostics")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION)

    def count_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    model_source = snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=["config.json", "pytorch_model.bin"])
    dtype = getattr(torch, args.dtype)
    model = MambaLMHeadModel.from_pretrained(model_source, device=args.device, dtype=dtype)
    model.eval()

    examples = make_examples(count_tokens)
    write_jsonl(out_dir / "diagnostic_examples.jsonl", examples)

    rows: list[dict[str, Any]] = []
    for item in diagnostic_items(examples):
        example = dict(item["example"])
        if item["prompt_variant"] == "strict":
            example["prompt"] = strict_prompt(example)

        row = run_one(
            model=model,
            tokenizer=tokenizer,
            torch=torch,
            example=example,
            variant=item["variant"],
            prompt_variant=item["prompt_variant"],
            decode_method=item["decode_method"],
            max_new_tokens=item["max_new_tokens"],
            device=args.device,
        )
        rows.append(row)
        print(
            f"{row['variant']} {row['example_id']} "
            f"correct={row['correct']} pred={row['pred_number']} "
            f"fail={row['failure_type']} gen={row['generated_text']!r}",
            flush=True,
        )

    write_rows(out_dir / "predictions.csv", rows)
    write_jsonl(out_dir / "predictions.jsonl", rows)
    write_summary(out_dir / "summary.csv", rows)
    write_json(
        out_dir / "manifest.json",
        {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_id": TOKENIZER_ID,
            "tokenizer_revision": TOKENIZER_REVISION,
            "diagnostics": [
                "boundary_generate_8",
                "long_generation_32",
                "recompute_8",
                "strict_prompt_generate_8",
            ],
        },
    )


def make_examples(count_tokens) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_length in [2048, 2304, 2560, 3072, 3584, 4096]:
        for index in range(3):
            rows.append(
                generate_single_example(
                    example_id=f"transformerpp_diag_single_{target_length}_{index:04d}",
                    target_length=target_length,
                    seed=982024 + target_length * 1000 + index,
                    count_length=count_tokens,
                    length_metric=f"tokens:{TOKENIZER_ID}",
                )
            )
    return rows


def diagnostic_items(examples: list[dict[str, Any]]):
    examples_4096 = [row for row in examples if row["target_length"] == 4096]
    for example in examples:
        yield {
            "variant": "boundary_generate_8",
            "prompt_variant": "baseline",
            "decode_method": "generate",
            "max_new_tokens": 8,
            "example": example,
        }
    for example in examples_4096:
        yield {
            "variant": "long_generation_32",
            "prompt_variant": "baseline",
            "decode_method": "generate",
            "max_new_tokens": 32,
            "example": example,
        }
    for example in examples_4096:
        yield {
            "variant": "recompute_8",
            "prompt_variant": "baseline",
            "decode_method": "recompute",
            "max_new_tokens": 8,
            "example": example,
        }
    for example in examples_4096:
        yield {
            "variant": "strict_prompt_generate_8",
            "prompt_variant": "strict",
            "decode_method": "generate",
            "max_new_tokens": 8,
            "example": example,
        }


def strict_prompt(example: dict[str, Any]) -> str:
    marker = "\n\nEnd of document.\n"
    prefix = example["prompt"].split(marker, 1)[0]
    return (
        prefix
        + marker
        + "Answer the query using the document above. Do not continue the document.\n"
        + "Write only the 7-digit numeric value for this key.\n"
        + f"PASSKEY_RECORD[{example['key']}] ="
    )


def run_one(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    example: dict[str, Any],
    variant: str,
    prompt_variant: str,
    decode_method: str,
    max_new_tokens: int,
    device: str,
) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    input_ids = tokenizer(example["prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    input_len = int(input_ids.shape[1])
    start = time.time()
    if decode_method == "generate":
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=input_len + max_new_tokens,
                top_k=1,
                top_p=0.0,
                min_p=0.0,
                temperature=1.0,
                return_dict_in_generate=False,
                output_scores=False,
            )
    elif decode_method == "recompute":
        output_ids = generate_recompute(model, torch, input_ids, max_new_tokens)
    else:
        raise ValueError(f"Unknown decode method: {decode_method}")

    elapsed_sec = time.time() - start
    generated_ids = output_ids[0, input_len:].detach().cpu()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    score = score_example(generated_text, example)
    peak_mem_gib = None
    if torch.cuda.is_available():
        peak_mem_gib = torch.cuda.max_memory_allocated() / 1024**3

    return {
        "variant": variant,
        "prompt_variant": prompt_variant,
        "decode_method": decode_method,
        "max_new_tokens": max_new_tokens,
        "example_id": example["example_id"],
        "target_length": example["target_length"],
        "input_tokens": input_len,
        "answer": str(example["answer"]),
        "pred_number": score["pred_number"],
        "correct": score["correct"],
        "failure_type": score["failure_type"],
        "generated_text": generated_text,
        "elapsed_sec": elapsed_sec,
        "peak_mem_gib": peak_mem_gib,
    }


def generate_recompute(model: Any, torch: Any, input_ids: Any, max_new_tokens: int) -> Any:
    output_ids = input_ids
    for _ in range(max_new_tokens):
        with torch.inference_mode():
            try:
                logits = model(output_ids, num_last_tokens=1).logits[:, -1, :]
            except TypeError:
                logits = model(output_ids).logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        output_ids = torch.cat([output_ids, next_token], dim=1)
    return output_ids


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "variant",
        "prompt_variant",
        "decode_method",
        "max_new_tokens",
        "example_id",
        "target_length",
        "input_tokens",
        "answer",
        "pred_number",
        "correct",
        "failure_type",
        "generated_text",
        "elapsed_sec",
        "peak_mem_gib",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    failures: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = (row["variant"], int(row["target_length"]))
        grouped[key].append(row)
        failures[key][str(row["failure_type"])] += 1

    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["variant", "target_length", "correct", "n", "accuracy_pct", "failures"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(grouped):
            group = grouped[key]
            correct = sum(1 for row in group if row["correct"])
            writer.writerow(
                {
                    "variant": key[0],
                    "target_length": key[1],
                    "correct": correct,
                    "n": len(group),
                    "accuracy_pct": round(100 * correct / len(group), 2),
                    "failures": json.dumps(dict(failures[key]), sort_keys=True),
                }
            )


if __name__ == "__main__":
    main()
