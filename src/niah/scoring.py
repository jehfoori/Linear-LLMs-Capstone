from __future__ import annotations

import re
from typing import Any


def extract_first_number(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"\d+", text)
    return match.group(0) if match else None


def score_generation(generated_text: str, answer: str) -> dict[str, Any]:
    pred_number = extract_first_number(generated_text)
    return {
        "pred_number": pred_number,
        "correct": pred_number == str(answer),
    }


def classify_failure(pred_number: str | None, answer: str, records: list[dict[str, Any]]) -> str:
    if pred_number is None:
        return "no_number_generated"

    pred = str(pred_number)
    target = str(answer)
    if pred == target:
        return "correct"

    distractor_answers = {str(record["answer"]) for record in records if not record.get("is_target", False)}
    if pred in distractor_answers:
        return "distractor_value"

    all_answers = {str(record["answer"]) for record in records}
    if pred in all_answers:
        return "other_record_value"

    return "not_in_document"


def score_example(generated_text: str, example: dict[str, Any]) -> dict[str, Any]:
    score = score_generation(generated_text, str(example["answer"]))
    score["failure_type"] = classify_failure(
        score["pred_number"],
        str(example["answer"]),
        list(example.get("records", [])),
    )
    return score
