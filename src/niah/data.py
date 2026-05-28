from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError:  # pragma: no cover - dependency is declared in pyproject
    yaml = None


FILLER_SENTENCES = [
    "The grass is green. The sky is blue. The sun is yellow.",
    "Here we go. There and back again. The story continues.",
    "A quiet room contains many ordinary objects and no useful clues.",
    "The passage is intentionally repetitive and mostly irrelevant.",
    "This sentence is filler text used to extend the context length.",
]

KEY_WORDS_A = [
    "silent",
    "silver",
    "crimson",
    "hidden",
    "frozen",
    "bright",
    "ancient",
    "gentle",
    "rapid",
    "lonely",
    "golden",
    "quiet",
    "distant",
    "blue",
]

KEY_WORDS_B = [
    "river",
    "forest",
    "planet",
    "window",
    "signal",
    "garden",
    "harbor",
    "mountain",
    "lantern",
    "ocean",
    "valley",
    "engine",
    "mirror",
    "cloud",
]

VARIABLE_WORDS_A = [
    "amber",
    "basil",
    "cobalt",
    "delta",
    "ember",
    "fable",
    "ginger",
    "hazel",
    "indigo",
    "jade",
    "kelp",
    "lilac",
    "marble",
    "nylon",
    "opal",
]

VARIABLE_WORDS_B = [
    "anchor",
    "beacon",
    "cipher",
    "drift",
    "echo",
    "flare",
    "grove",
    "hinge",
    "island",
    "jewel",
    "kernel",
    "ledger",
    "matrix",
    "notion",
    "orbit",
]


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if yaml is None:
        return parse_simple_yaml(text)
    data = yaml.safe_load(text)
    return data or {}


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by this repo's configs.

    This is intentionally conservative: top-level key/value pairs plus one
    level of nested mappings. It keeps the local pipeline dependency-light
    while still allowing PyYAML to handle richer configs when installed.
    """

    root: dict[str, Any] = {}
    current_map: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported config line: {raw_line!r}")
        key, raw_value = stripped.split(":", 1)
        raw_value = raw_value.strip()
        if indent == 0:
            if raw_value == "":
                current_map = {}
                root[key] = current_map
            else:
                root[key] = _parse_scalar(raw_value)
                current_map = None
        elif indent == 2 and current_map is not None:
            current_map[key] = _parse_scalar(raw_value)
        else:
            raise ValueError(f"Unsupported indentation in config line: {raw_line!r}")
    return root


def _parse_scalar(value: str) -> Any:
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if value == "{}":
        return {}
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")


def write_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_approx_tokens(text: str) -> int:
    return len(text.split())


def make_key(rng: random.Random) -> str:
    return f"{rng.choice(KEY_WORDS_A)}-{rng.choice(KEY_WORDS_B)}"


def make_answer(rng: random.Random) -> str:
    return str(rng.randint(1000000, 9999999))


def make_unique_key(rng: random.Random, used_keys: set[str]) -> str:
    while True:
        key = make_key(rng)
        if key not in used_keys:
            used_keys.add(key)
            return key


def make_variable(rng: random.Random) -> str:
    return f"{rng.choice(VARIABLE_WORDS_A)}_{rng.choice(VARIABLE_WORDS_B)}"


def make_unique_variable(rng: random.Random, used_variables: set[str]) -> str:
    while True:
        variable = make_variable(rng)
        if variable not in used_variables:
            used_variables.add(variable)
            return variable


def _filler_until(target_length: int, make_prompt, count_length=count_approx_tokens) -> tuple[str, int]:
    filler_unit = " ".join(FILLER_SENTENCES) + " "
    filler = ""
    while True:
        filler += filler_unit
        prompt = make_prompt(filler)
        length_count = count_length(prompt)
        if length_count >= target_length:
            return prompt, length_count


def generate_single_example(
    *,
    example_id: str,
    target_length: int,
    seed: int,
    position_fraction: float | None = None,
    count_length=count_approx_tokens,
    length_metric: str = "approx_words",
) -> dict[str, Any]:
    rng = random.Random(seed)
    key = make_key(rng)
    answer = make_answer(rng)
    if position_fraction is None:
        position_fraction = rng.uniform(0.05, 0.95)

    needle = f"PASSKEY_RECORD[{key}] = {answer}"

    def make_prompt(filler: str) -> str:
        split_idx = int(len(filler) * position_fraction)
        query = (
            "\n\nEnd of document.\n"
            "Repeat the matching passkey record from the document above.\n"
            f"PASSKEY_RECORD[{key}] ="
        )
        return "Document:\n\n" + filler[:split_idx] + "\n\n" + needle + "\n\n" + filler[split_idx:] + query

    prompt, length_count = _filler_until(target_length, make_prompt, count_length)
    return {
        "example_id": example_id,
        "task": "passkey_single",
        "target_length": target_length,
        "approx_tokens": length_count,
        "length_metric": length_metric,
        "seed": seed,
        "key": key,
        "answer": answer,
        "needle_sentence": needle,
        "needle_position_fraction": position_fraction,
        "target_record_index": 0,
        "num_distractors": 0,
        "records": [{"key": key, "answer": answer, "is_target": True, "line": needle}],
        "prompt": prompt,
    }


def generate_distractor_example(
    *,
    example_id: str,
    target_length: int,
    seed: int,
    num_distractors: int,
    count_length=count_approx_tokens,
    length_metric: str = "approx_words",
) -> dict[str, Any]:
    rng = random.Random(seed)
    used_keys: set[str] = set()

    target_key = make_unique_key(rng, used_keys)
    target_answer = make_answer(rng)
    records = [
        {
            "key": target_key,
            "answer": target_answer,
            "is_target": True,
            "line": f"PASSKEY_RECORD[{target_key}] = {target_answer}",
        }
    ]

    used_answers = {target_answer}
    for _ in range(num_distractors):
        key = make_unique_key(rng, used_keys)
        answer = make_answer(rng)
        while answer in used_answers:
            answer = make_answer(rng)
        used_answers.add(answer)
        records.append(
            {
                "key": key,
                "answer": answer,
                "is_target": False,
                "line": f"PASSKEY_RECORD[{key}] = {answer}",
            }
        )

    rng.shuffle(records)
    target_record_index = next(i for i, record in enumerate(records) if record["is_target"])

    def make_prompt(filler: str) -> str:
        segment_size = max(1, len(filler) // (len(records) + 1))
        parts = []
        for i, record in enumerate(records):
            parts.append(filler[i * segment_size : (i + 1) * segment_size])
            parts.append("\n" + record["line"] + "\n")
        parts.append(filler[len(records) * segment_size :])
        query = (
            "\n\nEnd of document.\n"
            "Repeat the matching passkey record from the document above.\n"
            f"PASSKEY_RECORD[{target_key}] ="
        )
        return "Document:\n\n" + "".join(parts) + query

    prompt, length_count = _filler_until(target_length, make_prompt, count_length)
    return {
        "example_id": example_id,
        "task": "passkey_distractors",
        "target_length": target_length,
        "approx_tokens": length_count,
        "length_metric": length_metric,
        "seed": seed,
        "key": target_key,
        "answer": target_answer,
        "needle_sentence": records[target_record_index]["line"],
        "needle_position_fraction": (target_record_index + 1) / (len(records) + 1),
        "target_record_index": target_record_index,
        "num_distractors": num_distractors,
        "records": records,
        "prompt": prompt,
    }


def generate_variable_tracking_example(
    *,
    example_id: str,
    target_length: int,
    seed: int,
    num_distractors: int,
    num_hops: int = 2,
    count_length=count_approx_tokens,
    length_metric: str = "approx_words",
) -> dict[str, Any]:
    rng = random.Random(seed)
    used_variables: set[str] = set()
    used_answers: set[str] = set()

    target_answer = make_answer(rng)
    used_answers.add(target_answer)
    chain_variables = [make_unique_variable(rng, used_variables) for _ in range(num_hops + 1)]
    target_variable = chain_variables[-1]

    records: list[dict[str, Any]] = [
        {
            "key": chain_variables[0],
            "answer": target_answer,
            "is_target": True,
            "record_type": "value",
            "line": f"VAR_RECORD[{chain_variables[0]}] = {target_answer}",
        }
    ]
    for index in range(1, len(chain_variables)):
        records.append(
            {
                "key": chain_variables[index],
                "answer": target_answer,
                "is_target": True,
                "record_type": "alias",
                "source_key": chain_variables[index - 1],
                "line": f"VAR_RECORD[{chain_variables[index]}] = VAR_RECORD[{chain_variables[index - 1]}]",
            }
        )

    for _ in range(num_distractors):
        variable = make_unique_variable(rng, used_variables)
        answer = make_answer(rng)
        while answer in used_answers:
            answer = make_answer(rng)
        used_answers.add(answer)
        records.append(
            {
                "key": variable,
                "answer": answer,
                "is_target": False,
                "record_type": "value",
                "line": f"VAR_RECORD[{variable}] = {answer}",
            }
        )

    rng.shuffle(records)
    source_record_index = next(i for i, record in enumerate(records) if record["key"] == chain_variables[0])
    query_record_index = next(i for i, record in enumerate(records) if record["key"] == target_variable)

    def make_prompt(filler: str) -> str:
        segment_size = max(1, len(filler) // (len(records) + 1))
        parts = [
            "Rules:\n"
            "A VAR_RECORD can store either a number or a reference to another VAR_RECORD.\n"
            "If a VAR_RECORD points to another VAR_RECORD, it has the same numeric value.\n\n"
        ]
        for i, record in enumerate(records):
            parts.append(filler[i * segment_size : (i + 1) * segment_size])
            parts.append("\n" + record["line"] + "\n")
        parts.append(filler[len(records) * segment_size :])
        query = (
            "\n\nEnd of document.\n"
            "Resolve the requested variable and write only its numeric value.\n"
            f"VAR_RECORD[{target_variable}] ="
        )
        return "Document:\n\n" + "".join(parts) + query

    prompt, length_count = _filler_until(target_length, make_prompt, count_length)
    return {
        "example_id": example_id,
        "task": "variable_tracking",
        "target_length": target_length,
        "approx_tokens": length_count,
        "length_metric": length_metric,
        "seed": seed,
        "key": target_variable,
        "answer": target_answer,
        "needle_sentence": records[source_record_index]["line"],
        "needle_position_fraction": (source_record_index + 1) / (len(records) + 1),
        "target_record_index": source_record_index,
        "query_record_index": query_record_index,
        "query_record_position_fraction": (query_record_index + 1) / (len(records) + 1),
        "num_distractors": num_distractors,
        "num_hops": num_hops,
        "chain_keys": chain_variables,
        "records": records,
        "prompt": prompt,
    }


def generate_dataset(config: dict[str, Any]) -> list[dict[str, Any]]:
    task = config.get("task", "passkey_distractors")
    target_lengths = [int(v) for v in config.get("target_lengths", [1024, 4096, 8192, 16384])]
    n_per_length = int(config.get("n_per_length", 50))
    base_seed = int(config.get("seed", 812345))
    raw_distractors = config.get("num_distractors", 20)
    num_distractor_values = [int(v) for v in raw_distractors] if isinstance(raw_distractors, list) else [int(raw_distractors)]
    num_hops = int(config.get("num_hops", 2))
    count_length, length_metric = build_length_counter(config)

    rows: list[dict[str, Any]] = []
    for target_length in target_lengths:
        for num_distractors in num_distractor_values:
            for index in range(n_per_length):
                seed = base_seed + target_length * 1000 + num_distractors * 100 + index
                distractor_suffix = f"_d{num_distractors}" if len(num_distractor_values) > 1 else ""
                example_id = f"{task}{distractor_suffix}_{target_length}_{index:04d}"
                if task == "passkey_single":
                    row = generate_single_example(
                        example_id=example_id,
                        target_length=target_length,
                        seed=seed,
                        count_length=count_length,
                        length_metric=length_metric,
                    )
                    row["num_distractors"] = num_distractors
                elif task == "passkey_distractors":
                    row = generate_distractor_example(
                        example_id=example_id,
                        target_length=target_length,
                        seed=seed,
                        num_distractors=num_distractors,
                        count_length=count_length,
                        length_metric=length_metric,
                    )
                elif task == "variable_tracking":
                    row = generate_variable_tracking_example(
                        example_id=example_id,
                        target_length=target_length,
                        seed=seed,
                        num_distractors=num_distractors,
                        num_hops=num_hops,
                        count_length=count_length,
                        length_metric=length_metric,
                    )
                else:
                    raise ValueError(f"Unknown dataset task: {task}")
                rows.append(row)
    return rows


def build_length_counter(config: dict[str, Any]):
    tokenizer_id = config.get("tokenizer_id")
    if not tokenizer_id:
        return count_approx_tokens, "approx_words"

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for tokenizer-calibrated dataset generation.") from exc

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_id), trust_remote_code=bool(config.get("trust_remote_code", False)))

    def count_with_tokenizer(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    return count_with_tokenizer, f"tokens:{tokenizer_id}"
