from niah.data import (
    build_length_counter,
    generate_dataset,
    generate_distractor_example,
    generate_variable_tracking_example,
)


def test_distractor_example_invariants():
    row = generate_distractor_example(
        example_id="d20_1024_0000",
        target_length=256,
        seed=123,
        num_distractors=20,
    )

    records = row["records"]
    target_records = [record for record in records if record["is_target"]]
    assert len(records) == 21
    assert len(target_records) == 1
    assert row["needle_sentence"] == target_records[0]["line"]
    assert row["prompt"].count(row["needle_sentence"]) == 1
    assert row["prompt"].count(row["answer"]) == 1
    assert f"PASSKEY_RECORD[{row['key']}] =" in row["prompt"].split("End of document.")[-1]
    assert len({record["key"] for record in records}) == len(records)
    assert len({record["answer"] for record in records}) == len(records)


def test_dataset_generation_is_deterministic():
    config = {
        "task": "passkey_distractors",
        "target_lengths": [128, 256],
        "n_per_length": 3,
        "num_distractors": 2,
        "seed": 99,
    }
    first = generate_dataset(config)
    second = generate_dataset(config)
    assert first == second
    assert len(first) == 6
    assert len({row["example_id"] for row in first}) == 6


def test_dataset_generation_accepts_multiple_distractor_counts():
    config = {
        "task": "passkey_distractors",
        "target_lengths": [128],
        "n_per_length": 2,
        "num_distractors": [0, 5],
        "seed": 99,
    }

    rows = generate_dataset(config)

    assert len(rows) == 4
    assert {row["num_distractors"] for row in rows} == {0, 5}
    assert len({row["example_id"] for row in rows}) == 4
    assert any("_d0_" in row["example_id"] for row in rows)
    assert any("_d5_" in row["example_id"] for row in rows)


def test_variable_tracking_example_invariants():
    row = generate_variable_tracking_example(
        example_id="variable_tracking_1024_0000",
        target_length=256,
        seed=456,
        num_distractors=5,
        num_hops=2,
    )

    records = row["records"]
    target_records = [record for record in records if record["is_target"]]
    assert row["task"] == "variable_tracking"
    assert row["num_hops"] == 2
    assert len(records) == 8
    assert len(target_records) == 3
    assert row["needle_sentence"].endswith(row["answer"])
    assert row["prompt"].count(row["needle_sentence"]) == 1
    assert row["answer"] in row["prompt"]
    assert f"VAR_RECORD[{row['key']}] =" in row["prompt"].split("End of document.")[-1]
    assert row["query_record_index"] != row["target_record_index"]
    assert len(row["chain_keys"]) == 3
    assert len({record["key"] for record in records}) == len(records)
    assert len({record["answer"] for record in records if not record["is_target"]}) == 5


def test_default_length_counter_is_approx_words():
    count_length, length_metric = build_length_counter({})
    assert length_metric == "approx_words"
    assert count_length("one two three") == 3
