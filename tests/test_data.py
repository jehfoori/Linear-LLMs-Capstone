from niah.data import generate_dataset, generate_distractor_example


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
