from niah.scoring import classify_failure, extract_first_number, score_example


def test_extract_first_number_handles_messy_text():
    assert extract_first_number(" 4839201\nThe grass is") == "4839201"
    assert extract_first_number("abc 12 then 34") == "12"
    assert extract_first_number("") is None
    assert extract_first_number(None) is None


def test_failure_classification():
    records = [
        {"answer": "1111111", "is_target": True},
        {"answer": "2222222", "is_target": False},
    ]
    assert classify_failure("1111111", "1111111", records) == "correct"
    assert classify_failure("2222222", "1111111", records) == "distractor_value"
    assert classify_failure("3333333", "1111111", records) == "not_in_document"
    assert classify_failure(None, "1111111", records) == "no_number_generated"


def test_score_example_uses_records():
    example = {
        "answer": "1111111",
        "records": [
            {"answer": "1111111", "is_target": True},
            {"answer": "2222222", "is_target": False},
        ],
    }
    score = score_example(" 2222222", example)
    assert score["pred_number"] == "2222222"
    assert score["correct"] is False
    assert score["failure_type"] == "distractor_value"
