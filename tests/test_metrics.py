from niah.metrics import accuracy_summary, position_summary, wilson_ci


def test_wilson_ci_known_value():
    lo, hi = wilson_ci(32, 50)
    assert round(100 * lo, 1) == 50.1
    assert round(100 * hi, 1) == 75.9


def test_accuracy_summary_groups_rows():
    rows = [
        {"model_label": "A", "target_length": 1024, "correct": True},
        {"model_label": "A", "target_length": 1024, "correct": False},
        {"model_label": "A", "target_length": 4096, "correct": True},
    ]
    summary = accuracy_summary(rows)
    by_length = {row["target_length"]: row for row in summary}
    assert by_length[1024]["correct"] == 1
    assert by_length[1024]["n"] == 2
    assert by_length[1024]["accuracy_pct"] == 50
    assert by_length[4096]["accuracy_pct"] == 100


def test_position_summary_bins_rows():
    rows = [
        {"model_label": "A", "target_length": 1024, "correct": True, "needle_position_fraction": 0.1},
        {"model_label": "A", "target_length": 1024, "correct": False, "needle_position_fraction": 0.5},
        {"model_label": "A", "target_length": 1024, "correct": True, "needle_position_fraction": 0.9},
    ]
    summary = position_summary(rows)
    bins = {row["position_bin"]: row for row in summary}
    assert bins["early"]["accuracy_pct"] == 100
    assert bins["middle"]["accuracy_pct"] == 0
    assert bins["late"]["accuracy_pct"] == 100
