from niah.models import HazyResearchLMRunner, MambaSSMLMRunner, build_runner


def test_build_runner_supports_mamba_ssm_without_importing_dependencies():
    runner = build_runner(
        {
            "runner": "mamba_ssm_lm",
            "model_id": "state-spaces/mamba2-370m",
            "tokenizer_id": "EleutherAI/gpt-neox-20b",
        }
    )

    assert isinstance(runner, MambaSSMLMRunner)
    assert runner.label == "state-spaces/mamba2-370m"


def test_build_runner_supports_hazyresearch_without_importing_dependencies():
    runner = build_runner(
        {
            "runner": "hazyresearch_lm",
            "model_id": "hazyresearch/based-360m",
            "architecture": "based",
        }
    )

    assert isinstance(runner, HazyResearchLMRunner)
    assert runner.label == "hazyresearch/based-360m"
    assert runner.decode_strategy == "recompute"


def test_hazyresearch_non_based_defaults_to_cached_decode():
    runner = build_runner(
        {
            "runner": "hazyresearch_lm",
            "model_id": "hazyresearch/mamba-360m",
            "architecture": "mamba",
        }
    )

    assert isinstance(runner, HazyResearchLMRunner)
    assert runner.decode_strategy == "cached"
