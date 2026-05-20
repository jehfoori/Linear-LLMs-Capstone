from niah.models import GatedDeltaNetConvertedRunner, HazyResearchLMRunner, MambaSSMLMRunner, build_runner


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


def test_build_runner_supports_converted_gated_deltanet_without_importing_dependencies():
    runner = build_runner(
        {
            "runner": "gated_deltanet_converted",
            "model_id": "linear-moe-hub/Gated-Deltanet-340M",
        }
    )

    assert isinstance(runner, GatedDeltaNetConvertedRunner)
    assert runner.load_report.manual_config_patch is True
    assert runner.load_report.manual_weight_conversion is True


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
