from niah.models import MambaSSMLMRunner, build_runner


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
