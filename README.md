# Linear LLM Synthetic Retrieval Experiments

This repository contains the code and report for a capstone study of synthetic
key-value retrieval in efficient language models. The experiments evaluate
public checkpoints on passkey-style needle-in-a-haystack tasks, with emphasis
on context length, distractor pressure, and prompt-aligned scoring.

The final report is available at:

- `report/main.tex`
- `report/main.pdf`

Generated datasets and full run outputs are intentionally ignored by git. The
scripts below regenerate the datasets and write results under `datasets/` and
`results/`.

## Repository Layout

- `src/niah/`: dataset generation, model runners, scoring, analysis, and CLI
  utilities.
- `configs/`: model and dataset configs used by the final experiments and
  retained diagnostics.
- `scripts/`: final experiment entry points.
- `scripts/diagnostics/`: targeted diagnostic probes discussed in the report.
- `requirements/`: GPU environment requirement files for the HazyResearch and
  State Spaces model families.
- `report/`: final LaTeX report and generated figures.
- `tests/`: local unit tests for dataset generation, scoring, analysis, and
  runner selection.

## Local Development

Local setup is meant for editing, testing, dataset generation without model
tokenization, and report work:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,plot]"
pytest
```

The GPU experiments require CUDA-capable remote compute. The scripts were run
on an A100 RunPod instance, but any comparable GPU VM should work if the same
Python environments are created.

## Remote GPU Environments

Clone the repository on the remote machine, then create separate environments
for the two checkpoint families. Keeping them separate avoids dependency
conflicts between the HazyResearch BASED package and the official State Spaces
Mamba stack.

### HazyResearch / BASED Environment

```bash
python3.11 -m venv /workspace/.venv-based
source /workspace/.venv-based/bin/activate
pip install --upgrade pip wheel
pip install -r requirements/based-base.txt
pip install --no-build-isolation -r requirements/based-kernels.txt
git config --global url.https://github.com/.insteadOf git@github.com:
pip install --no-build-isolation -r requirements/based-package.txt
pip install huggingface_hub==0.23.5
pip install -e .
```

The final HazyResearch experiment uses fp32 BASED configs with recompute
decoding. This avoids the cached-generation artifact observed in the public
BASED fallback path when optimized kernels are unavailable.

### State Spaces Mamba Environment

```bash
python3.11 -m venv /workspace/.venv-mamba-official23
source /workspace/.venv-mamba-official23/bin/activate
pip install --upgrade pip
pip install -r requirements/mamba-official.txt
pip install -e .
```

This environment is used for the official `state-spaces/mamba-2.8b`,
`state-spaces/mamba2-2.7b`, and `state-spaces/mamba2attn-2.7b` checkpoints.
The scripts default to `/workspace/.venv-mamba-official23/bin/python`; override
`MAMBA_PY` if your environment path differs.

## Final Experiments

### Experiment 1: HazyResearch Context Sweep

This experiment compares HazyResearch attention, Mamba, and BASED checkpoints
on single-needle and 20-distractor multi-key retrieval. It uses target context
lengths from 512 to 2560 GPT-2 tokens with 30 examples per setting.

```bash
scripts/run_hazy_context_sweep_n30_l512_2560_clean.sh
```

Outputs are written under:

```text
results/hazy_context_sweep_n30_l512_2560_clean/
```

The script generates token-calibrated datasets, evaluates all six checkpoints,
and writes comparison summaries for the two tasks.

### Experiment 2: State Spaces Mamba Family

This experiment compares official Mamba, Mamba-2, and Mamba-2-Attn checkpoints
at 2048, 4096, and 8192 GPT-NeoX-token contexts with 0, 5, and 10 distractors.

Run free-generation accuracy:

```bash
scripts/run_state_spaces_mamba_family_primary.sh
```

Run the prompt-aligned completion-likelihood diagnostic:

```bash
scripts/run_state_spaces_mamba_family_completion_likelihood.sh
```

Both scripts support `RUN_MODE=smoke`, `RUN_MODE=full`, or `RUN_MODE=all`.
For example:

```bash
RUN_MODE=smoke scripts/run_state_spaces_mamba_family_primary.sh
RUN_MODE=full scripts/run_state_spaces_mamba_family_primary.sh
```

Outputs are written under:

```text
results/state_spaces_mamba_family_primary/
results/state_spaces_mamba_family_completion_likelihood/
```

## Diagnostics

The final report references several methodological diagnostics. They are kept
separate from the main experiment scripts:

```bash
scripts/diagnostics/run_state_spaces_2p7b_probe.sh
scripts/diagnostics/run_state_spaces_ruler_diagnostic.sh
scripts/diagnostics/run_state_spaces_numeric_likelihood_diagnostic.sh
scripts/diagnostics/run_state_spaces_query_position_diagnostic.sh
python scripts/diagnostics/run_transformerpp_diagnostics.py --help
python scripts/diagnostics/run_attn1b_prompt_probe.py --help
```

These diagnostics are useful for inspecting prompt-format sensitivity,
Transformer++ task-mode collapse, and alternative likelihood scoring. They are
not required to reproduce the two main result matrices.

## Core CLI Utilities

The scripts above are thin wrappers around the package CLIs. The main commands
are:

```bash
python -m niah.cli.generate_dataset --config <dataset.yaml> --out <dataset.jsonl>
python -m niah.cli.evaluate --dataset <dataset.jsonl> --model-config <model.yaml> --out <run_dir>
python -m niah.cli.analyze_results --run-dir <run_dir>
python -m niah.cli.compare_runs --run-dirs <run_a> <run_b> --out <comparison_dir>
```

Every run writes `manifest.json`, `environment.json`, `model_load_report.json`,
`predictions.jsonl`, `predictions.csv`, and summary tables when analysis is
run.
