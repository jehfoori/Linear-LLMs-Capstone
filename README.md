# Linear LLM NIAH Reproduction Pipeline

This repo turns the original Colab notebooks into a lightweight, reproducible
pipeline for passkey / needle-in-a-haystack retrieval experiments.

The intended workflow is:

1. Generate a fixed paired dataset locally.
2. Run model inference on Colab or a GPU VM.
3. Bring the run folder back here.
4. Analyze, compare, plot, and write the report locally.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,plot]"
pytest
```

GPU machines should install only the extras they need:

```bash
pip install -e ".[mamba2]"
pip install -e ".[gated-deltanet]"
```

For the official `mamba-ssm` Mamba-2 path, use Python 3.11 and the matched
Torch 2.3 wheels. This avoids local CUDA extension builds:

```bash
python3.11 -m venv .venv-mamba-official
source .venv-mamba-official/bin/activate
pip install --upgrade pip
pip install torch==2.3.0 transformers==4.46.3 packaging ninja einops
pip install --no-deps \
  "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu122torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" \
  "https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
pip install -e .
```

For the Gated DeltaNet path, the clean import stack found on the A100 pod is:

```bash
python3.11 -m venv .venv-gdn
source .venv-gdn/bin/activate
pip install --upgrade pip
pip install -e ".[gated-deltanet]"
```

This loads FLA registrations with `torch==2.5.1`, `triton==3.1.0`, and
`flash-linear-attention==0.3.2`. The current public checkpoint still needs a
dedicated config/weight conversion loader before evaluation.

## Core Commands

Generate a tiny smoke dataset:

```bash
python -m niah.cli.generate_dataset \
  --config configs/dataset_d20_smoke.yaml \
  --out datasets/d20_smoke.jsonl
```

Generate a paired distractor dataset:

```bash
python -m niah.cli.generate_dataset \
  --config configs/dataset_d20_paired.yaml \
  --out datasets/d20_paired.jsonl
```

Generate a token-calibrated dataset on a GPU/remote machine:

```bash
python -m niah.cli.generate_dataset \
  --config configs/dataset_d20_paired.yaml \
  --tokenizer-id benchang1110/mamba2-370m-hf \
  --out datasets/d20_paired_mamba2tok.jsonl
```

Evaluate a model:

```bash
python -m niah.cli.evaluate \
  --dataset datasets/d20_paired.jsonl \
  --model-config configs/model_mamba2_official.yaml \
  --out results/mamba2_official_d20_paired
```

Run a cheap smoke evaluation before spending real GPU time:

```bash
python -m niah.cli.evaluate \
  --dataset datasets/d20_paired.jsonl \
  --model-config configs/model_mamba2_official.yaml \
  --out results/mamba2_official_smoke \
  --limit 2
```

Or target specific examples:

```bash
python -m niah.cli.evaluate \
  --dataset datasets/d20_paired.jsonl \
  --model-config configs/model_mamba2_official.yaml \
  --out results/mamba2_official_selected \
  --example-ids passkey_distractors_1024_0000 passkey_distractors_4096_0000
```

Analyze one run:

```bash
python -m niah.cli.analyze_results \
  --run-dir results/mamba2_official_d20_paired
```

Compare paired runs:

```bash
python -m niah.cli.compare_runs \
  --run-dirs results/mamba2_official_d20_paired results/gated_deltanet_d20_paired \
  --out results/comparison_d20_paired
```

## Design Notes

- Datasets are canonical JSONL prompt files.
- Predictions are JSONL plus CSV for spreadsheet-friendly inspection.
- Every run writes `manifest.json`, `environment.json`, and
  `model_load_report.json`.
- Model-specific loading stays isolated in `src/niah/models.py`.
- The report and archived notebooks live under `report/` and
  `notebooks/archived/`.
