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
pip install -r requirements/mamba-official.txt
pip install -e .
```

For the Gated DeltaNet path, the clean import stack found on the A100 pod is:

```bash
python3.11 -m venv .venv-gdn
source .venv-gdn/bin/activate
pip install --upgrade pip
pip install -r requirements/gated-deltanet.txt
pip install -e .
```

This loads FLA registrations with `torch==2.5.1`, `triton==3.1.0`,
`flash-linear-attention==0.3.2`, and a matched `causal-conv1d` wheel. The
public checkpoint is loaded through the dedicated `gated_deltanet_converted`
runner, which patches the missing `intermediate_size`, splits fused MLP gate/up
projection weights, and records missing/unexpected keys in
`model_load_report.json`. The config disables fused SwiGLU and patches the
SwiGLU activation to an equivalent pure PyTorch fallback because that Triton
kernel is unstable in the clean Torch 2.5 stack.

For the HazyResearch Based/Mamba/Attention comparison, use a separate Python
3.11 environment. The Based package and its CUDA extensions expect torch to be
installed before build metadata is prepared, so install it in phases:

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

The final `huggingface_hub` reinstall resolves a transitive
`accelerate`/`huggingface_hub` mismatch while preserving the Based model-loading
path.

The public BASED package can produce unstable cached recurrent generation in
the fallback path when the optimized kernels are unavailable. BASED model
configs therefore set `decode_strategy: recompute`, which greedily decodes by
recomputing the full sequence for each new token. This is slower, but keeps
accuracy runs on the full-forward path and avoids cache-specific artifacts.

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

## RULER-Style Synthetic Extension

The HazyResearch Based/Mamba/Attention comparison can also be run as a small
RULER-inspired synthetic suite at nominal 1k and 2k GPT-2-token contexts with
30 examples per length:

- `single_needle`: one passkey record and one exact numeric query.
- `multi_key`: one target passkey plus 20 same-format distractor records.
- `variable_tracking`: a two-hop variable reference chain plus 20 distractor bindings.

On the A100 pod, run:

```bash
scripts/run_hazy_ruler_n30_l1024_2048.sh
```

Outputs are written under `results/hazy_ruler_n30_l1024_2048/`, with one run
directory per task/model and comparison tables under `comparisons/`.

## HazyResearch Context Sweep

The context sweep narrows the task set to the two most diagnostic settings and
adds finer context-length resolution:

- `single_needle`
- `multi_key`: one target passkey plus 20 same-format distractor records

It uses nominal `512, 768, 1024, 1280, 1536, 1792, 2048, 2560` GPT-2-token
contexts with 20 examples per length:

```bash
scripts/run_hazy_context_sweep_n20_l512_2560.sh
```

Outputs are written under `results/hazy_context_sweep_n20_l512_2560/`.

## Design Notes

- Datasets are canonical JSONL prompt files.
- Predictions are JSONL plus CSV for spreadsheet-friendly inspection.
- Every run writes `manifest.json`, `environment.json`, and
  `model_load_report.json`.
- Model-specific loading stays isolated in `src/niah/models.py`.
- The report and archived notebooks live under `report/` and
  `notebooks/archived/`.
