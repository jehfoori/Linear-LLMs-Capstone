# Experiment Journal

Working notes for the capstone reproduction pipeline. This is not final report
prose; it is a chronological record of what we tried, what failed, what worked,
and what decisions came out of the process.

Last updated: 2026-05-21

## 1. Original Colab-Based Experiments

The project began in Google Colab, using notebooks to run synthetic
needle-in-a-haystack retrieval tests on long-context language models. The early
comparison centered on Mamba-2 and Gated DeltaNet.

The initial workflow was practical but brittle:

- Colab provided GPU access, but dependency state was hard to control.
- Model loading required several workarounds.
- Some official research-code paths were difficult to install directly.
- We often relied on Hugging Face checkpoints rather than exact publication
  artifacts.
- Memory pressure forced smaller batches, careful loading choices, and other
  notebook-level patches.

The first results suggested that Mamba-2 handled the passkey retrieval setting
substantially better than the public Gated DeltaNet checkpoints, especially
when distractor records were introduced. However, those results came with major
caveats because the Gated DeltaNet setup was not a clean reproduction of the
paper's private training/checkpoint conditions.

## 2. Report Review and Initial Concerns

The first report draft had a stronger hypothesis than the results could
support. In particular, the hypothesis implied architectural conclusions from
experiments that were affected by checkpoint availability, training-data
differences, and implementation constraints.

We identified two broad write-up issues:

- The comparison and hypothesis sections needed to be shortened and softened.
- The report needed clearer caveats around public checkpoints, especially for
  Gated DeltaNet.

We later revised the report to make the hypothesis more tentative and added a
short preface before results explaining model caveats.

## 3. Moving from Colab to a Local Repo Plus Remote Compute

Because Colab development was becoming messy, we decided to turn the notebooks
into a lightweight local repository with remote GPU execution.

The intended workflow became:

- Keep code, configs, datasets, analysis, and report artifacts in a normal git
  repository.
- Generate datasets deterministically from configs.
- Run expensive inference on a GPU pod.
- Persist result folders locally and on the pod.
- Analyze and compare results through scripts rather than notebook state.

The old project artifacts were moved into `OriginalArtifacts/`, while the new
pipeline was placed in `Linear-LLMs-Capstone/`. We initialized git, connected
the repository to GitHub, and organized the new code into configs, scripts,
source modules, tests, datasets, results, and report folders.

## 4. Dataset and Pipeline Setup

We implemented a lightweight NIAH pipeline with:

- JSON/YAML configs for datasets and models.
- Deterministic dataset generation.
- JSONL prompt files as canonical datasets.
- Model runners for several model families.
- Prediction outputs in JSONL and CSV.
- Run manifests, environment captures, and model load reports.
- Analysis scripts for summaries, failure modes, position summaries, and
  cross-run comparison tables.

The early dataset settings included:

- Single passkey retrieval.
- Passkey retrieval with distractor records.
- Later, RULER-style extensions such as multi-key retrieval and variable
  tracking.

## 5. Remote Compute Setup

We selected a RunPod secure cloud setup with an A100-SXM 80GB GPU. SSH access
was configured using the user's public key. There was an initial SSH issue
because the full public key line needed to include the `ssh-ed25519` prefix.
Once corrected, the pod became accessible.

The remote setup split environments by model family because dependencies were
not mutually clean:

- A Mamba/Mamba-2 path using the official `mamba-ssm` package.
- A Gated DeltaNet path using FLA-related dependencies.
- A HazyResearch Based/Mamba/Attention path using the public `based` package.

This split made the setup more verbose, but it avoided trying to force
incompatible research packages into one Python environment.

## 6. Mamba-2 Setup

The official Mamba-2 path became relatively stable. We installed the necessary
Mamba dependencies, pinned revisions where appropriate, and confirmed that the
models could load and run smoke tests.

Compared with Gated DeltaNet, Mamba-2 had fewer checkpoint and implementation
caveats. This made it a stronger candidate for clean reproduction-style
experiments.

## 7. Gated DeltaNet Setup and Caveats

Gated DeltaNet was substantially harder.

The key issues were:

- The paper's exact private checkpoints do not appear to be publicly available.
- Public checkpoints came from different Hugging Face sources.
- Some checkpoints had architectural/configuration differences that mattered,
  especially around `use_short_conv`.
- The FLA dependency stack had fragile version requirements.
- Some kernels or fused paths needed fallbacks.

We investigated both linear-moe-hub and m-a-p checkpoint sources. The m-a-p
source was preferable for same-source scaling because it offered similarly
sized 340M and 1.3B-style options and avoided some short-convolution concerns.

We added a dedicated converted Gated DeltaNet runner that:

- Loads public checkpoints through a controlled path.
- Patches missing config fields such as `intermediate_size`.
- Splits fused MLP gate/up projection weights.
- Records missing and unexpected keys.
- Uses a pure PyTorch SwiGLU fallback when the Triton path is unstable.

We also successfully installed `causal-conv1d` in the Gated DeltaNet
environment, improving the environment quality. Even so, the checkpoint
availability caveat remained serious enough that Gated DeltaNet became less
attractive for the main final experiment.

## 8. Early NIAH Results: Mamba-2 vs Gated DeltaNet

The early NIAH tests showed Gated DeltaNet performing much worse than Mamba-2,
especially in the presence of distractors.

We discussed whether this result reflected architecture or public-checkpoint
training differences. The conclusion was that it could not support a strong
architectural claim. Models of similar size and architecture can differ
substantially because of:

- Training data.
- Training recipe.
- Context-length curriculum.
- Checkpoint source.
- Tokenizer and formatting conventions.
- Public implementation quality.

This led us to reconsider the original comparison and look for a cleaner
experiment with matched checkpoint families.

## 9. Considering RetNet, Mamba, Mamba-2, and Other Linear-Time Models

We considered pivoting to RetNet, Mamba, and Mamba-2. The main concern was
whether RetNet had public official checkpoints suitable for a fair comparison.
If it lacked them, it would recreate the same problem as Gated DeltaNet.

We also clarified the Mamba vs Mamba-2 framing:

- Mamba-2 simplifies or restructures aspects of Mamba's state-space mechanics.
- One motivation is improved training parallelism and hardware efficiency.
- The potential tradeoff is reduced expressiveness relative to the original
  selective scan.
- A fair comparison would need to separate architectural quality from training
  and checkpoint differences.

This remained interesting, but the search for matched public checkpoints led us
toward the HazyResearch Based paper's released model family.

## 10. Pivot Toward the BASED Paper Model Family

We investigated the BASED paper and its public checkpoints. This became
promising because the paper provided a family of HazyResearch checkpoints:

- Attention baseline.
- Mamba baseline.
- BASED model.
- Two size regimes around 360M and 1.4B.

This gave us a cleaner comparison than Gated DeltaNet because the models were
from the same paper ecosystem and intended to support comparisons among
quadratic attention, Mamba, and BASED.

We decided not to directly recreate the BASED paper's throughput-centric
evaluation because:

- Their strongest claims involve recall relative to memory and speed costs.
- We could not get the optimized ThunderKittens-style kernel path working in a
  clean, faithful way.
- Exact speed comparisons would be misleading without those kernels.

Instead, we reframed the experiment as a long-context retrieval extension:
absolute NIAH/RULER-style accuracy across context length and distractor load.

## 11. HazyResearch Environment

We set up a separate HazyResearch environment on the A100 pod using:

- Python 3.11.
- Torch 2.1.2 with CUDA 11.8 wheels.
- The public `HazyResearch/based` package.
- `causal-conv1d` and `mamba-ssm` dependencies.

Some optimized BASED kernels were unavailable:

- Causal dot product kernel was not imported.
- FLA Triton kernels were not imported.
- ThunderKittens kernels were not installed.

This meant our BASED runs used fallback paths. That was acceptable for
accuracy-only experiments at first, but later became important when we found
generation-cache artifacts.

## 12. First HazyResearch NIAH Matrix

We ran a 1080-generation matrix:

- 6 models: attention, Mamba, BASED at 360M and 1.4B.
- 3 distractor settings: 0, 5, 20.
- 2 lengths: 1024 and 2048.
- 30 examples per setting.

The results showed:

- Attention was strongest, especially with distractors.
- Mamba and BASED struggled sharply when distractors were introduced.
- Based 360M showed suspicious behavior in single-needle settings across
  longer contexts.

This motivated a more RULER-style task expansion and then a finer context
sweep.

## 13. RULER-Style Extension

We then ran a RULER-inspired matrix with:

- `single_needle`.
- `multi_key` with 20 distractors.
- `variable_tracking` with 20 distractors.
- Lengths 1024 and 2048.
- 30 examples per length.

Variable tracking was too brittle and not very diagnostic; it broke even the
attention baseline in ways that looked format-sensitive. Multi-key remained
useful because attention still achieved meaningful results while compressed
memory models struggled.

We decided the most useful final task set was:

- Single-needle retrieval.
- Multi-key retrieval with distractors.

## 14. Finer Context Sweep

Because the 1024-to-2048 jump hid important behavior, we ran a finer context
sweep:

- Lengths: 512, 768, 1024, 1280, 1536, 1792, 2048, 2560.
- Tasks: single needle and multi-key with 20 distractors.
- Models: attention, Mamba, BASED at 360M and 1.4B.
- 20 examples per length.

This matrix exposed strange Based 360M behavior:

- In bfloat16, Based 360M appeared to collapse after roughly 1.1k tokens.
- In fp32, the collapse mostly disappeared, but a dramatic dip remained at
  length 1792.
- The dip then recovered at 2048 and 2560, which was not a plausible smooth
  context-length degradation curve.

This anomaly triggered a focused diagnostic pass.

## 15. Based 360M Precision and Cache Diagnostics

We first tested whether the Based 360M issue was simply bfloat16 precision.

Findings:

- bfloat16 caused a severe cliff around actual input lengths near 1.1k tokens.
- fp32 substantially improved Based 360M and removed the worst cliff.
- Disabling TF32 did not fix the remaining 1792 dip.

We then inspected individual generations. Many failed 1792 examples began with
the correct answer prefix and then drifted into nonnumeric junk. For example,
the answer might be `6503014`, while cached generation produced `650301...`
followed by unrelated tokens.

This pattern suggested that the model was finding the answer during prefill but
the cached recurrent generation path was unstable after the first few generated
tokens.

The decisive diagnostic compared two decoding strategies on the same examples:

- Cached generation using the package's normal `generate()` path.
- Recompute decoding, which reruns the full forward pass for each new token.

At length 1792, Based 360M fp32 changed from `2/20` with cached generation to
`20/20` with recompute decoding. Across the single-needle sweep, recompute
decoding produced a smooth high-accuracy curve:

- 512: 20/20
- 768: 20/20
- 1024: 20/20
- 1280: 20/20
- 1536: 20/20
- 1792: 20/20
- 2048: 19/20
- 2560: 18/20

Conclusion: the weird Based 360M valley was not a dataset, prompt, scoring, or
checkpoint problem. It was a cache/generation artifact in the public BASED
fallback path.

## 16. Custom Kernel Discussion

We considered trying to install the intended custom kernels, including
ThunderKittens-style BASED kernels.

Reasons not to make this the immediate path:

- The current pod is A100 with CUDA 11.8 runtime and no `nvcc`.
- Some newer Hazy kernel references are H100-oriented.
- The public research-code kernel path looks fragile.
- Our main issue is accuracy reliability, not speed measurement.
- Recompute decoding gives a clean accuracy path without depending on custom
  kernels.

Decision: do not block the experiment on custom kernels. Use recompute decoding
for BASED accuracy runs and document that speed/throughput claims are outside
the scope of our setup.

## 17. Recompute-Decoding Patch

We patched the pipeline so BASED runs use recompute decoding by default.

Changes:

- Added `decode_strategy` to the HazyResearch runner.
- Defaulted BASED models to `decode_strategy: recompute`.
- Left non-BASED Hazy models on cached generation.
- Recorded decode strategy in run manifests.
- Documented the choice in the README.
- Added lightweight tests for decode-strategy defaults.

Smoke test:

- Re-ran three previously failing Based 360M fp32 1792 examples through the
  normal CLI.
- All three passed.
- The manifest and model load report recorded `decode_strategy: recompute`.

This gives us a cleaner setup for future BASED accuracy runs.

## 18. Current Recommended Final Matrix

The next clean experiment should rerun the HazyResearch comparison from a clean
state using the patched pipeline.

Recommended final matrix:

- Models:
  - Attention 360M.
  - Mamba 360M.
  - BASED 360M.
  - Attention 1.4B.
  - Mamba 1.4B.
  - BASED 1.4B.
- Tasks:
  - Single needle.
  - Multi-key with 20 distractors.
- Context lengths:
  - 512, 768, 1024, 1280, 1536, 1792, 2048, 2560.
- Sample count:
  - Prefer 30 examples per length if runtime is acceptable.

This would create a clean, meaningful matrix using:

- Same code commit.
- Same dataset-generation pipeline.
- Same scoring logic.
- Same pod environment.
- Same manifest structure.
- BASED recompute decoding to avoid cache artifacts.

The final framing should be careful:

- We are not reproducing the BASED paper's speedup claims.
- We are extending the comparison to absolute long-context retrieval accuracy.
- Attention remains a strong baseline in exact retrieval.
- Compressed or linear-time memory architectures may struggle more with
  distractor-heavy retrieval, but claims should be tied to these public
  checkpoints and this synthetic task setting.

## 19. Clean n30 Context Sweep

We launched the clean final matrix after patching BASED recompute decoding.

Final matrix:

- Models:
  - Attention 360M.
  - Attention 1.4B.
  - Mamba 360M.
  - Mamba 1.4B.
  - BASED 360M fp32 with recompute decoding.
  - BASED 1.4B fp32 with recompute decoding.
- Tasks:
  - Single needle.
  - Multi-key with 20 distractors.
- Context lengths:
  - 512, 768, 1024, 1280, 1536, 1792, 2048, 2560.
- Sample count:
  - 30 examples per length.

This produced 2880 generations. Results were saved under
`results/hazy_context_sweep_n30_l512_2560_clean`.

Single-needle results were mostly strong:

- Attention 360M: nearly perfect across all lengths.
- Attention 1.4B: strong but unexpectedly weaker than Attention 360M at several
  shorter lengths.
- Mamba 1.4B, BASED 360M fp32, and BASED 1.4B fp32 were also strong.
- Mamba 360M degraded more clearly at longer lengths, reaching 19/30 at 2560.

Multi-key retrieval separated the models much more sharply:

- Attention 1.4B was strongest up to 2048 tokens, but collapsed at 2560.
- Attention 360M retained moderate performance, ending at 7/30 at 2560.
- Mamba and BASED models performed poorly under 20 distractors, often near
  floor at longer lengths.

The clean matrix therefore supports the main pattern that simple single-needle
retrieval is not enough to distinguish these architectures. Distractor-heavy
key-value binding is much more diagnostic.

## 20. Attention 1.4B Anomaly

The most surprising result in the clean matrix was Attention 1.4B on the
multi-key task at 2560 tokens. It dropped from 21/30 at 2048 to 1/30 at 2560,
while Attention 360M retained 7/30 at 2560.

We ran targeted probes to determine whether this was a run artifact.

Diagnostics:

- Exact rerun of the 30 Attention 1.4B multi-key 2560 examples reproduced
  1/30.
- Increasing generation length from 8 to 32 new tokens did not help.
- Recompute decoding did not help.
- Stricter prompt variants did not rescue the result.
- An explicit fallback sentinel prompt also did not produce the fallback token.

Prompt variants tested on a 10-example subset:

- Baseline prompt: 0/10.
- `Only output the numeric value...`: 0/10.
- `Return only the 7-digit number...`: 1/10.
- `Return either the matching 7-digit number or ZXQ_UNKNOWN...`: 1/10.

The model never emitted `ZXQ_UNKNOWN`. Instead, it generally resumed
document-style continuation, such as emitting `PASSKEY_RECORD[...]`, filler
text, or a wrong number.

Interpretation:

- This does not appear to be a hard context-length limit, since Attention 1.4B
  scored 30/30 on single-needle at 2560 and the model can execute at that
  context length.
- It also does not appear to be the same cache artifact observed in BASED 360M.
- The likely behavioral diagnosis is task-mode collapse: in the long
  distractor-heavy prompt, the final query no longer controls generation
  strongly enough, and the model reverts to high-probability document
  continuation.

The failure mode differs from the compressed-memory models. Mamba and BASED
usually still emit numbers, but the numbers are distractors or not in the
document. Attention 1.4B uniquely often exits the answer format entirely at the
2560 multi-key setting.

This anomaly should be kept in the results table rather than truncated away,
but it should be discussed carefully. It is a stable model/prompt behavior in
our setup, not a clean monotonic context-length trend.

