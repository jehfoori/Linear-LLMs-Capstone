# Report Part I Outline: Survey and Framing

Working outline for the first part of the report, before experiment setup and
results. The goal is to shift from the old "four-model progression" framing to
a broader survey of efficient long-context language models, while preserving the
project's original focus on compressed memory and retrieval behavior.

## Intended Shape

Survey and reproduction papers usually work best when they:

- motivate the shared problem;
- define comparison axes;
- organize the literature into a taxonomy;
- summarize many methods compactly in tables;
- then narrow to the specific experimental setting with a clear justification.

For this report, Part I should not try to fully explain every model's mechanism.
Instead, it should give readers enough architectural intuition to understand why
needle-in-a-haystack retrieval is a meaningful stress test.

## 1. Introduction and Motivation

Purpose: establish why long-context efficiency matters and why retrieval is a
useful lens.

Points to include:

- Decoder-only Transformers dominate modern LLMs because attention gives direct
  access to prior tokens.
- Standard attention is expensive for long contexts:
  - training uses quadratic token mixing;
  - inference requires a growing KV cache;
  - long prompts become memory- and bandwidth-heavy.
- Efficient long-context models try to reduce this cost by replacing explicit
  tokenwise history with cheaper structure:
  - sparse token access;
  - recurrent state;
  - linearized attention;
  - convolutional filters;
  - hybrids.
- The central tradeoff:
  - exact access to the full prefix is expensive;
  - compressed memory is efficient but may lose retrieval fidelity.
- This motivates the report's question:
  - how do efficient long-context architectures behave when asked to retrieve
    specific key-value facts under distractors?

Tone:

- Neutral survey framing.
- Avoid suggesting a single linear progression across the whole field.
- Present the field as multiple design families responding to the same pressure.

## 2. Transformer Baseline and Long-Context Bottleneck

Purpose: define the baseline and the cost problem.

Points to include:

- Self-attention:
  - constructs queries, keys, values;
  - computes all pairwise token interactions;
  - gives strong retrieval behavior because any token can attend to prior tokens.
- Complexity:
  - training attention matrix scales as `O(n^2)` in sequence length;
  - inference with KV cache avoids recomputing old keys/values but keeps memory
    growing with context length;
  - each new token attends over the full cache.
- Retrieval implication:
  - attention can preserve exact token access in principle;
  - long-context cost motivates approximations or compressed memory.
- Transition:
  - efficient models can be grouped by how they reduce or replace full
    attention.

Keep concise:

- Retain only one or two equations.
- Avoid a long textbook attention explanation.

Primary source:

- Transformer: `Paper Resources/transformer.pdf`.

## 3. Taxonomy of Efficient Long-Context Strategies

Purpose: organize 8-10 models without giving each a long subsection.

Suggested taxonomy:

1. Full attention baseline
   - Transformer.
   - Exact token-to-token access, but quadratic token mixing and growing cache.

2. Sparse attention
   - BigBird.
   - Restrict attention pattern to local/global/random links.
   - Reduces cost while preserving some explicit token access.
   - Retrieval risk: target may be unreachable or weakly connected depending on
     pattern.

3. Linear attention / associative memory
   - Linear Transformers; BASED.
   - Replace softmax attention with kernel/feature-map forms that accumulate
     compressed key-value statistics.
   - Retrieval risk: many associations share a compressed memory state, causing
     interference.

4. Recurrent and state-space models
   - RetNet, RWKV, Mamba, Mamba-2, Gated DeltaNet.
   - Maintain fixed-size state updated over time.
   - Strong inference efficiency.
   - Retrieval risk: finite state must decide what to retain, overwrite, or
     forget.

5. Convolutional / implicit long-filter models
   - Hyena.
   - Use long implicit convolutions and gating for sequence mixing.
   - Retrieval risk: long-range information is mixed through learned filters
     rather than directly indexed.

6. Hybrid architectures
   - Griffin, Jamba, Nemotron-H.
   - Combine local or sparse attention with recurrent/SSM/linear components.
   - Motivation: keep some explicit local token access while using efficient
     compressed memory for broader context.

Table rows:

- Transformer
- BigBird
- RetNet
- RWKV
- Hyena
- Mamba
- Mamba-2
- Gated DeltaNet
- BASED
- Griffin / Jamba / Nemotron-H as modern hybrid examples

Possible table columns:

- Model or family
- Memory strategy
- Training/inference efficiency idea
- What information is compressed
- Expected retrieval weakness
- Public checkpoint relevance to this project

Primary sources:

- `transformer.pdf`
- `bigbird.pdf`
- `retnet.pdf`
- `rwkv.pdf`
- `hyena.pdf`
- `mamba.pdf`
- `mamba2.pdf`
- `gateddelta.pdf`
- `based.pdf`
- `griffin.pdf`
- Optional context: `jamba.pdf`, `nemotron_h.pdf`

## 4. Mechanistic Themes

Purpose: explain the shared ideas once, rather than repeating them model by
model.

Theme A: Explicit access vs compressed state

- Full attention stores or recomputes tokenwise key-value information.
- Linear/recurrent models compress the prefix into fixed or structured state.
- The compression improves efficiency but may blur distinct key-value pairs.

Theme B: Decay, retention, and forgetting

- RetNet and related recurrent designs introduce decay-like memory.
- Useful for bounding state and suppressing stale information.
- Risk: old but relevant details can fade.

Theme C: Selectivity and content-dependent memory

- Mamba improves on fixed recurrent dynamics by making updates input-dependent.
- Mamba-2 reorganizes selective memory into a more hardware-friendly dual form.
- Selectivity helps decide what to keep, but does not remove finite-state
  pressure.

Theme D: Overwrite and interference

- Gated DeltaNet explicitly targets overwrite behavior with a delta-rule update.
- Retrieval-heavy tasks stress whether a model can update one association
  without corrupting others.

Theme E: Hybrids as a practical compromise

- BASED, Griffin, Jamba, and Nemotron-H show a trend toward mixing mechanisms:
  local attention plus compressed global memory, or Transformer layers mixed
  with recurrent/SSM layers.
- This suggests the field is moving away from pure replacement stories and
  toward architecture mixtures.

Need for the report:

- Keep this section short and conceptual.
- Use it to prepare the reader for failure modes:
  - wrong distractor value;
  - spurious number;
  - no-number/task-mode collapse;
  - degradation with length.

## 5. Representative Model Notes

Purpose: include compact descriptions of important models without a long
mechanism subsection for each.

Suggested format:

- One paragraph per family, not per paper when possible.
- Avoid full derivations except for attention and perhaps one compressed-memory
  example.
- Use the taxonomy table as the main reference.

Model notes to include:

Transformer:

- Baseline for exact retrieval and growing KV cache.
- Used experimentally through HazyResearch attention checkpoints.

BigBird:

- Sparse attention representative.
- Important historically because it preserves some explicit attention while
  reducing complexity.
- Not experimentally tested here.

RetNet:

- Retention/recurrent memory representative.
- Useful bridge between attention-like parallel form and recurrent inference.

RWKV:

- RNN-style language model intended to combine Transformer-like performance with
  recurrent inference.
- Useful as evidence that recurrent LLM design is broader than SSMs alone.

Hyena:

- Long convolutional sequence model.
- Represents non-attention sequence mixing through implicit filters.

Mamba / Mamba-2:

- Central SSM lineage.
- Mamba: selective state updates.
- Mamba-2: state-space duality and better hardware utilization.
- Relevant to original project plan and possible future comparison.

Gated DeltaNet:

- Originally central to our plan.
- Introduces more explicit overwrite-style memory update.
- In final report, explain checkpoint limitations and why it is not the main
  experimental comparison.

BASED:

- Main experimental linear-attention/hybrid model.
- Combines global linear attention, local sliding attention, and convolution.
- Important because public matched checkpoints exist for attention, Mamba, and
  BASED under the same paper ecosystem.

Modern hybrids:

- Griffin/Jamba/Nemotron-H show the current trend toward combining attention
  and recurrent/SSM blocks.
- Mention briefly to contextualize BASED as part of a broader hybrid direction.

## 6. Why the Experiment Narrows to HazyResearch Attention/Mamba/BASED

Purpose: justify the empirical scope.

Points to include:

- A broad survey can cover many architectures, but a fair reproduction-style
  experiment needs comparable checkpoints.
- The original Mamba-2 vs Gated DeltaNet plan ran into public-checkpoint and
  implementation caveats:
  - Gated DeltaNet paper weights were not publicly available in a directly
    comparable form;
  - public GDN checkpoints differed in source and training assumptions;
  - dependency and kernel paths were fragile.
- The BASED paper ecosystem provided a cleaner matched family:
  - attention baseline;
  - Mamba baseline;
  - BASED model;
  - two scales around 360M and 1.4B.
- Our experiment therefore becomes:
  - not a full reproduction of BASED throughput claims;
  - a controlled long-context retrieval extension using public checkpoints.

Important caveat:

- BASED optimized kernels were unavailable in our setup, so timing claims are
  not central.
- BASED accuracy runs used fp32 recompute decoding to avoid public fallback
  cache artifacts.

## 7. Experimental Expectations and Failure Modes

Purpose: bridge survey to experiment.

Expected behavior:

- Single-needle retrieval should be easier because only one relevant record is
  present.
- Multi-key retrieval with distractors should be harder because the model must
  bind the queried key to the correct value among many same-format alternatives.
- Full attention should have an advantage when exact key-value binding is
  needed, because tokenwise access is retained.
- Compressed-memory models may struggle through interference, overwriting, or
  finite-state bottlenecks.

Failure categories to define before results:

- Correct value.
- Distractor value:
  - model outputs a value that appears in the document but belongs to another
    key.
  - suggests key-value binding interference.
- Not-in-document number:
  - model outputs a plausible but absent number.
  - suggests generation or recall failure.
- No number generated:
  - model leaves answer format entirely.
  - suggests task-mode or prompt-continuation failure.

Tone:

- State these as expectations, not hard hypotheses.
- Avoid claiming architecture alone determines results.
- Note that checkpoint training, scale, prompt format, and implementation path
  all matter.

## 8. What to Cut from the Old Draft

Remove or shorten:

- The strong claim that RetNet, Mamba, Mamba-2, and Gated DeltaNet form the
  central progression of the whole paper.
- Long per-model derivations for models not experimentally tested.
- The progression figure as the main organizing device.
- Strong predictions that Gated DeltaNet should outperform Mamba/Mamba-2.

Keep or adapt:

- Attention cost explanation.
- Memory-compression framing.
- Comparison axes around training form, inference form, state growth, and
  retrieval risk.
- Error-mode discussion.
- Softened hypothesis language.

## 9. Suggested Section Structure for Part I

Final structure to draft:

1. Introduction
2. Long-context efficiency problem
3. Taxonomy of efficient sequence mixing
4. Representative model landscape table
5. Mechanistic themes in compressed memory
6. Scope of the empirical study
7. Experimental expectations

Approximate length target:

- 4 to 6 pages before the experiment section, depending on table size.
- One major taxonomy table.
- Possibly one small schematic or conceptual figure.
- Avoid more than two equations unless needed.

