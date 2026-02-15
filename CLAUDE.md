# CLAUDE.md

## Project Overview

Attention head zoo: manually cataloguing and classifying the functional roles of all 24 attention heads in a 2-layer attention-only transformer. Uses TransformerLens and circuitsvis for activation analysis and visualization.

## Project Structure

- `attention-head-zoo-2-layer-attention-only-transformer.ipynb` — main notebook with per-layer attention visualizations and programmatic summary tables
- `shared.py` — all shared data structures and utility functions (imported by every notebook)
- `generate_notebooks.py` — generates all 343 head/type/cross notebooks from `shared.py` data
- `heads/` — 24 per-head analysis notebooks (`l{layer}h{head}.ipynb`)
- `types/` — 30 per-type analysis notebooks (`{type_id}.ipynb`)
- `cross/` — 289 cross-type pair analysis notebooks (17×17 from/to pairs)
- `pyproject.toml` / `uv.lock` — dependencies managed with uv

## Setup & Running

```bash
uv venv && uv sync
```

Run any notebook with the `.venv` Python kernel from this directory.

To regenerate head/type notebooks after editing classifications or types in `shared.py`:
```bash
.venv/bin/python generate_notebooks.py
```

## Key Dependencies

- `transformer-lens` — mechanistic interpretability of transformer models
- `circuitsvis` — attention pattern visualization (`attention_pattern`, `attention_patterns`)
- `einops` — tensor reshaping
- `jaxtyping` — tensor type annotations
- `torch`, `plotly`, `numpy`, `nbformat`
- `nltk` — POS tagging for token classification
- `pandas`, `itables` — interactive tables in notebooks
- `tqdm` — progress bars

## Model

The model is a 2-layer attention-only transformer with:
- 768 d_model, 64 d_head, 12 heads per layer (24 heads total)
- No MLP layers, no LayerNorms, no biases
- Shortformer positional embeddings (added to Q/K but not V)
- Separate embed/unembed matrices
- Pretrained weights from `callummcdougall/attn_only_2L_half` on HuggingFace

## Attention Matrix Terminology

The attention pattern for each head is a matrix `attention[dest, src]` where:
- **Destination (dest)** — the token position that is *querying* (attending from). Each row sums to 1 after softmax.
- **Source (src)** — the token position being *attended to* (providing information). High `attention[dest, src]` means token at `dest` pulls information from token at `src`.

"Attention TO X" = X as source (columns), averaged over all dest. "Attention FROM X" = X as dest (rows).

## Key Data Structures in shared.py

- `HEAD_CLASSIFICATIONS` — `dict[(layer, head) -> str]`: one-line description per head
- `HEAD_TYPES` — `dict[type_id -> (display_name, description)]`: 30 attention head types
- `TYPE_TO_HEADS` — `dict[type_id -> list[((layer, head), activity_level)]]`: which heads exhibit each type. Measurable types are auto-populated by `populate_measurable_type_heads()`; non-measurable types have manual assignments.
- `MEASURABLE_TYPES` — `set[str]`: 21 type IDs with programmatic metrics (auto-populated at ≥20% threshold)
- `TYPE_ENTROPY_KEYS` — `dict[type_id -> entropy_key]`: maps type IDs to their entropy metric names
- `POS_CATEGORIES` — `dict[str -> set[str]]`: 8 POS tag categories (noun, verb, adjective, adverb, pronoun, preposition, determiner, conjunction)
- Word sets: `SALIENT_WORDS`, `AI_WORDS`, `SPOOKY_WORDS`, `GLUE_WORDS`, `CERTAINTY_WORDS`, `QUESTIONING_WORDS`
- `TYPE_ID_TO_POSITION_KEY` — `dict[type_id -> short_name]`: maps type IDs to cross-type position keys
- `CROSS_TYPE_NAMES` — `dict[short_name -> display_name]`: 17 cross-type display names
- Activity levels: `full` (90-100%), `fullish` (60-90%), `half` (40-60%), `partial` (10-40%), `almost_none` (0.1-10%)

## Key Functions in shared.py

- `load_model()` — loads the 2L attention-only transformer with pretrained weights
- `run_and_cache(model, text)` — forward pass returning `(str_tokens, logits, cache)`
- `get_attention_pattern(cache, layer, head)` — extracts `[dest_pos, src_pos]` attention matrix
- `show_head_pattern(str_tokens, cache, layer, head)` — displays both circuitsvis visualizations
- `show_attention_tables(str_tokens, attention, top_k)` — highest/lowest attention weight markdown tables
- `show_attention_to_position(cache, position, label)` — prints attention % to a position for all heads
- `compute_head_raw_pcts(cache)` — computes EOT%, self-attention%, previous-token% for all heads
- `compute_all_type_metrics(cache, str_tokens)` — computes metric % for all measurable types × all heads
- `populate_measurable_type_heads(cache, str_tokens)` — auto-populates TYPE_TO_HEADS for measurable types (≥20% threshold)
- `compute_cross_type_metrics(cache, str_tokens)` — computes cross-type attention between word categories for all heads
- `get_type_positions(str_tokens)` — gets token positions for all 17 cross-type categories
- `get_head_types(layer, head)` — reverse lookup: head -> list of `(type_id, activity_level)`
- `show_type_filtered_tables(str_tokens, attention, positions, label)` — attention tables filtered to type positions
- `attention_to_positions_pcts(cache, positions)` — mean attention to arbitrary source positions

## Conventions

- Use `einops.rearrange` / `einops.repeat` for tensor reshaping
- Use `jaxtyping` annotations for tensor-typed function signatures
- Use `t.` as the torch alias (i.e. `import torch as t`)
- All notebooks import from `shared.py` via `sys.path.insert(0, str(Path.cwd().parent))`
