# CLAUDE.md

## Project Overview

Attention head zoo: manually cataloguing and classifying the functional roles of all 24 attention heads in a 2-layer attention-only transformer. Uses TransformerLens and circuitsvis for activation analysis and visualization.

## Project Structure

- `attention-head-zoo-2-layer-attention-only-transformer.ipynb` — main notebook with per-layer attention visualizations and programmatic summary tables
- `shared.py` — all shared data structures and utility functions (imported by every notebook)
- `generate_notebooks.py` — generates the 39 head/type notebooks from `shared.py` data
- `heads/` — 24 per-head analysis notebooks (`l{layer}h{head}.ipynb`)
- `types/` — 15 per-type analysis notebooks (`{type_id}.ipynb`)
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

## Model

The model is a 2-layer attention-only transformer with:
- 768 d_model, 64 d_head, 12 heads per layer (24 heads total)
- No MLP layers, no LayerNorms, no biases
- Shortformer positional embeddings (added to Q/K but not V)
- Separate embed/unembed matrices
- Pretrained weights from `callummcdougall/attn_only_2L_half` on HuggingFace

## Key Data Structures in shared.py

- `HEAD_CLASSIFICATIONS` — `dict[(layer, head) -> str]`: one-line description per head
- `HEAD_TYPES` — `dict[type_id -> (display_name, description)]`: 15 attention head types
- `TYPE_TO_HEADS` — `dict[type_id -> list[((layer, head), activity_level)]]`: which heads exhibit each type
- Activity levels: `full` (90-100%), `fullish` (60-90%), `half` (40-60%), `partial` (10-40%), `almost_none` (0.1-10%)

## Key Functions in shared.py

- `load_model()` — loads the 2L attention-only transformer with pretrained weights
- `run_and_cache(model, text)` — forward pass returning `(str_tokens, logits, cache)`
- `get_attention_pattern(cache, layer, head)` — extracts `[dest_pos, src_pos]` attention matrix
- `show_head_pattern(str_tokens, cache, layer, head)` — displays both circuitsvis visualizations
- `show_attention_tables(str_tokens, attention, top_k)` — highest/lowest attention weight markdown tables
- `show_attention_to_position(cache, position, label)` — prints attention % to a position for all heads
- `compute_head_raw_pcts(cache)` — computes EOT%, self-attention%, previous-token% for all heads
- `get_head_types(layer, head)` — reverse lookup: head -> list of `(type_id, activity_level)`

## Conventions

- Use `einops.rearrange` / `einops.repeat` for tensor reshaping
- Use `jaxtyping` annotations for tensor-typed function signatures
- Use `t.` as the torch alias (i.e. `import torch as t`)
- All notebooks import from `shared.py` via `sys.path.insert(0, str(Path.cwd().parent))`
