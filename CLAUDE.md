# CLAUDE.md

## Project Overview

Attention head zoo: manually cataloguing and classifying the functional roles of all 24 attention heads in a 2-layer attention-only transformer (from [ARENA 3.0 exercises](https://github.com/callummcdougall/ARENA_3.0)). Uses TransformerLens and circuitsvis for activation analysis and visualization.

## Project Structure

- `attention-head-zoo-2-layer-attention-only-transformer.ipynb` — main notebook with visualization and classification of all attention heads
- `pyproject.toml` / `uv.lock` — dependencies managed with uv

## Setup & Running

```bash
uv venv && uv sync
```

Run the notebook with the `.venv` Python kernel from this directory.

## Key Dependencies

- `transformer-lens` — mechanistic interpretability of transformer models
- `circuitsvis` — attention pattern visualization
- `einops` — tensor reshaping
- `jaxtyping` — tensor type annotations
- `torch`, `plotly`, `numpy`

## Model

The model is a 2-layer attention-only transformer with:
- 768 d_model, 64 d_head, 12 heads per layer (24 heads total)
- No MLP layers, no LayerNorms, no biases
- Shortformer positional embeddings (added to Q/K but not V)
- Separate embed/unembed matrices
- Pretrained weights from `callummcdougall/attn_only_2L_half` on HuggingFace

## Conventions

- Use `einops.rearrange` / `einops.repeat` for tensor reshaping
- Use `jaxtyping` annotations for tensor-typed function signatures
- Use `t.` as the torch alias (i.e. `import torch as t`)
