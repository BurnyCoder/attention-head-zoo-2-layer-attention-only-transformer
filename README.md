# Attention Head Zoo: 2-Layer Attention-Only Transformer

Manually cataloguing and classifying the functional roles of all 24 attention heads in a 2-layer attention-only transformer, using [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) and [circuitsvis](https://github.com/alan-cooney/CircuitsVis) for mechanistic interpretability.

## Model

A toy 2-layer attention-only transformer designed for interpretability:

- **Architecture**: 768 d_model, 64 d_head, 12 heads/layer, 2 layers (24 heads total)
- **Simplifications**: No MLPs, no LayerNorms, no biases, separate embed/unembed
- **Positional embeddings**: Shortformer-style (added to Q/K only, not V) — the residual stream cannot directly encode position
- **Pretrained weights**: [`callummcdougall/attn_only_2L_half`](https://huggingface.co/callummcdougall/attn_only_2L_half)

## Attention Head Types Found

From analyzing attention patterns on natural language text:

| Type | Count | Description |
|------|-------|-------------|
| Glue word attender | 5x | Attends to function words like "are", "and", "if", "that", "were", "or" |
| BOS/end-of-text attender | 6x | Attends primarily to the beginning-of-sequence token |
| Previous token head | 3x | Attends to the immediately preceding token |
| Few previous tokens head | 2x | Attends to a small window of preceding tokens |
| Certainty/questioning attender | 3x | Attends to words expressing certainty or uncertainty ("likely", "think", "known") |
| Comma attender | 2x | Attends to comma tokens |
| Self-attender | 2x | Attends primarily to the current token position |
| Semantically salient attender | 2x | Attends to content words with high semantic salience |
| Glue-to-content connector | 3x | Connects function words to semantically rich content words |
| Context aggregator | 1x | Aggregates broad context into content-rich positions |
| Related token connector | 1x | Connects semantically related tokens (e.g., "machine" ↔ "intelligence") |

Many heads exhibit multiple behaviors (partial overlap between categories).

## Setup

```bash
uv venv && uv sync
```

Then run the notebook `attention-head-zoo-2-layer-attention-only-transformer.ipynb` with the `.venv` Python kernel.
