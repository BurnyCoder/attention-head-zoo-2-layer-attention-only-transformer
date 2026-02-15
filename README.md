# Attention Head Zoo: 2-Layer Attention-Only Transformer

Manually cataloguing and classifying the functional roles of all 24 attention heads in a 2-layer attention-only transformer, using [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) and [circuitsvis](https://github.com/alan-cooney/CircuitsVis) for mechanistic interpretability.

## Model

A toy 2-layer attention-only transformer designed for interpretability:

- **Architecture**: 768 d_model, 64 d_head, 12 heads/layer, 2 layers (24 heads total)
- **Simplifications**: No MLPs, no LayerNorms, no biases, separate embed/unembed
- **Positional embeddings**: Shortformer-style (added to Q/K only, not V) — the residual stream cannot directly encode position
- **Pretrained weights**: [`callummcdougall/attn_only_2L_half`](https://huggingface.co/callummcdougall/attn_only_2L_half)

## Project Structure

- `attention-head-zoo-2-layer-attention-only-transformer.ipynb` — main notebook with per-layer visualizations and a programmatic summary (per-head classification table with raw %, per-type summary, head-type activity heatmap)
- `heads/` — 24 per-head notebooks (`l0h0.ipynb` through `l1h11.ipynb`), each with the head's classification, attention pattern visualizations, and top-25 source/destination token tables
- `types/` — 15 per-type notebooks (e.g. `glue_words.ipynb`, `end_of_text.ipynb`), each showing all heads exhibiting that type sorted by activity level
- `shared.py` — shared data structures (classifications, type mappings, activity levels) and utility functions (model loading, attention extraction, visualization, tables)
- `generate_notebooks.py` — generates all 39 head/type notebooks from data in `shared.py`

## Attention Matrix Terminology

The attention pattern for each head is a matrix `attention[dest, src]` where:
- **Destination (dest)** — the token position that is *querying* (attending from). Each row sums to 1 after softmax.
- **Source (src)** — the token position being *attended to* (providing information). High `attention[dest, src]` means token at `dest` is pulling information from token at `src`.

For example, "attention TO commas" means commas appear as source tokens (columns), averaged over all destination positions. "Attention FROM commas" means commas are the querying/destination tokens (rows).

## Attention Head Types Found

15 types identified from analyzing attention patterns on natural language text. Each head can exhibit multiple types at different activity levels (full 90-100%, fullish 60-90%, half 40-60%, partial 10-40%).

| Type | Heads | Description |
|------|-------|-------------|
| Glue Word Attender | 6 | Attends to function words like "are", "and", "if", "that", "were", "or" |
| End-of-Text Attender | 11 | Attends primarily to the beginning-of-sequence / end-of-text token |
| Previous Token Head | 4 | Attends to the immediately preceding token |
| Few Previous Tokens Head | 2 | Attends to a small window of preceding tokens |
| Certainty/Questioning Attender | 3 | Attends to words expressing certainty or uncertainty ("likely", "think", "known") |
| Comma Attender | 2 | Attends to comma tokens |
| Period Attender | 1 | Attends to period (.) tokens |
| Self-Attender | 3 | Attends primarily to the current token position |
| Semantically Salient Attender | 2 | Attends to content words with high semantic salience |
| Context Aggregator | 2 | Aggregates broad context into content-rich positions |
| Dot-EOT Quirk | 1 | Period (.) token attends to end-of-text token |
| Glue-to-Semantic Connector | 3 | Connects function words to semantically rich content words |
| Glue-to-Glue Connector | 1 | Connects function words to other function words |
| Related Previous Token | 1 | Attends to the previous token when directly semantically related |
| Semantic Connector | 1 | Connects semantically related tokens (e.g., "machine" and "intelligence") |

Many heads exhibit multiple behaviors at different activity levels.

## Setup

```bash
uv venv && uv sync
```

Run any notebook with the `.venv` Python kernel. To regenerate the head/type notebooks after editing `shared.py`:

```bash
.venv/bin/python generate_notebooks.py
```
