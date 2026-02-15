# Attention Head Zoo: 2-Layer Attention-Only Transformer

Manually cataloguing and classifying the functional roles of all 24 attention heads in a 2-layer attention-only transformer, using [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) and [circuitsvis](https://github.com/alan-cooney/CircuitsVis) for mechanistic interpretability.

## Model

A toy 2-layer attention-only transformer designed for interpretability:

- **Architecture**: 768 d_model, 64 d_head, 12 heads/layer, 2 layers (24 heads total)
- **Simplifications**: No MLPs, no LayerNorms, no biases, separate embed/unembed
- **Positional embeddings**: Shortformer-style (added to Q/K only, not V) — the residual stream cannot directly encode position
- **Pretrained weights**: [`callummcdougall/attn_only_2L_half`](https://huggingface.co/callummcdougall/attn_only_2L_half)

## Project Structure

- `attention-head-zoo-2-layer-attention-only-transformer.ipynb` — main notebook with per-layer visualizations, programmatic summary tables, head-type heatmap, and cross-type attention matrix
- `heads/` — 24 per-head notebooks (`l0h0.ipynb` through `l1h11.ipynb`), each with the head's classification, attention pattern visualizations, and top-25 source/destination token tables
- `types/` — 30 per-type notebooks (e.g. `glue_words.ipynb`, `end_of_text.ipynb`, `noun_attention.ipynb`), each showing all heads exhibiting that type sorted by metric value
- `cross/` — 289 cross-type notebooks (17×17 from/to pairs, e.g. `glue_to_salient.ipynb`), each showing how much one word type attends to another across all 24 heads
- `shared.py` — shared data structures (classifications, type mappings, activity levels) and utility functions (model loading, attention extraction, visualization, tables)
- `generate_notebooks.py` — generates all 343 head/type/cross notebooks from data in `shared.py`

## Attention Matrix Terminology

The attention pattern for each head is a matrix `attention[dest, src]` where:
- **Destination (dest)** — the token position that is *querying* (attending from). Each row sums to 1 after softmax.
- **Source (src)** — the token position being *attended to* (providing information). High `attention[dest, src]` means token at `dest` is pulling information from token at `src`.

For example, "attention TO commas" means commas appear as source tokens (columns), averaged over all destination positions. "Attention FROM commas" means commas are the querying/destination tokens (rows).

## Attention Head Types Found

30 types identified from analyzing attention patterns on natural language text. Measurable types are auto-populated: any head with >= 20% metric value is classified into that type. Activity levels: full 90-100%, fullish 60-90%, half 40-60%, partial 10-40%.

| Type | # Heads | Description |
|------|---------|-------------|
| Few Previous Tokens Head | 19 | Attends to a small window of preceding tokens |
| Glue Word Attender (auto) | 19 | Fraction of attention to function/glue words |
| Salient Word Attender | 16 | Fraction of attention to semantically salient content words |
| End-of-Text Attender | 15 | Attends primarily to the beginning-of-sequence / end-of-text token |
| Verb Attender | 8 | Fraction of attention to verb positions |
| Glue Word Attender | 6 | Manually classified — attends to function words like "are", "and", "if", "that" |
| Previous Token Head | 3 | Attends to the immediately preceding token |
| Self-Attender | 3 | Attends primarily to the current token position |
| Certainty/Questioning Attender | 3 | Manually classified — attends to certainty/uncertainty words |
| Glue-to-Semantic Connector | 3 | Connects function words to semantically rich content words |
| Semantically Salient Attender | 2 | Attends to content words with high semantic salience |
| Noun Attender | 2 | Fraction of attention to noun positions |
| Context Aggregator | 2 | Aggregates broad context into content-rich positions |
| Preposition Attender | 1 | Fraction of attention to preposition/particle positions |
| Adjective Attender | 1 | Fraction of attention to adjective positions |
| AI Word Attender | 1 | Fraction of attention to AI/ML-related words |
| Dot-EOT Quirk | 1 | Period (.) token attends to end-of-text token |
| Glue-to-Glue Connector | 1 | Connects function words to other function words |
| Related Previous Token | 1 | Attends to the previous token when directly semantically related |
| Semantic Connector | 1 | Connects semantically related tokens (e.g., "machine" and "intelligence") |
| Period Attender | 0 | Fraction of attention to period (.) tokens |
| Comma Attender | 0 | Fraction of attention to comma (,) tokens |
| Pronoun Attender | 0 | Fraction of attention to pronoun positions |
| Adverb Attender | 0 | Fraction of attention to adverb positions |
| Conjunction Attender | 0 | Fraction of attention to conjunction positions |
| Determiner Attender | 0 | Fraction of attention to determiner positions |
| Spooky Word Attender | 0 | Fraction of attention to spooky/deceptive words |
| Certainty Word Attender | 0 | Fraction of attention to certainty/uncertainty words (think, likely, known, significantly) |
| Questioning Word Attender | 0 | Fraction of attention to questioning words (if, how) |

Entropy % (normalized entropy of attention distribution) is also tracked for all heads but omitted from the table as it's a distribution property rather than an attention target type. 22 of 24 heads have entropy >= 20%.

Many heads exhibit multiple behaviors at different activity levels.

## Setup

```bash
uv venv && uv sync
```

Run any notebook with the `.venv` Python kernel. To regenerate the head/type notebooks after editing `shared.py`:

```bash
.venv/bin/python generate_notebooks.py
```

## TODO

- Look at attention patterns and find stuff like attending to previous token if directly related
- Go through my manual classifications and compare with automatic and pick the better one