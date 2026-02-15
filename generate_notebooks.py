"""Generate per-head and per-type analysis notebooks."""

import json
import uuid
from pathlib import Path

from shared import (
    ACTIVITY_LABELS,
    ACTIVITY_ORDER,
    ACTIVITY_PCT_RANGES,
    HEAD_CLASSIFICATIONS,
    HEAD_TYPES,
    TYPE_TO_HEADS,
    get_head_types,
)

PROJECT_ROOT = Path(__file__).resolve().parent
HEADS_DIR = PROJECT_ROOT / "heads"
TYPES_DIR = PROJECT_ROOT / "types"

NOTEBOOK_METADATA = {
    "kernelspec": {
        "display_name": ".venv",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.12.3",
    },
}


def cell_id() -> str:
    return uuid.uuid4().hex[:8]


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id(),
        "metadata": {},
        "source": source,
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id(),
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def make_notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": NOTEBOOK_METADATA,
        "cells": cells,
    }


def write_notebook(path: Path, nb: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"  Written: {path.relative_to(PROJECT_ROOT)}")


SETUP_CODE = """\
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
import torch as t
from IPython.display import Markdown, display
from shared import (
    load_model, run_and_cache, get_attention_pattern,
    show_head_pattern, show_attention_tables, show_attention_to_position,
    show_self_attention_pcts, show_prev_token_pcts,
    show_attention_to_token, show_few_prev_tokens_pcts, TEXT,
)"""

LEVEL_EXPR = (
    "level = 'full' if pct >= 90 else 'fullish' if pct >= 60 else "
    "'half' if pct >= 40 else 'partial' if pct >= 10 else "
    "'almost none' if pct >= 0.1 else '-'"
)

# Types with programmatically computable % metrics.
# Each entry: summary_title, summary_desc, summary_code, head_pct_code (must assign `pct`).
MEASURABLE_TYPES: dict[str, tuple[str, str, str, str]] = {
    "end_of_text": (
        "EOT Attention % Across All 24 Heads",
        "Mean fraction of attention weight allocated to position 0 "
        "(end-of-text token), averaged across all destination positions. "
        "Sorted by raw % descending.",
        'show_attention_to_position(cache, position=0, label="EOT")',
        "pct = get_attention_pattern(cache, layer={l}, head={h})[:, 0].mean().item() * 100",
    ),
    "self_attention": (
        "Self-Attention % Across All 24 Heads",
        "Mean diagonal attention weight (attention[i, i]) across all positions. "
        "Sorted by raw % descending.",
        "show_self_attention_pcts(cache)",
        "pct = t.diagonal(get_attention_pattern(cache, layer={l}, head={h})).mean().item() * 100",
    ),
    "previous_token": (
        "Previous Token Attention % Across All 24 Heads",
        "Mean attention to the immediately preceding token (attention[i, i-1]). "
        "Sorted by raw % descending.",
        "show_prev_token_pcts(cache)",
        "attn = get_attention_pattern(cache, layer={l}, head={h})\n"
        "pct = t.tensor([attn[i, i-1].item() for i in range(1, attn.shape[0])]).mean().item() * 100",
    ),
    "comma_attention": (
        "Comma Attention % Across All 24 Heads",
        'Mean attention to positions containing comma (",") tokens. '
        "Sorted by raw % descending.",
        'show_attention_to_token(cache, str_tokens, ",", "Comma")',
        'comma_pos = [i for i, tok in enumerate(str_tokens) if "," in tok]\n'
        "pct = get_attention_pattern(cache, layer={l}, head={h})[:, comma_pos].sum(dim=-1).mean().item() * 100",
    ),
    "period_attention": (
        "Period Attention % Across All 24 Heads",
        'Mean attention to positions containing period (".") tokens. '
        "Sorted by raw % descending.",
        'show_attention_to_token(cache, str_tokens, ".", "Period")',
        'period_pos = [i for i, tok in enumerate(str_tokens) if "." in tok]\n'
        "pct = get_attention_pattern(cache, layer={l}, head={h})[:, period_pos].sum(dim=-1).mean().item() * 100",
    ),
    "few_previous_tokens": (
        "Few Previous Tokens Attention % Across All 24 Heads",
        "Mean attention to the 5 preceding tokens (excluding self). "
        "Sorted by raw % descending.",
        "show_few_prev_tokens_pcts(cache, k=5)",
        "attn = get_attention_pattern(cache, layer={l}, head={h})\n"
        "n = attn.shape[0]\n"
        "pct = sum(attn[i, max(0, i-5):i].sum().item() for i in range(n)) / n * 100",
    ),
}

LOAD_CODE = """\
model = load_model()
str_tokens, logits, cache = run_and_cache(model)"""


def generate_head_notebook(layer: int, head: int) -> None:
    """Generate a single head analysis notebook."""
    classification = HEAD_CLASSIFICATIONS.get((layer, head), "TODO: classify")
    head_types = get_head_types(layer, head)
    is_todo = classification.startswith("TODO")

    # Build type list markdown
    if head_types:
        type_lines = []
        for type_id, activity in head_types:
            display_name, _ = HEAD_TYPES[type_id]
            type_lines.append(f"- **{display_name}** ({ACTIVITY_LABELS[activity]})")
        types_md = "\n".join(type_lines)
    else:
        types_md = "- *No types assigned yet (needs classification)*"

    cells = [
        md_cell(
            f"# L{layer}H{head} — {classification}\n"
            f"\n"
            f"**Layer {layer}, Head {head}**\n"
            f"\n"
            f"## Types exhibited:\n"
            f"{types_md}"
        ),
        code_cell(SETUP_CODE),
        code_cell(LOAD_CODE),
        md_cell("## Attention Pattern Visualization"),
        code_cell(f"show_head_pattern(str_tokens, cache, layer={layer}, head={head})"),
        md_cell(
            "## Attention Weight Tables\n"
            "\n"
            "Source to destination token pairs sorted by attention weight."
        ),
        code_cell(
            f"attention = get_attention_pattern(cache, layer={layer}, head={head})\n"
            f"show_attention_tables(str_tokens, attention, top_k=25)"
        ),
    ]

    # Activity breakdown table
    if head_types:
        table_lines = [
            "## Attention Strength by Type\n",
            "| Type | Activity Level | Percentage Range |",
            "|------|---------------|-----------------|",
        ]
        for type_id, activity in head_types:
            display_name, _ = HEAD_TYPES[type_id]
            table_lines.append(
                f"| {display_name} | {activity} | {ACTIVITY_PCT_RANGES[activity]} |"
            )
        cells.append(md_cell("\n".join(table_lines)))

    # For unclassified heads, add helper code
    if is_todo:
        cells.append(
            md_cell(
                "## Auto-Classification Helper\n"
                "\n"
                "Inspect the attention pattern above and fill in the classification."
            )
        )
        cells.append(
            code_cell(
                f"# Helper statistics for classifying this head\n"
                f"import torch as t\n"
                f"attention = get_attention_pattern(cache, layer={layer}, head={head})\n"
                f'print(f"Mean attention to position 0 (EOT): {{attention[:, 0].mean().item():.4f}}")\n'
                f'print(f"Mean diagonal (self-attention): {{t.diagonal(attention).mean().item():.4f}}")\n'
                f"# Previous token attention: mean of attention[i, i-1] for i >= 1\n"
                f"prev_attn = t.tensor([attention[i, i-1].item() for i in range(1, attention.shape[0])])\n"
                f'print(f"Mean previous-token attention: {{prev_attn.mean().item():.4f}}")\n'
                f"# Entropy of attention distribution (lower = more focused)\n"
                f"entropy = -(attention * attention.clamp(min=1e-10).log()).sum(dim=-1).mean()\n"
                f'print(f"Mean attention entropy: {{entropy.item():.4f}}")'
            )
        )

    filename = f"l{layer}h{head}.ipynb"
    write_notebook(HEADS_DIR / filename, make_notebook(cells))


def generate_type_notebook(type_id: str) -> None:
    """Generate a single type analysis notebook."""
    display_name, description = HEAD_TYPES[type_id]
    heads_with_activity = TYPE_TO_HEADS.get(type_id, [])
    # Sort by activity level descending
    heads_sorted = sorted(
        heads_with_activity,
        key=lambda x: ACTIVITY_ORDER.get(x[1], 0),
        reverse=True,
    )

    measurable = MEASURABLE_TYPES.get(type_id)

    cells = [
        md_cell(
            f"# {display_name}\n"
            f"\n"
            f"{description}"
        ),
        code_cell(SETUP_CODE),
        code_cell(LOAD_CODE),
    ]

    # Add programmatic % summary for measurable types
    if measurable:
        summary_title, summary_desc, summary_code, _ = measurable
        cells.append(md_cell(f"## {summary_title}\n\n{summary_desc}"))
        cells.append(code_cell(summary_code))

    for (l, h), activity in heads_sorted:
        classification = HEAD_CLASSIFICATIONS.get((l, h), "TODO")
        if measurable:
            _, _, _, head_pct_code = measurable
            escaped = classification.replace("\\", "\\\\").replace('"', '\\"')
            cells.append(
                code_cell(
                    f"{head_pct_code.format(l=l, h=h)}\n"
                    f"{LEVEL_EXPR}\n"
                    f'display(Markdown(f"---\\n## L{l}H{h} — {{pct:.2f}}% ({{level}})\\n\\n{escaped}"))'
                )
            )
        else:
            cells.append(
                md_cell(
                    f"---\n"
                    f"## L{l}H{h} — {ACTIVITY_LABELS[activity]}\n"
                    f"\n"
                    f"{classification}"
                )
            )
        cells.append(
            code_cell(f"show_head_pattern(str_tokens, cache, layer={l}, head={h})")
        )
        cells.append(
            code_cell(
                f"attention = get_attention_pattern(cache, layer={l}, head={h})\n"
                f"show_attention_tables(str_tokens, attention, top_k=25)"
            )
        )

    write_notebook(TYPES_DIR / f"{type_id}.ipynb", make_notebook(cells))


MAIN_SETUP_CODE = """\
from IPython.display import display, Markdown, HTML
import circuitsvis as cv
import torch as t
from shared import (
    load_model, run_and_cache, get_attention_pattern,
    show_head_pattern, show_attention_tables, show_attention_to_position,
    show_self_attention_pcts, show_prev_token_pcts,
    show_attention_to_token, show_few_prev_tokens_pcts,
    compute_head_raw_pcts,
    HEAD_CLASSIFICATIONS, HEAD_TYPES, TYPE_TO_HEADS,
    ACTIVITY_LABELS, ACTIVITY_ORDER,
    get_head_types, TEXT,
)"""

MODEL_DESCRIPTION = """\
## Introducing Our Toy Attention-Only Model

Here we introduce a toy 2L attention-only transformer. Some changes to make them easier to interpret:
- It has only attention blocks.
- The positional embeddings are only added to the residual stream before calculating each key and query vector in the attention layers as opposed to the token embeddings - i.e. we compute queries as `Q = (resid + pos_embed) @ W_Q + b_Q` and same for keys, but values as `V = resid @ W_V + b_V`. This means that **the residual stream can't directly encode positional information**.
    - This turns out to make it *way* easier for induction heads to form, it happens 2-3x times earlier - [see the comparison of two training runs](https://wandb.ai/mechanistic-interpretability/attn-only/reports/loss_ewma-22-08-24-11-08-83---VmlldzoyNTI0MDMz?accessToken=8ap8ir6y072uqa4f9uinotdtrwmoa8d8k2je4ec0lyasf1jcm3mtdh37ouijgdbm) here. (The bump in each curve is the formation of induction heads.)
    - The argument that does this below is `positional_embedding_type="shortformer"`.
- It has no MLP layers, no LayerNorms, and no biases.
- There are separate embed and unembed matrices (i.e. the weights are not tied).

We now define our model with a `HookedTransformerConfig` object. """

LAYER_VIS_CODE = """\
attention_patterns_for_all_layers = t.stack([
    cache["pattern", layer] for layer in range(model.cfg.n_layers)
])

for layer in range(model.cfg.n_layers):
    attention_pattern = attention_patterns_for_all_layers[layer]
    print(f"Layer {layer} Head Attention Patterns:")
    display(
        cv.attention.attention_patterns(
            tokens=str_tokens,
            attention=attention_pattern,
        )
    )
    display(
        cv.attention.attention_heads(
            tokens=str_tokens,
            attention=attention_pattern,
            attention_head_names=[f"L{layer}H{i}" for i in range(12)],
        )
    )"""

PER_HEAD_TABLE_CODE = """\
# Per-head summary: classification + types with activity levels + raw %
raw_pcts = compute_head_raw_pcts(cache)

lines = [
    "| Head | Classification | Types | EOT % | Self % | Prev % |",
    "|------|---------------|-------|-------|--------|--------|",
]
for layer in range(2):
    for head in range(12):
        name = f"L{layer}H{head}"
        classification = HEAD_CLASSIFICATIONS.get((layer, head), "\\u2014")
        head_types = get_head_types(layer, head)
        if head_types:
            types_str = ", ".join(
                f"{HEAD_TYPES[tid][0]} ({ACTIVITY_LABELS[act]})"
                for tid, act in head_types
            )
        else:
            types_str = "\\u2014"
        pcts = raw_pcts[(layer, head)]
        lines.append(
            f"| **{name}** | {classification} | {types_str} "
            f"| {pcts['eot']:.1f}% | {pcts['self_attn']:.1f}% | {pcts['prev_token']:.1f}% |"
        )
display(Markdown("\\n".join(lines)))"""

PER_TYPE_TABLE_CODE = """\
# Per-type summary: sorted by number of heads descending
# Map type_ids to measurable raw % metrics
raw_pcts = compute_head_raw_pcts(cache)
TYPE_METRIC = {
    "end_of_text": "eot",
    "self_attention": "self_attn",
    "previous_token": "prev_token",
}

type_summary = []
for type_id, heads_list in TYPE_TO_HEADS.items():
    display_name, description = HEAD_TYPES[type_id]
    type_summary.append((type_id, display_name, description, heads_list))

type_summary.sort(key=lambda x: len(x[3]), reverse=True)

lines = [
    "| Type | Description | # Heads | Heads (by activity) |",
    "|------|-------------|---------|---------------------|",
]
for type_id, display_name, description, heads_list in type_summary:
    metric_key = TYPE_METRIC.get(type_id)
    if metric_key:
        # Sort by raw % descending when available
        sorted_heads = sorted(
            heads_list,
            key=lambda x: raw_pcts[(x[0][0], x[0][1])][metric_key],
            reverse=True,
        )
        heads_str = ", ".join(
            f"L{l}H{h} ({raw_pcts[(l, h)][metric_key]:.1f}%)"
            for (l, h), act in sorted_heads
        )
    else:
        sorted_heads = sorted(heads_list, key=lambda x: ACTIVITY_ORDER.get(x[1], 0), reverse=True)
        heads_str = ", ".join(f"L{l}H{h} ({act})" for (l, h), act in sorted_heads)
    lines.append(f"| **{display_name}** | {description} | {len(heads_list)} | {heads_str} |")
display(Markdown("\\n".join(lines)))"""

HEATMAP_CODE = """\
import plotly.graph_objects as go

# Build matrix: rows=types, cols=heads
type_ids = sorted(HEAD_TYPES.keys(), key=lambda tid: len(TYPE_TO_HEADS.get(tid, [])), reverse=True)
head_names = [f"L{l}H{h}" for l in range(2) for h in range(12)]

z = []
text_matrix = []
for tid in type_ids:
    row = []
    text_row = []
    heads_map = {(l, h): act for (l, h), act in TYPE_TO_HEADS.get(tid, [])}
    for l in range(2):
        for h in range(12):
            if (l, h) in heads_map:
                val = ACTIVITY_ORDER[heads_map[(l, h)]]
                row.append(val)
                text_row.append(str(val))
            else:
                row.append(0)
                text_row.append("")
    z.append(row)
    text_matrix.append(text_row)

type_labels = [HEAD_TYPES[tid][0] for tid in type_ids]

fig = go.Figure(data=go.Heatmap(
    z=z,
    x=head_names,
    y=type_labels,
    colorscale=[
        [0, "#f8f8f8"],
        [0.2, "#fde0dd"],
        [0.4, "#fa9fb5"],
        [0.6, "#f768a1"],
        [0.8, "#c51b8a"],
        [1.0, "#7a0177"],
    ],
    zmin=0,
    zmax=5,
    text=text_matrix,
    texttemplate="%{text}",
    hovertemplate="Head: %{x}<br>Type: %{y}<br>Activity: %{z}<extra></extra>",
))

fig.update_layout(
    title="Head-Type Activity Matrix",
    xaxis_title="Attention Head",
    yaxis_title="Type",
    height=600,
    width=1000,
    yaxis=dict(autorange="reversed"),
)
fig.show()"""

LAYER_COMPARISON_CODE = """\
# Count type assignments per layer
from collections import Counter

for layer in range(2):
    print(f"\\n=== Layer {layer} ===")
    type_counts = Counter()
    for type_id, heads_list in TYPE_TO_HEADS.items():
        for (l, h), activity in heads_list:
            if l == layer:
                type_counts[HEAD_TYPES[type_id][0]] += 1
    for type_name, count in type_counts.most_common():
        print(f"  {type_name}: {count} head(s)")

print("\\n=== Layer comparison ===")
layer0_types = set()
layer1_types = set()
for type_id, heads_list in TYPE_TO_HEADS.items():
    for (l, h), activity in heads_list:
        if l == 0:
            layer0_types.add(HEAD_TYPES[type_id][0])
        else:
            layer1_types.add(HEAD_TYPES[type_id][0])

only_l0 = layer0_types - layer1_types
only_l1 = layer1_types - layer0_types
print(f"Types only in Layer 0: {only_l0 or 'none'}")
print(f"Types only in Layer 1: {only_l1 or 'none'}")
print(f"Types in both layers: {layer0_types & layer1_types}")"""


def _build_classifications_md() -> str:
    """Build a markdown cell with per-head classifications from shared data."""
    lines = ["## Per-Head Classifications\n"]
    for layer in range(2):
        lines.append(f"\n**Layer {layer}:**\n")
        for head in range(12):
            classification = HEAD_CLASSIFICATIONS.get((layer, head), "TODO")
            lines.append(f"- **L{layer}H{head}**: {classification}")
    lines.append("\n## Type Summary\n")
    for type_id in sorted(TYPE_TO_HEADS.keys(), key=lambda tid: len(TYPE_TO_HEADS[tid]), reverse=True):
        display_name, _ = HEAD_TYPES[type_id]
        heads_list = TYPE_TO_HEADS[type_id]
        sorted_heads = sorted(heads_list, key=lambda x: ACTIVITY_ORDER.get(x[1], 0), reverse=True)
        head_strs = ", ".join(f"L{l}H{h} ({act})" for (l, h), act in sorted_heads)
        lines.append(f"- **{display_name}** ({len(heads_list)}x): {head_strs}")
    lines.append("\n## Activity Levels\n")
    for key, label in ACTIVITY_LABELS.items():
        lines.append(f"- {label}")
    return "\n".join(lines)


def generate_main_notebook() -> None:
    """Generate the main overview notebook."""
    cells = [
        code_cell(MAIN_SETUP_CODE),
        md_cell(MODEL_DESCRIPTION),
        code_cell("model = load_model()"),
        code_cell("str_tokens, logits, cache = run_and_cache(model)"),
        md_cell("## All Attention Patterns by Layer"),
        code_cell(LAYER_VIS_CODE),
        md_cell(_build_classifications_md()),
        md_cell(
            "## Summary: All 24 Attention Heads\n"
            "\n"
            "Programmatic summary pulling classifications and type mappings from "
            "`shared.py`. See `heads/` for per-head notebooks and `types/` for "
            "per-type notebooks."
        ),
        md_cell("### Per-Head Classification Table"),
        code_cell(PER_HEAD_TABLE_CODE),
        md_cell(
            "### Per-Type Summary\n"
            "\n"
            "Each type with the number of heads exhibiting it, sorted by head count descending."
        ),
        code_cell(PER_TYPE_TABLE_CODE),
        md_cell(
            "### EOT Attention % Across All Heads\n"
            "\n"
            "Mean attention weight allocated to position 0 (end-of-text token), "
            "computed per head and sorted by raw % descending."
        ),
        code_cell('show_attention_to_position(cache, position=0, label="EOT")'),
        md_cell(
            "### Head-Type Matrix\n"
            "\n"
            "Which heads exhibit which types. Numbers show activity level "
            "(5=full, 4=fullish, 3=half, 2=partial, 1=almost none)."
        ),
        code_cell(HEATMAP_CODE),
        md_cell(
            "### Layer-Level Observations\n"
            "\n"
            "Summary of functional differences between Layer 0 and Layer 1 heads."
        ),
        code_cell(LAYER_COMPARISON_CODE),
    ]

    write_notebook(
        PROJECT_ROOT / "attention-head-zoo-2-layer-attention-only-transformer.ipynb",
        make_notebook(cells),
    )


def main():
    print("Generating main notebook...")
    generate_main_notebook()

    print("\nGenerating head notebooks...")
    for layer in range(2):
        for head in range(12):
            generate_head_notebook(layer, head)

    print("\nGenerating type notebooks...")
    for type_id in HEAD_TYPES:
        generate_type_notebook(type_id)

    print(
        f"\nDone! Generated main notebook, 24 head notebooks, and {len(HEAD_TYPES)} type notebooks."
    )


if __name__ == "__main__":
    main()
