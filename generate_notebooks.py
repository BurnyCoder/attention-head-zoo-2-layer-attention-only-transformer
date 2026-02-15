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
    TYPE_ENTROPY_KEYS,
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
    show_head_pattern, show_attention_tables,
    compute_all_type_metrics, HEAD_TYPES, TYPE_ENTROPY_KEYS,
    ACTIVITY_LABELS, get_head_types, TEXT,
)"""

LEVEL_EXPR = (
    "level = 'full' if pct >= 90 else 'fullish' if pct >= 60 else "
    "'half' if pct >= 40 else 'partial' if pct >= 10 else "
    "'almost none' if pct >= 0.1 else '-'"
)

# Type IDs that have programmatically computable metrics via compute_all_type_metrics.
MEASURABLE_TYPES = {
    "end_of_text", "self_attention", "previous_token",
    "comma_attention", "period_attention",
    "few_previous_tokens", "entropy",
    "noun_attention", "verb_attention", "adjective_attention",
    "adverb_attention", "pronoun_attention", "preposition_attention",
    "determiner_attention", "conjunction_attention",
    "salient_word_attention",
}

LOAD_CODE = """\
model = load_model()
str_tokens, logits, cache = run_and_cache(model)"""


def generate_head_notebook(layer: int, head: int) -> None:
    """Generate a single head analysis notebook."""
    classification = HEAD_CLASSIFICATIONS.get((layer, head), "TODO: classify")

    cells = [
        md_cell(
            f"# L{layer}H{head} — {classification}\n"
            f"\n"
            f"**Layer {layer}, Head {head}**"
        ),
        code_cell(SETUP_CODE),
        code_cell(LOAD_CODE),
        md_cell("## Types exhibited"),
        code_cell(
            f"tm = compute_all_type_metrics(cache, str_tokens)\n"
            f"head_types = get_head_types({layer}, {head})\n"
            f"lines = []\n"
            f"for tid, act in head_types:\n"
            f"    pct_val = tm.get((tid, {layer}, {head}))\n"
            f"    ent_key = TYPE_ENTROPY_KEYS.get(tid)\n"
            f"    ent_val = tm.get((ent_key, {layer}, {head})) if ent_key else None\n"
            f"    if pct_val is not None:\n"
            f"        ent_str = f\", ent {{ent_val:.1f}}%\" if ent_val is not None else \"\"\n"
            f"        lines.append(f\"- **{{HEAD_TYPES[tid][0]}}** ({{pct_val:.1f}}%{{ent_str}})\")\n"
            f"    else:\n"
            f"        lines.append(f\"- **{{HEAD_TYPES[tid][0]}}** ({{ACTIVITY_LABELS[act]}})\")\n"
            f"if not lines:\n"
            f"    lines.append(\"- *No types assigned yet (needs classification)*\")\n"
            f"display(Markdown(\"\\n\".join(lines)))"
        ),
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

    # Programmatic metrics for all measurable types + manual types
    cells.append(md_cell("## Attention Metrics"))
    cells.append(
        code_cell(
            f"# Show all measurable metrics for this head (with entropy where available)\n"
            f"lines = []\n"
            f"for tid in HEAD_TYPES:\n"
            f"    key = (tid, {layer}, {head})\n"
            f"    if key in tm:\n"
            f"        ent_key = TYPE_ENTROPY_KEYS.get(tid)\n"
            f"        ent_val = tm.get((ent_key, {layer}, {head})) if ent_key else None\n"
            f"        ent_str = f\"{{ent_val:.2f}}%\" if ent_val is not None else \"\\u2014\"\n"
            f"        lines.append(f\"| {{HEAD_TYPES[tid][0]}} | {{tm[key]:.2f}}% | {{ent_str}} |\")\n"
            f"# Show non-measurable assigned types\n"
            f"for tid, act in head_types:\n"
            f"    if (tid, {layer}, {head}) not in tm:\n"
            f"        lines.append(f\"| {{HEAD_TYPES[tid][0]}} | {{ACTIVITY_LABELS[act]}} | \\u2014 |\")\n"
            f"table = \"| Type | Value | Entropy |\\n|------|-------|---------|\\n\" + \"\\n\".join(lines)\n"
            f"display(Markdown(table))"
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

    is_measurable = type_id in MEASURABLE_TYPES

    cells = [
        md_cell(
            f"# {display_name}\n"
            f"\n"
            f"{description}"
        ),
        code_cell(SETUP_CODE),
        code_cell(LOAD_CODE),
    ]

    # Summary table: show this type's metric for all 24 heads (sorted)
    cells.append(md_cell(f"## {display_name} — All 24 Heads"))
    cells.append(
        code_cell(
            f"tm = compute_all_type_metrics(cache, str_tokens)\n"
            f"ent_key = TYPE_ENTROPY_KEYS.get(\"{type_id}\")\n"
            f"is_measurable = (\"{type_id}\", 0, 0) in tm\n"
            f"if is_measurable:\n"
            f"    results = sorted(\n"
            f"        [((l, h), tm[(\"{type_id}\", l, h)]) for l in range(2) for h in range(12)],\n"
            f"        key=lambda x: x[1], reverse=True,\n"
            f"    )\n"
            f"    has_ent = ent_key and (ent_key, 0, 0) in tm\n"
            f"    if has_ent:\n"
            f"        lines = [\"| Head | {display_name} % | Entropy % |\", \"|------|-------|-------|\"]  \n"
            f"        for (l, h), pct in results:\n"
            f"            ent = tm[(ent_key, l, h)]\n"
            f"            lines.append(f\"| L{{l}}H{{h}} | {{pct:.2f}}% | {{ent:.2f}}% |\")\n"
            f"    else:\n"
            f"        lines = [\"| Head | {display_name} % |\", \"|------|-------|\"]  \n"
            f"        for (l, h), pct in results:\n"
            f"            lines.append(f\"| L{{l}}H{{h}} | {{pct:.2f}}% |\")\n"
            f"    display(Markdown(\"\\n\".join(lines)))\n"
            f"else:\n"
            f"    print(\"No programmatic metric available for this type.\")"
        )
    )

    for (l, h), activity in heads_sorted:
        classification = HEAD_CLASSIFICATIONS.get((l, h), "TODO")
        escaped = classification.replace("\\", "\\\\").replace('"', '\\"')
        if is_measurable:
            cells.append(
                code_cell(
                    f"pct = tm[(\"{type_id}\", {l}, {h})]\n"
                    f"ent_str = f\" | ent {{tm[(ent_key, {l}, {h})]:.2f}}%\" if ent_key and (ent_key, {l}, {h}) in tm else \"\"\n"
                    f"{LEVEL_EXPR}\n"
                    f'display(Markdown(f"---\\n## L{l}H{h} — {{pct:.2f}}% ({{level}}){{ent_str}}\\n\\n{escaped}"))'
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
import pandas as pd
from itables import init_notebook_mode, show as itshow
init_notebook_mode(all_interactive=True)
from shared import (
    load_model, run_and_cache, get_attention_pattern,
    show_head_pattern, show_attention_tables,
    compute_all_type_metrics,
    HEAD_CLASSIFICATIONS, HEAD_TYPES, TYPE_ENTROPY_KEYS,
    TYPE_TO_HEADS, ACTIVITY_LABELS, ACTIVITY_ORDER,
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
# Per-head summary: classification + types with activity levels + raw % + entropy
tm = compute_all_type_metrics(cache, str_tokens)

rows = []
for layer in range(2):
    for head in range(12):
        head_types = get_head_types(layer, head)
        type_parts = []
        for tid, act in head_types:
            pct_val = tm.get((tid, layer, head))
            ent_key = TYPE_ENTROPY_KEYS.get(tid)
            ent_val = tm.get((ent_key, layer, head)) if ent_key else None
            if pct_val is not None:
                ent_str = f", ent {ent_val:.1f}%" if ent_val is not None else ""
                type_parts.append(f"{HEAD_TYPES[tid][0]} ({pct_val:.1f}%{ent_str})")
            else:
                type_parts.append(f"{HEAD_TYPES[tid][0]} ({ACTIVITY_LABELS[act]})")
        types_str = ", ".join(type_parts) if type_parts else "\\u2014"
        row = {
            "Head": f"L{layer}H{head}",
            "Classification": HEAD_CLASSIFICATIONS.get((layer, head), "\\u2014"),
            "Types": types_str,
        }
        # Add columns for every measurable type + its entropy
        for tid in HEAD_TYPES:
            if (tid, 0, 0) in tm:
                row[HEAD_TYPES[tid][0]] = round(tm[(tid, layer, head)], 1)
                ent_key = TYPE_ENTROPY_KEYS.get(tid)
                if ent_key and (ent_key, layer, head) in tm:
                    row[HEAD_TYPES[tid][0] + " Ent"] = round(tm[(ent_key, layer, head)], 1)
        rows.append(row)
df = pd.DataFrame(rows)
itshow(df, paging=False, classes="display compact")"""

PER_TYPE_TABLE_CODE = """\
# Per-type summary: sorted by number of heads descending
tm = compute_all_type_metrics(cache, str_tokens)

rows = []
for type_id, heads_list in TYPE_TO_HEADS.items():
    display_name, description = HEAD_TYPES[type_id]
    has_metric = (type_id, 0, 0) in tm
    ent_key = TYPE_ENTROPY_KEYS.get(type_id)
    if has_metric:
        sorted_heads = sorted(
            heads_list,
            key=lambda x, tid=type_id: tm.get((tid, x[0][0], x[0][1]), 0),
            reverse=True,
        )
        parts = []
        for (l, h), act in sorted_heads:
            pct = tm[(type_id, l, h)]
            ent_val = tm.get((ent_key, l, h)) if ent_key else None
            if ent_val is not None:
                parts.append(f"L{l}H{h} ({pct:.1f}%, ent {ent_val:.1f}%)")
            else:
                parts.append(f"L{l}H{h} ({pct:.1f}%)")
        heads_str = ", ".join(parts)
    else:
        sorted_heads = sorted(heads_list, key=lambda x: ACTIVITY_ORDER.get(x[1], 0), reverse=True)
        heads_str = ", ".join(f"L{l}H{h} ({act})" for (l, h), act in sorted_heads)
    rows.append({
        "Type": display_name,
        "Description": description,
        "# Heads": len(heads_list),
        "Heads (by activity)": heads_str,
    })
df = pd.DataFrame(rows).sort_values("# Heads", ascending=False).reset_index(drop=True)
itshow(df, paging=False, classes="display compact")"""

HEATMAP_CODE = """\
import plotly.graph_objects as go

tm = compute_all_type_metrics(cache, str_tokens)

# Map activity levels to midpoint % for non-measurable types
ACTIVITY_MIDPOINT = {"full": 95, "fullish": 75, "half": 50, "partial": 25, "almost_none": 5}

type_ids = sorted(HEAD_TYPES.keys(), key=lambda tid: len(TYPE_TO_HEADS.get(tid, [])), reverse=True)
head_names = [f"L{l}H{h}" for l in range(2) for h in range(12)]

z = []
text_matrix = []
for tid in type_ids:
    row = []
    text_row = []
    heads_map = {(l, h): act for (l, h), act in TYPE_TO_HEADS.get(tid, [])}
    is_measurable = (tid, 0, 0) in tm
    for l in range(2):
        for h in range(12):
            if is_measurable:
                raw = tm[(tid, l, h)]
                row.append(raw)
                text_row.append(f"{raw:.0f}")
            elif (l, h) in heads_map:
                val = ACTIVITY_MIDPOINT[heads_map[(l, h)]]
                row.append(val)
                text_row.append(f"~{val}")
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
        [0.01, "#fff5f0"],
        [0.1, "#fde0dd"],
        [0.25, "#fa9fb5"],
        [0.4, "#f768a1"],
        [0.6, "#c51b8a"],
        [0.8, "#7a0177"],
        [1.0, "#49006a"],
    ],
    zmin=0,
    zmax=100,
    text=text_matrix,
    texttemplate="%{text}",
    hovertemplate="Head: %{x}<br>Type: %{y}<br>Activity: %{z:.1f}%<extra></extra>",
))

fig.update_layout(
    title="Head-Type Activity Matrix (% attention weight)",
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
            "### Head-Type Matrix\n"
            "\n"
            "Which heads exhibit which types. Values show raw attention % where "
            "computable, ~midpoint estimates for non-measurable types."
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
