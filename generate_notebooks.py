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
from shared import (
    load_model, run_and_cache, get_attention_pattern,
    show_head_pattern, show_attention_tables, TEXT,
)"""

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

    # Summary list
    summary_lines = []
    for (l, h), activity in heads_sorted:
        summary_lines.append(f"- **L{l}H{h}** — {ACTIVITY_LABELS[activity]}")
    summary_md = (
        "\n".join(summary_lines)
        if summary_lines
        else "- *No heads assigned to this type*"
    )

    cells = [
        md_cell(
            f"# {display_name}\n"
            f"\n"
            f"{description}\n"
            f"\n"
            f"## Heads exhibiting this type ({len(heads_sorted)} total):\n"
            f"{summary_md}"
        ),
        code_cell(SETUP_CODE),
        code_cell(LOAD_CODE),
    ]

    for (l, h), activity in heads_sorted:
        classification = HEAD_CLASSIFICATIONS.get((l, h), "TODO")
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


def main():
    print("Generating head notebooks...")
    for layer in range(2):
        for head in range(12):
            generate_head_notebook(layer, head)

    print("\nGenerating type notebooks...")
    for type_id in HEAD_TYPES:
        generate_type_notebook(type_id)

    print(
        f"\nDone! Generated 24 head notebooks and {len(HEAD_TYPES)} type notebooks."
    )


if __name__ == "__main__":
    main()
