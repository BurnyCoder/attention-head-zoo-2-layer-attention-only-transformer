"""Shared utilities for attention head zoo notebooks."""

import torch as t
from IPython.display import Markdown, display
from jaxtyping import Float
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer, HookedTransformerConfig

import circuitsvis as cv

# === Device setup ===
device = t.set_default_device(
    "cuda"
    if t.cuda.is_available()
    else "mps"
    if t.backends.mps.is_available()
    else "cpu"
)
t.set_grad_enabled(False)

# === Prompt text ===
TEXT = (
    "We think that powerful, significantly superhuman machine intelligence is more "
    "likely than not to be created this century. If current machine learning techniques "
    "were scaled up to this level, we think they would by default produce systems that "
    "are deceptive or manipulative, and that no solid plans are known for how to avoid this."
)

# === Activity levels ===
ACTIVITY_ORDER = {
    "full": 5,  # 100-90%
    "fullish": 4,  # 90-60%
    "half": 3,  # 60-40%
    "partial": 2,  # 40-10%
    "almost_none": 1,  # 10-0.1%
}

ACTIVITY_LABELS = {
    "full": "full (100-90%)",
    "fullish": "fullish (90-60%)",
    "half": "half (60-40%)",
    "partial": "partial (40-10%)",
    "almost_none": "almost none (10-0.1%)",
}

ACTIVITY_PCT_RANGES = {
    "full": "100-90%",
    "fullish": "90-60%",
    "half": "60-40%",
    "partial": "40-10%",
    "almost_none": "10-0.1%",
}

# === Per-head classifications ===
HEAD_CLASSIFICATIONS: dict[tuple[int, int], str] = {
    (0, 0): 'Attends to glue words like "are", "and", "if", "that", "were", "or"',
    (0, 1): "Attends from and to words representing certainty and questioning (if, likely, are, think, known to, how) and some stuff from . consistently",
    (0, 2): "Cares primarily about end of text token and some ,",
    (0, 3): "Cares only about end of text token",
    (0, 4): "Attends to few previous tokens",
    (0, 5): "Attends to just stuff like ,",
    (0, 6): "Attends to only end of text token",
    (0, 7): "Attends only to previous token, one to semantically salient (scaled up)",
    (0, 8): 'Attends to glue words like "are", "and", "if", "that", "were", "or"',
    (0, 9): "Attends to itself and to previous token and to few previous tokens",
    (0, 10): "Glue words, certainty, end of text",
    (0, 11): "To itself, glue words, end of text, semantically salient (scaled up, deceptive), certainty",
    (1, 0): 'End of text token, itself, token ","',
    (1, 1): "End of text, previous token, glue words",
    (1, 2): "End of text, self-attention, content words attend to EOT",
    (1, 3): "Primarily end of text token, diffuse across positions",
    (1, 4): "Cares only about end of text token",
    (1, 5): "Aggregating all context into rich words? and dot somehow cares about end of text wtf",
    (1, 6): "Connector between glue words and semantically rich words :D and some other types already listed",
    (1, 7): "Connector between glue words and semantically rich words (and other glue words?) (1x partial)",
    (1, 8): "Attending to previous token if directly related and other types already listed",
    (1, 9): "Diffuse attention across context, some end of text, broad aggregation",
    (1, 10): "Connector between glue words and semantically rich words, semantic connector between two related things (machine intelligence)",
    (1, 11): "End of text for function words, glue words, previous token",
}

# === Type definitions ===
HEAD_TYPES: dict[str, tuple[str, str]] = {
    "glue_words": (
        "Glue Word Attender",
        'Attends to function/glue words like "are", "and", "if", "that", "were", "or"',
    ),
    "certainty_questioning": (
        "Certainty/Questioning Attender",
        'Attends to words expressing certainty or uncertainty ("if", "likely", "think", "known", "how")',
    ),
    "end_of_text": (
        "End-of-Text Attender",
        "Attends primarily to the beginning-of-sequence / end-of-text token",
    ),
    "few_previous_tokens": (
        "Few Previous Tokens Head",
        "Attends to a small window of preceding tokens",
    ),
    "previous_token": (
        "Previous Token Head",
        "Attends to the immediately preceding token",
    ),
    "period_attention": (
        "Period Attender",
        "Attends to period (.) tokens",
    ),
    "comma_attention": (
        "Comma Attender",
        "Attends to comma (,) tokens",
    ),
    "self_attention": (
        "Self-Attender",
        "Attends primarily to the current token position (itself)",
    ),
    "semantically_salient": (
        "Semantically Salient Attender",
        "Attends to content words with high semantic salience (scaled up, deceptive)",
    ),
    "context_aggregation": (
        "Context Aggregator",
        "Aggregates broad context into semantically rich word positions",
    ),
    "dot_eot_quirk": (
        "Dot-EOT Quirk",
        "Period (.) token somehow attends to end-of-text token",
    ),
    "glue_semantic_connector": (
        "Glue-to-Semantic Connector",
        "Connects function/glue words to semantically rich content words",
    ),
    "glue_glue_connector": (
        "Glue-to-Glue Connector",
        "Connects function/glue words to other function/glue words",
    ),
    "related_previous_token": (
        "Related Previous Token",
        "Attends to the previous token when directly semantically related",
    ),
    "semantic_connector": (
        "Semantic Connector",
        'Connects semantically related tokens (e.g., "machine" and "intelligence")',
    ),
}

# === Type-to-heads mapping (sorted by activity level descending) ===
TYPE_TO_HEADS: dict[str, list[tuple[tuple[int, int], str]]] = {
    "glue_words": [
        ((0, 0), "full"),
        ((0, 8), "full"),
        ((0, 10), "partial"),
        ((0, 11), "partial"),
        ((1, 1), "partial"),
        ((1, 11), "partial"),
    ],
    "certainty_questioning": [
        ((0, 1), "half"),
        ((0, 10), "partial"),
        ((0, 11), "partial"),
    ],
    "end_of_text": [
        ((1, 4), "full"),
        ((0, 3), "full"),
        ((0, 6), "fullish"),
        ((1, 3), "half"),
        ((1, 2), "partial"),
        ((1, 11), "partial"),
        ((0, 2), "partial"),
        ((0, 10), "partial"),
        ((0, 11), "partial"),
        ((1, 0), "partial"),
        ((1, 9), "partial"),
    ],
    "few_previous_tokens": [
        ((0, 4), "full"),
        ((0, 9), "partial"),
    ],
    "previous_token": [
        ((0, 7), "full"),
        ((0, 9), "partial"),
        ((1, 1), "partial"),
        ((1, 11), "partial"),
    ],
    "period_attention": [
        ((1, 5), "half"),
    ],
    "comma_attention": [
        ((0, 5), "full"),
        ((0, 2), "partial"),
    ],
    "self_attention": [
        ((1, 2), "partial"),
        ((0, 9), "partial"),
        ((0, 11), "partial"),
    ],
    "semantically_salient": [
        ((0, 7), "half"),
        ((0, 11), "partial"),
    ],
    "context_aggregation": [
        ((1, 5), "half"),
        ((1, 9), "partial"),
    ],
    "dot_eot_quirk": [
        ((1, 5), "half"),
    ],
    "glue_semantic_connector": [
        ((1, 6), "half"),
        ((1, 7), "half"),
        ((1, 10), "half"),
    ],
    "glue_glue_connector": [
        ((1, 7), "partial"),
    ],
    "related_previous_token": [
        ((1, 8), "partial"),
    ],
    "semantic_connector": [
        ((1, 10), "partial"),
    ],
}


def get_head_types(layer: int, head: int) -> list[tuple[str, str]]:
    """Get all types exhibited by a given head, with their activity levels.

    Returns list of (type_id, activity_level) sorted by activity descending.
    """
    result = []
    for type_id, head_list in TYPE_TO_HEADS.items():
        for (l, h), activity in head_list:
            if l == layer and h == head:
                result.append((type_id, activity))
    result.sort(key=lambda x: ACTIVITY_ORDER.get(x[1], 0), reverse=True)
    return result


# === Model loading ===
def load_model() -> HookedTransformer:
    """Load the 2-layer attention-only transformer with pretrained weights."""
    from huggingface_hub import hf_hub_download

    cfg = HookedTransformerConfig(
        d_model=768,
        d_head=64,
        n_heads=12,
        n_layers=2,
        n_ctx=2048,
        d_vocab=50278,
        attention_dir="causal",
        attn_only=True,
        tokenizer_name="EleutherAI/gpt-neox-20b",
        seed=398,
        use_attn_result=True,
        normalization_type=None,
        positional_embedding_type="shortformer",
    )
    weights_path = hf_hub_download(
        repo_id="callummcdougall/attn_only_2L_half",
        filename="attn_only_2L_half.pth",
    )
    model = HookedTransformer(cfg)
    pretrained_weights = t.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(pretrained_weights)
    return model


def run_and_cache(
    model: HookedTransformer, text: str = TEXT
) -> tuple[list[str], t.Tensor, ActivationCache]:
    """Run model on text and return (str_tokens, logits, cache)."""
    logits, cache = model.run_with_cache(text, remove_batch_dim=True)
    str_tokens = model.to_str_tokens(text)
    return str_tokens, logits, cache


def get_attention_pattern(
    cache: ActivationCache, layer: int, head: int
) -> Float[Tensor, "dest_pos src_pos"]:
    """Extract attention pattern for a single head."""
    return cache["pattern", layer][head]


# === Visualization ===
def show_head_pattern(
    str_tokens: list[str],
    cache: ActivationCache,
    layer: int,
    head: int,
) -> None:
    """Display attention pattern visualizations for a single head."""
    attention = get_attention_pattern(cache, layer, head)
    # Interactive token-highlighting view (click tokens to see attention flow)
    display(
        cv.attention.attention_pattern(
            tokens=str_tokens,
            attention=attention,
        )
    )
    # Attention head thumbnail + detail view (expects [num_heads, dest, src])
    display(
        cv.attention.attention_patterns(
            tokens=str_tokens,
            attention=attention.unsqueeze(0),
        )
    )


def attention_pairs_table(
    str_tokens: list[str],
    attention: Float[Tensor, "dest_pos src_pos"],
    top_k: int = 25,
    ascending: bool = False,
) -> str:
    """Generate a markdown table of (dest_token, src_token, weight) triples.

    Args:
        str_tokens: List of token strings.
        attention: Attention matrix [dest_pos, src_pos] for a single head.
        top_k: Number of rows to include.
        ascending: If True, sort smallest-to-largest.

    Returns:
        Markdown-formatted table string.
    """
    n_tokens = len(str_tokens)
    rows = []
    for dest_pos in range(n_tokens):
        for src_pos in range(dest_pos + 1):  # causal: src <= dest
            weight = attention[dest_pos, src_pos].item()
            rows.append((dest_pos, src_pos, weight))

    rows.sort(key=lambda r: r[2], reverse=not ascending)
    rows = rows[:top_k]

    lines = [
        "| Rank | Dest Token | Src Token | Dest Pos | Src Pos | Weight |",
        "|------|-----------|-----------|----------|---------|--------|",
    ]
    for i, (dp, sp, w) in enumerate(rows, 1):
        dt = str_tokens[dp].replace("|", "\\|")
        st = str_tokens[sp].replace("|", "\\|")
        lines.append(f"| {i} | `{dt}` | `{st}` | {dp} | {sp} | {w:.4f} |")

    return "\n".join(lines)


def show_attention_tables(
    str_tokens: list[str],
    attention: Float[Tensor, "dest_pos src_pos"],
    top_k: int = 25,
) -> None:
    """Display both highest and lowest attention weight tables."""
    display(Markdown("### Highest attention weights (destination <- source)"))
    display(
        Markdown(
            attention_pairs_table(str_tokens, attention, top_k=top_k, ascending=False)
        )
    )
    display(Markdown("### Lowest attention weights (destination <- source)"))
    display(
        Markdown(
            attention_pairs_table(str_tokens, attention, top_k=top_k, ascending=True)
        )
    )
