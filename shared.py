"""Shared utilities for attention head zoo notebooks."""

from collections.abc import Callable

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
        "Fraction of total attention where period (.) is the source (attended to)",
    ),
    "comma_attention": (
        "Comma Attender",
        "Fraction of total attention where comma (,) is the source (attended to)",
    ),
    "self_attention": (
        "Self-Attender",
        "Attends primarily to the current token position (itself)",
    ),
    "entropy": (
        "Entropy %",
        "Normalized entropy of attention distribution (0%=concentrated on one token, 100%=uniform)",
    ),
    "noun_attention": (
        "Noun Attender",
        "Fraction of attention directed to noun positions (NN, NNS, NNP, NNPS)",
    ),
    "verb_attention": (
        "Verb Attender",
        "Fraction of attention directed to verb positions (VB, VBD, VBG, VBN, VBP, VBZ, MD)",
    ),
    "adjective_attention": (
        "Adjective Attender",
        "Fraction of attention directed to adjective positions (JJ, JJR, JJS)",
    ),
    "adverb_attention": (
        "Adverb Attender",
        "Fraction of attention directed to adverb positions (RB, RBR, RBS, WRB)",
    ),
    "pronoun_attention": (
        "Pronoun Attender",
        "Fraction of attention directed to pronoun positions (PRP, WP, WDT)",
    ),
    "preposition_attention": (
        "Preposition Attender",
        "Fraction of attention directed to preposition/particle positions (IN, TO, RP)",
    ),
    "determiner_attention": (
        "Determiner Attender",
        "Fraction of attention directed to determiner positions (DT)",
    ),
    "conjunction_attention": (
        "Conjunction Attender",
        "Fraction of attention directed to conjunction positions (CC)",
    ),
    "salient_word_attention": (
        "Salient Word Attender",
        "Fraction of attention directed to semantically salient content words (powerful, superhuman, intelligence, deceptive, etc.)",
    ),
    "ai_word_attention": (
        "AI Word Attender",
        "Fraction of attention directed to AI/ML-related words (superhuman, machine, intelligence, learning, scaled, up)",
    ),
    "spooky_word_attention": (
        "Spooky Word Attender",
        "Fraction of attention directed to spooky/deceptive words (deceptive, manipulative)",
    ),
    "glue_word_attention": (
        "Glue Word Attender (auto)",
        "Fraction of attention directed to function/glue words (we, that, is, are, to, be, if, and, or, etc.)",
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
    "entropy": [],
    "noun_attention": [],
    "verb_attention": [],
    "adjective_attention": [],
    "adverb_attention": [],
    "pronoun_attention": [],
    "preposition_attention": [],
    "determiner_attention": [],
    "conjunction_attention": [],
    "salient_word_attention": [],
    "ai_word_attention": [],
    "spooky_word_attention": [],
    "glue_word_attention": [],
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


# Mapping from parent type_id to its entropy key in compute_all_type_metrics.
# Entropy measures how spread out the behavior is across positions (0%=concentrated, 100%=uniform).
TYPE_ENTROPY_KEYS: dict[str, str] = {
    "end_of_text": "eot_entropy",
    "self_attention": "self_attention_entropy",
    "previous_token": "prev_token_entropy",
    "comma_attention": "comma_attention_entropy",
    "period_attention": "period_attention_entropy",
    "few_previous_tokens": "few_prev_tokens_entropy",
    "noun_attention": "noun_attention_entropy",
    "verb_attention": "verb_attention_entropy",
    "adjective_attention": "adjective_attention_entropy",
    "adverb_attention": "adverb_attention_entropy",
    "pronoun_attention": "pronoun_attention_entropy",
    "preposition_attention": "preposition_attention_entropy",
    "determiner_attention": "determiner_attention_entropy",
    "conjunction_attention": "conjunction_attention_entropy",
    "salient_word_attention": "salient_word_attention_entropy",
    "ai_word_attention": "ai_word_attention_entropy",
    "spooky_word_attention": "spooky_word_attention_entropy",
    "glue_word_attention": "glue_word_attention_entropy",
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


def _classify_pct(pct: float) -> str:
    """Classify a percentage into an activity level string."""
    if pct >= 90:
        return "full"
    elif pct >= 60:
        return "fullish"
    elif pct >= 40:
        return "half"
    elif pct >= 10:
        return "partial"
    elif pct >= 0.1:
        return "almost none"
    else:
        return "-"


def _show_metric_table(results: list[tuple[int, int, float, str]], label: str) -> None:
    """Print a sorted table of (layer, head, pct, level) tuples."""
    print(f"{'Head':<8} {label + ' %':>12}  {'Level'}")
    print("-" * 35)
    for layer, head, pct, level in results:
        print(f"L{layer}H{head:<5} {pct:>11.2f}%  {level}")


def _compute_metric_pcts(
    cache: ActivationCache,
    metric_fn: Callable[[Float[Tensor, "dest src"]], float],
    n_layers: int = 2,
    n_heads: int = 12,
) -> list[tuple[int, int, float, str]]:
    """Compute a metric for every head and return sorted results.

    Args:
        cache: Activation cache from model forward pass.
        metric_fn: Function taking a single head's attention matrix [dest, src]
            and returning a fraction (0-1) to be displayed as %.

    Returns:
        List of (layer, head, pct, level) sorted by pct descending.
    """
    results = []
    for layer in range(n_layers):
        for head in range(n_heads):
            attention = get_attention_pattern(cache, layer, head)
            pct = metric_fn(attention) * 100
            results.append((layer, head, pct, _classify_pct(pct)))
    results.sort(key=lambda x: x[2], reverse=True)
    return results


# --- Specific metric functions (all average over all positions) ---

def attention_to_position_pct(
    cache: ActivationCache, position: int, **kwargs
) -> list[tuple[int, int, float, str]]:
    """Mean attention to a given source position, averaged over all dest positions."""
    return _compute_metric_pcts(
        cache, lambda a: a[:, position].mean().item(), **kwargs
    )

def show_attention_to_position(cache: ActivationCache, position: int, label: str = "position") -> None:
    _show_metric_table(attention_to_position_pct(cache, position), label)

def self_attention_pcts(cache: ActivationCache, **kwargs) -> list[tuple[int, int, float, str]]:
    """Mean diagonal attention (self-attention), averaged over all positions."""
    return _compute_metric_pcts(
        cache, lambda a: t.diagonal(a).mean().item(), **kwargs
    )

def show_self_attention_pcts(cache: ActivationCache) -> None:
    _show_metric_table(self_attention_pcts(cache), "Self-attn")

def prev_token_pcts(cache: ActivationCache, **kwargs) -> list[tuple[int, int, float, str]]:
    """Mean previous-token attention, averaged over all positions."""
    def metric(a):
        n = a.shape[0]
        if n <= 1:
            return 0.0
        return t.tensor([a[i, i - 1].item() for i in range(1, n)]).mean().item()
    return _compute_metric_pcts(cache, metric, **kwargs)

def show_prev_token_pcts(cache: ActivationCache) -> None:
    _show_metric_table(prev_token_pcts(cache), "Prev-tok")

def attention_to_token_pcts(
    cache: ActivationCache, str_tokens: list[str], token: str, **kwargs
) -> list[tuple[int, int, float, str]]:
    """Mean attention to positions containing token, averaged over all dest positions."""
    positions = [i for i, tok in enumerate(str_tokens) if token in tok]
    if not positions:
        return []
    return _compute_metric_pcts(
        cache, lambda a: a[:, positions].sum(dim=-1).mean().item(), **kwargs
    )

def show_attention_to_token(
    cache: ActivationCache, str_tokens: list[str], token: str, label: str
) -> None:
    _show_metric_table(attention_to_token_pcts(cache, str_tokens, token), label)

def few_prev_tokens_pcts(
    cache: ActivationCache, k: int = 5, **kwargs
) -> list[tuple[int, int, float, str]]:
    """Mean attention to k preceding tokens, averaged over all positions."""
    def metric(a):
        n = a.shape[0]
        return sum(a[i, max(0, i - k):i].sum().item() for i in range(n)) / n
    return _compute_metric_pcts(cache, metric, **kwargs)

def entropy_pcts(cache: ActivationCache, **kwargs) -> list[tuple[int, int, float, str]]:
    """Normalized mean entropy of attention (0%=one token, 100%=uniform)."""
    def metric(a):
        # Per-row entropy, averaged over all dest positions
        # Normalize by log(n_visible) per row for causal attention
        n = a.shape[0]
        total = 0.0
        for i in range(n):
            row = a[i, :i + 1]  # only causal positions (0..i)
            row = row.clamp(min=1e-10)
            ent = -(row * row.log()).sum().item()
            max_ent = t.tensor(float(i + 1)).log().item()  # log(n_visible)
            total += ent / max_ent if max_ent > 0 else 0.0
        return total / n
    return _compute_metric_pcts(cache, metric, **kwargs)

def show_few_prev_tokens_pcts(cache: ActivationCache, k: int = 5) -> None:
    _show_metric_table(few_prev_tokens_pcts(cache, k), f"Prev-{k}tok")


def _values_entropy_normalized(values: Float[Tensor, "n"]) -> float:
    """Compute normalized entropy of a 1D tensor of non-negative values.

    Returns 0-1 (0=all weight on one position, 1=uniform across all).
    """
    s = values.sum()
    if s <= 0:
        return 0.0
    p = values / s
    p = p.clamp(min=1e-10)
    ent = -(p * p.log()).sum().item()
    max_ent = t.tensor(float(len(values))).log().item()
    return ent / max_ent if max_ent > 0 else 0.0


def _metric_entropy(
    cache: ActivationCache,
    per_pos_fn: Callable[[Float[Tensor, "dest src"]], Float[Tensor, "n"]],
    **kwargs,
) -> list[tuple[int, int, float, str]]:
    """Entropy of per-position metric values (how spread the behavior is).

    per_pos_fn: takes attention matrix, returns a 1D tensor of per-position values.
    """
    def metric(a):
        values = per_pos_fn(a)
        return _values_entropy_normalized(values)
    return _compute_metric_pcts(cache, metric, **kwargs)


def eot_entropy_pcts(cache: ActivationCache, **kwargs):
    """How spread out EOT attention is across positions."""
    return _metric_entropy(cache, lambda a: a[:, 0], **kwargs)

def self_attention_entropy_pcts(cache: ActivationCache, **kwargs):
    """How spread out self-attention is across positions."""
    return _metric_entropy(cache, lambda a: t.diagonal(a), **kwargs)

def prev_token_entropy_pcts(cache: ActivationCache, **kwargs):
    """How spread out previous-token attention is across positions."""
    def per_pos(a):
        n = a.shape[0]
        if n <= 1:
            return t.zeros(1)
        return t.tensor([a[i, i - 1].item() for i in range(1, n)])
    return _metric_entropy(cache, per_pos, **kwargs)

def token_attention_entropy_pcts(
    cache: ActivationCache, str_tokens: list[str], token: str, **kwargs
):
    """How spread out attention-to-token is across dest positions."""
    positions = [i for i, tok in enumerate(str_tokens) if token in tok]
    if not positions:
        return []
    return _metric_entropy(
        cache, lambda a: a[:, positions].sum(dim=-1), **kwargs
    )

def few_prev_tokens_entropy_pcts(cache: ActivationCache, k: int = 5, **kwargs):
    """How spread out few-prev-tokens attention is across positions."""
    def per_pos(a):
        n = a.shape[0]
        return t.tensor([a[i, max(0, i - k):i].sum().item() for i in range(n)])
    return _metric_entropy(cache, per_pos, **kwargs)


# === POS-based attention metrics ===

POS_CATEGORIES: dict[str, set[str]] = {
    "noun": {"NN", "NNS", "NNP", "NNPS"},
    "verb": {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"},
    "adjective": {"JJ", "JJR", "JJS"},
    "adverb": {"RB", "RBR", "RBS", "WRB"},
    "pronoun": {"PRP", "PRP$", "WP", "WP$", "WDT"},
    "preposition": {"IN", "TO", "RP"},
    "determiner": {"DT"},
    "conjunction": {"CC"},
}


SALIENT_WORDS = {
    "powerful", "significantly", "superhuman", "machine", "intelligence",
    "century", "learning", "scaled", "systems", "level",
    "deceptive", "manipulative", "plans", "avoid", "think",
}

AI_WORDS = {"superhuman", "machine", "intelligence", "learning", "scaled", "up"}

SPOOKY_WORDS = {"deceptive", "manipulative"}

GLUE_WORDS = {
    "we", "that", "is", "more", "than", "to", "be", "this", "if",
    "were", "they", "by", "are", "or", "and", "for", "how",
}


def _reconstruct_words(str_tokens: list[str]) -> list[tuple[str, list[int]]]:
    """Reconstruct words from subword tokens.

    Returns list of (word_string, [token_indices]).
    Skips the first token (assumed to be <|endoftext|>).
    """
    words: list[tuple[str, list[int]]] = []
    current_word = ""
    current_indices: list[int] = []

    for i, tok in enumerate(str_tokens):
        if i == 0:  # skip <|endoftext|>
            continue
        if tok.startswith(" "):
            if current_indices:
                words.append((current_word, current_indices))
            current_word = tok[1:]
            current_indices = [i]
        elif tok[0].isalpha():
            if current_indices:
                current_word += tok
                current_indices.append(i)
            else:
                current_word = tok
                current_indices = [i]
        else:
            if current_indices:
                words.append((current_word, current_indices))
            current_word = tok
            current_indices = [i]
    if current_indices:
        words.append((current_word, current_indices))
    return words


def _get_pos_positions(str_tokens: list[str]) -> dict[str, list[int]]:
    """Map POS categories to token position lists using NLTK POS tagging."""
    import nltk
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)

    words = _reconstruct_words(str_tokens)
    tagged = nltk.pos_tag([w for w, _ in words])

    result: dict[str, list[int]] = {cat: [] for cat in POS_CATEGORIES}
    for (_, indices), (_, pos) in zip(words, tagged):
        for cat, pos_set in POS_CATEGORIES.items():
            if pos in pos_set:
                result[cat].extend(indices)
                break
    return result


def _get_word_set_positions(str_tokens: list[str], word_set: set[str]) -> list[int]:
    """Get token positions for words in the given set (case-insensitive)."""
    positions: list[int] = []
    for word, indices in _reconstruct_words(str_tokens):
        if word.lower() in word_set:
            positions.extend(indices)
    return positions


def attention_to_positions_pcts(
    cache: ActivationCache, positions: list[int], **kwargs
) -> list[tuple[int, int, float, str]]:
    """Mean attention to given source positions, averaged over all dest positions."""
    if not positions:
        return []
    return _compute_metric_pcts(
        cache, lambda a: a[:, positions].sum(dim=-1).mean().item(), **kwargs
    )


def positions_attention_entropy_pcts(
    cache: ActivationCache, positions: list[int], **kwargs
) -> list[tuple[int, int, float, str]]:
    """Entropy of attention to given positions across dest positions."""
    if not positions:
        return []
    return _metric_entropy(
        cache, lambda a: a[:, positions].sum(dim=-1), **kwargs
    )


def compute_head_raw_pcts(
    cache: ActivationCache,
    n_layers: int = 2,
    n_heads: int = 12,
) -> dict[tuple[int, int], dict[str, float]]:
    """Compute raw attention % for measurable types for all heads.

    Returns dict mapping (layer, head) -> {metric_name: pct}.
    Metrics: eot, self_attn, prev_token.
    """
    results = {}
    for layer in range(n_layers):
        for head in range(n_heads):
            attention = get_attention_pattern(cache, layer, head)
            n = attention.shape[0]
            eot_pct = attention[:, 0].mean().item() * 100
            self_pct = t.diagonal(attention).mean().item() * 100
            prev_pct = (
                t.tensor([attention[i, i - 1].item() for i in range(1, n)]).mean().item() * 100
                if n > 1 else 0.0
            )
            results[(layer, head)] = {
                "eot": eot_pct,
                "self_attn": self_pct,
                "prev_token": prev_pct,
            }
    return results


def compute_all_type_metrics(
    cache: ActivationCache,
    str_tokens: list[str],
) -> dict[tuple[str, int, int], float]:
    """Compute raw % for all measurable type_ids for all heads.

    Returns dict mapping (type_id, layer, head) -> pct.
    """
    pos_positions = _get_pos_positions(str_tokens)
    salient_positions = _get_word_set_positions(str_tokens, SALIENT_WORDS)
    ai_positions = _get_word_set_positions(str_tokens, AI_WORDS)

    metric_calls: dict[str, list[tuple[int, int, float, str]]] = {
        "end_of_text": attention_to_position_pct(cache, position=0),
        "self_attention": self_attention_pcts(cache),
        "previous_token": prev_token_pcts(cache),
        "comma_attention": attention_to_token_pcts(cache, str_tokens, ","),
        "period_attention": attention_to_token_pcts(cache, str_tokens, "."),
        "few_previous_tokens": few_prev_tokens_pcts(cache, k=5),
        "entropy": entropy_pcts(cache),
        "eot_entropy": eot_entropy_pcts(cache),
        "self_attention_entropy": self_attention_entropy_pcts(cache),
        "prev_token_entropy": prev_token_entropy_pcts(cache),
        "comma_attention_entropy": token_attention_entropy_pcts(cache, str_tokens, ","),
        "period_attention_entropy": token_attention_entropy_pcts(cache, str_tokens, "."),
        "few_prev_tokens_entropy": few_prev_tokens_entropy_pcts(cache, k=5),
    }
    # Word-set metrics (salient + AI)
    for type_id, positions in [
        ("salient_word_attention", salient_positions),
        ("ai_word_attention", ai_positions),
        ("spooky_word_attention", _get_word_set_positions(str_tokens, SPOOKY_WORDS)),
        ("glue_word_attention", _get_word_set_positions(str_tokens, GLUE_WORDS)),
    ]:
        metric_calls[type_id] = attention_to_positions_pcts(cache, positions)
        ent_key = TYPE_ENTROPY_KEYS.get(type_id)
        if ent_key:
            metric_calls[ent_key] = positions_attention_entropy_pcts(cache, positions)

    # POS-based metrics
    for pos_cat, positions in pos_positions.items():
        type_id = f"{pos_cat}_attention"
        metric_calls[type_id] = attention_to_positions_pcts(cache, positions)
        ent_key = TYPE_ENTROPY_KEYS.get(type_id)
        if ent_key:
            metric_calls[ent_key] = positions_attention_entropy_pcts(cache, positions)
    result = {}
    for type_id, entries in metric_calls.items():
        for layer, head, pct, _ in entries:
            result[(type_id, layer, head)] = pct
    return result


# === Cross-type attention metrics ===

CROSS_TYPE_NAMES: dict[str, str] = {
    "eot": "EOT",
    "comma": "Comma",
    "period": "Period",
    "noun": "Noun",
    "verb": "Verb",
    "adjective": "Adjective",
    "adverb": "Adverb",
    "pronoun": "Pronoun",
    "preposition": "Preposition",
    "determiner": "Determiner",
    "conjunction": "Conjunction",
    "salient": "Salient",
    "ai": "AI",
    "spooky": "Spooky",
    "glue": "Glue",
}


def get_type_positions(str_tokens: list[str]) -> dict[str, list[int]]:
    """Get token positions for all cross-type categories.

    Returns dict mapping short type name -> list of token positions.
    """
    pos_positions = _get_pos_positions(str_tokens)
    result: dict[str, list[int]] = {
        "eot": [0],
        "comma": [i for i, tok in enumerate(str_tokens) if "," in tok],
        "period": [i for i, tok in enumerate(str_tokens) if "." in tok],
        "salient": _get_word_set_positions(str_tokens, SALIENT_WORDS),
        "ai": _get_word_set_positions(str_tokens, AI_WORDS),
        "spooky": _get_word_set_positions(str_tokens, SPOOKY_WORDS),
        "glue": _get_word_set_positions(str_tokens, GLUE_WORDS),
    }
    result.update(pos_positions)
    return result


def compute_cross_type_metrics(
    cache: ActivationCache,
    str_tokens: list[str],
    n_layers: int = 2,
    n_heads: int = 12,
) -> dict[tuple[str, int, int], float]:
    """Compute cross-type attention metrics for all type pairs and heads.

    Returns dict mapping (cross_key, layer, head) -> pct.
    cross_key is "{from}_to_{to}" for pct, "{from}_to_{to}_entropy" for entropy.
    """
    type_positions = get_type_positions(str_tokens)
    result: dict[tuple[str, int, int], float] = {}

    for layer in range(n_layers):
        for head in range(n_heads):
            a = get_attention_pattern(cache, layer, head)
            for from_type, dest_pos in type_positions.items():
                if not dest_pos:
                    continue
                for to_type, src_pos in type_positions.items():
                    if not src_pos:
                        continue
                    key = f"{from_type}_to_{to_type}"
                    pct = a[dest_pos][:, src_pos].sum(dim=-1).mean().item() * 100
                    result[(key, layer, head)] = pct
                    values = a[dest_pos][:, src_pos].sum(dim=-1)
                    ent = _values_entropy_normalized(values) * 100
                    result[(f"{key}_entropy", layer, head)] = ent
    return result


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
