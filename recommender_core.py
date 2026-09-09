"""
Shared recommendation logic for the Fashion Recommender.

Imported by both `streamlit_app.py` (the deployed app) and `recommender.py`
(the local test script) so the ranking behaviour can never drift between them.

Core idea (Phase 1):
    - The BERT heads give a *distribution* over each attribute axis, not just a
      single label. We treat that distribution as the user's *target profile*.
    - Every product's CLIP attribute scores are turned into a distribution over
      the same labels.
    - Products are ranked by how *close* their profile is to the target
      (cosine similarity), not by which product has the single highest score.
      A product that is 100% "party" is now worse than one that is 70% party /
      30% casual when that blend is what was asked for.
"""

import math
import os

# Axis blend weights (must sum to ~1.0). Occasion + formality dominate.
WEIGHTS = {
    "occasion": 0.30,
    "formality": 0.30,
    "constraint": 0.25,
    "color_tone": 0.15,
}

# BERT label  ->  the exact CLIP prompt string stored in ChromaDB metadata.
OCCASION_LABEL_MAP = {
    "casual": "casual everyday wear",
    "party": "party outfit",
    "wedding": "wedding guest outfit",
    "office": "office wear",
    "date": "date night outfit",
    # `vacation` is not a BERT class yet (Phase 2) but every product already
    # carries this CLIP score, so wiring it now is harmless and forward-ready.
    "vacation": "beach vacation outfit",
}

FORMALITY_LABEL_MAP = {
    "formal": "very formal outfit",
    "semi-formal": "semi formal outfit",
    "casual": "casual informal outfit",
}

BOLDNESS_LABEL_MAP = {
    "understated": "subtle and understated outfit",
    "bold": "bold and loud statement outfit",
    "no_constraint": None,  # no CLIP string -> treated as neutral
}

COLOR_LABEL_MAP = {
    "soft": "soft muted pastel colors",
    "bright": "bright bold colors",
    "dark": "dark moody colors",
    "neutral": "neutral minimal colors",
}

AXIS_TO_CLIP_MAP = {
    "occasion": OCCASION_LABEL_MAP,
    "formality": FORMALITY_LABEL_MAP,
    "constraint": BOLDNESS_LABEL_MAP,
    "color_tone": COLOR_LABEL_MAP,
}

AXIS_TO_PRODUCT_KEY = {
    "occasion": "occasion_scores",
    "formality": "formality_scores",
    "constraint": "boldness_scores",
    "color_tone": "color_scores",
}

# Tunable defaults (env-overridable; the Streamlit UI can override per request).
DEFAULT_TEMPERATURE = float(os.environ.get("INTENT_TEMPERATURE", "2.0"))
DEFAULT_ALPHA = float(os.environ.get("RANK_ALPHA", "0.8"))   # target-match vs raw magnitude
DEFAULT_MMR = float(os.environ.get("RANK_MMR", "0.7"))       # 1.0 = no diversity re-rank


# ------------------------------------------------------------------
# Distribution helpers
# ------------------------------------------------------------------

def retemper(probs, temperature):
    """Re-soften/sharpen an existing probability vector. T > 1 softens, T < 1 sharpens."""
    t = max(float(temperature), 1e-6)
    powered = [max(float(p), 1e-12) ** (1.0 / t) for p in probs]
    s = sum(powered)
    if s <= 0:
        return [1.0 / len(probs)] * len(probs)
    return [p / s for p in powered]


def _l1_normalize(vec):
    s = sum(vec)
    if s <= 0:
        return [1.0 / len(vec)] * len(vec)
    return [v / s for v in vec]


def product_axis_vector(product, axis, reverse_maps):
    """A product's CLIP scores for one axis, as a distribution aligned to label order."""
    clip_map = AXIS_TO_CLIP_MAP[axis]
    raw = product.get(AXIS_TO_PRODUCT_KEY[axis], {}) or {}
    n = len(reverse_maps[axis])

    vec = []
    for i in range(n):
        label = reverse_maps[axis][i]
        clip_label = clip_map.get(label)
        vec.append(None if clip_label is None else float(raw.get(clip_label, 0.0)))

    known = [v for v in vec if v is not None]
    fill = (sum(known) / len(known)) if known else 0.0
    vec = [fill if v is None else v for v in vec]
    return _l1_normalize(vec)


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


# ------------------------------------------------------------------
# Scoring & ranking
# ------------------------------------------------------------------

def score_product(product, intent_scores, reverse_maps, alpha=DEFAULT_ALPHA):
    """
    Returns (final_score, per_axis_similarity).

    final_score = alpha * (weighted closeness to the target profile)
                + (1 - alpha) * (weighted raw strength on the user's top label)

    Both terms are in [0, 1], so final_score is too.
    """
    details = {}
    target_term = 0.0
    raw_term = 0.0

    for axis, w in WEIGHTS.items():
        tvec = intent_scores[axis]
        pvec = product_axis_vector(product, axis, reverse_maps)

        sim = cosine_sim(tvec, pvec)
        details[axis] = round(sim, 3)
        target_term += w * sim

        top_idx = max(range(len(tvec)), key=tvec.__getitem__)
        raw_term += w * pvec[top_idx]

    final = alpha * target_term + (1.0 - alpha) * raw_term
    return round(final, 4), details


def _full_vector(product, reverse_maps):
    v = []
    for axis in WEIGHTS:
        v.extend(product_axis_vector(product, axis, reverse_maps))
    return v


def rank_products(products, intent_scores, reverse_maps, top_k=5,
                  alpha=DEFAULT_ALPHA, mmr_lambda=DEFAULT_MMR):
    """Rank by closeness to the target profile, with an optional MMR diversity pass."""
    scored = []
    for p in products:
        s, d = score_product(p, intent_scores, reverse_maps, alpha=alpha)
        scored.append({
            **p,
            "match_score": s,
            "score_details": d,
            "_vec": _full_vector(p, reverse_maps),
        })
    scored.sort(key=lambda x: x["match_score"], reverse=True)

    if mmr_lambda >= 1.0 or len(scored) <= top_k:
        picked = scored[:top_k]
    else:
        pool = scored[:]
        picked = [pool.pop(0)]
        while pool and len(picked) < top_k:
            best_i, best_val = 0, float("-inf")
            for i, cand in enumerate(pool):
                diversity = max(cosine_sim(cand["_vec"], q["_vec"]) for q in picked)
                val = mmr_lambda * cand["match_score"] - (1.0 - mmr_lambda) * diversity
                if val > best_val:
                    best_val, best_i = val, i
            picked.append(pool.pop(best_i))

    for p in picked:
        p.pop("_vec", None)
    return picked


# ------------------------------------------------------------------
# Retrieval query text
# ------------------------------------------------------------------

def build_query_text(intent_scores, reverse_maps, prompt="", item_type=None):
    """
    Query text for ChromaDB: the user's own words first (so 'beach', 'colorful',
    'rooftop' actually influence retrieval), then the top-2 labels per axis
    repeated in proportion to their probability.
    """
    parts = []
    if prompt and prompt.strip():
        parts.append(prompt.strip())

    for axis in ("occasion", "formality"):
        dist = intent_scores[axis]
        # Skip an axis the model is basically undecided on — adding its labels
        # would just inject noise.
        if max(dist) - min(dist) < 0.10:
            continue
        order = sorted(range(len(dist)), key=lambda i: dist[i], reverse=True)[:2]
        for i in order:
            if dist[i] < 0.20:
                continue
            reps = max(1, int(round(dist[i] * 3)))
            parts += [reverse_maps[axis][i]] * reps

    if item_type:
        parts.append(item_type)
    parts.append("outfit")
    return " ".join(parts)
