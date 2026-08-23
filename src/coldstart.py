"""Cold start with real text embeddings, and a fair test of what they buy.

WHAT THE PREVIOUS PASS SAID
---------------------------
"Cold start is content-only -- no vendor metadata, no image or text embedding,
no borrowing a vector from the family centroid."

All three of those are here now, and the point of the section is not that
embeddings are better. It is that **cold start is the only part of a recommender
that can be evaluated honestly without an A/B test**, because the counterfactual
is available: hold a product out of training entirely, then ask each method to
place it. A product the model has never seen is a product whose recommendations
you can score without any deployment.

THE THREE FALLBACKS, IN INCREASING ORDER OF WHAT THEY ASSUME
------------------------------------------------------------
  family centroid  -- borrow the mean learned vector of the product's family.
                      Assumes someone has correctly assigned the family, which
                      in a real catalogue is a merchandising task that is often
                      wrong and always late.
  text embedding   -- encode the product's own description. Assumes only that
                      somebody wrote a description, which is true on day one
                      because it is what goes on the page.
  content rules    -- same family, then same aisle, ranked by price and pack
                      proximity. Assumes the taxonomy and nothing else.

The comparison that matters is against the LEARNED vector the product would have
had with full history. That is the ceiling, and every cold-start method is scored
as a fraction of it rather than against each other, because "better than the
other fallback" is not the question a launch team is asking.
"""
from __future__ import annotations

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_CACHE: dict = {}


def load(name: str = MODEL_NAME):
    if name in _CACHE:
        return _CACHE[name]
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(name)
    except Exception:
        model = None
    _CACHE[name] = model
    return model


def embed_texts(texts: list[str], name: str = MODEL_NAME):
    """L2-normalised, so a dot product is a cosine."""
    model = load(name)
    if model is None:
        return None
    v = model.encode(list(texts), convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(v, np.float32)


def family_centroid(vectors: np.ndarray, members: list[int],
                    exclude: int | None = None) -> np.ndarray:
    """Mean learned vector of a family, optionally excluding the cold product.

    Excluding it is not a detail: leaving the held-out product in its own
    centroid is the cold-start equivalent of training on the test set, and it
    would make this method look perfect for a reason that cannot happen on day
    one.
    """
    idx = [m for m in members if m != exclude]
    if not idx:
        return np.zeros(vectors.shape[1], np.float32)
    v = vectors[idx].mean(axis=0)
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


def rank_by_vector(query_vec: np.ndarray, vectors: np.ndarray,
                   exclude: set[int] | None = None, k: int = 10) -> list[int]:
    s = vectors @ query_vec
    if exclude:
        s[list(exclude)] = -np.inf
    top = np.argpartition(-s, min(k, len(s) - 1))[:k]
    return [int(i) for i in top[np.argsort(-s[top])]]


def content_rules(item_meta: dict, target: int, all_items: list[int],
                  k: int = 10) -> list[int]:
    """Same family first, then same aisle, ranked by price and pack proximity.

    Strictly worse than the learned answer and it is what you serve on day one
    of a SKU's life. Every recommender needs this path; most portfolio projects
    skip it, because an offline evaluation never contains an item the model has
    not seen.
    """
    t = item_meta[target]
    same_fam, same_aisle = [], []
    for i in all_items:
        if i == target:
            continue
        m = item_meta[i]
        if m["family"] == t["family"]:
            same_fam.append(i)
        elif m["aisle"] == t["aisle"]:
            same_aisle.append(i)

    def prox(i):
        m = item_meta[i]
        pr = abs(np.log(max(m["price"], 0.01) / max(t["price"], 0.01)))
        pk = abs(np.log(max(m["pack_size"], 0.5) / max(t["pack_size"], 0.5)))
        return pr + 0.5 * pk

    same_fam.sort(key=prox)
    same_aisle.sort(key=prox)
    out = same_fam + same_aisle

    # THIRD TIER, and it exists because the first two can come up empty.
    # A product whose family has one member and whose aisle is small returns
    # almost nothing from the rules above -- and "almost nothing" on day one is a
    # blank recommendation slot on a live product page. The last resort is the
    # nearest products by price and pack across the whole catalogue: a weak
    # answer, and weaker than an empty one is not.
    if len(out) < k:
        rest = [i for i in all_items if i != target and i not in set(out)]
        rest.sort(key=prox)
        out = out + rest[:k - len(out)]
    return out[:k]


def overlap_at_k(a: list[int], b: list[int], k: int = 10) -> float:
    """Set overlap with the full-history answer -- the ceiling every fallback is
    scored against."""
    return len(set(a[:k]) & set(b[:k])) / float(k)
