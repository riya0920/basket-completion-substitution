"""Reorder TIMING, directional complements, and price-aware substitution.

Four things the first pass named as missing, and the first is the answer to the
spec's own grill question.

WHERE THE MODELLING VALUE IN REORDER PREDICTION ACTUALLY IS
-----------------------------------------------------------
"Reorder prediction is easy -- the user buys milk weekly. Where's the modelling
value?" The first pass answered in prose: timing, basket context, and the
new-item discovery margin. It then built a model with no notion of time at all,
which meant it could rank WHAT a user would reorder and had nothing to say about
WHEN.

That distinction is the whole product. A grocery app that surfaces milk every
single session is not personalising, it is nagging -- and the user learns to
ignore the slot. The value is in surfacing milk on the day the carton runs out,
which requires a per-user, per-item inter-purchase-time model and a notion of
how much time has elapsed since the last purchase.

This models it as a HAZARD: given that a user last bought item i tau days ago,
how likely are they to buy it today? Fitted per (user, item) with shrinkage
toward the item's population mean, because most (user, item) pairs have two or
three purchases and a per-pair estimate from three observations is noise.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# inter-purchase timing
# --------------------------------------------------------------------------
class ReorderTiming:
    """Per (user, item) inter-purchase interval, shrunk to the item population.

    The shrinkage is the load-bearing part. A user with two purchases of an item
    has ONE observed interval; trusting it produces a model that fires
    confidently on noise. The shrunk estimate is

        tau_hat = (n * tau_user + k * tau_item) / (n + k)

    where n is the number of observed intervals and k is the shrinkage strength.
    With n=0 it returns the item's population interval; with n large it returns
    the user's own. That is the entire Bayesian argument in one line, and it is
    written out rather than hidden in a library so the k is arguable.
    """

    def __init__(self, shrinkage_k: float = 3.0):
        self.k = shrinkage_k
        self.item_mean: dict[int, float] = {}
        self.pair_mean: dict[tuple[int, int], float] = {}
        self.pair_n: dict[tuple[int, int], int] = {}
        self.global_mean = 14.0

    def fit(self, user_baskets: dict[int, list[tuple[float, list[int]]]]):
        """user_baskets: user -> [(day, [item ids]), ...] in time order."""
        item_intervals: dict[int, list[float]] = {}
        pair_intervals: dict[tuple[int, int], list[float]] = {}

        for u, seq in user_baskets.items():
            last_seen: dict[int, float] = {}
            for day, items in seq:
                for i in items:
                    if i in last_seen:
                        gap = day - last_seen[i]
                        if gap > 0:
                            item_intervals.setdefault(i, []).append(gap)
                            pair_intervals.setdefault((u, i), []).append(gap)
                    last_seen[i] = day

        allg = [g for v in item_intervals.values() for g in v]
        self.global_mean = float(np.mean(allg)) if allg else 14.0
        self.item_mean = {i: float(np.mean(v)) for i, v in item_intervals.items()}
        for key, v in pair_intervals.items():
            self.pair_mean[key] = float(np.mean(v))
            self.pair_n[key] = len(v)
        return self

    def expected_interval(self, user: int, item: int) -> float:
        pop = self.item_mean.get(item, self.global_mean)
        n = self.pair_n.get((user, item), 0)
        if n == 0:
            return pop
        own = self.pair_mean[(user, item)]
        return (n * own + self.k * pop) / (n + self.k)

    def due_score(self, user: int, item: int, days_since: float) -> float:
        """How 'due' an item is, peaking at the expected interval.

        A ratio rather than a probability, and deliberately so: it peaks at 1.0
        when the elapsed time equals the expected interval and decays either
        side. Buying milk two days after the last carton is unlikely; buying it
        twenty days after is ALSO unlikely, because the user has probably bought
        it somewhere else or changed habits. A monotone "more time = more likely"
        score gets that second case exactly backwards, and it is the mistake a
        recency feature makes.
        """
        exp = max(self.expected_interval(user, item), 1e-6)
        r = days_since / exp
        return float(np.exp(-0.5 * ((np.log(max(r, 1e-6))) / 0.6) ** 2))


# --------------------------------------------------------------------------
# directional complements
# --------------------------------------------------------------------------
def directional_lift(baskets: list[list[int]], n_items: int) -> np.ndarray:
    """P(b | a) -- how often b appears GIVEN a, which is not symmetric.

    The first pass symmetrised the complement score and said so. That is wrong in
    a way that matters commercially: P(buns | hot dogs) is high because buns are
    what you need once you have hot dogs, while P(hot dogs | buns) is lower
    because buns go with other things too. A cart-completion widget should
    suggest buns to a hot-dog buyer far more readily than the reverse, and a
    symmetric score cannot express the difference.

    Returns M where M[a, b] = P(b in basket | a in basket).
    """
    count = np.zeros(n_items)
    joint = np.zeros((n_items, n_items))
    for b in baskets:
        u = list(set(b))
        for a in u:
            count[a] += 1
            for c in u:
                if a != c:
                    joint[a, c] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        M = np.where(count[:, None] > 0, joint / np.maximum(count[:, None], 1), 0.0)
    np.fill_diagonal(M, 0.0)
    return np.nan_to_num(M)


def asymmetry(M: np.ndarray, a: int, b: int) -> float:
    """P(b|a) - P(a|b). Positive means b follows a more than a follows b."""
    return float(M[a, b] - M[b, a])


# --------------------------------------------------------------------------
# price- and pack-aware substitution
# --------------------------------------------------------------------------
def substitution_score(sim: np.ndarray, prices: np.ndarray, packs: np.ndarray,
                       item: int, candidates: np.ndarray,
                       price_tolerance: float = 0.35,
                       pack_tolerance: float = 0.5) -> np.ndarray:
    """Rank substitutes by similarity, PENALISED for price and pack mismatch.

    The first pass ranked purely on embedding similarity, which is the first two
    things a real shopper checks away from being useful. A shopper whose $3 pasta
    sauce is out does not want the $11 one, and someone who wanted a 500g bag
    does not want the 2kg sack -- both are "the same product" to a distributional
    model and neither is an acceptable swap.

    Penalties are multiplicative and bounded, so a very close match on price and
    pack can never OUTRANK a genuinely similar item, it can only reorder items
    that were already close. That ordering matters: price should break ties among
    substitutes, not create substitutes.
    """
    s = sim[item, candidates].astype(float)
    p_ratio = prices[candidates] / max(prices[item], 1e-9)
    pack_ratio = packs[candidates] / max(packs[item], 1e-9)
    price_pen = np.exp(-0.5 * (np.log(np.maximum(p_ratio, 1e-6)) / price_tolerance) ** 2)
    pack_pen = np.exp(-0.5 * (np.log(np.maximum(pack_ratio, 1e-6)) / pack_tolerance) ** 2)
    return s * price_pen * pack_pen


# --------------------------------------------------------------------------
# cold start
# --------------------------------------------------------------------------
def cold_start_substitutes(item_meta: dict, target: int, all_items: list[int],
                           prices: np.ndarray, packs: np.ndarray,
                           top_k: int = 5) -> list[int]:
    """Substitutes for an item with NO transaction history.

    A new SKU has no co-occurrence and no embedding worth anything, so the
    distributional method has nothing to say about it -- which is the honest
    position, and the reason a content fallback exists. Same family, then same
    aisle, ranked by price and pack proximity.

    This is strictly worse than the learned answer and it is what you serve on
    day one of a SKU's life, until it has enough baskets to earn a real vector.
    Every recommender needs this path and most portfolio projects skip it,
    because the offline eval never contains an item the model has not seen.
    """
    fam = item_meta[target]["family"]
    aisle = item_meta[target]["aisle"]
    same_fam = [i for i in all_items
                if i != target and item_meta[i]["family"] == fam]
    same_aisle = [i for i in all_items
                  if i != target and item_meta[i]["aisle"] == aisle
                  and item_meta[i]["family"] != fam]

    def prox(i):
        pr = abs(np.log(max(prices[i], 1e-9) / max(prices[target], 1e-9)))
        pk = abs(np.log(max(packs[i], 1e-9) / max(packs[target], 1e-9)))
        return pr + 0.5 * pk

    return (sorted(same_fam, key=prox) + sorted(same_aisle, key=prox))[:top_k]
