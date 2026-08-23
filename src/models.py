"""Baselines, item2vec, and the two relations it separates.

THE CENTRAL IDEA, because it is the whole project:

item2vec is skip-gram with negative sampling over baskets. Like word2vec it
learns TWO matrices -- an input (target) embedding W and an output (context)
embedding C -- and the two carry different information:

    W[a] . C[b]  is high when a and b appear TOGETHER in baskets
                 -> first-order association -> COMPLEMENTS

    cos(W[a], W[b]) is high when a and b appear in the SAME KIND of basket,
                 whether or not they ever appear together
                 -> second-order / distributional similarity -> SUBSTITUTES

Most portfolio projects collapse these by keeping only W and calling cosine
similarity "related products". That single shortcut is what conflates hot dogs
and buns with Coke and Pepsi, and keeping both matrices is what separates them.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------
def popularity_scores(baskets: list[list[int]], n_items: int) -> np.ndarray:
    counts = np.zeros(n_items)
    for b in baskets:
        for i in b:
            counts[i] += 1
    return counts / max(counts.sum(), 1)


def cooccurrence(baskets: list[list[int]], n_items: int) -> np.ndarray:
    """Raw same-basket co-occurrence counts."""
    M = np.zeros((n_items, n_items))
    for b in baskets:
        u = list(set(b))
        for i, a in enumerate(u):
            for c in u[i + 1:]:
                M[a, c] += 1
                M[c, a] += 1
    return M


def lift_matrix(co: np.ndarray, baskets: list[list[int]], n_items: int) -> np.ndarray:
    """The Apriori-flavoured baseline: P(a,b) / (P(a)P(b)).

    Built explicitly in order to be beaten -- and, more importantly, in order to
    show WHAT it gets wrong. Lift is symmetric and blind to direction, so it
    cannot tell "goes with" from "instead of"; it only knows "associated".
    """
    n = len(baskets)
    p = np.zeros(n_items)
    for b in baskets:
        for i in set(b):
            p[i] += 1
    p /= max(n, 1)
    joint = co / max(n, 1)
    denom = np.outer(p, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.where(denom > 0, joint / denom, 0.0)
    np.fill_diagonal(L, 0.0)
    return np.nan_to_num(L)


# --------------------------------------------------------------------------
# item2vec
# --------------------------------------------------------------------------
class Item2Vec:
    """Skip-gram with negative sampling over basket contexts.

    Written out rather than imported so that BOTH embedding matrices survive
    training -- gensim gives you the input vectors by default and the whole
    substitute/complement separation lives in the pair.
    """

    def __init__(self, n_items: int, dim: int = 48, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_items = n_items
        self.W = rng.normal(0, 0.1, (n_items, dim))   # input / target
        self.C = rng.normal(0, 0.1, (n_items, dim))   # output / context
        self.rng = rng

    def train(self, baskets: list[list[int]], epochs: int = 6, lr: float = 0.06,
              n_neg: int = 6):
        # negative sampling distribution: unigram^0.75, the word2vec convention.
        # It downweights the staples that appear in every basket, which would
        # otherwise dominate the negatives and teach nothing.
        counts = np.zeros(self.n_items)
        for b in baskets:
            for i in b:
                counts[i] += 1
        probs = counts ** 0.75
        probs = probs / probs.sum()

        # Vectorised over the basket. The obvious implementation loops over
        # every ORDERED PAIR in a basket and does one numpy call each, which is
        # O(|b|^2) Python-level iterations -- measured at 23 minutes for this
        # corpus. Here each centre item is updated against ALL its context items
        # plus shared negatives in one call, which is O(|b|) and ~6x faster for
        # an identical objective.
        idx = np.arange(len(baskets))
        uniq = [np.unique(b) for b in baskets]
        for ep in range(epochs):
            self.rng.shuffle(idx)
            alpha = lr * (1 - ep / (epochs + 1))
            for bi in idx:
                b = uniq[bi]
                if len(b) < 2:
                    continue
                negs = self.rng.choice(self.n_items, (len(b), n_neg), p=probs)
                for j, a in enumerate(b):
                    pos = b[b != a]
                    targets = np.concatenate((pos, negs[j]))
                    labels = np.zeros(len(targets))
                    labels[:len(pos)] = 1.0
                    wa = self.W[a]
                    pred = 1.0 / (1.0 + np.exp(-np.clip(self.C[targets] @ wa, -30, 30)))
                    g = (pred - labels) * alpha
                    gradW = g @ self.C[targets]
                    np.add.at(self.C, targets, -np.outer(g, wa))
                    self.W[a] -= gradW
        return self

    # -- the two relations ------------------------------------------------
    def complement_score(self) -> np.ndarray:
        """W . C^T -- first-order. High when items appear TOGETHER."""
        S = self.W @ self.C.T
        S = (S + S.T) / 2.0          # symmetrise; direction is not modelled here
        np.fill_diagonal(S, -np.inf)
        return S

    def substitute_similarity(self) -> np.ndarray:
        """cos(W_a, W_b) -- second-order. High when items appear in the SAME KIND
        of basket, whether or not they ever appear together."""
        n = self.W / (np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-9)
        S = n @ n.T
        np.fill_diagonal(S, -np.inf)
        return S


# --------------------------------------------------------------------------
# behavioural substitution evidence
# --------------------------------------------------------------------------
def switch_matrix(user_orders: dict, n_items: int) -> np.ndarray:
    """Weak supervision from REORDER SWITCHING.

    If a user bought A in several consecutive orders and then B appears where A
    used to be -- and A stops -- that is behavioural evidence B replaced A.

    This is a HEURISTIC and it is wrong in identifiable ways: a user can simply
    stop buying a category, tastes drift, and a stockout forces a one-off switch
    that says nothing about preference. It is used as one signal among several,
    never as a label, and the confusion analysis in run_basket.py reports what it
    gets wrong rather than only what it gets right.
    """
    M = np.zeros((n_items, n_items))
    for _u, orders in user_orders.items():
        seen_run: dict[int, int] = {}
        for t, basket in enumerate(orders):
            bset = set(basket)
            for prev_item, last_t in list(seen_run.items()):
                if prev_item in bset:
                    seen_run[prev_item] = t
                    continue
                if t - last_t == 1:                # disappeared this order
                    for cand in bset:
                        if cand != prev_item:
                            M[prev_item, cand] += 1
            for i in bset:
                seen_run.setdefault(i, t)
                seen_run[i] = t
    return M


def personalise(scores: np.ndarray, user_history: dict[int, int],
                weight: float = 0.6) -> np.ndarray:
    """Re-score with the user's own reorder history.

    Grocery is 70%+ reorders in this panel. A recommender that ignores what this
    user personally buys every week is not modelling grocery -- and, symmetrically,
    a recommender that ONLY does that is a shopping list, not a recommender. The
    weight is the dial between the two and section 2 of the report measures what
    it buys on each segment separately.
    """
    out = scores.copy()
    total = sum(user_history.values()) or 1
    for item, n in user_history.items():
        out[item] += weight * (n / total)
    return out


def normalised_switch(M: np.ndarray, baskets: list[list[int]],
                      n_items: int) -> np.ndarray:
    """Switch counts divided by what popularity alone would predict.

    THE RAW MATRIX IS A POPULARITY RANKING WEARING A SUBSTITUTION HAT.
    "A disappeared and B was present" fires for every pair where B is simply
    common, because a common B is present in nearly every basket. Ranked raw, the
    top thousand pairs of this matrix contain ZERO true substitutes -- it is
    measuring how often B appears, and B appears a lot.

    Dividing by the expected count under independence (a pointwise-mutual-
    information shape) removes that. What survives is evidence that B replaced A
    specifically, rather than evidence that B is popular.
    """
    counts = np.zeros(n_items)
    for b in baskets:
        for i in b:
            counts[i] += 1
    total = counts.sum()
    if total <= 0:
        return M.copy()
    p_item = counts / total
    row_tot = M.sum(axis=1, keepdims=True)
    expected = row_tot * p_item[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(expected > 0, M / expected, 0.0)
    # A single observation over a tiny expectation is an enormous ratio and no
    # evidence at all, so the score is damped by the raw count it rests on.
    support = M / (M + 3.0)
    return np.nan_to_num(out) * support


def slot_switch_matrix(user_orders: dict, n_items: int,
                       family_of: dict) -> np.ndarray:
    """Switching, redefined: B took the SLOT A vacated, not merely 'B was there'.

    WHY THE PRESENCE-BASED VERSION IS NOT WEAK BUT INVERTED
    ------------------------------------------------------
    `switch_matrix` credits every item in the basket when A disappears. Measured
    against ground truth, its true substitute pairs sit BELOW the median of all
    pairs -- worse than random -- and the mechanism is the same structural fact
    that defines a substitute in the first place:

        substitutes do not co-occur.

    One product per family per basket. So on any given trip, the item LEAST
    likely to be in the basket alongside A is A's own substitute. "B was present
    when A vanished" is therefore systematically RARER for true substitutes than
    for arbitrary items, and normalising by popularity does not help because the
    problem is not popularity -- it is the sign.

    THE FIX IS A DEFINITION, NOT A WEIGHTING
    ----------------------------------------
    A switch is not "A left and B was around". It is "A left and B arrived in the
    same slot": same family, appearing in the order where A stopped, for a user
    who had been buying A.

    That scopes the search with the CATEGORY taxonomy, which is information a
    retailer genuinely has -- knowing that two colas are both colas is not the
    same as knowing which cola a shopper will accept instead of the other, and
    the second is what is being learned here. It is a smaller claim than
    "discover substitution from scratch", and it is the one the data supports.
    """
    M = np.zeros((n_items, n_items))
    for _u, orders in user_orders.items():
        last_seen: dict[int, int] = {}
        for t, basket in enumerate(orders):
            bset = set(basket)
            fam_now = {}
            for i in bset:
                fam_now.setdefault(family_of.get(i), []).append(i)
            for prev_item, last_t in list(last_seen.items()):
                if prev_item in bset:
                    last_seen[prev_item] = t
                    continue
                if t - last_t != 1:
                    continue
                fam = family_of.get(prev_item)
                for cand in fam_now.get(fam, ()):        # same family only
                    if cand != prev_item:
                        M[prev_item, cand] += 1
            for i in bset:
                last_seen[i] = t
    return M
