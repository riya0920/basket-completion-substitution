"""A sequential model over within-basket order -- the gap item2vec cannot see.

WHAT THE PREVIOUS PASS SAID
---------------------------
"No sequential model. item2vec still ignores within-basket order; timing is now
modelled but as a per-item hazard, not a sequence."

That was true twice over: the model ignored order, and the DATA had no order to
ignore -- baskets were emitted in whatever sequence the generator happened to
build them. The generator now emits a store walk (trip starter first, then aisle
order with jitter), so "does sequence carry signal" is finally a question with an
answer rather than a hypothesis.

WHY SEQUENCE MATTERS COMMERCIALLY
---------------------------------
The cart-completion endpoint fires on a PARTIAL basket. What is already in the
cart is not an unordered set to the shopper -- the last thing they added is the
freshest evidence of what they are doing right now. A shopper who has just put
pasta sauce in the cart is in a different state from one who put it in six items
ago and has since moved to cleaning products.

item2vec, by construction, cannot represent that: it shuffles the context window
and learns a symmetric co-occurrence geometry. That is a strength for finding
substitutes and a blind spot for predicting the next add.

THE MODEL
---------
A first-order Markov chain over add-to-cart transitions, backed off to the
overall item popularity, with the backoff weight set by how much evidence the
transition row actually has. Deliberately not a neural sequence model:

  * the comparison of interest is ORDER vs NO ORDER, and a Markov chain isolates
    exactly that -- a GRU would change the model class at the same time and the
    delta could not be attributed;
  * it is inspectable. `P(next = buns | last = hot dogs)` is a number a
    merchandiser can be shown and can argue with.

Its weakness is stated rather than discovered: first-order means it forgets
everything before the last item, so a basket that is clearly a cookout is
represented only by whatever went in most recently.
"""
from __future__ import annotations

import numpy as np


class SequenceModel:
    """First-order transitions over add-to-cart order, with popularity backoff."""

    def __init__(self, n_items: int, backoff_k: float = 5.0):
        self.n_items = n_items
        self.backoff_k = backoff_k
        self.trans = np.zeros((n_items, n_items), dtype=np.float32)
        self.pop = np.zeros(n_items, dtype=np.float32)

    def fit(self, ordered_baskets: list[list[int]]):
        """`ordered_baskets` must be in ADD-TO-CART ORDER.

        There is no way for this class to check that, which is precisely why it
        is stated here and asserted in the tests: handed shuffled baskets it will
        train happily and learn nothing, and the failure looks like "sequence
        does not help" rather than like a bug.
        """
        for b in ordered_baskets:
            for it in b:
                self.pop[it] += 1
            for a, c in zip(b[:-1], b[1:]):
                self.trans[a, c] += 1
        tot = self.pop.sum()
        self.pop = self.pop / tot if tot > 0 else self.pop
        return self

    def next_scores(self, last_item: int | None) -> np.ndarray:
        """P(next | last), shrunk toward popularity by the evidence available.

        weight = n / (n + k): a transition row with 200 observations is trusted
        almost entirely, one with 3 is almost entirely popularity. Without the
        backoff the tail of the catalogue -- which is most of it -- returns
        whatever single transition it happened to see.
        """
        if last_item is None:
            return self.pop.copy()
        row = self.trans[last_item]
        n = float(row.sum())
        if n <= 0:
            return self.pop.copy()
        w = n / (n + self.backoff_k)
        return w * (row / n) + (1 - w) * self.pop

    def score_basket(self, basket_in_order: list[int]) -> np.ndarray:
        """Score candidates given a partial basket, using only the LAST item.

        Using only the last item is the model's defining limitation and it is
        left visible rather than patched with an average over the basket --
        averaging would quietly turn this into a bag-of-items model and the
        comparison against item2vec would stop meaning anything.
        """
        last = basket_in_order[-1] if basket_in_order else None
        s = self.next_scores(last)
        out = s.copy()
        out[list(basket_in_order)] = -np.inf      # never recommend what is in the cart
        return out


class BagOfItemsControl:
    """The same counts with the ORDER DESTROYED -- the control for the experiment.

    Trained on shuffled copies of the same baskets, so it sees identical
    co-occurrence and identical popularity and differs from `SequenceModel` in
    exactly one respect: it cannot know which item came last. Any gap between the
    two is attributable to order and to nothing else.
    """

    def __init__(self, n_items: int, backoff_k: float = 5.0, seed: int = 0):
        self.inner = SequenceModel(n_items, backoff_k)
        self.rng = np.random.default_rng(seed)

    def fit(self, ordered_baskets: list[list[int]]):
        shuffled = []
        for b in ordered_baskets:
            c = list(b)
            self.rng.shuffle(c)
            shuffled.append(c)
        self.inner.fit(shuffled)
        return self

    def score_basket(self, basket_in_order: list[int]) -> np.ndarray:
        c = list(basket_in_order)
        self.rng.shuffle(c)
        return self.inner.score_basket(c)


def evaluate_next_item(model, baskets: list[list[int]], k: int = 10,
                       min_prefix: int = 2) -> dict:
    """Hide the LAST item of each basket and try to recover it.

    This is the honest formulation of cart completion: the prefix is what the
    shopper has already added and the held-out item is what they add next. Note
    it is a harder task than the basket-completion evaluation elsewhere in this
    project, which hides a RANDOM item -- hiding the last one removes the
    easiest-to-guess member of the basket about half the time.
    """
    hits, mrr, n = 0, 0.0, 0
    for b in baskets:
        if len(b) < min_prefix + 1:
            continue
        prefix, target = b[:-1], b[-1]
        scores = model.score_basket(prefix)
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        n += 1
        if target in top:
            hits += 1
            mrr += 1.0 / (1 + int(np.where(top == target)[0][0]))
    return dict(hit_rate=hits / max(n, 1), mrr=mrr / max(n, 1), n=n)
