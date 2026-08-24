"""Real Instacart baskets — the dataset this project said it could not have.

WHAT THIS PROJECT SAID, IN EVERY PASS
-------------------------------------
"No Instacart data. It is not downloadable here, and the planted ground truth is
what lets substitutes and complements be scored rather than eyeballed."

The first clause was false and took five passes to check. The competition's own
download endpoint returns 403 until the rules are accepted in a browser, but the
whole dataset has been republished as a plain Kaggle DATASET, and datasets carry
no rules gate.

PROVENANCE, STATED BECAUSE IT MATTERS
-------------------------------------
`psparks/instacart-market-basket-analysis`, a third-party republication of the
competition files rather than the official archive: 3.2M orders, 32M
order-product rows, 49,688 products across 134 aisles. It has not been diffed
against the official CSVs, because those are the thing that is gated.

THE TWO EXPERIMENTS THIS IS FOR
-------------------------------
1. ORDER. This project measured that add-to-cart order is worth +0.1229 hit@10
   against a control with the order destroyed. Instacart records
   `add_to_cart_order` for real, so the same experiment runs on real sequences.

2. THE SUBSTITUTE MECHANISM. This project's sharpest finding was that a
   presence-based switch matrix scores BELOW CHANCE, because

       "substitutes do not co-occur -- one product per family per basket, so on
        any given trip the item LEAST likely to be beside A is A's own
        substitute."

   In the generator that is true BY CONSTRUCTION: the generator puts one item per
   family in each basket. Whether it is true of real shoppers is a different
   question and this is the first chance to ask it. Instacart's 134 aisles are a
   real taxonomy, so same-aisle co-occurrence can be measured against chance.

   If real same-aisle pairs co-occur BELOW chance, the mechanism is real and the
   generator was reproducing something. If they co-occur ABOVE chance, the
   finding was an artifact of how the generator builds baskets and the conclusion
   drawn from it does not transfer.

WHAT INSTACART STILL CANNOT DO
------------------------------
It has no ground-truth substitute set. Aisles are a proxy for "similar", not a
label for "this shopper would swap these", so the precision/recall numbers
against planted pairs stay on the generator.
"""
from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(HERE, os.pardir, ".vendor", "kaggle")
ORDERS = os.path.join(VENDOR, "orders.csv")
PRIOR = os.path.join(VENDOR, "order_products__prior.csv")
PRODUCTS = os.path.join(VENDOR, "products.csv")
AISLES = os.path.join(VENDOR, "aisles.csv")


def available() -> bool:
    return all(os.path.exists(p) for p in (ORDERS, PRIOR, PRODUCTS))


def load_baskets(max_orders: int = 60_000, seed: int = 0,
                 min_len: int = 3) -> dict:
    """Baskets in real add-to-cart order, plus the aisle each product sits in.

    Sampled by ORDER, and the sample is drawn before the 32M-row product file is
    read so only the wanted rows are materialised.

    `add_to_cart_order` is the column the whole order experiment rests on, and it
    is sorted on explicitly rather than trusted to arrive sorted -- the file is
    grouped by order but nothing in the format guarantees the within-order
    sequence, and a silently mis-ordered basket would make the sequence model
    look exactly like its own control.
    """
    import pandas as pd

    orders = pd.read_csv(ORDERS, usecols=["order_id", "user_id", "eval_set"])
    prior = orders[orders.eval_set == "prior"]
    rng = np.random.default_rng(seed)
    take = min(max_orders, len(prior))
    keep_ids = set(rng.choice(prior.order_id.to_numpy(), size=take,
                              replace=False).tolist())
    user_of = dict(zip(prior.order_id, prior.user_id))

    rows = []
    for chunk in pd.read_csv(
            PRIOR, usecols=["order_id", "product_id", "add_to_cart_order"],
            chunksize=2_000_000):
        rows.append(chunk[chunk.order_id.isin(keep_ids)])
    op = pd.concat(rows, ignore_index=True)
    op = op.sort_values(["order_id", "add_to_cart_order"])

    prods = pd.read_csv(PRODUCTS, usecols=["product_id", "product_name",
                                           "aisle_id", "department_id"])
    aisle_of = dict(zip(prods.product_id, prods.aisle_id))
    name_of = dict(zip(prods.product_id, prods.product_name))

    # dense reindex: the models allocate an n_items x n_items matrix, and
    # Instacart's ids run to 49,688 whether or not the sample uses them.
    used = np.sort(op.product_id.unique())
    dense = {int(p): i for i, p in enumerate(used)}

    baskets, users = [], []
    for oid, grp in op.groupby("order_id", sort=False):
        b = [dense[int(p)] for p in grp.product_id]
        if len(b) >= min_len:
            baskets.append(b)
            users.append(int(user_of.get(oid, -1)))
    return dict(baskets=baskets, users=users,
                n_items=len(used),
                aisle=[int(aisle_of.get(int(p), -1)) for p in used],
                name=[str(name_of.get(int(p), "")) for p in used],
                product_ids=[int(p) for p in used])


def cooccurrence_vs_chance(data: dict, n_pairs: int = 200_000,
                           seed: int = 0, min_baskets: int = 25) -> dict:
    """Do same-aisle products co-occur MORE or LESS than chance?

    THE TEST OF THE GENERATOR'S MECHANISM. If real same-aisle pairs co-occur
    below chance, "substitutes do not co-occur" is a property of shopping and the
    generator was reproducing it. If above, it was an artifact of the generator
    putting one item per family in each basket.

    Chance is the product of the two items' basket frequencies -- what would be
    expected if membership were independent -- rather than a shuffle, so the
    comparison does not inherit whatever a shuffle happens to preserve.
    """
    baskets = data["baskets"]
    n_items = data["n_items"]
    aisle = np.asarray(data["aisle"])
    freq = np.zeros(n_items)
    for b in baskets:
        freq[list(set(b))] += 1
    freq /= max(len(baskets), 1)

    from collections import Counter
    co = Counter()
    for b in baskets:
        u = sorted(set(b))
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                co[(u[i], u[j])] += 1

    # Restricted to items appearing in at least `min_baskets` baskets. Among
    # 27,440 products most random pairs never co-occur at all, and a mean lift
    # dominated by zeros measures how rare the catalogue is rather than whether
    # same-aisle items avoid each other. The floor is stated rather than tuned:
    # the comparison is between two groups drawn from the SAME candidate pool.
    rng = np.random.default_rng(seed)
    same, diff = [], []
    cand = np.flatnonzero(freq * max(len(baskets), 1) >= min_baskets)
    for _ in range(n_pairs):
        a, b = rng.choice(cand, size=2, replace=False)
        a, b = (int(a), int(b)) if a < b else (int(b), int(a))
        exp = freq[a] * freq[b]
        if exp <= 0:
            continue
        obs = co.get((a, b), 0) / max(len(baskets), 1)
        lift = obs / exp
        (same if aisle[a] == aisle[b] and aisle[a] >= 0 else diff).append(lift)
    return dict(candidates=int(len(cand)), min_baskets=min_baskets,
                same_aisle_pairs=len(same), other_pairs=len(diff),
                same_aisle_mean_lift=float(np.mean(same)) if same else float("nan"),
                other_mean_lift=float(np.mean(diff)) if diff else float("nan"),
                same_aisle_median_lift=float(np.median(same)) if same else float("nan"),
                other_median_lift=float(np.median(diff)) if diff else float("nan"))


def stats(data: dict) -> dict:
    lens = [len(b) for b in data["baskets"]]
    return dict(baskets=len(data["baskets"]), items=data["n_items"],
                mean_basket=float(np.mean(lens)),
                median_basket=float(np.median(lens)),
                aisles=int(len(set(a for a in data["aisle"] if a >= 0))))


def variant_cooccurrence(data: dict, seed: int = 0, min_baskets: int = 25,
                         jaccard: float = 0.6, n_cross: int = 400_000) -> dict:
    """The mechanism test at the granularity the claim is actually about.

    An Instacart AISLE is not the generator's FAMILY. "Yogurt" holds hundreds of
    products a shopper cheerfully buys four of at once; the generator's family is
    a set of near-identical variants of which a shopper takes one. Measuring
    same-aisle co-occurrence and calling it a test of "substitutes do not
    co-occur" would be answering a question nobody asked.

    A VARIANT PAIR is two products in the same aisle whose names share most of
    their words -- "Chobani Greek Yogurt Strawberry" against "Chobani Greek
    Yogurt Blueberry". That is a rough proxy: it catches different sizes of the
    same thing and misses substitutes that are branded differently. It is stated
    rather than smoothed over, because the conclusion depends on it.

    Same-aisle pairs are ENUMERATED EXHAUSTIVELY rather than sampled. Random
    pairs among 2,832 candidates find a variant pair roughly once in 4,000 draws,
    and the first version of this reported a headline on 161 of them. Cross-aisle
    pairs are still sampled -- there are ~4 million and they are the easy group.
    """
    from collections import Counter

    baskets = data["baskets"]
    n_items = data["n_items"]
    aisle = np.asarray(data["aisle"])
    names = [str(x).lower() for x in data["name"]]
    toks = [set(n.replace("&", " ").split()) for n in names]

    freq = np.zeros(n_items)
    for b in baskets:
        freq[list(set(b))] += 1
    nb = max(len(baskets), 1)
    freq /= nb

    co = Counter()
    for b in baskets:
        u = sorted(set(b))
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                co[(u[i], u[j])] += 1

    cand = np.flatnonzero(freq * nb >= min_baskets)
    by_aisle: dict[int, list[int]] = {}
    for i in cand:
        by_aisle.setdefault(int(aisle[i]), []).append(int(i))

    def lift(a, b):
        exp = freq[a] * freq[b]
        return (co.get((a, b), 0) / nb) / exp if exp > 0 else None

    groups = {"variant": [], "same_aisle": [], "cross_aisle": []}
    for ai, members in by_aisle.items():
        if ai < 0:
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                a, b = sorted((members[x], members[y]))
                v = lift(a, b)
                if v is None:
                    continue
                ta, tb = toks[a], toks[b]
                j = len(ta & tb) / max(len(ta | tb), 1)
                groups["variant" if j >= jaccard else "same_aisle"].append(v)

    rng = np.random.default_rng(seed)
    for _ in range(n_cross):
        a, b = rng.choice(cand, size=2, replace=False)
        a, b = (int(a), int(b)) if a < b else (int(b), int(a))
        if aisle[a] >= 0 and aisle[a] == aisle[b]:
            continue
        v = lift(a, b)
        if v is not None:
            groups["cross_aisle"].append(v)

    out = dict(min_baskets=min_baskets, jaccard=jaccard,
               candidates=int(len(cand)))
    for g, v in groups.items():
        out[g] = dict(n=len(v),
                      mean_lift=float(np.mean(v)) if v else float("nan"),
                      median_lift=float(np.median(v)) if v else float("nan"),
                      share_cooccurring=float(np.mean([x > 0 for x in v]))
                      if v else float("nan"))
    return out
