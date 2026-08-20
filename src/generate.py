"""Instacart-shaped grocery orders with PLANTED complement and substitute structure.

The Instacart Online Grocery dataset is not downloadable in this offline
environment. What makes it uniquely good for this problem is not its size, it is
three structural properties, and all three are reproduced here:

  1. baskets, not sessions -- the unit of economics is the whole order
  2. reorder flags -- grocery is 60%+ repeat purchase, and a model that ignores
     that is not modelling grocery
  3. aisle / department structure -- the taxonomy substitution reasoning needs

On top of that, this generator PLANTS the thing the project is about:

  COMPLEMENTS  co-occur in the SAME basket        (hot dogs + buns)
  SUBSTITUTES  almost NEVER co-occur in a basket, but the same user alternates
               between them ACROSS baskets        (Coke vs Pepsi)

Both relations produce high aggregate association, which is exactly why naive
co-occurrence conflates them. Because the truth is planted, the conflation can be
MEASURED rather than asserted, and a substitution model can be scored against
something better than intuition.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

RNG = np.random.default_rng(90210)

# aisle -> list of product families. Each family holds mutually SUBSTITUTABLE
# products (same job, different brand/variant).
AISLES = {
    "soft_drinks": [["cola_a", "cola_b", "cola_c"], ["lemon_lime_a", "lemon_lime_b"]],
    "bread": [["hotdog_buns_a", "hotdog_buns_b"], ["sandwich_bread_a", "sandwich_bread_b",
                                                   "sandwich_bread_c"]],
    "meat": [["hot_dogs_a", "hot_dogs_b"], ["ground_beef_a", "ground_beef_b"],
             ["chicken_breast_a", "chicken_breast_b"]],
    "dairy": [["milk_whole_a", "milk_whole_b"], ["butter_a", "butter_b"],
              ["cheese_slices_a", "cheese_slices_b"]],
    "produce": [["bananas_a"], ["lettuce_a", "lettuce_b"], ["tomatoes_a"]],
    "condiments": [["ketchup_a", "ketchup_b"], ["mustard_a", "mustard_b"],
                   ["mayo_a", "mayo_b"]],
    "pasta": [["spaghetti_a", "spaghetti_b"], ["pasta_sauce_a", "pasta_sauce_b"]],
    "snacks": [["chips_a", "chips_b", "chips_c"], ["salsa_a", "salsa_b"]],
    "breakfast": [["cereal_a", "cereal_b"], ["coffee_a", "coffee_b"], ["eggs_a"]],
    "baking": [["flour_a"], ["sugar_a"], ["choc_chips_a"]],
}

# Themes are the COMPLEMENT structure: a shopper with an intent buys one product
# from several different families together.
THEMES = {
    "cookout":      [["hot_dogs"], ["hotdog_buns"], ["ketchup"], ["mustard"], ["chips"]],
    "pasta_night":  [["spaghetti"], ["pasta_sauce"], ["ground_beef"], ["cheese_slices"]],
    "breakfast":    [["cereal"], ["milk_whole"], ["coffee"], ["eggs"]],
    "sandwiches":   [["sandwich_bread"], ["cheese_slices"], ["mayo"], ["lettuce"],
                     ["tomatoes"]],
    "baking_day":   [["flour"], ["sugar"], ["butter"], ["choc_chips"], ["eggs"]],
    "snack_run":    [["chips"], ["salsa"], ["cola"], ["lemon_lime"]],
}

N_USERS = 3000
ORDERS_PER_USER = (4, 22)
BASKET_SIZE = (4, 14)


def build_catalogue():
    products, families, family_of, aisle_of = [], {}, {}, {}
    pid = 0
    for aisle, fams in AISLES.items():
        for fam in fams:
            fam_name = fam[0].rsplit("_", 1)[0]
            families[fam_name] = []
            for name in fam:
                # price and pack size are needed to rank substitutes the way a
                # shopper does -- a $3 sauce is not substitutable by an $11 one,
                # and a 500g bag is not swapped for a 2kg sack. Within a family
                # they VARY, which is what makes the penalty do work.
                base_price = float(RNG.uniform(1.5, 12.0))
                products.append(dict(product_id=pid, name=name, aisle=aisle,
                                     family=fam_name,
                                     price=round(base_price * float(RNG.uniform(0.75, 1.4)), 2),
                                     pack_size=float(RNG.choice([1, 1, 1, 2, 4, 6, 12]))))
                families[fam_name].append(pid)
                family_of[pid] = fam_name
                aisle_of[pid] = aisle
                pid += 1
    return products, families, family_of, aisle_of


def build(out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    products, families, family_of, aisle_of = build_catalogue()
    n_items = len(products)

    # Per-user BRAND LOYALTY within a family. This is what makes substitutes
    # substitutes: a user picks (mostly) one member of a family per basket, and
    # occasionally switches -- which is the behavioural evidence the weak
    # supervision downstream mines.
    loyalty = {}
    for u in range(N_USERS):
        loyalty[u] = {fam: int(RNG.choice(members))
                      for fam, members in families.items()}

    theme_names = list(THEMES)
    theme_pref = RNG.dirichlet(np.ones(len(theme_names)) * 0.7, size=N_USERS)

    # staple families a user buys regardless of theme -- the reorder backbone
    staples = {u: list(RNG.choice(list(families), size=int(RNG.integers(2, 6)),
                                  replace=False)) for u in range(N_USERS)}

    orders, order_products = [], []
    oid = 0
    user_prev_items = defaultdict(set)

    # Each user shops on their own cadence -- weekly, fortnightly, erratic. The
    # first pass had order_number only, so the models could rank WHAT a user
    # reorders and had nothing to say about WHEN, which is where the value is.
    user_cadence = {u: float(RNG.uniform(4.0, 16.0)) for u in range(N_USERS)}

    for u in range(N_USERS):
        n_orders = int(RNG.integers(*ORDERS_PER_USER))
        day = float(RNG.uniform(0, 30))
        for seq in range(n_orders):
            day += max(1.0, float(RNG.gamma(4.0, user_cadence[u] / 4.0)))
            theme = theme_names[int(RNG.choice(len(theme_names), p=theme_pref[u]))]
            target = int(RNG.integers(*BASKET_SIZE))
            fams_wanted = []

            for grp in THEMES[theme]:
                if RNG.random() < 0.82:          # theme adherence
                    fams_wanted.append(grp[0])
            for fam in staples[u]:
                if RNG.random() < 0.55:
                    fams_wanted.append(fam)
            all_fams = list(families)
            while len(fams_wanted) < target:
                fams_wanted.append(str(RNG.choice(all_fams)))
            fams_wanted = list(dict.fromkeys(fams_wanted))[:target]

            basket = []
            for fam in fams_wanted:
                # ONE product per family per basket. This is the structural fact
                # that makes substitutes anti-correlated WITHIN a basket: you buy
                # cola, or you buy the other cola, not both.
                if RNG.random() < 0.88:
                    item = loyalty[u][fam]                 # loyal choice
                else:
                    item = int(RNG.choice(families[fam]))  # switch
                    if RNG.random() < 0.45:
                        loyalty[u][fam] = item             # switch sticks
                basket.append(item)

            basket = list(dict.fromkeys(basket))
            prev = user_prev_items[u]
            orders.append(dict(order_id=oid, user_id=u, order_number=seq,
                               day=day, n_items=len(basket)))
            for pos, item in enumerate(basket):
                order_products.append(dict(order_id=oid, product_id=item,
                                           add_to_cart_order=pos + 1,
                                           reordered=int(item in prev)))
            user_prev_items[u].update(basket)
            oid += 1

    # ---- ground truth ----
    true_substitutes = set()
    for fam, members in families.items():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                true_substitutes.add((min(a, b), max(a, b)))

    true_complements = set()
    for theme, grps in THEMES.items():
        fams = [g[0] for g in grps]
        for i, fa in enumerate(fams):
            for fb in fams[i + 1:]:
                for a in families[fa]:
                    for b in families[fb]:
                        true_complements.add((min(a, b), max(a, b)))

    truth = dict(
        substitutes=[list(p) for p in sorted(true_substitutes)],
        complements=[list(p) for p in sorted(true_complements)])

    np.save(os.path.join(out_dir, "orders.npy"),
            np.array([(o["order_id"], o["user_id"], o["order_number"], o["day"])
                      for o in orders], dtype=np.float64))
    np.save(os.path.join(out_dir, "order_products.npy"),
            np.array([(r["order_id"], r["product_id"], r["add_to_cart_order"],
                       r["reordered"]) for r in order_products], dtype=np.int32))
    with open(os.path.join(out_dir, "products.json"), "w") as f:
        json.dump(products, f)
    with open(os.path.join(out_dir, "TRUTH.json"), "w") as f:
        json.dump(truth, f)

    reorder_rate = float(np.mean([r["reordered"] for r in order_products]))
    stats = dict(n_users=N_USERS, n_orders=len(orders), n_products=n_items,
                 n_lines=len(order_products),
                 mean_basket=round(len(order_products) / len(orders), 2),
                 reorder_rate=round(reorder_rate, 4),
                 n_true_substitute_pairs=len(true_substitutes),
                 n_true_complement_pairs=len(true_complements),
                 mean_days_between_orders=round(float(np.mean(
                     [user_cadence[u] for u in range(N_USERS)])), 2),
                 span_days=round(float(max(o["day"] for o in orders)), 1))
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(json.dumps(build(os.path.join(here, "data")), indent=2))
