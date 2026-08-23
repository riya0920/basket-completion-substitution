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

# --------------------------------------------------------------------------
# SCALE
#
# The hand-authored skeleton above is 49 products, which was enough to
# demonstrate the complement/substitute mechanism and far too few to say
# anything about catalogue-scale retrieval -- the previous README said exactly
# that. The skeleton is kept because it carries the MEANING (which families
# substitute, which co-occur in a theme) and is extended programmatically:
#
#   * each family gets more brand variants, so substitution sets are realistic
#     sizes rather than pairs;
#   * each aisle gets filler families that belong to no theme, which is the
#     honest majority of a real grocery catalogue and the thing that makes
#     retrieval hard -- a model that only ever sees themed items has never had
#     to ignore anything.
#
# Filler items are NOT noise. They participate in baskets at a low rate, so they
# generate exactly the weak, ambiguous co-occurrence that a complement model has
# to learn not to trust.
BRANDS_PER_FAMILY = (3, 7)
FILLER_FAMILIES_PER_AISLE = (6, 12)
FILLER_NOUNS = [
    "rice", "oats", "granola", "yoghurt", "cream", "soup", "beans", "tuna",
    "crackers", "biscuits", "olive_oil", "vinegar", "soy_sauce", "honey",
    "jam", "peanut_butter", "tea", "juice", "sparkling_water", "napkins",
    "foil", "cling_film", "washing_up_liquid", "sponges", "bin_bags",
    "kitchen_roll", "batteries", "light_bulbs", "shampoo", "soap",
    "toothpaste", "razors", "vitamins", "painkillers", "plasters",
    "dog_food", "cat_food", "cat_litter", "birthday_candles", "greeting_card",
]

# Within-basket ORDER. Shoppers do not add items in a random order: they walk
# the store, so items from the same aisle arrive together, and the item that
# STARTED the trip tends to come first. Emitting the order is what makes a
# sequential model possible at all -- the previous pass could not build one
# because the data had no notion of sequence to learn from.
AISLE_WALK_ORDER = ["produce", "bread", "dairy", "meat", "breakfast", "pasta",
                    "condiments", "snacks", "soft_drinks", "baking",
                    "household", "personal_care", "pets"]

# PER-FAMILY CONSUMPTION CADENCE.
#
# Every family previously had the same purchase dynamics, so measured
# inter-purchase intervals came out within 5% of each other across every aisle --
# which made "the due-score weight should not be one number" an argument with no
# evidence behind it. Real grocery is nothing like that: milk is weekly, bin bags
# are quarterly, and a single urgency curve applied to both fires far too early
# on one and far too late on the other.
#
# The multiplier scales how often a family enters a basket. It is the
# CONSUMPTION rate, not a preference: a household gets through milk faster than
# it gets through vinegar, and no amount of liking vinegar changes that.
FAMILY_CADENCE = {
    # fast movers -- bought nearly every trip
    "milk_whole": 0.30, "bananas": 0.35, "bread": 0.35, "sandwich_bread": 0.40,
    "eggs": 0.45, "lettuce": 0.50, "tomatoes": 0.55, "yoghurt": 0.5,
    # mid
    "cheese_slices": 0.8, "butter": 0.9, "coffee": 0.9, "cereal": 1.0,
    "chicken_breast": 0.8, "ground_beef": 0.9, "chips": 0.9, "cola": 0.8,
    # slow movers -- a jar lasts months
    "ketchup": 2.6, "mustard": 3.2, "mayo": 2.8, "flour": 3.0, "sugar": 3.4,
    "choc_chips": 3.6, "olive_oil": 3.2, "vinegar": 4.2, "soy_sauce": 4.0,
    "honey": 3.8, "jam": 2.6, "peanut_butter": 2.4,
    # household / personal care -- the slowest of all
    "bin_bags": 4.5, "foil": 4.8, "cling_film": 4.8, "kitchen_roll": 2.2,
    "washing_up_liquid": 3.0, "sponges": 3.5, "batteries": 6.0,
    "light_bulbs": 7.0, "shampoo": 3.5, "soap": 3.0, "toothpaste": 3.5,
    "razors": 4.0, "vitamins": 4.5, "painkillers": 5.0, "plasters": 6.0,
    "cat_litter": 1.6, "dog_food": 1.2, "cat_food": 1.2,
}
DEFAULT_CADENCE = 1.6


def _variant_names(fam_name: str, k: int) -> list[str]:
    suffixes = "abcdefghij"
    return ["%s_%s" % (fam_name, suffixes[i]) for i in range(k)]


def build_catalogue():
    """The hand-authored themes, widened, plus filler that belongs to no theme."""
    products, families, family_of, aisle_of = [], {}, {}, {}
    pid = 0

    def add_family(aisle, fam_name, n_variants):
        nonlocal pid
        families.setdefault(fam_name, [])
        for name in _variant_names(fam_name, n_variants):
            # Price and pack size are needed to rank substitutes the way a
            # shopper does -- a $3 sauce is not substitutable by an $11 one, and
            # a 500g bag is not swapped for a 2kg sack. Within a family they
            # VARY, which is what makes the penalty do work.
            base_price = float(RNG.uniform(1.5, 12.0))
            products.append(dict(
                product_id=pid, name=name, aisle=aisle, family=fam_name,
                price=round(base_price * float(RNG.uniform(0.75, 1.4)), 2),
                pack_size=float(RNG.choice([1, 1, 1, 2, 4, 6, 12])),
                # Free-text description, so a cold-start model has something to
                # embed. A brand-new SKU has no co-occurrence by definition; text
                # is the only signal that exists on day one.
                text="%s %s from the %s aisle" % (
                    name.replace("_", " "), fam_name.replace("_", " "),
                    aisle.replace("_", " ")),
                themed=False))
            families[fam_name].append(pid)
            family_of[pid] = fam_name
            aisle_of[pid] = aisle
            pid += 1

    for aisle, fams in AISLES.items():
        for fam in fams:
            fam_name = fam[0].rsplit("_", 1)[0]
            add_family(aisle, fam_name,
                       int(RNG.integers(*BRANDS_PER_FAMILY)))

    themed = set(families)
    for p in products:
        p["themed"] = True

    fillers = list(FILLER_NOUNS)
    RNG.shuffle(fillers)
    cursor = 0
    extra_aisles = ["household", "personal_care", "pets"]
    for aisle in list(AISLES) + extra_aisles:
        n = int(RNG.integers(*FILLER_FAMILIES_PER_AISLE))
        for _ in range(n):
            if cursor >= len(fillers):
                cursor = 0
                fillers = ["%s_x" % f for f in fillers]
            fam_name = fillers[cursor]
            cursor += 1
            if fam_name in families:
                continue
            add_family(aisle, fam_name, int(RNG.integers(2, 5)))

    for p in products:
        p["themed"] = p["family"] in themed
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

    # Each user's REPERTOIRE over families: a sparse Dirichlet, so most of a
    # user's long-tail purchases come from a handful of families they return to
    # rather than from the whole catalogue. Multiplied by the inverse cadence, so
    # a family a user likes but consumes slowly still appears rarely.
    all_fams = list(families)
    inv_cad = np.array([1.0 / FAMILY_CADENCE.get(f, DEFAULT_CADENCE)
                        for f in all_fams])
    fam_w = {}
    for u in range(N_USERS):
        pref = RNG.dirichlet(np.ones(len(all_fams)) * 0.12)
        w = pref * inv_cad
        fam_w[u] = w / w.sum()

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
                # even a user's own staples respect consumption: you do not buy
                # bin bags every week however loyal you are to the brand
                if RNG.random() < min(0.95, 0.55 / FAMILY_CADENCE.get(
                        fam, DEFAULT_CADENCE)):
                    fams_wanted.append(fam)
            # Filler families are drawn INVERSELY to their cadence multiplier, so
            # a family that lasts four times as long enters a basket a quarter as
            # often. That is what produces genuinely different inter-purchase
            # intervals per aisle, and it is the whole reason a per-category
            # due-weight has anything to bite on.
            # Filler draws are weighted two ways at once, and both are needed:
            #
            #   1/cadence   -- a family that lasts four times as long enters a
            #                  basket a quarter as often, which is what produces
            #                  genuinely different inter-purchase intervals;
            #   user habit  -- a shopper does not sample uniformly from 200
            #                  families every trip. They have a repertoire.
            #
            # Without the habit term the reorder rate collapsed to 0.49 when the
            # catalogue was widened from 49 to 420 products: users were touching
            # a new long-tail family every trip and never coming back. Real
            # grocery is 60%+ reorders, and the reason is repertoire, not memory.
            while len(fams_wanted) < target:
                fams_wanted.append(str(RNG.choice(all_fams, p=fam_w[u])))
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

            # WITHIN-BASKET ORDER, and it is not arbitrary.
            #
            # The trip starter -- the first themed item chosen -- stays first,
            # because it is the reason the shopper came. Everything after it is
            # ordered by the AISLE WALK, with jitter, because a shopper moves
            # through a store rather than teleporting. That gives the sequence
            # real structure for a sequential model to find and for an
            # order-agnostic model (item2vec, which shuffles context) to throw
            # away -- which is the comparison the previous pass could not run,
            # because the data had no order to ignore.
            if len(basket) > 1:
                anchor, rest = basket[0], basket[1:]
                walk = {a: i for i, a in enumerate(AISLE_WALK_ORDER)}
                rest.sort(key=lambda it: walk.get(aisle_of[it], 99)
                          + float(RNG.normal(0, 0.9)))
                basket = [anchor] + rest

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
