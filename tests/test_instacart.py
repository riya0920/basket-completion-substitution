"""Real Instacart baskets, and the generator claim they refute.

Skips when the CSVs are absent, so the suite passes without 700 MB on disk.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import instacart as I  # noqa: E402
from src import sequence as S   # noqa: E402


needs_ic = pytest.mark.skipif(not I.available(),
                              reason="no Instacart CSVs in .vendor/kaggle/")


@pytest.fixture(scope="module")
def data():
    if not I.available():
        pytest.skip("no Instacart CSVs")
    return I.load_baskets(max_orders=12_000, seed=0)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
@needs_ic
def test_baskets_are_in_real_add_to_cart_order(data):
    """The whole order experiment rests on this column. The file is grouped by
    order but nothing in the format guarantees the within-order sequence, and a
    silently mis-ordered basket would make the sequence model look exactly like
    its own control."""
    import pandas as pd
    keep = set()
    for chunk in pd.read_csv(I.PRIOR,
                             usecols=["order_id", "product_id",
                                      "add_to_cart_order"],
                             chunksize=500_000):
        keep = chunk
        break
    g = keep.groupby("order_id").head(50)
    for _oid, grp in g.groupby("order_id"):
        assert list(grp.add_to_cart_order) == sorted(grp.add_to_cart_order)


@needs_ic
def test_product_ids_are_densely_reindexed(data):
    """The models allocate an n_items x n_items matrix and Instacart's ids run
    to 49,688 whether or not the sample uses them."""
    flat = [i for b in data["baskets"] for i in b]
    assert min(flat) >= 0
    assert max(flat) < data["n_items"]
    assert len(data["aisle"]) == data["n_items"]
    assert len(data["name"]) == data["n_items"]


@needs_ic
def test_every_basket_meets_the_minimum_length(data):
    assert all(len(b) >= 3 for b in data["baskets"])


@needs_ic
def test_the_corpus_looks_like_instacart(data):
    st = I.stats(data)
    assert st["aisles"] > 100
    assert 5 < st["mean_basket"] < 20


# --------------------------------------------------------------------------
# the order claim -- direction survives, magnitude does not
# --------------------------------------------------------------------------
@needs_ic
def test_order_still_carries_signal_on_real_baskets(data):
    b = data["baskets"]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(b))
    cut = int(0.8 * len(b))
    tr = [b[i] for i in idx[:cut]]
    te = [b[i] for i in idx[cut:]]
    seq = S.SequenceModel(data["n_items"]).fit(tr)
    bag = S.BagOfItemsControl(data["n_items"], seed=0).fit(tr)
    rs = S.evaluate_next_item(seq, te, k=10)
    rb = S.evaluate_next_item(bag, te, k=10)
    assert rs["hit_rate"] > rb["hit_rate"], "order should still help"


@needs_ic
def test_the_real_order_effect_is_far_smaller_than_the_generators(data):
    """+0.1229 on the generator. Real shoppers are messier, and the generator's
    number was an upper bound on a real effect rather than an estimate of one."""
    b = data["baskets"]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(b))
    cut = int(0.8 * len(b))
    tr = [b[i] for i in idx[:cut]]
    te = [b[i] for i in idx[cut:]]
    seq = S.SequenceModel(data["n_items"]).fit(tr)
    bag = S.BagOfItemsControl(data["n_items"], seed=0).fit(tr)
    delta = (S.evaluate_next_item(seq, te, k=10)["hit_rate"]
             - S.evaluate_next_item(bag, te, k=10)["hit_rate"])
    assert 0 < delta < 0.1229 / 2


# --------------------------------------------------------------------------
# the mechanism claim -- refuted
# --------------------------------------------------------------------------
@needs_ic
def test_similar_products_co_occur_MORE_not_less(data):
    """The generator says 'substitutes do not co-occur' and builds that in by
    placing one item per family per basket. Real shoppers do the opposite: the
    more similar two products are, the more often they share a basket."""
    r = I.variant_cooccurrence(data, jaccard=0.6, n_cross=40_000)
    assert r["variant"]["mean_lift"] > r["same_aisle"]["mean_lift"]
    assert r["same_aisle"]["mean_lift"] > r["cross_aisle"]["mean_lift"]
    assert r["variant"]["mean_lift"] > 2.0, "variants should be well above chance"


@needs_ic
def test_the_refutation_survives_every_similarity_threshold(data):
    """The proxy is doing real work in the argument, so its knob is swept rather
    than chosen."""
    lifts = []
    for jac in (0.5, 0.6, 0.7):
        r = I.variant_cooccurrence(data, jaccard=jac, n_cross=30_000)
        lifts.append(r["variant"]["mean_lift"])
        assert r["variant"]["mean_lift"] > r["cross_aisle"]["mean_lift"]
    assert min(lifts) > 2.0, lifts


@needs_ic
def test_cross_aisle_lift_is_near_chance(data):
    """The baseline has to behave, or the contrast means nothing."""
    r = I.variant_cooccurrence(data, jaccard=0.6, n_cross=60_000)
    assert 0.5 < r["cross_aisle"]["mean_lift"] < 3.0


@needs_ic
def test_same_aisle_pairs_are_enumerated_not_sampled(data):
    """Random pairs find a variant roughly once in 4,000 draws, and the first
    version of this reported a headline on 161 of them."""
    small = I.variant_cooccurrence(data, jaccard=0.6, n_cross=1_000)
    big = I.variant_cooccurrence(data, jaccard=0.6, n_cross=50_000)
    assert small["variant"]["n"] == big["variant"]["n"]
    assert big["cross_aisle"]["n"] > small["cross_aisle"]["n"]
