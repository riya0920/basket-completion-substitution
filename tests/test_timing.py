"""Tests for the second tranche: timing, directionality, price/pack, cold start."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import timing as T  # noqa: E402


# --------------------------------------------------------------------------
# reorder timing
# --------------------------------------------------------------------------
def _weekly_user(n=12, gap=7.0, item=0, other=1):
    return [(gap * i, [item, other]) for i in range(n)]


def test_interval_recovers_a_regular_cadence():
    tm = T.ReorderTiming(shrinkage_k=1.0).fit({0: _weekly_user(gap=7.0)})
    assert tm.expected_interval(0, 0) == pytest.approx(7.0, abs=0.5)


def test_due_score_peaks_at_the_expected_interval():
    """The whole argument for a hazard rather than a recency feature."""
    tm = T.ReorderTiming(shrinkage_k=0.0).fit({0: _weekly_user(gap=7.0)})
    at_interval = tm.due_score(0, 0, 7.0)
    too_soon = tm.due_score(0, 0, 1.0)
    too_late = tm.due_score(0, 0, 40.0)
    assert at_interval > too_soon
    assert at_interval > too_late
    assert at_interval == pytest.approx(1.0, abs=0.05)


def test_due_score_is_not_monotone_in_elapsed_time():
    """A recency feature says 'more time = more likely' and is wrong past the
    interval. This asserts the non-monotonicity that fixes it."""
    tm = T.ReorderTiming(shrinkage_k=0.0).fit({0: _weekly_user(gap=7.0)})
    scores = [tm.due_score(0, 0, d) for d in (1, 4, 7, 14, 30, 60)]
    assert scores != sorted(scores), "due score must not be monotone in recency"
    assert scores[-1] < scores[2]


def test_shrinkage_pulls_a_sparse_pair_toward_the_population():
    """A user with ONE observed interval must not be trusted over the item's
    population mean; a user with many must be."""
    users = {u: _weekly_user(n=12, gap=10.0, item=0, other=1) for u in range(30)}
    users[99] = [(0.0, [0, 1]), (2.0, [0, 1])]      # one interval of 2 days
    tm = T.ReorderTiming(shrinkage_k=3.0).fit(users)
    pop = tm.item_mean[0]
    sparse = tm.expected_interval(99, 0)
    dense = tm.expected_interval(0, 0)
    assert abs(sparse - pop) < abs(2.0 - pop), "sparse pair must shrink to population"
    assert abs(dense - 10.0) < abs(sparse - 10.0)


def test_unseen_pair_falls_back_to_the_item_population():
    tm = T.ReorderTiming().fit({0: _weekly_user(gap=9.0)})
    assert tm.expected_interval(12345, 0) == pytest.approx(tm.item_mean[0])


def test_unseen_item_falls_back_to_the_global_mean():
    tm = T.ReorderTiming().fit({0: _weekly_user(gap=9.0)})
    assert tm.expected_interval(0, 9999) == pytest.approx(tm.global_mean)


# --------------------------------------------------------------------------
# directional complements
# --------------------------------------------------------------------------
def test_directional_lift_is_asymmetric_when_the_data_is():
    """b always appears with a; a appears without b half the time. So
    P(b|a) = 1.0 and P(a|b) = 1.0 -- to make it asymmetric, b must also appear
    without a."""
    baskets = [[0, 1]] * 10 + [[0, 2]] * 10 + [[1, 3]] * 2
    M = T.directional_lift(baskets, 4)
    assert M[0, 1] == pytest.approx(10 / 20)     # a appears 20x, with b 10x
    assert M[1, 0] == pytest.approx(10 / 12)     # b appears 12x, with a 10x
    assert T.asymmetry(M, 0, 1) < 0


def test_directional_lift_rows_are_conditional_probabilities():
    baskets = [[0, 1, 2], [0, 1], [0], [1, 2]]
    M = T.directional_lift(baskets, 3)
    assert 0.0 <= M.min() and M.max() <= 1.0
    assert np.allclose(np.diag(M), 0.0)
    assert M[0, 1] == pytest.approx(2 / 3)       # 0 appears 3x, with 1 twice


def test_symmetric_data_gives_zero_asymmetry():
    baskets = [[0, 1]] * 20
    M = T.directional_lift(baskets, 2)
    assert T.asymmetry(M, 0, 1) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# price- and pack-aware substitution
# --------------------------------------------------------------------------
def test_price_penalty_demotes_a_far_priced_substitute():
    sim = np.array([[0.0, 0.9, 0.9], [0.9, 0.0, 0.5], [0.9, 0.5, 0.0]])
    prices = np.array([3.0, 3.2, 11.0])
    packs = np.array([1.0, 1.0, 1.0])
    out = T.substitution_score(sim, prices, packs, 0, np.array([1, 2]))
    assert out[0] > out[1], "the similarly-priced substitute must win"


def test_pack_penalty_demotes_a_far_sized_substitute():
    sim = np.array([[0.0, 0.9, 0.9], [0.9, 0.0, 0.5], [0.9, 0.5, 0.0]])
    prices = np.array([3.0, 3.0, 3.0])
    packs = np.array([1.0, 1.0, 12.0])
    out = T.substitution_score(sim, prices, packs, 0, np.array([1, 2]))
    assert out[0] > out[1]


def test_penalties_cannot_promote_a_dissimilar_item():
    """Bounded and multiplicative: price breaks ties among substitutes, it does
    not create substitutes. An additive price term would fail this."""
    sim = np.array([[0.0, 0.95, 0.05], [0.95, 0.0, 0.0], [0.05, 0.0, 0.0]])
    prices = np.array([5.0, 9.0, 5.0])       # item 2 is a perfect price match
    packs = np.array([1.0, 1.0, 1.0])
    out = T.substitution_score(sim, prices, packs, 0, np.array([1, 2]))
    assert out[0] > out[1], "a similar item must beat a dissimilar price twin"


def test_identical_price_and_pack_leaves_similarity_untouched():
    sim = np.array([[0.0, 0.8], [0.8, 0.0]])
    prices = np.array([4.0, 4.0])
    packs = np.array([2.0, 2.0])
    out = T.substitution_score(sim, prices, packs, 0, np.array([1]))
    assert out[0] == pytest.approx(0.8)


# --------------------------------------------------------------------------
# cold start
# --------------------------------------------------------------------------
def _meta():
    return {
        0: dict(family="cola", aisle="drinks"),
        1: dict(family="cola", aisle="drinks"),
        2: dict(family="lemon", aisle="drinks"),
        3: dict(family="bread", aisle="bakery"),
    }


def test_cold_start_prefers_the_same_family():
    prices = np.array([3.0, 3.1, 3.0, 3.0])
    packs = np.ones(4)
    out = T.cold_start_substitutes(_meta(), 0, [0, 1, 2, 3], prices, packs, top_k=3)
    assert out[0] == 1, "same-family item must come first"
    assert 3 not in out[:2], "a different aisle must not outrank the same aisle"


def test_cold_start_never_returns_the_item_itself():
    prices = np.ones(4)
    packs = np.ones(4)
    out = T.cold_start_substitutes(_meta(), 0, [0, 1, 2, 3], prices, packs)
    assert 0 not in out


def test_cold_start_orders_same_aisle_by_price_proximity():
    meta = {0: dict(family="a", aisle="z"), 1: dict(family="b", aisle="z"),
            2: dict(family="c", aisle="z")}
    prices = np.array([5.0, 20.0, 5.2])
    packs = np.ones(3)
    out = T.cold_start_substitutes(meta, 0, [0, 1, 2], prices, packs)
    assert out[0] == 2, "the closer-priced aisle-mate must rank first"


def test_cold_start_works_with_no_same_family_items():
    meta = {0: dict(family="solo", aisle="z"), 1: dict(family="b", aisle="z")}
    out = T.cold_start_substitutes(meta, 0, [0, 1], np.ones(2), np.ones(2))
    assert out == [1]
