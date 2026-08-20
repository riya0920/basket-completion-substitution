"""Guards on the two relations and on the evaluation protocol.

The relation tests are the point: this project's entire claim is that complements
and substitutes are OPPOSITE relations that naive co-occurrence conflates. If
that separation is not asserted mechanically it is just a paragraph.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import generate as G  # noqa: E402
from src import models as M  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@pytest.fixture(scope="module")
def world():
    if not os.path.exists(os.path.join(DATA, "TRUTH.json")):
        pytest.skip("run `python src/generate.py` first")
    op = np.load(os.path.join(DATA, "order_products.npy"))
    with open(os.path.join(DATA, "products.json")) as f:
        products = json.load(f)
    with open(os.path.join(DATA, "TRUTH.json")) as f:
        truth = json.load(f)
    baskets = {}
    for oid, pid, _pos, _r in op:
        baskets.setdefault(int(oid), []).append(int(pid))
    return list(baskets.values()), products, truth


# --------------------------------------------------------------------------
# the generator plants what it claims to plant
# --------------------------------------------------------------------------
def test_substitutes_share_a_family_and_complements_do_not(world):
    _, products, truth = world
    fam = {p["product_id"]: p["family"] for p in products}
    for a, b in truth["substitutes"]:
        assert fam[a] == fam[b], "substitutes must be same-family"
    for a, b in truth["complements"]:
        assert fam[a] != fam[b], "complements must be cross-family"


def test_substitutes_essentially_never_co_occur(world):
    """The structural fact the whole project rests on. If this fails, the
    generator is not producing substitutes."""
    baskets, products, truth = world
    co = M.cooccurrence(baskets, len(products))
    sub_co = np.mean([co[a, b] for a, b in truth["substitutes"]])
    comp_co = np.mean([co[a, b] for a, b in truth["complements"]])
    assert sub_co == 0.0
    assert comp_co > 100


def test_reorder_rate_is_grocery_like(world):
    op = np.load(os.path.join(DATA, "order_products.npy"))
    assert 0.5 < op[:, 3].mean() < 0.9, "grocery is 60%+ reorders"


# --------------------------------------------------------------------------
# the conflation, asserted
# --------------------------------------------------------------------------
def test_lift_ranks_substitutes_below_unrelated_pairs(world):
    """The failure mode, pinned. Co-occurrence lift does not merely miss
    substitutes -- it ranks them BELOW random pairs, because they are
    anti-correlated within a basket."""
    baskets, products, truth = world
    n = len(products)
    co = M.cooccurrence(baskets, n)
    lift = M.lift_matrix(co, baskets, n)
    subs = {(min(a, b), max(a, b)) for a, b in truth["substitutes"]}
    comps = {(min(a, b), max(a, b)) for a, b in truth["complements"]}
    rng = np.random.default_rng(0)
    other = [(a, b) for a in range(n) for b in range(a + 1, n)
             if (a, b) not in subs and (a, b) not in comps]
    rng.shuffle(other)
    sub_lift = np.mean([lift[a, b] for a, b in subs])
    oth_lift = np.mean([lift[a, b] for a, b in other[:200]])
    assert sub_lift < oth_lift


@pytest.fixture(scope="module")
def trained(world):
    baskets, products, _ = world
    return M.Item2Vec(len(products), dim=32, seed=0).train(baskets[:12000], epochs=3)


def test_second_order_similarity_separates_substitutes(trained, world):
    """cos(W_a, W_b) must be higher for substitutes than for complements. This is
    the fix, and it is the assertion that keeps the fix honest."""
    _, products, truth = world
    S = trained.substitute_similarity()
    sub = np.mean([S[a, b] for a, b in truth["substitutes"]])
    comp = np.mean([S[a, b] for a, b in truth["complements"]])
    assert sub > comp, "second-order similarity must rank substitutes above complements"


def test_first_order_score_separates_complements(trained, world):
    """...and the OTHER matrix must do the opposite. Both directions, or the
    separation is an accident."""
    _, products, truth = world
    S = trained.complement_score()
    sub = np.mean([S[a, b] for a, b in truth["substitutes"]])
    comp = np.mean([S[a, b] for a, b in truth["complements"]])
    assert comp > sub, "W.C^T must rank complements above substitutes"


def test_the_two_relations_disagree(trained, world):
    """The sharp version: the top-1 by first-order score and the top-1 by
    second-order similarity must be DIFFERENT items for most products. If they
    agree, one matrix would have been enough and the project has no thesis."""
    _, products, _ = world
    comp_S, sub_S = trained.complement_score(), trained.substitute_similarity()
    disagree = 0
    for i in range(len(products)):
        c = int(np.argmax(np.where(np.isfinite(comp_S[i]), comp_S[i], -9e9)))
        s = int(np.argmax(np.where(np.isfinite(sub_S[i]), sub_S[i], -9e9)))
        disagree += (c != s)
    assert disagree > 0.8 * len(products)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
def test_lift_is_symmetric_and_diagonal_free(world):
    baskets, products, _ = world
    n = len(products)
    lift = M.lift_matrix(M.cooccurrence(baskets, n), baskets, n)
    assert np.allclose(lift, lift.T)
    assert np.allclose(np.diag(lift), 0)


def test_personalise_promotes_the_users_own_history():
    scores = np.zeros(10)
    out = M.personalise(scores, {3: 5, 7: 1}, weight=1.0)
    assert out[3] > out[7] > out[0]


def test_popularity_scores_sum_to_one(world):
    baskets, products, _ = world
    p = M.popularity_scores(baskets, len(products))
    assert p.sum() == pytest.approx(1.0)


def test_item2vec_keeps_both_matrices():
    """gensim hands you the input vectors and drops the context vectors. Half the
    thesis lives in the ones it drops."""
    m = M.Item2Vec(10, dim=4, seed=0)
    assert m.W.shape == m.C.shape == (10, 4)
    assert not np.allclose(m.W, m.C)


def test_switch_matrix_records_a_switch():
    """user buys 0 for three orders, then 1 appears in its place"""
    orders = {0: [[0, 5], [0, 5], [0, 5], [1, 5]]}
    S = M.switch_matrix(orders, 6)
    assert S[0, 1] > 0
