"""Tests for the completion pass: sequence, cold start, the slot-switch
redefinition, and the cart service."""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import coldstart as CS   # noqa: E402
from src import models as M       # noqa: E402
from src import sequence as SEQ   # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# --------------------------------------------------------------------------
# sequence
# --------------------------------------------------------------------------
def _chain_baskets(n=800, seed=0):
    """Baskets where the next item is a deterministic function of the last.
    Order carries ALL the signal; an order-blind model cannot recover it."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        start = int(rng.integers(0, 5))
        b = [start]
        for _ in range(4):
            b.append((b[-1] + 1) % 10)
        out.append(b)
    return out


def test_the_sequence_model_learns_a_deterministic_chain():
    b = _chain_baskets()
    m = SEQ.SequenceModel(10).fit(b)
    r = SEQ.evaluate_next_item(m, b, k=1)
    assert r["hit_rate"] > 0.9


def test_the_order_blind_control_cannot():
    """The control has identical co-occurrence and identical popularity. If it
    matched the sequence model, the experiment would be measuring nothing."""
    b = _chain_baskets()
    seq = SEQ.SequenceModel(10).fit(b)
    bag = SEQ.BagOfItemsControl(10, seed=0).fit(b)
    a = SEQ.evaluate_next_item(seq, b, k=1)["hit_rate"]
    c = SEQ.evaluate_next_item(bag, b, k=1)["hit_rate"]
    assert a > c + 0.3


def test_backoff_returns_popularity_for_an_unseen_last_item():
    m = SEQ.SequenceModel(6).fit([[0, 1, 2], [0, 1, 2]])
    s = m.next_scores(5)                       # item 5 never started a transition
    assert np.allclose(s, m.pop)


def test_backoff_weight_grows_with_evidence():
    thin = SEQ.SequenceModel(4, backoff_k=5.0).fit([[0, 1]])
    thick = SEQ.SequenceModel(4, backoff_k=5.0).fit([[0, 1]] * 200)
    # with more evidence the transition should dominate popularity
    assert thick.next_scores(0)[1] > thin.next_scores(0)[1]


def test_items_already_in_the_cart_are_never_suggested():
    m = SEQ.SequenceModel(6).fit([[0, 1, 2, 3]] * 20)
    s = m.score_basket([0, 1])
    assert s[0] == -np.inf and s[1] == -np.inf


def test_evaluation_hides_the_last_item_not_a_random_one():
    """A harder and more honest formulation of cart completion: the prefix is
    what the shopper has already added."""
    m = SEQ.SequenceModel(10).fit(_chain_baskets())
    r = SEQ.evaluate_next_item(m, [[0, 1, 2, 3, 4]], k=1)
    assert r["n"] == 1


# --------------------------------------------------------------------------
# cold start
# --------------------------------------------------------------------------
def test_the_family_centroid_excludes_the_held_out_product():
    """Leaving it in is the cold-start equivalent of training on the test set."""
    v = np.eye(4, dtype=np.float32)
    c = CS.family_centroid(v, [0, 1, 2], exclude=0)
    assert c[0] == pytest.approx(0.0)
    assert c[1] > 0 and c[2] > 0


def test_the_centroid_of_a_singleton_family_after_exclusion_is_zero():
    v = np.eye(3, dtype=np.float32)
    assert np.allclose(CS.family_centroid(v, [1], exclude=1), 0.0)


def test_content_rules_prefer_the_same_family_then_the_same_aisle():
    meta = {0: dict(family="cola", aisle="drinks", price=2.0, pack_size=1),
            1: dict(family="cola", aisle="drinks", price=2.1, pack_size=1),
            2: dict(family="juice", aisle="drinks", price=2.0, pack_size=1),
            3: dict(family="soap", aisle="cleaning", price=2.0, pack_size=1)}
    out = CS.content_rules(meta, 0, list(meta), k=3)
    assert out[0] == 1                      # same family first
    assert out.index(2) < out.index(3)      # same aisle before another aisle


def test_content_rules_always_return_k_even_from_a_tiny_family():
    """A blank recommendation slot on a live product page is worse than a weak
    one, so the third tier fills from the whole catalogue rather than returning
    an empty list."""
    meta = {0: dict(family="lonely", aisle="tiny", price=3.0, pack_size=1),
            1: dict(family="other", aisle="elsewhere", price=3.2, pack_size=1),
            2: dict(family="other", aisle="elsewhere", price=9.0, pack_size=4)}
    out = CS.content_rules(meta, 0, list(meta), k=2)
    assert len(out) == 2 and out[0] == 1


def test_content_rules_rank_by_price_and_pack_proximity():
    meta = {0: dict(family="f", aisle="a", price=3.0, pack_size=1),
            1: dict(family="f", aisle="a", price=3.1, pack_size=1),
            2: dict(family="f", aisle="a", price=30.0, pack_size=12)}
    assert CS.content_rules(meta, 0, list(meta), k=2)[0] == 1


def test_overlap_at_k_is_a_fraction_of_k():
    assert CS.overlap_at_k([1, 2, 3], [3, 4, 5], k=3) == pytest.approx(1 / 3)


# --------------------------------------------------------------------------
# the slot-switch redefinition
# --------------------------------------------------------------------------
def _switch_history():
    """A user who buys A three times then switches to B, its family sibling,
    with an unrelated staple present in every basket."""
    A, B, STAPLE, OTHER = 0, 1, 2, 3
    return {0: [[A, STAPLE, OTHER]] * 3 + [[B, STAPLE, OTHER]] * 3}


FAMILY_OF = {0: "f", 1: "f", 2: "staple", 3: "other"}


def test_the_presence_matrix_credits_the_bystanders_too():
    """The defect: 'B was present when A vanished' fires for every item in the
    basket, so the staple that is always there scores as high as the real
    substitute."""
    M_ = M.switch_matrix(_switch_history(), 4)
    assert M_[0, 1] > 0
    assert M_[0, 2] >= M_[0, 1], "the staple is credited at least as much"


def test_the_slot_matrix_credits_only_the_family_sibling():
    M_ = M.slot_switch_matrix(_switch_history(), 4, FAMILY_OF)
    assert M_[0, 1] > 0
    assert M_[0, 2] == 0 and M_[0, 3] == 0


def test_the_slot_matrix_emits_only_within_family_pairs():
    """Which is exactly why its precision is bounded by the taxonomy rather than
    earned -- the report says so rather than quoting the 1.0000."""
    M_ = M.slot_switch_matrix(_switch_history(), 4, FAMILY_OF)
    for a, b in np.argwhere(M_ > 0):
        assert FAMILY_OF[int(a)] == FAMILY_OF[int(b)]


def test_normalised_switch_damps_a_single_observation():
    raw = np.zeros((3, 3))
    raw[0, 1] = 1.0
    raw[0, 2] = 60.0
    out = M.normalised_switch(raw, [[0, 1, 2]] * 60, 3)
    assert out[0, 1] < out[0, 2] or out[0, 1] < 1.0


def test_substitutes_are_anti_correlated_in_the_generated_data():
    """The structural fact that defeats the presence heuristic. If this stops
    holding, section 4's whole explanation stops being true."""
    if not os.path.exists(os.path.join(DATA, "TRUTH.json")):
        pytest.skip("run `python src/generate.py` first")
    products = json.load(open(os.path.join(DATA, "products.json")))
    truth = json.load(open(os.path.join(DATA, "TRUTH.json")))
    op = np.load(os.path.join(DATA, "order_products.npy"))
    by = defaultdict(set)
    for oid, pid, _p, _r in op:
        by[int(oid)].add(int(pid))
    n = len(products)
    present = np.zeros(n)
    for b in by.values():
        for i in b:
            present[i] += 1
    n_orders = len(by)
    subs = [tuple(p) for p in truth["substitutes"][:200]]
    co = 0
    for a, b in subs:
        co += sum(1 for s in by.values() if a in s and b in s)
    observed = co / max(len(subs) * n_orders, 1)
    expected = float(np.mean([(present[a] / n_orders) * (present[b] / n_orders)
                              for a, b in subs]))
    assert observed < expected, "substitutes must co-occur LESS than chance"


# --------------------------------------------------------------------------
# the service
# --------------------------------------------------------------------------
def _client():
    tc = pytest.importorskip("fastapi.testclient")
    import serve
    return tc.TestClient(serve.app), serve


def test_service_health():
    client, _ = _client()
    assert client.get("/health").status_code == 200


def test_complete_never_suggests_something_already_in_the_cart():
    client, serve = _client()
    if not serve.ENGINE.ok:
        pytest.skip("run `python src/generate.py` first")
    items = [0, 1, 2]
    body = client.post("/complete", json={"items": items, "k": 5}).json()
    assert body["suggestions"]
    assert not (set(s["product_id"] for s in body["suggestions"]) & set(items))


def test_complete_reports_which_scorer_it_used():
    client, serve = _client()
    if not serve.ENGINE.ok:
        pytest.skip("run `python src/generate.py` first")
    body = client.post("/complete", json={"items": [0, 1], "k": 3}).json()
    assert body["scorer"] in {"symmetric lift", "content_rules (cold cart)"}
    assert isinstance(body["cold_fallback"], bool)


def test_unknown_product_is_a_404():
    client, serve = _client()
    if not serve.ENGINE.ok:
        pytest.skip("run `python src/generate.py` first")
    assert client.post("/complete", json={"items": [999999]}).status_code == 404


def test_an_empty_cart_is_a_400_not_a_crash():
    client, serve = _client()
    if not serve.ENGINE.ok:
        pytest.skip("run `python src/generate.py` first")
    assert client.post("/complete", json={"items": []}).status_code == 400


def test_substitutes_are_always_from_the_same_family():
    client, serve = _client()
    if not serve.ENGINE.ok:
        pytest.skip("run `python src/generate.py` first")
    for pid in (0, 5, 20):
        body = client.get("/substitute/%d" % pid).json()
        fam = serve.ENGINE.meta[pid]["family"]
        for s in body["substitutes"]:
            assert serve.ENGINE.meta[s["product_id"]]["family"] == fam


def test_substitution_says_what_it_ranked_on():
    client, serve = _client()
    if not serve.ENGINE.ok:
        pytest.skip("run `python src/generate.py` first")
    body = client.get("/substitute/0").json()
    assert body["basis"] in {"observed switches",
                             "price and pack proximity (no switch evidence)"}


def test_why_returns_both_conditionals_and_they_can_differ():
    client, serve = _client()
    if not serve.ENGINE.ok:
        pytest.skip("run `python src/generate.py` first")
    body = client.get("/why", params={"a": 0, "b": 1}).json()
    assert "p_b_given_a" in body and "p_a_given_b" in body
    assert "reading" in body


def test_next_uses_the_last_item():
    client, serve = _client()
    if not serve.ENGINE.ok:
        pytest.skip("run `python src/generate.py` first")
    body = client.get("/next", params={"items": "0,1,2"}).json()
    assert body["last_item"] == serve.ENGINE.meta[2]["name"]


def test_the_cart_ui_renders():
    client, serve = _client()
    if not serve.ENGINE.ok:
        pytest.skip("run `python src/generate.py` first")
    r = client.get("/", params={"items": "0,1"})
    assert r.status_code == 200 and "Complete the basket" in r.text
