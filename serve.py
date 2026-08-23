"""The cart service: complete a basket, substitute an out-of-stock line, explain both.

WHY THIS EXISTS
---------------
"No API and no cart UI. The endpoints are Python functions; the spec asks for a
demo cart that exercises both."

Both endpoints are the ones a grocery site actually calls, and putting them
behind HTTP forces three decisions the offline evaluation never had to make:

  * WHICH SCORE FOR WHICH JOB. The measurement in `run_complete.py` section 5
    found that the directional score LOSES as a whole-basket scorer and is
    obviously right for a pairwise widget. So the two endpoints use different
    scores on purpose, and each says which and why.
  * WHAT TO DO WITH A PRODUCT NOBODY HAS EVER BOUGHT. A cold SKU has no
    co-occurrence at all, so `/complete` has to fall back rather than return an
    empty list, and it has to SAY it fell back.
  * WHAT AN EXPLANATION IS. A merchandiser asking "why is this suggested" wants
    the pair evidence, not a feature importance table.

Run:  uvicorn serve:app --port 8013     then open http://127.0.0.1:8013/
"""
from __future__ import annotations

import html
import json
import os
import sys
from collections import defaultdict

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import coldstart as CS   # noqa: E402
from src import models as M       # noqa: E402
from src import sequence as SEQ   # noqa: E402
from src import timing as T       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

app = FastAPI(title="ML-3 cart service",
              description="Basket completion, substitution, and why each was suggested.")


class Engine:
    def __init__(self):
        self.ok = False
        try:
            self.products = json.load(open(os.path.join(DATA, "products.json")))
            self.n = len(self.products)
            self.meta = {int(p["product_id"]): p for p in self.products}
            self.by_name = {p["name"]: int(p["product_id"]) for p in self.products}
            self.fam = defaultdict(list)
            for p in self.products:
                self.fam[p["family"]].append(int(p["product_id"]))

            orders = np.load(os.path.join(DATA, "orders.npy"))
            op = np.load(os.path.join(DATA, "order_products.npy"))
            op = op[np.lexsort((op[:, 2], op[:, 0]))]
            by_order = defaultdict(list)
            for oid, pid, _pos, _re in op:
                by_order[int(oid)].append(int(pid))
            self.baskets = [by_order[o] for o in sorted(by_order)]
            user_of = {int(r[0]): int(r[1]) for r in orders}
            self.user_orders = defaultdict(list)
            for oid in sorted(by_order):
                self.user_orders[user_of[oid]].append(by_order[oid])

            self.lift = T.directional_lift(self.baskets, self.n)
            self.sym = np.minimum(self.lift, self.lift.T)
            self.seq = SEQ.SequenceModel(self.n).fit(self.baskets)
            self.i2v = M.Item2Vec(self.n, dim=48, seed=0)
            self.i2v.train(self.baskets, epochs=4)
            W = self.i2v.W
            self.Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
            self.counts = np.zeros(self.n)
            for b in self.baskets:
                for i in b:
                    self.counts[i] += 1
            self.slot = M.slot_switch_matrix(
                self.user_orders, self.n,
                {int(p["product_id"]): p["family"] for p in self.products})
            self.ok = True
        except Exception as exc:                       # pragma: no cover
            self.error = str(exc)


ENGINE = Engine()
COLD_THRESHOLD = 5


def _require():
    if not ENGINE.ok:
        raise HTTPException(503, "data missing -- run `python src/generate.py` first")


def _name(pid: int) -> str:
    return ENGINE.meta[pid]["name"]


class Cart(BaseModel):
    items: list[int] = Field(..., description="product ids, IN ADD-TO-CART ORDER")
    k: int = Field(5, ge=1, le=20)


@app.get("/health")
def health():
    return {"ok": ENGINE.ok, "products": getattr(ENGINE, "n", 0)}


@app.post("/complete")
def complete(cart: Cart):
    """Suggest what else belongs in this basket.

    Uses the SYMMETRIC score, and that is a measured decision rather than a
    default. Summing the directional P(b|a) over the items already in the cart
    is a mixture of conditionals, and a globally popular b scores well under
    every conditional -- so directional scoring rewards popularity here. The
    symmetrised min(P(b|a), P(a|b)) demands the relationship hold both ways,
    which is the cheapest specificity filter available. Section 5 of
    run_complete.py measures the gap.
    """
    _require()
    bad = [i for i in cart.items if i not in ENGINE.meta]
    if bad:
        raise HTTPException(404, "unknown product ids: %s" % bad)
    if not cart.items:
        raise HTTPException(400, "empty cart")

    warm = [i for i in cart.items if ENGINE.counts[i] >= COLD_THRESHOLD]
    fell_back = False
    if warm:
        s = ENGINE.sym[warm].sum(axis=0)
    else:
        # Nothing in the cart has co-occurrence history, so the distributional
        # model has literally nothing to say. Falling back is not an error path;
        # it is the day-one path for every new SKU.
        fell_back = True
        s = np.zeros(ENGINE.n)
        for i in cart.items:
            for j in CS.content_rules(ENGINE.meta, i, list(range(ENGINE.n)), k=20):
                s[j] += 1.0
    s[list(cart.items)] = -np.inf
    top = np.argpartition(-s, min(cart.k, len(s) - 1))[:cart.k]
    top = [int(i) for i in top[np.argsort(-s[top])]]

    return {
        "cart": [_name(i) for i in cart.items],
        "suggestions": [
            {"product_id": i, "name": _name(i), "score": round(float(s[i]), 5),
             "family": ENGINE.meta[i]["family"], "price": ENGINE.meta[i]["price"]}
            for i in top],
        "scorer": "content_rules (cold cart)" if fell_back else "symmetric lift",
        "cold_fallback": fell_back,
        "note": ("no item in this cart has co-occurrence history, so the "
                 "distributional model was skipped entirely"
                 if fell_back else
                 "symmetric rather than directional: see run_complete.py section 5"),
    }


@app.get("/next")
def next_item(items: str, k: int = 5):
    """What the shopper is most likely to add NEXT, given the order they added in.

    A different question from `/complete` and it deserves a different model. The
    sequence model keys on the LAST item only, which is its stated limitation --
    a shopper who just added pasta sauce is in a different state from one who
    added it six items ago.
    """
    _require()
    try:
        basket = [int(x) for x in items.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(400, "items must be a comma-separated list of ids")
    if any(i not in ENGINE.meta for i in basket):
        raise HTTPException(404, "unknown product id")
    s = ENGINE.seq.score_basket(basket)
    top = np.argpartition(-s, min(k, len(s) - 1))[:k]
    top = [int(i) for i in top[np.argsort(-s[top])]]
    return {"cart": [_name(i) for i in basket],
            "last_item": _name(basket[-1]) if basket else None,
            "next": [{"product_id": i, "name": _name(i),
                      "score": round(float(s[i]), 5)} for i in top],
            "note": "first-order: keys on the LAST item only"}


@app.get("/substitute/{product_id}")
def substitute(product_id: int, k: int = 5):
    """This line is out of stock -- what do we offer instead?

    Ranked by OBSERVED SWITCHES within the family, not by embedding similarity.
    The taxonomy says which products could substitute; the switch evidence says
    which one shoppers actually accept, and only the second is a recommendation.
    A cold product has no switch evidence and falls back to price and pack
    proximity within its family.
    """
    _require()
    if product_id not in ENGINE.meta:
        raise HTTPException(404, "unknown product")
    fam = ENGINE.meta[product_id]["family"]
    siblings = [i for i in ENGINE.fam[fam] if i != product_id]
    if not siblings:
        return {"product": _name(product_id), "substitutes": [],
                "note": "single-member family: nothing in the catalogue replaces it"}

    evidence = {i: float(ENGINE.slot[product_id, i] + ENGINE.slot[i, product_id])
                for i in siblings}
    observed = any(v > 0 for v in evidence.values())
    if observed:
        ranked = sorted(siblings, key=lambda i: -evidence[i])[:k]
        basis = "observed switches"
    else:
        ranked = CS.content_rules(ENGINE.meta, product_id, ENGINE.fam[fam], k=k)
        basis = "price and pack proximity (no switch evidence)"
    return {
        "product": _name(product_id), "family": fam,
        "substitutes": [
            {"product_id": i, "name": _name(i),
             "price": ENGINE.meta[i]["price"],
             "pack_size": ENGINE.meta[i]["pack_size"],
             "observed_switches": evidence.get(i, 0.0)} for i in ranked],
        "basis": basis,
    }


@app.get("/why")
def why(a: int, b: int):
    """Why is b suggested for a? The pair evidence, both directions.

    A merchandiser asking this wants the numbers for THIS pair, not a global
    feature-importance table -- which answers what the model uses on average and
    is a different question.
    """
    _require()
    if a not in ENGINE.meta or b not in ENGINE.meta:
        raise HTTPException(404, "unknown product")
    return {
        "a": _name(a), "b": _name(b),
        "p_b_given_a": round(float(ENGINE.lift[a, b]), 5),
        "p_a_given_b": round(float(ENGINE.lift[b, a]), 5),
        "symmetric": round(float(ENGINE.sym[a, b]), 5),
        "same_family": ENGINE.meta[a]["family"] == ENGINE.meta[b]["family"],
        "observed_switches": float(ENGINE.slot[a, b] + ENGINE.slot[b, a]),
        "reading": ("same family: these are SUBSTITUTES, and a cart that already "
                    "holds one should not be offered the other"
                    if ENGINE.meta[a]["family"] == ENGINE.meta[b]["family"] else
                    "different families: a COMPLEMENT relationship, and the two "
                    "conditionals differ because the direction is real"),
    }


@app.get("/", response_class=HTMLResponse)
def cart_ui(items: str = ""):
    if not ENGINE.ok:
        return HTMLResponse("<h1>data missing</h1><p>run <code>python "
                            "src/generate.py</code></p>", status_code=503)
    sample = ",".join(str(ENGINE.by_name[n]) for n in
                      [p["name"] for p in ENGINE.products[:3]])
    body = ""
    if items:
        ids = [int(x) for x in items.split(",") if x.strip()]
        comp = complete(Cart(items=ids, k=6))
        nxt = next_item(items=items, k=6)
        sub = substitute(ids[-1], k=4)
        body = (
            "<h2>Cart</h2><ul>%s</ul>"
            "<h2>Complete the basket <small>(%s)</small></h2><ol>%s</ol>"
            "<h2>Most likely next add <small>(sequence, keys on %s)</small></h2><ol>%s</ol>"
            "<h2>If <code>%s</code> is out of stock <small>(%s)</small></h2><ol>%s</ol>"
            % ("".join("<li><code>%s</code></li>" % html.escape(c) for c in comp["cart"]),
               html.escape(comp["scorer"]),
               "".join("<li>%s <small>%.4f</small></li>"
                       % (html.escape(r["name"]), r["score"]) for r in comp["suggestions"]),
               html.escape(str(nxt["last_item"])),
               "".join("<li>%s <small>%.4f</small></li>"
                       % (html.escape(r["name"]), r["score"]) for r in nxt["next"]),
               html.escape(sub["product"]), html.escape(sub["basis"]),
               "".join("<li>%s <small>$%.2f, %d observed switches</small></li>"
                       % (html.escape(r["name"]), r["price"],
                          int(r["observed_switches"])) for r in sub["substitutes"])))
    return """<!doctype html><meta charset=utf-8><title>ML-3 cart</title>
<style>
 body{font:15px/1.55 system-ui,sans-serif;max-width:48rem;margin:2rem auto;padding:0 1rem}
 input{font-size:1rem;padding:.4rem;width:22rem}
 h2{font-size:1.05rem;margin-top:1.4rem}
 code{background:#f4f4f4;padding:.1rem .3rem}
 small{color:#666}
</style>
<h1>ML-3 &mdash; cart</h1>
<form><input name=items value="%s" placeholder="product ids, in add-to-cart order">
<button>go</button></form>
<p><small>Try <code>%s</code>. Three endpoints, three different scores, on purpose:
completion is symmetric (directional scoring rewards popularity when summed over a
basket), next-add is sequential, substitution is ranked by observed switches rather
than embedding similarity. API: <a href="/docs">/docs</a>.</small></p>
%s
""" % (html.escape(items), sample, body)
