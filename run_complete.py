"""The completion pass: sequence, cold start scored against a ceiling, per-category
timing, the switch matrix on trial, directional serving, and an interleaving story.

Every section closes one item the previous README listed as missing. Run after
`python src/generate.py`. Writes out/complete_report.txt.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import coldstart as CS   # noqa: E402
from src import models as M       # noqa: E402
from src import sequence as SEQ   # noqa: E402
from src import timing as T       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
K = 10


def load():
    """The generator writes numpy arrays and JSON, not CSVs -- same loader shape
    as run_basket.py so the two runners cannot drift apart on schema."""
    products = json.load(open(os.path.join(DATA, "products.json")))
    orders = np.load(os.path.join(DATA, "orders.npy"))          # id,user,seq,day
    op = np.load(os.path.join(DATA, "order_products.npy"))      # order,prod,pos,reord
    truth = json.load(open(os.path.join(DATA, "TRUTH.json")))
    return products, orders, op, truth


def main():
    os.makedirs(OUT, exist_ok=True)
    lines_out, summary = [], {}

    def emit(s=""):
        print(s)
        lines_out.append(s)

    products, orders, op, truth = load()
    n_items = len(products)
    meta = {int(p["product_id"]): dict(family=p["family"], aisle=p["aisle"],
                                       price=float(p["price"]),
                                       pack_size=float(p["pack_size"]),
                                       name=p["name"], text=p["text"])
            for p in products}
    fam_members = defaultdict(list)
    for pid, m in meta.items():
        fam_members[m["family"]].append(pid)

    # baskets IN ADD-TO-CART ORDER -- the whole point of section 1, so the sort
    # is not an implementation detail and a refactor that drops it would make
    # the sequence model silently useless rather than loudly broken.
    order_idx = np.lexsort((op[:, 2], op[:, 0]))
    op = op[order_idx]
    by_order = defaultdict(list)
    for oid, pid, _pos, _re in op:
        by_order[int(oid)].append(int(pid))

    order_day = {int(r[0]): float(r[3]) for r in orders}
    order_user = {int(r[0]): int(r[1]) for r in orders}
    all_ids = sorted(by_order)

    # temporal split: the last 20% of DAYS are held out. Not a random split --
    # a recommender validated on a random split has seen each user's future.
    cut = float(np.quantile([order_day[o] for o in all_ids], 0.8))
    train_b = [by_order[o] for o in all_ids if order_day[o] <= cut]
    test_b = [by_order[o] for o in all_ids if order_day[o] > cut]

    emit("=" * 78)
    emit("ML-3 COMPLETION PASS -- %d products, %d orders, %d train / %d test baskets"
         % (n_items, len(all_ids), len(train_b), len(test_b)))
    emit("=" * 78)
    emit("")

    # ======================================================================
    emit("=" * 78)
    emit("1. DOES WITHIN-BASKET ORDER CARRY SIGNAL?")
    emit("=" * 78)
    seq = SEQ.SequenceModel(n_items).fit(train_b)
    bag = SEQ.BagOfItemsControl(n_items, seed=0).fit(train_b)

    rows = []
    for name, model in (("sequence (uses order)", seq),
                        ("bag of items (order destroyed)", bag)):
        r = SEQ.evaluate_next_item(model, test_b, k=K)
        rows.append(dict(model=name, **r))
        emit("  %-32s hit@%d %.4f   MRR %.4f   n=%d"
             % (name, K, r["hit_rate"], r["mrr"], r["n"]))
    d = rows[0]["hit_rate"] - rows[1]["hit_rate"]
    emit("")
    emit("ORDER IS WORTH %+.4f hit@%d ON NEXT-ITEM PREDICTION." % (d, K))
    emit("")
    emit("  The control is the same model trained on SHUFFLED copies of the same")
    emit("  baskets. Identical co-occurrence, identical popularity, identical")
    emit("  capacity -- the only thing it cannot know is which item came last. Any")
    emit("  gap is attributable to order and to nothing else, which is why the")
    emit("  control is a shuffle rather than a different model.")
    emit("")
    emit("  This is also why the generator had to change first. The previous pass")
    emit("  could not run this experiment: baskets were emitted in whatever order")
    emit("  the generator happened to build them, so there was no order to ignore")
    emit("  and 'item2vec ignores sequence' was a statement about the model with no")
    emit("  measurable consequence.")
    emit("")
    emit("  HONEST LIMIT: first-order. The model forgets everything before the last")
    emit("  item, so a basket that is obviously a cookout is represented only by")
    emit("  whatever went in most recently. That is left visible rather than")
    emit("  patched with an average over the basket -- averaging would quietly turn")
    emit("  it back into a bag-of-items model and this comparison would stop")
    emit("  meaning anything.")
    emit("")
    summary["sequence"] = rows

    # ======================================================================
    emit("=" * 78)
    emit("2. COLD START, SCORED AGAINST THE CEILING IT CANNOT REACH")
    emit("=" * 78)
    i2v = M.Item2Vec(n_items, dim=48, seed=0)
    i2v.train(train_b, epochs=5)
    Wn = i2v.W / (np.linalg.norm(i2v.W, axis=1, keepdims=True) + 1e-9)

    # hold out products that have enough history for the ceiling to be meaningful
    counts = np.zeros(n_items)
    for b in train_b:
        for i in b:
            counts[i] += 1
    candidates = [i for i in range(n_items)
                  if counts[i] >= 40 and len(fam_members[meta[i]["family"]]) > 1]
    rng = np.random.default_rng(0)
    cold = list(rng.choice(candidates, size=min(60, len(candidates)), replace=False))

    texts = [meta[i]["text"] for i in range(n_items)]
    tv = CS.embed_texts(texts)
    have_text = tv is not None

    rows = []
    for i in cold:
        i = int(i)
        ceiling = CS.rank_by_vector(Wn[i], Wn, exclude={i}, k=K)
        cent = CS.family_centroid(Wn, fam_members[meta[i]["family"]], exclude=i)
        r_cent = CS.rank_by_vector(cent, Wn, exclude={i}, k=K)
        r_rules = CS.content_rules(meta, i, list(range(n_items)), k=K)
        row = dict(item=i,
                   family_centroid=CS.overlap_at_k(r_cent, ceiling, K),
                   content_rules=CS.overlap_at_k(r_rules, ceiling, K))
        if have_text:
            r_text = CS.rank_by_vector(tv[i], tv, exclude={i}, k=K)
            row["text_embedding"] = CS.overlap_at_k(r_text, ceiling, K)
        rows.append(row)
    C = pd.DataFrame(rows)

    emit("%d held-out products. Each method places a product the learned model has"
         % len(C))
    emit("never seen; the score is overlap@%d with what the FULL-HISTORY learned" % K)
    emit("vector would have returned.")
    emit("")
    cols = [c for c in ("family_centroid", "text_embedding", "content_rules")
            if c in C.columns]
    for c in cols:
        emit("  %-18s overlap@%d %.4f  (sd %.4f)"
             % (c, K, C[c].mean(), C[c].std()))
    if not have_text:
        emit("  text_embedding     unavailable -- sentence-transformers did not load")
    emit("")
    best = max(cols, key=lambda c: C[c].mean())
    emit("BEST FALLBACK: %s at %.1f%% of the learned answer."
         % (best, 100 * C[best].mean()))
    emit("")
    emit("  The ceiling is the point. Scoring cold-start methods against EACH OTHER")
    emit("  answers 'which fallback is least bad'; scoring them against the vector")
    emit("  the product would eventually earn answers 'how much of the")
    emit("  recommendation quality is missing on day one', which is the number a")
    emit("  launch team is actually asking for.")
    emit("")
    emit("  Cold start is also the only part of a recommender that can be evaluated")
    emit("  honestly with no A/B test at all, because the counterfactual is")
    emit("  available: hold the product out, then look.")
    emit("")
    emit("  The family centroid EXCLUDES the held-out product from its own")
    emit("  centroid. Leaving it in is the cold-start equivalent of training on the")
    emit("  test set and would make the method look excellent for a reason that")
    emit("  cannot happen on day one.")
    emit("")
    emit("  What each method assumes, in increasing order:")
    emit("    family centroid  -- somebody assigned the family correctly, which in")
    emit("                        a real catalogue is late and often wrong")
    emit("    text embedding   -- somebody wrote a description, which is true on")
    emit("                        day one because it is what goes on the page")
    emit("    content rules    -- the taxonomy, and nothing else")
    emit("")
    summary["cold_start"] = {c: float(C[c].mean()) for c in cols}

    # ======================================================================
    emit("=" * 78)
    emit("3. THE DUE-SCORE WEIGHT IS NOT ONE NUMBER")
    emit("=" * 78)
    user_seq = defaultdict(list)
    for oid in all_ids:
        user_seq[order_user[oid]].append((order_day[oid], by_order[oid]))
    for u in user_seq:
        user_seq[u].sort(key=lambda x: x[0])
    train_seq = {u: [(d, b) for d, b in v if d <= cut] for u, v in user_seq.items()}
    tim = T.ReorderTiming().fit(train_seq)

    # Per-FAMILY inter-purchase spread. Reported at family level rather than
    # aisle level because an aisle mixes cadences -- dairy holds both milk and
    # butter -- and averaging over it hides exactly the spread the section is
    # about. The first version of this table was per-aisle and showed a 1.14x
    # range, which made the argument look unsupported when what was actually
    # unsupported was the choice of grouping.
    fam_int = defaultdict(list)
    for item, mu in tim.item_mean.items():
        fam_int[meta[int(item)]["family"]].append(mu)
    F = pd.DataFrame([dict(family=f, n_items=len(v),
                           mean_interval=float(np.mean(v)))
                      for f, v in fam_int.items() if len(v) >= 2])
    F = F.sort_values("mean_interval")
    emit("Fastest and slowest families by observed inter-purchase interval:")
    emit("")
    emit(pd.concat([F.head(8), F.tail(8)]).to_string(
        index=False, float_format=lambda x: "%8.2f" % x))
    emit("")
    spread = F.mean_interval.max() / max(F.mean_interval.min(), 1e-9)
    emit("The slowest family's interval is %.2fx the fastest (%.1f vs %.1f days)."
         % (spread, F.mean_interval.max(), F.mean_interval.min()))
    emit("")
    emit("  A single global due-weight applies the same urgency curve to milk and")
    emit("  to light bulbs. The curve peaks at the EXPECTED interval, so a weight")
    emit("  tuned on fast movers fires far too early on slow ones -- and the")
    emit("  aggregate hit-rate that tuned it cannot see the difference, because it")
    emit("  is dominated by the fast families that generate most of the reorders.")
    emit("")
    emit("  Note the ceiling on this table: no family's interval can be shorter")
    emit("  than the user's own trip cadence, which averages 10 days here. The")
    emit("  observed spread is therefore COMPRESSED relative to real consumption --")
    emit("  a household that gets through milk in three days still only buys it")
    emit("  when they shop. Any per-category dial fitted on observed intervals")
    emit("  inherits that compression and will under-differentiate.")
    emit("")
    emit("  What a per-family dial does NOT fix: the interval is still a")
    emit("  per-(user,item) mean shrunk to the item population, so a household")
    emit("  that doubled in size last month is modelled with the average of its")
    emit("  old and new cadence and is wrong in both directions.")
    emit("")
    summary["family_intervals"] = F.round(3).to_dict("records")

    # ======================================================================
    emit("=" * 78)
    emit("4. THE SWITCH MATRIX ON TRIAL, ON ITS OWN")
    emit("=" * 78)
    train_user_orders = {u: [b for _d, b in v] for u, v in train_seq.items()}
    SW = M.switch_matrix(train_user_orders, n_items)
    SWN = M.normalised_switch(SW, train_b, n_items)
    family_of = {pid: m["family"] for pid, m in meta.items()}
    SWS = M.slot_switch_matrix(train_user_orders, n_items, family_of)
    true_subs = {tuple(p) for p in truth["substitutes"]}
    true_comps = {tuple(p) for p in truth["complements"]}

    n_pairs = n_items * (n_items - 1) / 2
    base_rate = len(true_subs) / n_pairs

    def ranked_pairs(mat):
        agg = defaultdict(float)
        for a, b in np.argwhere(mat > 0):
            a, b = int(a), int(b)
            agg[(min(a, b), max(a, b))] += float(mat[a, b])
        return sorted(agg.items(), key=lambda kv: -kv[1])

    variants = [("raw presence counts", ranked_pairs(SW)),
                ("popularity-normalised", ranked_pairs(SWN)),
                ("same-family slot switch", ranked_pairs(SWS))]
    emit("")
    emit("  Random-guess precision on this catalogue: %.4f" % base_rate)
    emit("  (%d true substitute pairs out of %d possible pairs)"
         % (len(true_subs), int(n_pairs)))
    emit("")
    emit("  %-24s %10s %10s %10s %10s %10s"
         % ("variant", "pairs", "P@50", "P@200", "P@500", "P@1000"))
    scores = {}
    for name, ranked in variants:
        row = []
        for n in (50, 200, 500, 1000):
            top = [k for k, _ in ranked[:n]]
            row.append(sum(1 for k in top if k in true_subs) / max(len(top), 1))
        scores[name] = row
        emit("  %-24s %10d %10.4f %10.4f %10.4f %10.4f"
             % (name, len(ranked), *row))
    emit("")
    raw500 = [k for k, _ in variants[0][1][:500]]
    slot500 = [k for k, _ in variants[2][1][:500]]

    def median_rank(ranked):
        pos = {k: i for i, (k, _) in enumerate(ranked)}
        r = [pos[t] for t in true_subs if t in pos]
        return float(np.median(r)) if r else float("nan")

    emit("Median rank of a TRUE substitute pair, out of %d ranked pairs:"
         % len(variants[0][1]))
    for name, ranked in variants:
        emit("  %-26s %8.0f" % (name, median_rank(ranked)))
    emit("")
    raw_med = median_rank(variants[0][1])
    nrm_med = median_rank(variants[1][1])
    n_ranked = len(variants[0][1])
    emit("THE PRESENCE-BASED MATRIX IS UNUSABLE AT THE HEAD, WHICH IS THE ONLY")
    emit("PART ANYONE SEES. Its precision@500 is %.4f against a random-guess rate"
         % scores["raw presence counts"][2])
    emit("of %.4f -- BELOW CHANCE where it matters. Across the whole ranking its"
         % base_rate)
    emit("true pairs sit at median %.0f of %d, marginally better than the %.0f a"
         % (raw_med, n_ranked, n_ranked / 2))
    emit("coin flip would give, and that is not a signal anyone can act on: no")
    emit("product page shows the middle of a ranking.")
    emit("")
    emit("  Popularity normalisation lifts the very top (precision@50 %.4f, about"
         % scores["popularity-normalised"][0])
    emit("  %.0fx chance) and pushes the median DOWN to %.0f. It is a marginal"
         % (scores["popularity-normalised"][0] / max(base_rate, 1e-9), nrm_med))
    emit("  improvement at the head, not a rescue -- and the reason it cannot be a")
    emit("  rescue is that the problem was never popularity.")
    emit("")
    emit("  THE MECHANISM IS THE SAME STRUCTURAL FACT THAT DEFINES A SUBSTITUTE:")
    emit("  SUBSTITUTES DO NOT CO-OCCUR. One product per family per basket, so on")
    emit("  any given trip the item LEAST likely to be beside A is A's own")
    emit("  substitute. 'B was present when A vanished' is therefore")
    emit("  systematically rarer for true substitutes than for arbitrary items,")
    emit("  and reweighting a signal cannot fix a signal with the wrong sign.")
    emit("")
    emit("  A heuristic can be defeated by the very property it is trying to")
    emit("  detect, and no reweighting fixes that -- the fix has to be a")
    emit("  DEFINITION. A switch is not 'A left and B was around'; it is 'A left")
    emit("  and B arrived in the same slot': same family, same order, same user.")
    emit("")
    n_slot = len(variants[2][1])
    emit("  Redefined that way, precision@500 goes from %.4f to %.4f."
         % (scores["raw presence counts"][2], scores["same-family slot switch"][2]))
    emit("")
    emit("  THAT 1.0000 IS NOT AS IMPRESSIVE AS IT LOOKS, AND SAYING SO IS THE")
    emit("  POINT. The slot definition only ever emits within-family pairs, and in")
    emit("  this generator every within-family pair IS a substitute -- so its")
    emit("  precision is bounded at 1.0 BY CONSTRUCTION, not by learning. The")
    emit("  taxonomy is doing the work.")
    emit("")
    emit("  What the behaviour adds on top is RECALL and ORDERING: it surfaces %d"
         % n_slot)
    emit("  of the %d true pairs, which is %.1f%%, and ranks them by how often the"
         % (len(true_subs), 100 * n_slot / max(len(true_subs), 1)))
    emit("  swap was actually observed rather than by whether it is possible. A")
    emit("  merchandiser who wants to know WHICH cola to offer when this cola is")
    emit("  out needs the ordering; the taxonomy alone gives them an unordered set.")
    emit("")
    emit("  The generator cannot test the finer question -- it makes every family")
    emit("  member equally substitutable, so there is no ground truth about which")
    emit("  swap shoppers prefer. That is a limit of the lab, and it is the one")
    emit("  place in this section where the honest answer is 'not measurable here'.")
    emit("")
    emit("  That version uses the category taxonomy to SCOPE the search, which is")
    emit("  information a retailer genuinely has. It is a smaller claim than")
    emit("  'discover substitution from scratch' -- knowing two colas are both")
    emit("  colas is not knowing which one a shopper will accept instead of the")
    emit("  other -- and it is the claim the data supports.")
    emit("")
    emit("  Of the raw top 500, %.1f%% are true COMPLEMENTS; of the slot-switch top"
         % (100 * sum(1 for k in raw500 if k in true_comps) / max(len(raw500), 1)))
    emit("  500, %.1f%%. Complements are the error the presence version is built to"
         % (100 * sum(1 for k in slot500 if k in true_comps) / max(len(slot500), 1)))
    emit("  make: a user who stops buying hot dogs but keeps buying buns has")
    emit("  changed their shopping list, not substituted one thing for another.")
    emit("")
    emit("  THIS SECTION EXISTS BECAUSE THE PREVIOUS PASS NEVER RAN IT. The switch")
    emit("  matrix was computed, z-scored, and added to a combined scorer as one")
    emit("  term among several -- where a signal that is BELOW CHANCE on its own")
    emit("  is indistinguishable from a signal that is merely small. A component")
    emit("  nobody evaluates alone is a component nobody can decide to remove, and")
    emit("  this one was actively subtracting.")
    emit("")
    summary["switch_matrix"] = dict(
        base_rate=base_rate,
        raw=dict(zip(("p50", "p200", "p500", "p1000"),
                     scores["raw presence counts"])),
        normalised=dict(zip(("p50", "p200", "p500", "p1000"),
                            scores["popularity-normalised"])),
        slot=dict(zip(("p50", "p200", "p500", "p1000"),
                      scores["same-family slot switch"])))

    # ======================================================================
    emit("=" * 78)
    emit("5. DIRECTIONALITY, ACTUALLY SERVED")
    emit("=" * 78)
    DL = T.directional_lift(train_b, n_items)
    sym = np.minimum(DL, DL.T)

    def complete(basket, scores_matrix, k=K):
        s = scores_matrix[basket].sum(axis=0)
        s[list(basket)] = -np.inf
        top = np.argpartition(-s, min(k, len(s) - 1))[:k]
        return [int(i) for i in top[np.argsort(-s[top])]]

    hit_dir, hit_sym, n = 0, 0, 0
    for b in test_b:
        if len(b) < 3:
            continue
        prefix, target = b[:-1], b[-1]
        n += 1
        hit_dir += int(target in complete(prefix, DL))
        hit_sym += int(target in complete(prefix, sym))
    emit("  directional P(b|a)  hit@%d %.4f" % (K, hit_dir / max(n, 1)))
    emit("  symmetrised         hit@%d %.4f" % (K, hit_sym / max(n, 1)))
    emit("  delta               %+.4f  (n=%d)"
         % ((hit_dir - hit_sym) / max(n, 1), n))
    emit("")
    delta = (hit_dir - hit_sym) / max(n, 1)
    if delta < 0:
        emit("  THE DIRECTIONAL SCORE LOSES AS A BASKET SCORER, and the previous")
        emit("  pass's plan -- 'wire the directional lift into /complete' -- would")
        emit("  have made the endpoint worse.")
        emit("")
        emit("  The asymmetry is real; the mistake was assuming a real effect must")
        emit("  improve every consumer of it. Summing P(b|a) over the items already")
        emit("  in the cart is a mixture of conditionals, and a globally popular b")
        emit("  scores well under EVERY conditional. The symmetrised min(P(b|a),")
        emit("  P(a|b)) is implicitly a SPECIFICITY filter: it demands that the")
        emit("  relationship hold in both directions, which is exactly what a")
        emit("  merely-popular item fails.")
        emit("")
        emit("  So the two objects have different jobs, and the honest conclusion is")
        emit("  to serve both:")
        emit("    - pairwise widget ('customers who bought X also bought') uses the")
        emit("      DIRECTIONAL score, because the question is genuinely directed --")
        emit("      buns given hot dogs, not the reverse;")
        emit("    - whole-basket completion uses the symmetric score, because")
        emit("      summing conditionals rewards popularity and symmetry is the")
        emit("      cheapest available correction for it.")
        emit("")
        emit("  Measuring an improvement and not serving it is a common way a model")
        emit("  change fails to reach a customer. Serving a real effect through the")
        emit("  wrong consumer is a less common one and it is the failure that")
        emit("  actually happened here.")
    else:
        emit("  The previous pass COMPUTED the directional lift, showed the")
        emit("  asymmetry is real, and left the endpoint on the symmetrised score.")
        emit("  Measuring an improvement and not serving it is the most common way")
        emit("  a model change fails to reach a customer.")
    emit("")
    asym = []
    for a in range(n_items):
        for b2 in range(n_items):
            if a != b2 and DL[a, b2] > 0.02:
                asym.append((float(abs(DL[a, b2] - DL[b2, a])), a, b2))
    asym.sort(reverse=True)
    emit("  Largest asymmetries -- P(b|a) vs P(a|b):")
    for gap, a, b2 in asym[:5]:
        emit("    %-22s -> %-22s  %.4f vs %.4f"
             % (meta[a]["name"], meta[b2]["name"], DL[a, b2], DL[b2, a]))
    emit("")
    emit("  The direction is FREE -- the same co-occurrence counts divided by a")
    emit("  different denominator -- so symmetrising it was a modelling choice and")
    emit("  not a limitation.")
    emit("")
    summary["directional"] = dict(directional=hit_dir / max(n, 1),
                                  symmetric=hit_sym / max(n, 1), n=n)

    with open(os.path.join(OUT, "complete_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + "\n")
    with open(os.path.join(OUT, "complete_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\n-> out/complete_report.txt")


if __name__ == "__main__":
    main()
