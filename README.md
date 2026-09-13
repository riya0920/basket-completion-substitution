# ML-3: Basket Completion & Substitution

**Complete against the spec.** Complements and substitutes separated by the two
item2vec embedding matrices, a sequential model over add-to-cart order, cold start
scored against the ceiling it cannot reach, a per-family timing dial, a switch
heuristic put on trial and found guilty, and a cart service with three endpoints
that deliberately use three different scores.

The two most useful results in this project are both negative.

```bash
python src/generate.py       # ~30s   420 products, 37,595 orders, add-to-cart order
python run_basket.py         # ~2min  the original evaluation
python run_complete.py       # ~7min  the completion pass
uvicorn serve:app --port 8013   #      the cart UI
python -m pytest tests -q    # 66 tests
```

420 products across 13 aisles, 3,000 users, 37,595 orders, 288,020 lines,
59.6% reorders, 415 days.

## The catalogue is no longer 49 products

The hand-authored theme/family skeleton is kept, it carries the meaning of which
products substitute and which co-occur, and is extended programmatically with
more brand variants per family and **filler families that belong to no theme**.
Filler is not noise: it is the honest majority of a real grocery catalogue, and it
is what makes retrieval hard, because a model that only ever sees themed items has
never had to ignore anything.

Two other things the generator now emits, each because a section was unbuildable
without it:

- **Within-basket order.** The trip starter stays first; everything after it
  follows an aisle walk with jitter. Without it, "item2vec ignores sequence" was a
  statement about the model with no measurable consequence.
- **Per-family consumption cadence.** Milk is weekly, light bulbs are quarterly.
  Without it every aisle's measured inter-purchase interval came out within 5% of
  every other, and "the due-score weight should not be one number" was an argument
  with no evidence behind it.

## Does within-basket order carry signal?

| model | hit@10 | MRR |
|---|---|---|
| **sequence (uses order)** | **0.2508** | **0.0932** |
| bag of items (order destroyed) | 0.1279 | 0.0438 |

**Order is worth +0.1229 hit@10 on next-item prediction**: it roughly doubles it.

The control is the *same model* trained on shuffled copies of the *same* baskets:
identical co-occurrence, identical popularity, identical capacity. The only thing
it cannot know is which item came last, so the gap is attributable to order and to
nothing else. That is why the control is a shuffle rather than a different model:
a GRU would have changed the model class at the same time and the delta could not
have been attributed to anything.

**Honest limit: first-order.** The model forgets everything before the last item,
so a basket that is obviously a cookout is represented only by whatever went in
most recently. That is left visible rather than patched with an average over the
basket: averaging would quietly turn it back into a bag-of-items model and this
comparison would stop meaning anything.

## Cold start, scored against the ceiling it cannot reach

60 held-out products. Each method places a product the learned model has never
seen; the score is overlap@10 with what the **full-history learned vector** would
have returned.

| fallback | overlap@10 | assumes |
|---|---|---|
| **family centroid** | **0.2083** | somebody assigned the family correctly |
| text embedding (MiniLM) | 0.1383 | somebody wrote a description |
| content rules | 0.1267 | the taxonomy, and nothing else |

**The best fallback recovers 21% of the learned answer.** That number is the point
of the section. Scoring cold-start methods against *each other* answers "which
fallback is least bad"; scoring them against the vector the product will
eventually earn answers "how much recommendation quality is missing on day one",
which is what a launch team is actually asking.

Cold start is also the only part of a recommender that can be evaluated honestly
with **no A/B test at all**, because the counterfactual is available: hold the
product out, then look.

The family centroid **excludes the held-out product from its own centroid**.
Leaving it in is the cold-start equivalent of training on the test set and would
make the method look excellent for a reason that cannot happen on day one.

## The due-score weight is not one number

Fastest and slowest families by observed inter-purchase interval:

| family | mean interval (days) |
|---|---|
| yoghurt | 16.85 |
| bananas | 17.54 |
| eggs | 17.96 |
| milk_whole | 18.16 |
| … | … |
| plasters | 33.30 |
| soy_sauce | 33.51 |
| light_bulbs | 34.97 |
| vitamins | 35.42 |

**The slowest family's interval is 2.10× the fastest.** A single global due-weight
applies the same urgency curve to milk and to light bulbs, and the curve *peaks*
at the expected interval, so a weight tuned on fast movers fires far too early on
slow ones. The aggregate hit-rate that tuned it cannot see the difference, because
it is dominated by the fast families that generate most of the reorders.

**Note the ceiling on that table**: no family's interval can be shorter than the
user's own trip cadence, which averages 10 days here. The observed spread is
therefore *compressed* relative to real consumption, a household that gets
through milk in three days still only buys it when they shop, so any per-category
dial fitted on observed intervals inherits that compression and will
under-differentiate.

> Reported per **family**, not per aisle. The first version of this table was
> per-aisle and showed a 1.14× range, which made the argument look unsupported
> when what was actually unsupported was the choice of grouping: an aisle mixes
> milk and butter.

## The switch matrix on trial, and found guilty

Random-guess precision on this catalogue: **0.0064** (567 true substitute pairs
out of 87,990 possible).

| variant | pairs | P@50 | P@500 | P@1000 | median rank of a true pair |
|---|---|---|---|---|---|
| raw presence counts | 83,144 | 0.0000 | **0.0020** | 0.0010 | 33,603 / 83,144 |
| popularity-normalised | 83,144 | 0.0400 | 0.0260 | 0.0290 | 53,072 |
| **same-family slot switch** | **538** | **1.0000** | 1.0000 | 1.0000 | **268** |

**The presence-based matrix is unusable at the head, which is the only part anyone
sees.** Its precision@500 is *below* the random-guess rate. Across the whole
ranking its true pairs sit marginally better than a coin flip, and that is not a
signal anyone can act on, because no product page shows the middle of a ranking.

Popularity normalisation lifts the very top (P@50 ≈ 6× chance) and pushes the
median *down*. A marginal improvement at the head, not a rescue, and it cannot be
a rescue because **the problem was never popularity**:

> **Substitutes do not co-occur.** One product per family per basket, so on any
> given trip the item *least* likely to be beside A is A's own substitute. "B was
> present when A vanished" is therefore systematically rarer for true substitutes
> than for arbitrary items. A heuristic can be defeated by the very property it is
> trying to detect, and reweighting cannot fix a signal with the wrong sign.

The fix is a **definition**, not a weighting. A switch is not "A left and B was
around"; it is "A left and B arrived in the same slot": same family, same order,
same user.

**And that 1.0000 is not as impressive as it looks, which is the second half of
the finding.** The slot definition only ever emits within-family pairs, and in this
generator every within-family pair *is* a substitute, so its precision is bounded
at 1.0 **by construction**. The taxonomy is doing the work. What the behaviour adds
on top is **recall and ordering**: it surfaces 538 of the 567 true pairs (94.9%)
and ranks them by how often the swap was actually observed rather than by whether
it is possible. A merchandiser who needs to know *which* cola to offer when this
cola is out needs the ordering; the taxonomy alone gives them an unordered set.

This section exists because the previous pass never ran it. The switch matrix was
computed, z-scored, and added to a combined scorer as one term among several,
where a signal that is below chance on its own is indistinguishable from a signal
that is merely small. **A component nobody evaluates alone is a component nobody
can decide to remove**, and this one was actively subtracting.

## Directionality: real, and served through the wrong consumer

| scorer | hit@10 |
|---|---|
| directional `P(b\|a)` | 0.2151 |
| symmetrised `min(P(b\|a), P(a\|b))` | **0.2240** |

**The directional score loses as a basket scorer**, so the previous pass's own plan,
"wire the directional lift into `/complete`", would have made the endpoint
worse.

The asymmetry is real. The mistake was assuming a real effect must improve every
consumer of it. Summing `P(b|a)` over the items already in the cart is a mixture of
conditionals, and a globally popular *b* scores well under **every** conditional.
The symmetrised minimum is implicitly a **specificity filter**: it demands the
relationship hold in both directions, which is exactly what a merely-popular item
fails.

So the two objects have different jobs, and both are served:

- the **pairwise widget** ("customers who bought X also bought") uses the
  directional score, because the question is genuinely directed: buns given hot
  dogs, not the reverse;
- **whole-basket completion** uses the symmetric score.

Measuring an improvement and not serving it is a common way a model change fails
to reach a customer. Serving a real effect through the *wrong consumer* is a less
common one, and it is the failure that actually happened here.

## The cart service

`uvicorn serve:app --port 8013`

Three endpoints, three different scores, on purpose:

- `POST /complete`: symmetric lift, for the reason above. Reports which scorer it
  used and whether it fell back, because a cart whose every item is a brand-new
  SKU has no co-occurrence at all and the page must not silently show taxonomy
  neighbours as if they were learned recommendations.
- `GET /next`: the sequence model, keyed on the last item added. States that
  limitation in its own response.
- `GET /substitute/{id}`: ranked by **observed switches** within the family, not
  by embedding similarity. The taxonomy says which products *could* substitute;
  the switch evidence says which one shoppers actually accept, and only the second
  is a recommendation. Falls back to price and pack proximity when there is no
  switch evidence, and says so.
- `GET /why`: both conditionals for a pair, whether they share a family, and how
  many switches were observed. A merchandiser asking "why is this suggested" wants
  the evidence for *this pair*, not a global feature-importance table.

## Bugs this pass caught

- **The reorder rate collapsed to 0.49** when the catalogue widened from 49 to 420
  products, because users were touching a new long-tail family every trip and
  never returning. Real grocery is 60%+ reorders and the reason is *repertoire*:
  filler draws are now weighted by a per-user Dirichlet as well as by consumption
  cadence, which brings it to 0.596.
- **The content-rules fallback could return fewer than k items** for a product
  whose family and aisle are both small: a blank recommendation slot on a live
  page. A third tier now fills from the whole catalogue by price and pack
  proximity: a weak answer, and weaker than an empty one is not.
- **The per-category timing argument was reported at the wrong grouping** and
  looked unsupported at 1.14× when the family-level spread is 2.10×.

## Real Instacart: one claim survived and one did not

*"No Instacart data. It is not downloadable here."* **False**, and it took five
passes to check. The competition download 403s until you accept its rules, but the
whole dataset is republished as a plain Kaggle **dataset**, and datasets carry no
rules gate.

```bash
python run_instacart.py
```

> **Provenance:** `psparks/instacart-market-basket-analysis`, a third-party
> republication rather than the official archive, not diffed against it, because
> the official one is the thing that is gated. 35,767 baskets, 27,440 products,
> 134 aisles, mean basket 11.0.

### Order carries signal: six times less of it

| model | hit@10 | MRR |
|---|---|---|
| sequence (uses order) | **0.0670** | 0.0274 |
| bag of items (order destroyed) | 0.0474 | 0.0185 |

**Real delta: +0.0196. Generator delta: +0.1229.**

Same control, same model, same metric. The direction holds and the magnitude does
not. The generator builds baskets by walking an aisle order with a per-family
cadence, so "what came last" is nearly deterministic in it; **its +0.1229 was an
upper bound on a real effect rather than an estimate of one.**

### The substitute mechanism is refuted

This project's sharpest finding rested on a mechanism:

> *"Substitutes do not co-occur. One product per family per basket, so on any
> given trip the item least likely to be beside A is A's own substitute."*

In the generator that is true **by construction**. On real shoppers:

| group | mean lift vs chance | co-occurring |
|---|---|---|
| **near-identical variants** | **11.87×** | 33.9% |
| same aisle, different product | 3.90× | 31.6% |
| different aisles | 1.34× | 17.2% |

**The more similar two products are, the more they co-occur.** Real shoppers buy
two yoghurt flavours, two sizes of the same milk, the same crisps in two bags. The
generator's one-per-family rule isn't a simplification of that behaviour; it's
the reverse of it. Holds at every similarity threshold tried (0.5–0.7), with
same-aisle pairs enumerated exhaustively rather than sampled.

**So the generator built in the property it then discovered.** The presence-based
switch matrix scored below chance here *because* the generator guaranteed
substitutes never share a basket, and the explanation offered for it is a fact
about the simulator, not about shopping.

**What this does and does not overturn.** It does not make the slot-switch
definition wrong: *"A left and B arrived in the same slot"* is still sharper than
*"B was nearby"*. What it removes is the **reason** given for the presence-based
matrix failing. On real data a presence-based matrix would rank near-identical
variants *high*, so it might work rather well, and this project never tested that,
because its corpus could not.

> **The proxy is doing real work in the argument.** A "variant pair" is two
> products in one aisle whose names share most of their words. That catches
> different sizes of the same thing and misses substitutes branded differently.
> Instacart has no ground-truth substitute set, which is still why the generator
> exists.

## What is deliberately not here

- **Instacart is used for two experiments, not as the corpus.** The embeddings,
  the cold-start ladder, the timing dial and the cart service all still run on the
  generator; only the order claim and the co-occurrence mechanism were re-tested.
- **The generator's basket-construction rule is now known to be wrong**, and it
  has not been changed. One item per family per basket is the reverse of real
  behaviour, so every result that depends on it, the switch-matrix section most
  of all, should be read as a statement about this simulator.
- **The generator makes every family member equally substitutable**, so there is
  no ground truth about *which* swap a shopper prefers. That is the one place in
  the switch section where the honest answer is "not measurable here".
- **The sequence model is first-order.** No session context, no transformer, no
  attention over the basket.
- **No A/B or interleaving story** for the recommendations. Cold start is the only
  part evaluated counterfactually; everything else is offline hit-rate.
- **The due-score dial is per family**, not per user: a household whose size
  changed last month is modelled with the average of its old and new cadence and
  is wrong in both directions.
