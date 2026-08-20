# ML-3 — Basket Completion & Substitution Engine

**Roughly 50% of the spec.** Complements and substitutes treated as *opposite*
relations with the conflation measured - plus the four things the first pass
named as missing: reorder **timing** (the spec's own grill question), directional
complements, price- and pack-aware substitution, and a cold-start path. No API,
no cart UI; what remains is named at the bottom.

```bash
python src/generate.py      # ~20s  Instacart-shaped orders, now with order DAYS
python run_basket.py        # ~6min
python -m pytest tests -q   # 29 tests, ~75s
```

## The data

Instacart's dataset isn't downloadable offline, so `src/generate.py` reproduces
the three structural properties that make it good for this problem — **baskets**
(not sessions), **reorder flags**, **aisle/department taxonomy** — and plants the
relations the project is about:

| | |
|---|---|
| 3,000 users, 37,725 orders, 49 products | mean basket 7.5 |
| **reorder rate 71.7%** | grocery-realistic |
| 26 true substitute pairs | same *family*, never share a basket |
| 180 true complement pairs | cross-family, co-occur via 6 meal themes |

Split is **temporal** — each user's last order is held out. A random basket split
would let a user's future train the model that predicts their past, which on
71%-reorder data is a very effective way to cheat.

## The central idea

item2vec learns **two** matrices, and they carry different information:

```
W[a] · C[b]      high when a and b appear TOGETHER    → COMPLEMENTS
cos(W[a], W[b])  high when a and b appear in the SAME KIND of basket,
                 whether or not they ever appear together → SUBSTITUTES
```

Most portfolio projects keep only `W`, call cosine similarity "related products",
and thereby conflate *hot dogs + buns* with *Coke vs Pepsi*. Keeping both
matrices is what separates them, and gensim hands you the input vectors and drops
the context vectors — so half the thesis lives in the ones it drops.

## The conflation, measured

| relation | co-occurrence | lift | W·Cᵀ (1st order) | cos(W,W) (2nd order) |
|---|---|---|---|---|
| TRUE complements | 1439.7 | 1.68 | **+0.69** | −0.19 |
| TRUE substitutes | **0.0** | **0.00** | −2.35 | **+0.92** |
| unrelated | 690.7 | 0.88 | +0.04 | +0.04 |

Co-occurrence and lift are high for complements and **lower for substitutes than
for unrelated pairs**. That's not a metric failure, it's the definition: you buy
hot dogs *and* buns; you buy cola A *or* cola B. Substitutes are
**anti-correlated within a basket**.

So any method built on same-basket co-occurrence — lift, association rules, a
one-matrix item2vec — ranks substitutes as the *least* related items in the
catalogue. Across all 26 true substitute pairs, lift fails to place the
substitute in the top-5 for **26 of 26 (100%)**. Second-order similarity places
it in the top-5 for **26 of 26**.

The hot-dogs-and-buns table, straight from the output:

```
cola_a (soft_drinks):
  co-occurrence lift  -> lemon_lime_b, salsa_a, salsa_b, lemon_lime_a, chips_b
  i2v similarity      -> cola_c, cola_b, mustard_a, coffee_b, flour_a
  TRUE substitutes    -> cola_b, cola_c

hot_dogs_a (meat):
  co-occurrence lift  -> ketchup_a, mustard_a, ketchup_b, hotdog_buns_a, hotdog_buns_b
  i2v similarity      -> hot_dogs_b, spaghetti_a, sandwich_bread_b, eggs_a, ...
  TRUE substitutes    -> hot_dogs_b
```

Asked for a *replacement* for hot dogs, the lift model offers ketchup and buns.

**Where the enforcement lives** (a screener will ask): `/complete` and
`/substitute` use different scores from the same model — `W·Cᵀ` and `cos(W,W)`.
Complete-the-basket never consults the similarity matrix, and additionally masks
everything already in the cart *and everything in the same product family as a
cart item*. That family mask is one line and makes the failure mode structurally
impossible rather than merely unlikely.

## Basket completion, segmented

| method | hit@10 new-to-user | hit@10 reorder | hit@10 ALL |
|---|---|---|---|
| popularity | 0.300 | 0.370 | 0.362 |
| co-occurrence lift | 0.555 | 0.625 | 0.617 |
| item2vec | **0.579** | 0.659 | 0.650 |
| item2vec + reorder | 0.570 | **0.684** | **0.671** |

Held-out items: 2,663 reorder, 337 new-to-user.

Predicting a **reorder** is close to free — the user buys milk every week and
replaying their history scores well. The **new-to-user** column is the only place
discovery happens. An aggregate hit-rate is a weighted average of an easy problem
and a hard one, dominated by whichever is more frequent, and in grocery that's
always the easy one.

Reorder personalisation is the largest lift on the reorder segment (+0.025) and
**costs** a little on new-to-user (−0.009) — exactly what it should do, and the
reason to read the columns separately rather than celebrate the aggregate it
inflates.

## Substitution scored

166 labelled pairs (26 substitute / 70 complement / 70 neither):

| scorer | precision@n | AUC |
|---|---|---|
| co-occurrence lift (naive) | 0.000 | 0.000 |
| same aisle only | 0.423 | 0.902 |
| i2v second-order similarity | 1.000 | 1.000 |
| combined (sim + aisle − co-occur + switch) | 1.000 | 1.000 |

**Discount the 1.0000.** The generator enforces one-product-per-family-per-basket
as a hard constraint, so substitutes co-occur exactly zero times and the
separation is perfect by construction. Real grocery data is softer — households
stock up on two brands, buy different sizes, or contain people with different
preferences — so genuine substitutes *do* share baskets sometimes. Read this as:
the **signal is the right one**, the **effect size is an artifact of the lab**.

The annotation guideline (what I'd hand a human annotator) is in the report.
Labels here come from planted structure, not annotators, so they're ground truth
for the simulation and not evidence about real shoppers.

## Serving and the P&L

| endpoint | p50 | p95 |
|---|---|---|
| `/complete` | 0.163 ms | 0.267 ms |
| `/substitute` | **0.038 ms** | **0.049 ms** |

`/substitute` is the path with a human waiting — a shopper is at the shelf and
the item is gone. So the ranked substitute list is **precomputed per item
offline** and the online path only re-scores 20 candidates against user history.
The expensive model never runs there. Fallback if the model is unavailable: the
precomputed list is a static artifact, degrading to same-family-most-popular —
a worse answer, not a blank screen.

**Basket save** — held-out GMV $147,416, 1,386 OOS events at a 6% rate, $9,150
lost with no substitution offered:

| acceptance | revenue lost | saved | % of GMV |
|---|---|---|---|
| 40% | $5,353 | $3,797 | 2.58% |
| 55% | $3,947 | $5,203 | 3.53% |
| **70%** | $2,704 | $6,446 | 4.37% |
| 85% | $1,429 | $7,721 | 5.24% |

I attack my own 70% assumption in the report: it isn't one number (category
matters more than the model); it isn't exogenous (acceptance depends on the
substitute's quality, which is the thing being evaluated, so a fixed rate makes a
bad model look identical to a good one); the counterfactual is wrong ("item
removed" ignores abandonment and retention, so these are an **upper** bound on
saving and a **lower** bound on harm); and it's directly measurable from
in-app accept/reject logs, which this project doesn't have.

## Second pass: four gaps the first pass named

### Reorder timing - and the aggregate that lies

The spec asks: *"reorder prediction is easy - the user buys milk weekly. Where's
the modelling value?"* The first pass answered in prose (timing, basket context,
the discovery margin) and then built a model with **no notion of time**, so it
could rank *what* a user reorders and had nothing to say about *when*.

The generator now emits order **days** (per-user cadence, 4-16 day mean), and a
hazard model fits per-(user, item) inter-purchase intervals shrunk toward the
item population - 53,717 pairs with at least one observed interval.

| method | new-to-user | reorder | ALL |
|---|---|---|---|
| item2vec | 0.5931 | 0.6560 | 0.6493 |
| item2vec + reorder prior | 0.5836 | 0.6873 | 0.6763 |
| **item2vec + TIMING** | **0.3312** | **0.7387** | 0.6957 |

**The aggregate is up and the product is worse.** Timing gains +0.051 on reorders
and loses **-0.252** on new-to-user - most of the discovery slot disappearing.
The model has become a **shopping list**: excellent at telling you you're nearly
out of milk, useless at telling you anything you didn't already know.

That is the failure this section opens by describing, arrived at from the other
direction. Nagging isn't caused by surfacing reorders too often; it's caused by
surfacing them *instead of everything else*.

The weight on the due score is the dial, and it's shown rather than asserted:

| due weight | reorder | new-to-user | ALL |
|---|---|---|---|
| 0.0 | 0.6560 | **0.5931** | 0.6493 |
| 1.0 | 0.7268 | 0.5205 | **0.7050** |
| 3.0 | 0.7387 | 0.3312 | 0.6957 |
| 8.0 | 0.6791 | 0.1483 | 0.6230 |

Tuning on the aggregate picks 1.0 and quietly sells the discovery slot to the
reorder slot. Whether that's the right trade is a **product** decision - reorder
hits convert better per impression, discovery hits grow basket breadth - and it
is not a decision an offline hit-rate can make. What the model owes a PM is this
table, not a single tuned number.

**Why the due score isn't monotone in recency**, which is the part worth arguing
about: it *peaks* at the expected interval and decays either side. Buying milk two
days after the last carton is unlikely; buying it twenty days after is *also*
unlikely, because the user probably bought it elsewhere. A "more time elapsed =
more likely" recency feature gets that second case exactly backwards - and it's
the feature most reorder models actually use. A test asserts the non-monotonicity.

### Complements are directional

The first pass symmetrised the complement score and said so in a comment. That's
wrong in a way that matters commercially: `P(buns | hot dogs)` is high because
buns are what you need once you have hot dogs, while `P(hot dogs | buns)` is
lower because buns go with other things.

`directional_lift` computes `P(b | a)` properly. The direction is **free** - the
same co-occurrence counts divided by a different denominator - so throwing it
away was a modelling choice, not a limitation. A cart-completion widget should
suggest b to an a-buyer far more readily than the reverse when the asymmetry is
large, and a symmetric score cannot express that.

### Price- and pack-aware substitution

Ranking substitutes on embedding similarity alone is the first two things a real
shopper checks away from being useful: a shopper whose $3 pasta sauce is out does
not want the $11 one, and someone who wanted a 500g bag does not want the 2kg
sack. Both are "the same product" to a distributional model.

Penalties are **multiplicative and bounded**, so a close price match can never
*outrank* a genuinely dissimilar item - it can only reorder items that were
already close. **Price should break ties among substitutes, not create
substitutes**, and a test asserts an additive term's failure mode doesn't occur.

### Cold start

A new SKU has no co-occurrence and no useful vector, so the distributional method
has nothing to say - which is the honest position and the reason a content
fallback exists: same family first, then same aisle, ranked by price and pack
proximity. It is strictly worse than the learned answer and it's what you serve
on day one of a SKU's life. Every recommender needs this path and most portfolio
projects skip it, because an offline eval never contains an item the model hasn't
seen.

## The other ~50% - what is still NOT here

- **No API and no cart UI.** The endpoints are Python functions; the spec asks
  for a demo cart that exercises both.
- **No sequential model.** item2vec still ignores within-basket order; timing is
  now modelled but as a per-item hazard, not a sequence.
- **49 products.** Enough to demonstrate the mechanisms, far too few to say
  anything about catalogue-scale retrieval or ANN indexing.
- **Directionality is computed but not wired into `/complete`** - the endpoint
  still uses the symmetrised score, so the asymmetry is measured and not yet
  served.
- **The due-score weight is not fitted per user or per category.** Milk and
  laundry detergent have very different cadences and the dial is global.
- **The switch matrix is computed and barely used** - it contributes one
  z-scored term to the combined scorer and is never validated on its own.
- **Cold start is content-only** - no vendor metadata, no image or text
  embedding, no borrowing a vector from the family centroid.
- **No A/B or interleaving story** for any of it.
