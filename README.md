# ML-3 — Basket Completion & Substitution Engine

**This is not deployable.** It is the first ~20% of the spec: complements and
substitutes treated as *opposite* relations, with the conflation measured rather
than described. No API, no cart UI. Missing 80% at the bottom.

```bash
python src/generate.py      # ~15s  Instacart-shaped orders with planted truth
python run_basket.py        # ~4min
python -m pytest tests -q   # 12 tests, ~50s
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

## The other 80% — what is NOT here

- **No API and no cart UI.** The endpoints are Python functions; the spec asks
  for a demo cart that exercises both.
- **No sequential/basket-aware model.** item2vec ignores within-basket order and
  inter-purchase timing entirely; the spec's "where's the modelling value in
  reorder prediction" answer is *timing*, and timing is not modelled.
- **49 products.** Enough to demonstrate the mechanism, far too few to say
  anything about catalogue-scale retrieval, cold start, or ANN indexing.
- **No cold start.** Every product has training data.
- **No directionality.** `complement_score` is symmetrised; in reality "buns
  given hot dogs" and "hot dogs given buns" are different propensities.
- **Substitution ignores price and pack size**, which are the first two things a
  real shopper checks.
- **The switch matrix is computed and barely used** — it contributes one
  z-scored term to the combined scorer and is never validated on its own.
- **No A/B or interleaving story** for any of it.
