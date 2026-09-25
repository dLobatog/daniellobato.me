# Ranking Family: Causal Teaching Pass

Scope: the eight existing ranking-family concepts. The shared lesson API, native
disclosures, math/code panels, and optional quiz slot are unchanged. No new
libraries, shared shell edits, or Pretext changes are required. Scores and labels
are deterministic toy data, not a production dataset or performance claim.

## Matrix Factorization (`matrix-factorization`)

- **Concrete change:** the main scene now follows the observed Morgan-A click
  through the shared user vector to the unexposed Morgan-D prediction. It no
  longer leads with six unexplained changing percentages. D's vector remains
  visibly fixed; the full matrix and other predictions are secondary.
- **Action:** **Learn from this click** performs the existing simultaneous
  logistic-factor update. An observed non-click gets its corresponding action
  label. An unexposed pair cannot be trained as a negative.
- **Equation linkage:** the acted-on gradient is retained from the pre-update
  factors. Highlighted user coordinates enter D's substituted dot product. The
  displayed attribution is `delta score(D) = delta p dot q_D`; its two coordinate
  contributions sum to the actual score difference. The scene distinguishes the
  logit from its sigmoid click probability.
- **Transfer question:** if the user vector were frozen and D's vector remained
  unchanged, would learning only A change Morgan-D? No. Shared parameters must
  actually change for transfer to occur.
- **Limitations:** two latent dimensions, exposed binary labels, one-pair SGD,
  fixed learning rate and L2 coefficient. No semantic names are assigned to
  latent axes. Generalization does not establish that an unexposed item is liked.

## Two-Tower Retrieval (`two-tower`)

- **Concrete change:** query and item branches display the operands feeding the
  inspected similarity. A request change retains its previous query and score,
  while the cached item encoding stays fixed. The initial selected item is A;
  the first request switch changes A from rank 1 to rank 6 rather than relying
  on a nearly invisible score perturbation.
- **Action:** **Try request: Optimize an RL policy** encodes a different real toy
  feature vector and recomputes the catalog scores and order. Later requests
  cycle through the three existing inputs; this is not an explanatory slide.
- **Equation linkage:** highlighted query coordinates multiply the selected
  item's fixed coordinates directly beneath the two encoder branches. Before
  and after similarity and rank use the same inspected item. Cosine mode labels
  unit vectors and distinguishes normalization from an encoder/cache change.
- **Transfer question:** if the item encoder consumed the live query, could its
  output still be cached once for all queries? Not generally.
- **Limitations:** fixed linear encoders, six items, exact search rather than a
  simulated ANN index. Similarity is not a click probability. An item-encoder
  update requires refreshing its cache; compatible query-only updates may not.

## Ranking Objectives (`rank-objectives`)

- **Concrete change:** learning retains the exact pre-update scores, objective,
  pair, and cutoff. The causal equation explains the gradient just applied,
  rather than accidentally attributing the last movement to the newly
  recomputed next gradient. Untrained items are subdued in pairwise mode;
  before/after rank markers remain visible, including unchanged ranks.
- **Action:** **Learn this preference** updates the selected pair. Pointwise and
  listwise alternatives use **Learn from these labels**, since their updates
  involve the complete slate. Manual swaps are explicitly identified as swaps,
  not optimization steps.
- **Equation linkage:** substituted margin operands produce the preferred
  score's gradient, followed by `new score = old score - 0.7 * gradient`.
  Lambda mode highlights the detached swap-NDCG multiplier. Other modes show
  the actual probability-minus-target term and their reduction convention.
- **Transfer question:** add the same constant to every score. Pairwise loss
  and NDCG stay fixed, while pointwise BCE generally changes.
- **Limitations:** independently trainable logits, not a shared neural ranker;
  the listwise target is normalized exponential gain. Lambda weighting is a
  local rank-weighted gradient, not differentiation through sorting.

## Retrieval Funnel (`retrieval-funnel`)

- **Concrete change:** candidate-set changes and displayed-slot changes are
  compared explicitly. Recovering D highlights D's new membership and A's
  displacement; unaffected catalog entries are subdued. The oracle ceiling
  and relevant-candidate count update with the actual candidate set.
- **Action:** **Let the ranker see D** unions the existing complementary source.
  Removing it reverses membership. Adding it when D is already retrieved does
  not claim extra recall.
- **Equation linkage:** the highlighted numerator in candidate recall uses
  actual relevant survivors over the fixed three relevant catalog items. The
  before/after oracle NDCG ceiling retains the full-catalog ideal denominator.
  Grades 0-3 and the relevance threshold of 2 remain visible.
- **Transfer question:** if reranking becomes perfect but the same shortlist
  omits D, can D be served? No; ordering cannot invent a missing candidate.
- **Limitations:** the extra source is a deterministic candidate union, not a
  lexical search-engine simulation. The oracle is diagnostic, not deployable;
  the toy contextual score does not use evaluation grades as features.

## Cold Start (`cold-start`)

- **Concrete change:** each replayed record exposes the exact additions to
  clicks and shown counts. The selected item's posterior is compared with its
  value immediately before that event. Success, shown non-click, and
  non-exposure have different visible consequences; other items are secondary.
- **Action:** **Next exposure record** updates the evidence and follows that
  event's item. Previous rewinds the same deterministic log.
- **Equation linkage:** `(4 * prior + previous clicks + added clicks) /
  (4 + previous shown + added shown)` highlights the actual new evidence. A
  non-exposure displays zero additions and an unchanged estimate.
- **Transfer question:** replace a non-exposure record with a shown non-click.
  The denominator grows without a success, so the posterior mean falls.
- **Limitations:** Beta-Bernoulli shrinkage for a fixed user segment, four prior
  pseudo-observations, not a collaborative embedding or causal CTR estimator.
  Outcomes still depend on exposure and position. Priors are synthetic.

## Threshold Metrics (`threshold-metrics`)

- **Concrete change:** the scene isolates the records that crossed the cutoff,
  names their confusion-cell transitions, and subdues unaffected records. Cell
  counts show before/after values. The first admission is record 7: TN to FP,
  precision 4/6 to 4/7, and recall unchanged at 4/6.
- **Action:** **Admit the next impression** moves the threshold across one
  actual record. Clicking a score can cross several; the explanation then
  reports all changed records instead of claiming a single crossing.
- **Equation linkage:** the selected-count precision fraction highlights the
  false-positive term and compares it with the previous fraction. TP and FP
  changes explain why recall did or did not change.
- **Transfer question:** increase false-negative cost while holding the cutoff
  fixed. Precision and recall do not change until decisions change.
- **Limitations:** twelve exposed held-out examples, artificial prevalence,
  and hypothetical downstream selection. No-selection precision is undefined.
  The sample's minimum decision cost is not a recommended deployment threshold.

## Calibration (`calibration`)

- **Concrete change:** the scene follows one immutable click label through a
  temperature transform. Initially record 5 moves from probability .68 in bin
  3 to about .593 in bin 2 when T becomes 2. The scene distinguishes changed
  bin composition from changed outcomes; unrelated bins and records are subdued.
- **Action:** **Make the scores less confident** transforms the actual logits
  and recomputes probabilities, bin memberships, ECE, and Brier score. The
  subsequent actions sharpen or restore the same scores.
- **Equation linkage:** the inspected record's raw logit and highlighted
  temperature are substituted into `sigmoid(logit / T)` next to its before/after
  probability and bin. Predicted and observed click-rate legends stay visible.
- **Transfer question:** with a decision threshold exactly .5, can positive T
  change binary decisions? No; logit signs stay fixed, even if calibration changes.
- **Limitations:** temperature is not fitted. Twelve examples and bin-sensitive
  ECE do not establish population calibration. Brier includes discrimination.
  Displayed numbers are rounded; computations use unrounded values.

## Ranking Metrics (`ranking-metrics`)

- **Concrete change:** a move attributes DCG change to both the selected result
  and the displaced neighbor. The first D-up action adds 3.5 from D but removes
  1.5 from A: net DCG gain 2, not 3.5. Unaffected results are subdued and both
  changed contributions remain visible on their ranked rows.
- **Action:** **Move D up one rank** swaps actual positions without modifying
  grades. Direct item inspection changes which result the next move affects.
- **Equation linkage:** highlighted selected-item contribution changes and the
  neighbor's change sum to delta DCG. Dividing by unchanged IDCG gives the exact
  delta NDCG. Before any move, the selected gain and cutoff discount are shown.
- **Transfer question:** swap two items wholly below K. NDCG@K stays fixed
  because both discounts are zero, even if their grades differ.
- **Limitations:** four fully judged items, exponential gains, one query.
  Displayed RR is not an average across queries. Unjudged results and click
  propensity correction are outside this example.

## Verification

Run `node --test atelier/tests/ranking-labs.test.cjs`.

The owned suite checks analytic gradients, immutable transitions, exact
before/after attributions, frozen-vector and score-shift counterfactuals,
candidate ceilings, missing-exposure handling, cutoff crossings, temperature
bin crossings, and displaced-neighbor DCG changes. First-view checks require
one enabled primary action, visible highlighted numeric causal linkage, a
populated mechanism, and secondary calculation tables. All 42 tests pass;
`node --check atelier/ranking-labs.js` also passes.

## Rendered Review And Follow-up

Reviewed actual initial and after-action images at both 1440 and 390 pixels
for every owned concept, first in `/tmp/atelier-causal-teaching-final` and then
in the isolated rerender `/tmp/atelier-causal-ranking-final`. The screenshot
section name for `threshold-metrics` is `thresholds`. The three chapter folders
are `recommendation-depth`, `metrics-eval`, and `systems-retrieval`.

Also reviewed all available parent bottom captures: all eight at 390 pixels,
and six at 1440 pixels. Desktop matrix-factorization and cold-start bottom
files were absent, so those are not claimed as reviewed. The parent's desktop
tower capture had the earlier D/query transition; the final isolated desktop
capture verifies the current A rank 1-to-6 transition. Initial 320-pixel rank
objectives and ranking-metrics captures were additionally visually inspected.

Concrete fixes after the parent screenshots, in the owned JS and CSS only:

- Ranking objectives: move the ordered score cards before the equation; use
  a compact two-column reading order and remove redundant rank/status labels.
  Keep both before/after ranks, score rails, and next gradient directions.
- Retrieval: put the selected item's complete catalog-to-retrieval-to-served
  path directly below the action. Full candidate inspection remains interactive
  inside a disclosure, with grade/relevance legends still visible outside it.
- Thresholds: place the boundary record and four actual confusion cells before
  the formula. Keep all twelve selectable records in an optional disclosure.
- Calibration: move actual predicted/observed bin bars ahead of the derivation.
  Outcome chips and the fixed-label explanation remain available in the stage.
- Ranking metrics: put all four ordered contributions before the equation and
  retain an explicit cutoff/dashed-border legend. Shorten repeated labels.
- Across concepts: remove whole-node opacity fading. Context uses readable
  muted text and reduced font emphasis instead of looking disabled.

The final isolated run completed 32 checks (eight concepts at 1440, 390, 320,
and 720 pixels) with zero automated errors. It checks widths, card clipping,
primary-action changes, disclosure toggles, math rendering, quiz responses,
and modal closing. These automated checks supplement, not replace, the image
review above. No overlap was seen in the reviewed final before/after images.
Narrow scenes scroll vertically; formulas are not guaranteed to fit into a
single 900-pixel viewport. No shared browser or parent report was modified.
