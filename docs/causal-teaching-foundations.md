# Causal Teaching: Foundations and Linear Algebra

Scope: `atelier/foundations-labs.js`, `atelier/foundations-labs.css`, and
`atelier/tests/foundations-labs.test.cjs`. No shared shell, other family, or
public HTML changes. All examples are small computed fixtures, not measured
production results.

## Interaction Contract

Every initial stage supplies its own question and one primary action. Native
disclosures retain optional settings and full arithmetic; there are no summary
action handlers. The reducer retains only the immediately previous state for
comparisons. Initial actions change a probability boundary, population, surviving
subset, inspected outcome, gradient update, vector geometry, or matrix component.
They do not merely advance explanatory prose.

Selected formula terms use an orange underline, not browser-default yellow.
Labels occupy a separate line with a colon. Essential axis units and legends
remain outside disclosures. Muting applies to shapes and connecting lines, not
whole nodes containing labels. Direct SVG controls retain keyboard/touch access.

## Distribution

- Causal change: a routing model's selected logit gains 1 while the same sampler
  draw stays at `u=0.8`. Retrieval mass grows from 0.6652 to 0.8438; the sampled
  route changes from ranking to retrieval. Dashed boundaries retain the old map.
- Action: **Raise this score**, reversible; selecting a region changes the target.
- Equation linkage: the selected `exp(2)` becomes `exp(3)` in the shared softmax
  denominator. Probability widths and inverse-CDF selection use those exact values.
- Transfer question: adding 5 to every logit changes neither probability nor the
  result at a fixed draw, because the shared exponential factor cancels.
- Limitations: a seeded inverse-CDF example is not a claim that draws equal 0.8.
  Optional continuous density remains distinct: quarter-interval area is `b^2-a^2`,
  not density height; optional sample counts reset when scores change.

## Expectation

- Causal change: switch arrival weights from `[0.8,0.2]` to `[0.3,0.7]` while
  conditional mean losses remain fixed. Ten of twenty cohort units change segment;
  outlines identify them. Model A's risk rises from 0.24 to 0.59 and it loses to B.
- Action: **Send more new users**; cohort selection highlights its weighted term.
- Equation linkage: `0.8*0.1 + 0.2*0.8` becomes `0.3*0.1 + 0.7*0.8`, in nats
  per arrival. Both model risks remain visible for the current mix.
- Transfer question: if A beats B in both segments, changing only nonnegative
  mixture weights cannot make B win.
- Limitations: fixed segment-conditional mean log-losses; no within-segment
  variance, estimation error, or changed segment behavior is simulated.

## Bayes

- Causal change: start with 100 spam and 9,900 non-spam emails. Applying the filter
  keeps 90 and 198 respectively; discarded counts are struck out. Renormalizing
  uses these 288 survivors as the new whole, making spam's share 31.25%.
- Action: **Apply the filter**, then **Use the surviving group**. Both steps change
  the actual displayed population, not merely its description.
- Equation linkage: counts come first: `100/(100+9900)`, then `90+198`, then
  `90/(90+198)`. The filtering diagram shows the 90% and 2% retention rates.
- Transfer question: a 10% base rate with identical classifier rates yields
  `900/(900+180)=83.33%` spam among flags, not unchanged precision.
- Limitations: synthetic expected integer counts for the available cases; fixed
  recall/false-positive rates model prior shift only. Strip widths are exact shares.

## Entropy

- Causal change: inspect rare `glass`, then common `cup` for one toy next-token
  context. The selected area and surprise ruler move while the distribution and
  total entropy stay fixed. Cup is less surprising but contributes more on average.
- Action: **Compare common cup**, then **Compare rare glass**. Model mismatch is
  clearly available under **What if the model overpredicts cup?**, not the first step.
- Equation linkage: `0.1*(-log2(0.1))=0.3322` versus
  `0.6*(-log2(0.6))=0.4422` bits per continuation. Width is reference frequency;
  height is bits per occurrence. Sum of areas is 1.2955 bits per continuation.
- Transfer question: halving glass's model probability adds one bit per glass,
  hence 0.1 bits per continuation to that token's contribution. It does not imply
  a 0.1 change to total cross-entropy after other probabilities are reallocated.
- Limitations: illustrative counts `[60,30,10]`, not empirical language frequencies.
  Optional mismatch preserves exact `cross-entropy = entropy + KL`, a fixed visual
  scale, and the distinction between bits and natural-log training's nats.

## Loss

- Causal change: two independent scalar logit updates start from the same 1% click
  prediction and observed click, with learning rate 2. Applying them yields
  approximately 6.82% under log-loss versus 1.04% under squared probability error.
- Action: **Apply one learning step**; hollow markers retain the common start.
- Equation linkage: highlight the extra factor `2*0.01*0.99` in the squared-loss
  gradient. The computed gradients are -0.99 and -0.019602; updates use
  `z' = z - 2*gradient` and transform back with sigmoid.
- Transfer question: as a confidently wrong click prediction tends to zero, the
  log-loss logit gradient tends to -1, while the squared-loss gradient tends to 0.
- Limitations: one illustrative scalar step, not training a network. Both losses
  are proper scores. Nats and unitless squared loss are not comparable scales;
  the shared learning rate does not establish universal optimizer superiority.

## Vectors

- Causal change: normalize item B `[2,-0.1]`; the selected arrow moves to the unit
  circle. Its original representation remains dashed on the same ray.
- Action: **Remove length**, reversible. Other vectors and zero remain selectable.
- Equation linkage: `[2,-0.1]/2.0025` gives approximately `[0.9988,-0.04994]`.
  The before/after norm is 2.0025 to 1; cosine with the original is exactly 1.
- Transfer question: multiplying a nonzero vector by positive 3 before normalizing
  gives the same unit vector; a negative multiplier reverses its direction.
- Limitations: two toy coordinates, no invented semantic axis meanings; zero has
  no unit direction. Norm is not assumed to represent popularity or importance.

## Dot Products

- Causal change: remove norm factors while holding embeddings and query fixed.
  The visible rank comparison switches the active rule from dot (B first) to
  cosine (A first). Geometry/projection and selected numerical score update too.
- Action: **Remove norm advantage**, reversible; rank nodes and geometry points
  inspect the same item. On mobile the ranking comparison precedes geometry.
- Equation linkage: selected coordinate products yield A's dot score 1.2; dividing
  by the two displayed norms yields cosine 0.9923. The removed denominator is marked.
- Transfer question: scaling only the query by positive 10 preserves both rankings;
  every raw score scales together, and cosine cancels that common factor.
- Limitations: exact exhaustive comparison of three 2D items, not approximate
  nearest-neighbor behavior. Metric changes are modeling choices, not guaranteed gains.

## Matrix Multiplication

- Causal change: the initial accumulator is the first weighted column `[1,0]`.
  Adding the second weighted column `[2,2]` at its tip moves the endpoint to `[3,2]`.
  The previous contribution is muted; the active contribution is orange.
- Action: **Add second column**. The output diagram is first on mobile. Other
  trace steps place the first column or replace the decomposition with the sum.
- Equation linkage: `[1,0] + 2*[1,1] = [3,2]`; the active weighted column is marked.
  Full row-dot-input arithmetic and the actual matrix remain inspectable.
- Transfer question: under `[[1,1],[1,1]]`, `[1,2]` and `[2,1]` both map to `[3,3]`;
  their difference lies in the nullspace, so inversion cannot recover the input.
- Limitations: a 2D linear map, no bias or nonlinearity; determinant reasoning is
  specific to square maps. Alternative rotation/collapse fixtures are exact.

## Eigenvectors

- Causal change: display the green perpendicular residual from the output's
  projection onto the input line to the actual output. Feeding normalized output
  back changes the direction and shrinks its length from 1 to 0.6 on the default step.
- Action: **Feed output back**; exact eigenvectors can also be selected directly.
- Equation linkage: `Av - rho*v` initially gives `[2,1]-2*[1,0]=[0,1]`.
  The projected component is highlighted and the residual is drawn at its endpoint.
- Transfer question: a start exactly on the weaker eigenvector stays on that
  eigenline in exact arithmetic; power iteration does not always escape it.
- Limitations: symmetric 2x2 examples, with explicit indefinite and repeated-value
  cases. A unique dominant absolute eigenvalue and initial component are necessary
  for the usual power-iteration convergence argument. Covariance is not Hessian.

## SVD

- Causal change: rank one collapses the mapped unit circle onto a line. The full
  ellipse remains dashed; an orange vector shows the missing output for `[1,1]`.
  Restoring the component fills out the ellipse and removes that missing vector.
- Action: **Restore direction**, reversible; the factorization trace is secondary.
- Equation linkage: approximately `[4.061,0.446] + [0,0]` becomes
  `[4.061,0.446] + [-0.061,0.554] = [4,1]`. The restored component is marked.
  Whole-matrix Frobenius error falls from 0.9435 to zero, distinctly labeled from
  the missing output-vector length (about 0.558) for this single input.
- Transfer question: changing singular values from `[5,1]` to `[5,2]` quadruples
  rank-one squared Frobenius error, from 1 to 4.
- Limitations: computed 2x2 decomposition, no storage-saving claim for this tiny
  example; orthogonal factors may include reflections. Retained matrix energy is
  not ranking quality. Restoring an already-zero component adds no information.

## Verification

- `node --test atelier/tests/foundations-labs.test.cjs`: 32 tests passed, zero
  failures. The final scoped browser render completed 40 width checks with zero
  automated errors, including disclosure toggles, primary actions, and formula/quiz
  integration. `git diff --check` passed for the owned files.
- Owned Node tests cover conservation, inverse-CDF boundaries, population-weighted
  risk, all Bayes subsets, entropy-area/decomposition identities, finite-difference
  loss gradients, actual gradient steps, normalization/ranking counterfactuals,
  basis maps, eigen residuals, exact eig/SVD reconstruction, rank-one error, extreme
  SVD scaling, bounded history, native disclosures, and actual first-action geometry.
- Visual review uses the parent's real integrated pages, not a mock DOM. The scoped
  render runs both chapters at 1440, 720, 390, and 320 pixels; images/reports stay in
  `/tmp/atelier-causal-foundations-final`, separate from the parent report.
- Desktop and mobile initial captures were inspected for all ten concepts. This
  caught missing formula/rank-node styling and a below-fold mobile rank comparison.
  Subsequent fixes also preserve text contrast, expose Bayes rates, and retain enough
  loss-probability precision to avoid rounding away the small squared-loss update.
- All ten mobile first-action captures were also inspected. The final desktop and
  mobile dot-product before/after captures confirm readable rank labels, one filled
  scoring-rule column, and the mobile ranking comparison above the fold. Bayes and
  loss after-captures confirm visible filter rates and the 6.82% versus 1.04% updates.
