# Tabular Causal Teaching Pass

Scope: `atelier/tabular-labs.js`, `atelier/tabular-labs.css`, and
`atelier/tests/tabular-labs.test.cjs`. No shared shell, HTML, or other family edits.

Each stage has one concrete question and one primary computation-changing action.
The selected term, a numerical comparison, and its mechanism are visible beside
the diagram. Full tables, derivations, and alternative settings remain optional.
Existing optional quizzes now ask counterfactual questions rather than button
recall. "Causal" here means controlled changes to the specified toy computation,
not causal identification from observational data.

## MLE And MAP (`mle-map`)

- Change: the evidence-weight band and posterior mode move together while the
  observed click rate and prior stay fixed. A one-step history supplies the real
  before/after estimate rather than a fabricated curve.
- Action: see ten times more observations, from 2/3 to 20/30 clicks/impressions.
- Equation linkage: highlighted observed counts in `(k+2)/(n+10)`; MAP moves from
  `4/13 = 0.3077` to `22/40 = 0.55`. The prior mode is 0.2.
- Transfer question: strengthen a prior without moving its mode. MAP shrinks
  toward 0.2 while the same-data MLE remains 2/3.
- Limits: independent stationary Bernoulli trials, specified prior, no exposure
  correction. MAP is not the posterior mean. Curves in the disclosure are
  individually peak-normalized, not comparable density heights.

## Bias And Variance (`bias-variance`)

- Change: refit after exactly one label changes. A Gray-code cycle visits all
  eight noise patterns; the moved point and fit are bright, other fits are faint.
  Training points also support direct keyboard/touch label flips.
- Action: change one label and refit. Inputs, estimator, and probe remain fixed.
- Equation linkage: at probe 0.5 the quadratic prediction is
  `-0.125*y_left + 0.75*y_center + 0.375*y_right`; changed label terms are
  highlighted. The selected prediction and previous fit are both shown.
- Transfer question: increasing only the center label by 1 moves the probe
  prediction by 0.75 even though both quadratic fits have zero training error.
- Limits: fixed design, known truth `x^2`, independent symmetric +/-0.5 noise.
  The exact bias/variance/noise decomposition averages all eight datasets and
  independent future noise; it is not a universal complexity curve.

## Regularization (`regularization`)

- Change: selectable coefficient lanes show the previous and current weight on
  one scale. Unselected feature bars recede. The selected row's contribution
  changes with the actual solved coefficient.
- Action: compare L2 shrinkage with the initial L1 solution at the same lambda.
- Equation linkage: for the weak feature, `max(0.4-0.5,0)=0` becomes
  `0.4/(1+0.5)=0.2667`; the selected `x_j*w_j` term and total score stay visible.
- Transfer question: raising lambda from 0.25 to 0.5 thresholds L1's weak weight
  from 0.15 to zero, but changes L2 from 0.32 to about 0.267.
- Limits: centered orthogonal design with unit mean-square columns; these closed
  forms do not apply to general correlated features. The constructed holdout is
  illustrative, not an independent deployment-quality estimate.

## Tree Splits (`tree-split`)

- Change: moving the cut reroutes the actual row tiles. R3 and R4 are marked as
  moved; unaffected rows recede. Children expose class counts and Gini values.
- Action: try the weaker cut 2.5 instead of the initial best cut 4.5.
- Equation linkage: the selected row highlights its child's weighted impurity.
  At 4.5, `(4/8)*0.375 + (4/8)*0 = 0.1875`; gain is 0.28125. At 2.5 gain is
  0.2604167. Histogram settings expose the actual bin counts and allowed cuts.
- Transfer question: uniformly duplicating rows leaves class fractions and
  child-size proportions unchanged, so normalized Gini gain does not double.
- Limits: binary Gini, one feature, eight rows; not second-order gradient/Hessian
  gain. Histogram boundaries happen to retain the best cut in this fixture.

## Boosting (`boosting`)

- Change: each example has a target, current prediction, and previous-prediction
  tick. Rows outside the selected leaf recede; only member residuals enter the
  highlighted leaf-mean calculation.
- Action: fit the next residual tree, then add its scaled output to the ensemble.
- Equation linkage: first leaf mean `(-2-2-1-1)/4=-1.5`; R1 updates as
  `2 + 0.5*(-1.5)=1.25`. Subsequent rounds recompute both residuals and best split.
- Transfer question: hold the first fitted tree fixed and use rate 1; R1 becomes
  0.5, not zero. Later trees would see different residuals.
- Limits: exact squared-loss depth-one trees, deterministic split ties, no
  subsampling or leaf penalty. Displayed loss is training MSE, not validation.

## Feature Leakage (`feature-leakage`)

- Change: enforcing the request-time gate visibly blocks unreadable timeline
  events and substitutes the explicit fallback. The request split is unchanged.
- Action: replay the hindsight snapshot at prediction time.
- Equation linkage: Q5 tests `(52 <= 50) AND (52 <= 50) = false`; replay rejects
  value 1 and predicts 0. The fixed held-out audit accuracy changes 100% to 50%.
- Transfer question: moving a past event from minute 49 to 48 cannot repair a
  value first readable at minute 51 for a request at minute 50.
- Limits: a fixed rule, not a fitted model; six requests and explicit zero
  fallback. Chronological train/test membership does not repair a future join.

## Feature Shift (`feature-shift`)

- Change: null rows show whether the imputed zero retains its missing flag.
  Non-null rows recede. The same raw row is compared under both feature policies.
- Action: coerce nulls to zero, erasing the indicator without changing raw data.
- Equation linkage: selected null has `sigmoid(-2+0.4*0-0.8*1)=0.0573`; erasing
  the highlighted flag term yields `sigmoid(-2)=0.1192`. Raw missingness stays
  5/8 while post-transform missingness falls to zero.
- Transfer question: if the indicator coefficient were zero, erasing it would
  not change this score, but would still remove the monitoring signal.
- Limits: snapshots are not paired users; the cross-policy comparison uses the
  same selected raw value. No labels or accuracy estimate are supplied.

## Train/Serve Skew (`serving-skew`)

- Change: paired paths emphasize the transformed input for the same request,
  with fixed model weights and an actual previous/now score comparison.
- Action: apply the shared feature transform to repair the missing-log1p path.
- Equation linkage: U4 changes model input from 9 to `log1p(9)=2.3026` inside
  `sigmoid(-1.5+1.2*x)`; the score drops from about 99.99% to 78.0%. All five
  fixture requests then pass parity. Stale-source and null-default cases remain.
- Transfer question: all-zero test counts miss this bug because `log1p(0)=0`;
  positive counts and null cases are needed to exercise divergent behavior.
- Limits: model artifact and requests are fixed. Parity is not evidence of good
  model quality; the fixture does not model serving latency or race conditions.

## Online/Offline Evaluation (`online-offline`)

- Change: audience weights change while observed cohort rates remain fixed.
  Selecting a cohort highlights its weighted contribution; the other recedes.
- Action: apply the live 20/80 audience instead of the offline 90/10 mix.
- Equation linkage: `0.9*(17/20-16/20)+0.1*(5/20-8/20)=+0.03` becomes
  `0.2*0.05+0.8*(-0.15)=-0.11`. Percentage-point units and audience colors remain
  visible, not hidden in the calculation disclosure.
- Transfer question: positive differences in every cohort cannot become a
  negative weighted sum under common nonnegative weights.
- Limits: observational counts and a stable-rate projection, not a randomized
  launch effect. Reweighting does not remove selection or exposure bias.

## Data Drift (`data-drift`)

- Change: input buckets stay in place while arriving labels reveal which
  predictions were correct. The selected bucket highlights its correct-count
  term; unseen label-dependent metrics remain unknown rather than fabricated.
- Action: reveal delayed labels, not change the model or input distribution.
- Equation linkage: `(?+?)/20` becomes `(2+2)/20=0.2`; input total variation stays
  zero. Reference accuracy is 0.8. Labels 0/1, pending marks, and correct outlines
  have visible explanations.
- Transfer question: unchanged input and label marginals can conceal a reversed
  conditional relationship and an accuracy drop from 80% to 20%.
- Limits: exact small batches, no sampling uncertainty. The label-shift fixture
  preserves `P(x|y)`; these distribution descriptions need not be exclusive.

## Verification

- `node --test atelier/tests/tabular-labs.test.cjs`: 32 passed, zero failures;
  one opt-in standalone browser test skipped in that invocation.
- Final invocation with `TABULAR_BROWSER_TEST=1`: all 33 tests passed, no skips.
  This also checks direct SVG label flips with Enter, focus restoration, and
  synchronized roots at all four viewport widths in an isolated browser.
- Reachability checks every rendered transition including one-level prior-state
  histories and SVG label controls, and separately verifies all 507 parameter
  configurations. It no longer asserts a brittle history-dependent grand total.
- Independent headless integration rerender: 40 checks at 320/390/720/1440,
  zero reported errors. Output: `/tmp/atelier-causal-tabular-final`; the parent
  report at `/tmp/atelier-causal-teaching-final` was not overwritten.
- The parent's early captures had new markup but pre-update CSS. The scoped
  rerender includes block-separated equation labels/reasons, semantic underlines
  instead of default yellow marks, visible coefficient bars, and wider wrapping
  tree row tiles. Essential legends were also restored in stage HTML.
- Visually reviewed the scoped rerender's 390px and 1440px initial and primary-
  action screenshots for all ten concepts (40 images), plus all nine available
  bottom captures at those widths. No overlap or concatenated annotations remain
  in those reviewed images. No renderer/CSS changes followed this capture;
  only the owned test and this verification document were updated afterward.
