# Atelier: Causal Teaching Pass

Date: 2026-09-25

## Preserve the baseline

The deployed starting point is commit `1fe1e429ecac7ba898b760730aab5e53c4bfb692`.
It is preserved as `codex/atelier-before-causal-teaching-20260925`.
Existing uncommitted prototype/review work is outside this pass and must not be
reverted or included in its commit.

## Six acceptance criteria

1. A short, concrete hook leads quickly to the actual interaction.
2. The relevant mechanism sits beside the input or object the learner selects.
3. A primary action produces a visible causal change, not just different prose.
4. Plain-language actions come first; optional settings follow the experiment.
5. The selected visual and its real substituted calculation refer to each other.
6. Optional checks ask for transfer or a counterfactual, not recall of button order.

These are design criteria, not proof of measured learning improvement. A tidy
screen is necessary but insufficient. Catchiness means a revealing contrast,
not exaggerated claims, fake measurements, or unrelated analogies.

## Complete public scope

68 public concepts use 67 mechanisms; the bandit appears in two chapters.
The family ledgers contain the individual action, equation link, transfer check,
and limitations for every concept, rather than a generic chapter sign-off.

| Chapter | Concepts in this pass | Individual ledger |
| --- | --- | --- |
| Foundations | Distribution; expectation; Bayes; entropy/cross-entropy; loss | [Foundations](causal-teaching-foundations.md) |
| Linear algebra | Vectors; dot products; matrix multiplication; eigenvectors; SVD | [Foundations](causal-teaching-foundations.md) |
| Classical ML/statistics | MLE/MAP; bias-variance; regularization | [Tabular](causal-teaching-tabular.md) |
| Neural-network basics | Neuron; activations; outputs; forward pass; chain rule; backprop; gradient descent; optimizers; LR schedules | [Neural](causal-teaching-neural.md) |
| Deep learning | Learning rate; initialization; gradient flow; normalization; residuals; FlashAttention | [Training](causal-teaching-training.md) |
| Transformers/RAG | Tokenization; embeddings; positional encoding; block architecture; attention; KV cache; RAG | [Transformers](causal-teaching-transformers.md) |
| Adaptation/serving | Pretraining/fine-tuning; PEFT; LoRA; quantization; distillation; serving | [Training](causal-teaching-training.md) |
| Metrics/evaluation | Thresholds; calibration; ranking metrics | [Ranking](causal-teaching-ranking.md) |
| Recommendations | Matrix factorization; two-tower; ranking objectives | [Ranking](causal-teaching-ranking.md) |
| Retrieval systems | Retrieval funnel; cold start | [Ranking](causal-teaching-ranking.md) |
| Production | Serving skew; offline/online evaluation; drift | [Tabular](causal-teaching-tabular.md) |
| Trees | Tree splits; boosting | [Tabular](causal-teaching-tabular.md) |
| Data/features | Leakage; feature shift | [Tabular](causal-teaching-tabular.md) |
| Generative/RL | Diffusion; bandits | [Decision](causal-teaching-decision.md) |
| Reinforcement learning | MDP; value functions; TD; Q-learning; DQN; exploration | [Decision](causal-teaching-decision.md) |
| Alignment | Guidance; GRPO; DPO; reward hacking | [Decision](causal-teaching-decision.md) |

## Shared presentation

- Remove repeated concept ordinals and introductory summaries above the stage.
  Keep the meaningful heading; retain explanations, annotated formulas, code,
  and optional quizzes in the existing secondary layer.
- Reduce chapter-entry spacing without shrinking reading text.
- Use real before/after values and highlight only the term causing the change.
- Keep essential legends and units visible even when settings are collapsed.
- Preserve click-to-expand, pinned close, Escape, backdrop close and focus return.
- Version changed public assets so cached styles cannot mismatch new markup.

## Verification gates

- Pure numerical and state-transition tests for each renderer family.
- Whole-library browser run at 1440, 720 and 390 pixels.
- Screenshot inspection of all concepts on desktop and phone, including applied
  actions and scrollable lower content. Family workers review their own output;
  the coordinating pass checks integration and cross-family consistency.
- Disclosures open and close; controls visibly update; formulas parse; quizzes
  return feedback; no horizontal overflow or uncaught browser exceptions.
- Additional actual neural interaction tests distinguish inference, gradients,
  parameter updates, independent probabilities and shared probability mass.

## Completed verification

All 68 public sections are covered by the individual family ledgers above.
No new-topic expansion or prototype/review-page work is included in this pass.

- Full suite with all optional browser tests enabled: **315 passed, 0 failed,
  0 skipped**. Log: `/tmp/atelier-causal-browser-tests.log`.
- Final whole-library render: **204 checks**, all 68 sections at 1440, 720,
  and 390 pixels; every primary action changes visible content; zero reported
  overflow, disclosure, formula, quiz, viewer-dismissal, or browser errors.
  Report and captures: `/tmp/atelier-causal-release`.
- Dark-theme phone pass: **68 checks**, zero reported errors. Captures:
  `/tmp/atelier-causal-release-dark`. Contrast spot checks supplement each
  family's desktop/mobile screenshot review.
- Final neural layout correction keeps pixels and their calculation adjacent
  at half-screen width. A regression test covers full-width, half-width, and
  phone layouts. The subsequent **27-check** neural render passed; captures:
  `/tmp/atelier-causal-neural-release`.
- Each family reviewed its desktop/mobile initial and applied states, plus
  scrollable lower content where present. Coordinating screenshot review
  checked the integrated presentation and caught styling, contrast, formula
  spacing, and unnecessary stacking issues before release.
- Family-specific narrow-screen checks also include 320 pixels; see their
  ledgers for exact coverage and intentional mathematical simplifications.
- `git diff --check` passes. Public stylesheet/script versions are updated
  to avoid mixing cached presentation with new markup.

Screenshots are local QA artifacts, not public-site assets. Reproduce the
integrated pass against a local static server with
`node atelier/tests/render-review.cjs`; set `PLAYWRIGHT_MODULE`, `CHROME_PATH`,
and `ATELIER_BASE_URL` when needed. Enable `ATELIER_BROWSER_TESTS=1`,
`TABULAR_BROWSER_TEST=1`, and `DECISION_BROWSER_TEST=1` for the full test suite.

The implementation and technical acceptance pass are complete. Actual learner
comprehension is not proven by these checks. Use observed misunderstandings to
choose the next iteration, preserving the baseline for side-by-side comparison.
