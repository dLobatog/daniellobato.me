# Atelier Mechanism Rebuild

## Scope and rollback

The public product remains the static Atelier in this repository. This pass replaces the mechanisms in all 16 existing chapters, not every possible ML topic. There are 67 existing sections (the bandit appears twice) and one new GRPO section.

Baseline: `29b528a9084d181a163abaeab9817db24dd23f69`, preserved by branch `codex/atelier-before-mechanism-rebuild-20260925`. Legacy renderers remain available in Git; public chapter files load the new native modules.

## Attention-first acceptance gate

- The first view asks one plain question and makes the next action obvious.
- The visual shows an actual computation or decision, not a fabricated quality curve.
- One click or step creates a visible, explainable consequence.
- Inspect nodes, examples, points or transitions directly. Do not substitute a row of unrelated knobs.
- Detailed arithmetic and alternative settings are secondary disclosures. Do not hide the core phenomenon with them.
- Keep text and controls readable. Reflow narrow layouts instead of shrinking a desktop diagram.
- A formula defines its symbols and uses the same numerical example as the mechanism and code.
- Preserve math, code, and retrieval-practice questions as optional depth.
- Explicitly distinguish illustrative models from measured production results.
- These are editorial acceptance criteria, not proof that learners understand. User review remains necessary.

## Implementation Coverage

| Chapter | Mechanisms | Module | Status |
| --- | --- | --- | --- |
| Foundations | Distribution, expectation, Bayes, entropy, loss | foundations-labs | Implemented; desktop + phone screenshots reviewed |
| Linear algebra | Vectors, dot products, matrix multiplication, eigenvectors, SVD | foundations-labs | Implemented; desktop + phone screenshots reviewed |
| Classical ML | MLE/MAP, bias-variance, regularization | tabular-labs | Implemented; desktop + phone screenshots reviewed |
| Neural basics | Image neuron, activations, output functions, forward pass, chain rule, backprop, gradient descent, optimizers, schedules | neural-mechanisms / neural-functions / optimization-lessons | Implemented; desktop + phone screenshots reviewed |
| Deep learning | Learning rate, initialization, gradient flow, normalization, residuals, Flash Attention | training-labs | Implemented; desktop + phone screenshots reviewed |
| Transformers | BPE, embeddings, positions, block, attention, KV cache, RAG | transformer-labs | Implemented; desktop + phone screenshots reviewed |
| Adaptation | Pretraining/SFT, PEFT, LoRA, quantization, distillation, serving | training-labs | Implemented; desktop + phone screenshots reviewed |
| Metrics | Thresholds, calibration, ranking metrics | ranking-labs | Implemented; desktop + phone screenshots reviewed |
| Recommendation | Factorization, two towers, ranking objectives | ranking-labs | Implemented; desktop + phone screenshots reviewed |
| Retrieval | Candidate ceiling, cold start | ranking-labs | Implemented; desktop + phone screenshots reviewed |
| Production | Serving skew, cohort effects, delayed-label drift | tabular-labs | Implemented; desktop + phone screenshots reviewed |
| Trees | Gini/histogram splitting, residual-fitting boosting | tabular-labs | Implemented; desktop + phone screenshots reviewed |
| Data | Point-in-time leakage, feature shift | tabular-labs | Implemented; desktop + phone screenshots reviewed |
| Generation | Diffusion, bandits | decision-labs | Implemented; desktop + phone screenshots reviewed |
| Reinforcement learning | MDP, values, TD, Q-learning/SARSA, DQN, bandits | decision-labs | Implemented; desktop + phone screenshots reviewed |
| Alignment | CFG, GRPO, DPO, reward hacking | decision-labs | Implemented; desktop + phone screenshots reviewed |

## Verification

- Numerical tests cover finite-difference gradients, probability conservation, matrix reconstruction, attention/cache equivalence, evaluation metrics, terminal handling and policy updates.
- Integration tests verify every public section resolves to one native mechanism and retains annotated math, code and a quiz.
- `atelier/tests/render-review.cjs` checks actual chapter pages at desktop, half-screen and mobile widths, exercises interactions, formulas, quizzes and Escape, and saves screenshots outside the repository.
- Screenshot review completed for all 68 public sections on desktop and phone. Half-width layouts were additionally exercised by the browser suite, with representative manual screenshot checks.
- Final unit/integration run: 232 tests, 229 passing, 3 optional browser tests skipped, 0 failures. The independent browser suite ran all 68 sections at 1440, 720 and 390px: 204 checks, 0 errors.
- Browser coverage includes meaningful actions, internal disclosure open/close, math rendering, quiz feedback, focus-view entry, Escape, close buttons and desktop backdrop dismissal. Screenshots and the detailed report are saved in /tmp/atelier-render-review, not published.
- Visual review caught and fixed collapsed mobile decision-flow cards and unnecessarily stacked two-tower branches. Detailed calculations no longer dominate the initial view.
- This verifies implementation and rendering, not learner comprehension. User testing remains the next acceptance gate; larger ranking lists and multi-stage diagrams can still be refined for faster first-read understanding.

## Next After User Review

1. Refine weak initial states based on what the learner predicts and notices, not additional explanation volume.
2. Add standalone logistic-regression and unsupervised/k-means mechanisms with actual training examples.
3. Add CNN/convolution, recurrence/gating and multi-task learning only with equally concrete mechanisms.
4. Expand trees to bagging/random forests, data splits and feature importance.
5. Expand generative architecture, GNNs and experiment design as separate high-quality chapters rather than superficial additions.
6. Continue source curation using public primary references. Private recruiting materials and personal notes must never become public citations.
