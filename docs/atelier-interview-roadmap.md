# Atelier: concept-by-concept rebuild

## Acceptance gate

A visualization is ready only when the learner can answer one concrete question by interacting with it. A polished shell alone does not pass.

- Familiar example before notation; one idea in the initial scene.
- Every visible quantity comes from the actual computation, not an illustrative independent slider.
- Direct interaction with the visual: inspect a node, make a prediction, or advance one computation. Keyboard and touch must work too.
- Explain the change, not just the resulting number. Keep persistent labels outside plotted geometry.
- Formula, annotated numerical example, executable code sketch, and a misconception check agree with the visual.
- Capture and personally inspect desktop, half-width, mobile, and expanded screenshots. Check closing, focus, state retention, and overflow.
- Distinguish the small teaching example from what follows at scale. No universal optimizer or architecture claims.

## Priority order

### 1. Neural-network learning loop (implemented, ready for learner review)

Replace the nine disconnected demos in Neural Network Basics. One prediction example connects the neuron, forward pass, chain rule, and backward pass. Optimization then shows actual update trajectories. The reading UI gets a light default and an optional dark theme, without recoloring legacy plots incorrectly.

- [x] Neuron: identify inputs, learned weights, bias, activation, and prediction.
- [x] Activation functions: same input, different outputs and local derivatives.
- [x] Output functions: independent labels versus mutually exclusive classes; binary equivalence explained.
- [x] Forward pass: reveal each intermediate computation and its units.
- [x] Chain rule: multiply actual local sensitivities and verify numerically.
- [x] Backpropagation: compute gradients first, then update weights, then recompute loss.
- [x] Gradient descent: take a real step and expose overshoot rather than hiding it.
- [x] Optimizers: compare SGD, momentum, and bias-corrected Adam on the same objective.
- [x] Learning-rate schedules: compare actual training traces under an identical update rule.

Implementation and render checks are recorded in [the validation log](atelier-neural-validation.md). These checkmarks mean implemented and tested, not that learner comprehension has been measured. The next review should ask someone to predict an output or update before clicking Next.

### 2. Fundamentals that make evaluation trustworthy

Data splits and information availability at prediction time; temporal/group leakage; baselines; bias/variance and regularization; conditional probability; entropy/cross-entropy/KL; calibration versus ranking; uncertainty, confidence intervals, power and A/B tests. Show a plausible wrong conclusion and the experiment that exposes it.

Next concrete lesson: a forecasting example where a feature is available today but was not available at the historical prediction time. Let the learner inspect a row, move the prediction cutoff, and see which features become inadmissible. Contrast a convincing but leaked validation result with an honest baseline. Keep probability/loss fundamentals in this same priority tier, not behind advanced architectures.

### 3. Retrieval, ranking, and personalization

Retrieval versus ranking, two-tower training and ANN recall, in-batch negatives and sampling corrections, exposure/selection/position bias, cold start and sparse/dense features, sequential user representations, pairwise/listwise losses and LambdaRank, multitask tradeoffs, calibrated probabilities, offline versus online evaluation, feedback loops, latency/freshness, and generative retrieval. In particular: an unseen item is not a known disliked item.

### 4. LLM mechanisms and post-training

Attention as score/normalize/mix; residual stream and the job of each transformer sublayer; causal masking and positional information; pretraining versus SFT; policy gradients, baseline/advantage, importance ratios and clipping; PPO versus GRPO; group-relative rewards and zero-variance groups; token/sequence credit assignment; KL penalties and reward hacking; DPO; judge reliability and held-out evaluation. Teach the objective with a small set of inspectable completions before presenting a large equation.

### 5. Staff-level systems tradeoffs

Batching, KV cache, serving latency and throughput, LoRA/PEFT, quantization and distillation, training/inference mismatch, data and model freshness, distribution shift, launch guardrails, experiment design, failure slices, and identifying the binding constraint. Tie each tradeoff to a measurable decision instead of architecture name recognition.

### 6. Broader depth and public references

Trees/GBDT, linear algebra/SVD, normalization/initialization, CNNs, GNNs, generative models, and deeper RL. Attach public primary references and selected publicly accessible videos at the concept level. Never expose private preparation notes or personal source collections.

## Review and rollback

Preserve existing experiments until a replacement passes the gate. Use explicit commits for each reviewed milestone. Do not fold unrelated prototype or review changes into a release. Keep public pages about the concepts, not the development process.

Remaining shared cleanup: older renderer families still expose duplicated outer controls and internal controls in some chapters. Remove those as each lesson receives a coherent state owner; do not imply that the theme change has repaired those lessons pedagogically.
