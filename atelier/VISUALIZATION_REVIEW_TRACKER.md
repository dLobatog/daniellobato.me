# Atelier Visualization Review Tracker

Status legend:
- todo
- reviewing
- fixed
- verified

Scoring:
- 1 = broken / misleading / incomplete
- 2 = weak
- 3 = usable but needs work
- 4 = strong with polish left
- 5 = publish-ready

Non-negotiables:
- Pedagogy: easy to understand, builds intuition
- Design: clean, quiet, Apple-like clarity
- Completeness: do not leave out key details or move them elsewhere for now

| Chapter | Page | Viz key | Status | Pedagogy | Design | Completeness | Priority | Notes |
|---|---|---|---|---:|---:|---:|---|---|
| foundations | foundations.html | distribution | verified | 4 | 4 | 4 | P2 | Wider stage and stronger readout hierarchy make the default discrete view easier to parse at a glance. The sample-vs-distribution story now lands faster without dropping the continuous mode or supporting detail. |
| foundations | foundations.html | expectation | verified | 4 | 4 | 4 | P2 | The center-of-mass framing is still the right teaching hook, and the bigger viz/readout treatment makes the triangle-vs-running-average relationship easier to scan. Remaining work is mostly polish, not conceptual repair. |
| foundations | foundations.html | bayes | verified | 4 | 4 | 4 | P2 | Updated copy now foregrounds the “crop down to positive cases” mental model, and the chapter-wide layout pass gives the posterior story more room. Biggest remaining opportunity is an even more explicit live before/after posterior sentence inside the viz itself. |
| foundations | foundations.html | entropy | verified | 4 | 4 | 4 | P2 | The review found that uncertainty vs mismatch was present but too implicit. New summary/what/why copy plus larger readout metrics make the blue-task vs gold-mismatch decomposition much easier to read without hiding the formal detail. |
| foundations | foundations.html | loss | verified | 4 | 4 | 4 | P2 | Loss now reads more directly as “penalty score, lower is better,” and the wider, calmer panel treatment helps the residual/loss comparison stay legible. The remaining improvements are mostly deeper per-viz polish rather than chapter-level clarity fixes. |
| linear-algebra | linear-algebra.html | vectors | todo |  |  |  |  |  |
| linear-algebra | linear-algebra.html | dot-products | todo |  |  |  |  |  |
| linear-algebra | linear-algebra.html | matrix-multiply | todo |  |  |  |  |  |
| linear-algebra | linear-algebra.html | eigen | todo |  |  |  |  |  |
| linear-algebra | linear-algebra.html | svd | todo |  |  |  |  |  |
| neural-basics | neural-network-basics.html | neuron | reviewing | 4 | 3 | 4 | P1 | Good mental model: one hyperplane plus one squash. The decision boundary and weight vector are present, but small text and crowded controls keep it from feeling effortlessly clear. |
| neural-basics | neural-network-basics.html | activation-basics | reviewing | 4 | 3 | 4 | P1 | Very strong intuition around saturation and gradient flow. The four-function comparison is valuable, but the panel is visually busy and the signal-to-noise ratio suffers from too many small annotations at once. |
| neural-basics | neural-network-basics.html | output-functions | todo |  |  |  |  |  |
| neural-basics | neural-network-basics.html | forward-pass | todo |  |  |  |  |  |
| neural-basics | neural-network-basics.html | chain-rule | todo |  |  |  |  |  |
| neural-basics | neural-network-basics.html | backprop | reviewing | 3 | 3 | 4 | P1 | Completeness is strong and the reverse-error-flow story is present, but it is still the hardest of the batch to parse quickly. It needs clearer hierarchy so the learner sees the main causal path before the notation details. |
| neural-basics | neural-network-basics.html | gradient-descent | reviewing | 4 | 4 | 4 | P1 | Strong overall. The surface presets, LR control, and trajectory story build good intuition. Biggest opportunity is to simplify the surrounding text and metrics so the path animation remains the hero. |
| neural-basics | neural-network-basics.html | optimizers | todo |  |  |  |  |  |
| neural-basics | neural-network-basics.html | lr-schedule | todo |  |  |  |  |  |
| deep-learning | deep-learning.html | learning-rate | todo |  |  |  |  |  |
| deep-learning | deep-learning.html | initialization | todo |  |  |  |  |  |
| deep-learning | deep-learning.html | gradient-flow | todo |  |  |  |  |  |
| deep-learning | deep-learning.html | normalization | todo |  |  |  |  |  |
| deep-learning | deep-learning.html | residuals | todo |  |  |  |  |  |
| deep-learning | deep-learning.html | flash-attention | todo |  |  |  |  |  |
| transformers-rag | transformers-rag.html | tokenization | todo |  |  |  |  |  |
| transformers-rag | transformers-rag.html | embeddings | todo |  |  |  |  |  |
| transformers-rag | transformers-rag.html | positional | todo |  |  |  |  |  |
| transformers-rag | transformers-rag.html | transformer-block | todo |  |  |  |  |  |
| transformers-rag | transformers-rag.html | attention | todo |  |  |  |  |  |
| transformers-rag | transformers-rag.html | kv-cache | todo |  |  |  |  |  |
| transformers-rag | transformers-rag.html | rag | todo |  |  |  |  |  |
| adaptation | adaptation-serving.html | pretrain-finetune | todo |  |  |  |  |  |
| adaptation | adaptation-serving.html | peft | todo |  |  |  |  |  |
| adaptation | adaptation-serving.html | lora | todo |  |  |  |  |  |
| adaptation | adaptation-serving.html | quantization | todo |  |  |  |  |  |
| adaptation | adaptation-serving.html | distillation | todo |  |  |  |  |  |
| adaptation | adaptation-serving.html | serving-tradeoffs | todo |  |  |  |  |  |
| rl | reinforcement-learning.html | mdp | todo |  |  |  |  |  |
| rl | reinforcement-learning.html | value-functions | todo |  |  |  |  |  |
| rl | reinforcement-learning.html | td-learning | todo |  |  |  |  |  |
| rl | reinforcement-learning.html | q-learning | todo |  |  |  |  |  |
| rl | reinforcement-learning.html | dqn | todo |  |  |  |  |  |
| rl | reinforcement-learning.html | bandit | todo |  |  |  |  |  |
| metrics | metrics-eval.html | threshold-metrics | todo |  |  |  |  |  |
| metrics | metrics-eval.html | calibration | todo |  |  |  |  |  |
| metrics | metrics-eval.html | ranking-metrics | todo |  |  |  |  |  |
| systems | systems-retrieval.html | retrieval-funnel | todo |  |  |  |  |  |
| systems | systems-retrieval.html | cold-start | todo |  |  |  |  |  |
| generative | generative-and-rl.html | diffusion | todo |  |  |  |  |  |
| generative | generative-and-rl.html | bandit | todo |  |  |  |  |  |
| recommendation | recommendation-depth.html | matrix-factorization | todo |  |  |  |  |  |
| recommendation | recommendation-depth.html | two-tower | todo |  |  |  |  |  |
| recommendation | recommendation-depth.html | ranking-metrics | todo |  |  |  |  |  |
| classical | classical-ml-stats.html | mle-map | todo |  |  |  |  |  |
| classical | classical-ml-stats.html | bias-variance | todo |  |  |  |  |  |
| classical | classical-ml-stats.html | regularization | todo |  |  |  |  |  |
| production | production-systems.html | serving-skew | todo |  |  |  |  |  |
| production | production-systems.html | online-offline | todo |  |  |  |  |  |
| production | production-systems.html | data-drift | todo |  |  |  |  |  |
| gbdt | gbdt-tabular.html | tree-split | todo |  |  |  |  |  |
| gbdt | gbdt-tabular.html | boosting | todo |  |  |  |  |  |
| data-features | data-features.html | feature-leakage | todo |  |  |  |  |  |
| data-features | data-features.html | feature-shift | todo |  |  |  |  |  |
| alignment-depth | alignment-depth.html | guidance | todo |  |  |  |  |  |
| alignment-depth | alignment-depth.html | dpo | todo |  |  |  |  |  |
| alignment-depth | alignment-depth.html | reward-hacking | todo |  |  |  |  |  |

## First batch recommendation

Start here:
- foundations/distribution
- foundations/expectation
- neural-basics/neuron
- neural-basics/activation-basics
- neural-basics/backprop
- neural-basics/gradient-descent

Reason:
- highest leverage for intuition
- strongest influence on the rest of the library
- easiest place to establish the visual language standard
