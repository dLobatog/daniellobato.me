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
| linear-algebra | linear-algebra.html | vectors | verified | 4 | 4 | 4 | P2 | The vector story was already solid, but the default state hid the direction-vs-magnitude split too implicitly. Adding the live angle cue and reframing the readout around length, unit direction, and full vector makes normalization land faster without removing any detail. |
| linear-algebra | linear-algebra.html | dot-products | verified | 4 | 4 | 4 | P2 | Strong pedagogical core: alignment, orthogonality, and negative overlap are all present and the projection picture does real work. Remaining opportunity is mostly polish around making the projection/readout hierarchy even more dominant at first glance. |
| linear-algebra | linear-algebra.html | matrix-multiply | verified | 4 | 4 | 4 | P2 | The key issue was that the square-and-basis view explained the map, but not quickly enough for a first pass. Adding a persistent sample x → Ax arrow makes the “same rule for every input” story much easier to see while keeping determinant and shear visible. |
| linear-algebra | linear-algebra.html | eigen | verified | 4 | 4 | 4 | P2 | Completeness was already good, but the special-direction idea stayed slightly abstract. Labeling v and Av directly on the surviving lines makes the same-line/stretch-only behavior easier to parse without dumbing down the concept. |
| linear-algebra | linear-algebra.html | svd | verified | 4 | 4 | 4 | P2 | This was the chapter’s hardest first-read concept because the rotate-stretch-rotate story was present but under-signposted. The new right-to-left formula framing, step connectors, and explicit singular-value labeling make the decomposition sequence much more teachable while preserving low-rank intuition. |
| neural-basics | neural-network-basics.html | neuron | reviewing | 4 | 3 | 4 | P1 | Good mental model: one hyperplane plus one squash. The decision boundary and weight vector are present, but small text and crowded controls keep it from feeling effortlessly clear. |
| neural-basics | neural-network-basics.html | activation-basics | reviewing | 4 | 3 | 4 | P1 | Very strong intuition around saturation and gradient flow. The four-function comparison is valuable, but the panel is visually busy and the signal-to-noise ratio suffers from too many small annotations at once. |
| neural-basics | neural-network-basics.html | output-functions | todo |  |  |  |  |  |
| neural-basics | neural-network-basics.html | forward-pass | todo |  |  |  |  |  |
| neural-basics | neural-network-basics.html | chain-rule | todo |  |  |  |  |  |
| neural-basics | neural-network-basics.html | backprop | reviewing | 3 | 3 | 4 | P1 | Completeness is strong and the reverse-error-flow story is present, but it is still the hardest of the batch to parse quickly. It needs clearer hierarchy so the learner sees the main causal path before the notation details. |
| neural-basics | neural-network-basics.html | gradient-descent | reviewing | 4 | 4 | 4 | P1 | Strong overall. The surface presets, LR control, and trajectory story build good intuition. Biggest opportunity is to simplify the surrounding text and metrics so the path animation remains the hero. |
| neural-basics | neural-network-basics.html | optimizers | todo |  |  |  |  |  |
| neural-basics | neural-network-basics.html | lr-schedule | todo |  |  |  |  |  |
| deep-learning | deep-learning.html | learning-rate | verified | 4 | 4 | 4 | P2 | The quadratic-bowl viz was already mechanically strong, but the chapter copy now makes the target behavior explicit: look for the largest step that still stays stable, then decay later. The main remaining work is visual polish, not conceptual repair. |
| deep-learning | deep-learning.html | initialization | verified | 4 | 4 | 4 | P2 | The initialization bars already showed the right healthy-band story; the biggest gap was first-read guidance. Updated copy now frames initialization as a signal-budget problem and explicitly anchors Xavier vs He to tanh vs ReLU so the learner can act on the comparison faster. |
| deep-learning | deep-learning.html | gradient-flow | verified | 4 | 4 | 4 | P2 | This card teaches the right mechanism, but it benefited from sharper framing around repeated multipliers and why the rest of the chapter exists. The new copy turns it into the chapter’s bridge concept instead of a standalone warning about vanishing/exploding gradients. |
| deep-learning | deep-learning.html | normalization | verified | 4 | 4 | 4 | P2 | Completeness was solid, but the old wording left BatchNorm, LayerNorm, and RMSNorm slightly too name-heavy on first pass. The revised explanation now foregrounds the common job — stop internal scale drift — and makes the architecture tradeoff clearer without dropping detail. |
| deep-learning | deep-learning.html | residuals | verified | 4 | 4 | 4 | P2 | The residual visualization already carried the identity-path intuition well. Copy updates now make the “preserve the signal, then learn the correction” story more explicit, which helps the skip connection read as an optimization tool instead of just an architectural trick. |
| deep-learning | deep-learning.html | flash-attention | verified | 4 | 4 | 4 | P2 | Flash Attention already had the right exact-vs-approximate stance, but the first-read story needed a clearer hardware mental model. The updated copy now foregrounds IO-aware tiling, online softmax state, and the fact that the win is memory traffic rather than a new modeling objective. |
| transformers-rag | transformers-rag.html | tokenization | verified | 4 | 4 | 4 | P1 | The biggest blocker was default-state rendering: the card could sit on a placeholder instead of teaching. Routing the chapter through the calmer local canvas view and reframing the copy around context budget now makes the vocabulary-versus-sequence trade legible inline without dropping rare-word coverage. |
| transformers-rag | transformers-rag.html | embeddings | verified | 4 | 4 | 4 | P2 | The semantic-geometry core was already strong, but the first-read story needed to land faster and the inline state needed to render reliably. The updated meaning-vs-keyword framing plus immediate canvas rendering now make the retrieval lesson easier to scan in one glance. |
| transformers-rag | transformers-rag.html | positional | verified | 4 | 4 | 4 | P1 | The concept was correct but too easy to read as abstract transformer plumbing. Stronger copy around “same words, different order” plus reliable inline rendering now make the content-vs-arrangement split visible before the learner has to parse the formula. |
| transformers-rag | transformers-rag.html | transformer-block | verified | 4 | 4 | 4 | P1 | This was the chapter’s densest systems concept. Reframing the shell around two jobs — context mixing then local feature rewriting — and keeping the card in the chapter-native inline canvas view makes the block feel like one causal sequence instead of memorized plumbing. |
| transformers-rag | transformers-rag.html | attention | verified | 4 | 4 | 4 | P1 | The highest-leverage fix was making the visible shell and the live stage agree on the same soft-lookup story. With inline canvas rendering and plainer score -> softmax -> mix language, Q/K/V now support the main teaching gesture instead of crowding it out. |
| transformers-rag | transformers-rag.html | kv-cache | verified | 4 | 4 | 4 | P2 | The key systems tradeoff is now clearer in the default state: reuse past attention state to save compute, then pay the memory bill. Reliable inline rendering plus “one fresh slice instead of replaying the prefix” language make the speed-vs-memory lesson land more quickly. |
| transformers-rag | transformers-rag.html | rag | verified | 4 | 4 | 4 | P1 | The most important gain was making RAG feel debuggable instead of aspirational. Immediate inline rendering and the stronger retrieval-vs-context-use framing now make the chapter’s practical lesson explicit: first ask whether the evidence fetched was weak or whether the model failed to use good evidence. |
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
| production | production-systems.html | serving-skew | verified | 4 | 4 | 4 | P2 | The main issue was a shell/renderer mismatch: the chapter copy and controls loaded, but the inline stage could sit on a placeholder or diverge into a different React teaching model. Routing the card through the local canvas view restores an immediate default-state lesson, a connected current-read, and a much clearer “same model, different feature reality” story. |
| production | production-systems.html | online-offline | verified | 4 | 4 | 4 | P2 | This card now teaches in the inline state instead of hiding behind a blank shell. The chapter-native bar view makes the proxy-vs-product split legible at a glance, and the shell controls/readout now match the stage instead of competing with a separate renderer model. |
| production | production-systems.html | data-drift | verified | 4 | 4 | 4 | P2 | Drift was strongest once the inline card showed the moving live distribution immediately rather than waiting on a hybrid placeholder. The repaired local canvas view keeps the environment-changed-not-code-broke lesson visible in one glance while preserving the population-vs-concept distinction. |
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
