# Training Labs: Causal Teaching Pass

Scope: `atelier/training-labs.js`, `atelier/training-labs.css`, and
`atelier/tests/training-labs.test.cjs`. Shared rendering, chapter data, Pretext,
HTML, and other lesson families are not changed by this pass.

Every stage starts with one concrete question and one primary action. A selected
element links its actual before/after values to a substituted equation with the
causal term highlighted. Full tables, longer derivations, and alternative
settings remain in native disclosures. Direct cell, token, layer, and request
inspection remains available. Counterfactuals replace the existing optional quiz;
no additional quiz boxes are added. Displayed operands are rounded, while the
underlying computations and numeric regression assertions use full precision.

## Learning Rate

- Causal change: separate flat and sharp coordinate tracks show an update crossing zero while loss decreases; the unrelated coordinate is de-emphasized.
- Action: **Next update** advances the actual two-coordinate quadratic trajectory. The hollow point is the preceding position, not a second synthetic trajectory.
- Equation link: the selected coordinate highlights `eta * gradient` in `theta_new = theta_old - eta * gradient`; curvature 8 gives multiplier `1 - 8 * 0.22 = -0.76`.
- Transfer: doubling sharp-direction curvature to 16 makes the same rate unstable. Limitation: fixed diagonal quadratic, no stochasticity, momentum, or changing Hessian.

## Initialization

- Causal change: paired unit bars compare actual activations from the same random draws at the same selected layer; a scale change compounds through depth rather than merely changing a chart caption.
- Action: **Use smaller starting weights** switches He to Xavier with input, seed, width, and depth held fixed. The default layer-6 selected activation changes from about 1.814 to 0.227.
- Equation link: highlight `(std_current / std_reference)^depth`; the exact factor at six layers is 1/8 when moving He to Xavier in this network.
- Transfer: doubling fan-in requires dividing He standard deviation by `sqrt(2)`. Limitation: eight-wide, bias-free ReLU network with deterministic draws. Positive homogeneity makes the scaling identity exact here; variance preservation in general is an expectation, not a per-layer guarantee.

## Gradient Flow

- Causal change: two signed contribution routes show the incoming gradient, activation derivative, and connecting weight before their sum; norms no longer stand alone as unexplained data.
- Action: **Backprop one layer** traverses the real Jacobian chain. The previous outgoing gradient becomes the next layer's incoming gradient.
- Equation link: highlight each activation derivative in `g_in[j] = sum_i W[i,j] * phi_prime[i] * g_out[i]`; inspect either input gradient directly.
- Transfer: replacing saturated tanh with a linear activation removes the small local derivative but does not guarantee preservation of the total gradient. Limitation: tied 2x2 weights, six layers, terminal loss equal to the sum of outputs; ReLU's derivative at zero follows the zero convention.

## Normalization

- Causal change: only the row or column that supplies the selected output's statistics is emphasized. Changing example 3 can change example 1 under training-mode BatchNorm even though example 1's input is unchanged.
- Action: **Change example 3** changes `[3,6,10]` to `[11,14,18]`; the selected before/after normalization and highlighted statistics update together.
- Equation link: substituted selected input, mean, variance, and epsilon; the mean and variance terms are highlighted. The active method is visible even with settings closed.
- Transfer: another example cannot alter LayerNorm output with fixed affine parameters. Limitation: a 3x3 batch, population moments, gamma 1, beta 0, epsilon 1e-5. RMSNorm uses a mean square without centering; inference BatchNorm running averages are not simulated.

## Residual Connections

- Causal change: the same branch and depth are shown with an inactive or active identity bypass, actual last-block vectors, and the selected stack sensitivity. The default sensitivity changes from -0.001 to 0.729.
- Action: **Add skip paths** changes only the identity contribution, not the learned branch or number of blocks.
- Equation link: highlight identity entries in `J_stack = (W + skip * I) J_previous`; the selected 2x2 Jacobian entry is reproduced by the displayed recurrence.
- Transfer: with branch W=0, six plain blocks erase the input while six residual blocks preserve it. Limitation: linear blocks; residual paths can amplify or cancel, demonstrated by positive branches and W=-I, rather than guaranteeing stability.

## Flash Attention

- Causal change: the active key tile is distinguished from stored tiles, and the denominator bar separates rescaled old exponential mass from the current tile's mass. Output vectors are computed from the rescaled numerator as well.
- Action: **Process tile 2** introduces a larger score maximum; the old mass contracts by about 0.243 instead of being discarded or carried in the wrong scale. Selecting a key also selects its tile.
- Equation link: highlight `exp(old_max - new_max)` and the selected key's `exp(score - new_max)` in the online denominator recurrence. Full numerator and dense-equivalence checks are secondary.
- Transfer: omitting old-mass rescaling is generally wrong even after dividing by the new denominator; identical values can conceal the error. Limitation: one query, four keys, dimension two, two-key tiles, no causal mask or GPU kernel. Score-buffer byte savings are a partial IO accounting, not total kernel traffic or measured speed.

## Pretraining and Fine-Tuning

- Causal change: a selected prompt target starts with mask zero; switching to all-token training visibly gives it a nonzero contribution. Prompt and response groups remain distinct and probabilities stay fixed.
- Action: **Train on every token** changes the actual supervision mask and denominator from four response targets to nine targets.
- Equation link: highlight both `mask` and `supervised_count` in the selected token's normalized `mask * -ln(p) / count`. Masking a target does not delete its contextual representation.
- Transfer: adding masked prompt targets with fixed response probabilities leaves response-only average loss unchanged. Limitation: fixed toy conditional probabilities and a simplified conversation template; this isolates objective differences, not learned model quality.

## Parameter-Efficient Fine-Tuning

- Causal change: the parameter-group map distinguishes trainable from frozen tensors and links the selected cell to its raw gradient, mask, actual update, and resulting loss. Inspecting another tensor no longer undoes an applied step.
- Action: **Apply one update** runs the analytic SGD step. Default B[0,0] changes from 0 to 0.00065; A stays fixed despite being trainable because initial B is zero.
- Equation link: highlight the trainability mask in `delta = -0.1 * mask * gradient`; selected matrix values retain enough precision to agree with the update equation.
- Transfer: trainable A has zero initial gradient when B=0 under plain SGD without weight decay. Limitation: one example, two linear layers, one step; reported Adam moment bytes are theoretical m/v storage only, not a simulated Adam optimizer or total memory.

## LoRA

- Causal change: the selected B row and A column are shown directly above their correction and effective weight. Unrelated merged cells remain inspectable without competing for attention.
- Action: **Apply example adapter** replaces zero B with fixed example factors, changing `W + BA` without modifying frozen W.
- Equation link: highlight the selected products in `W_effective[i,j] = W[i,j] + sum_k B[i,k] A[k,j]`. Merged and unmerged forward paths remain checked in the calculation disclosure.
- Transfer: multiplying B by two and dividing A by two preserves BA, though training dynamics need not be identical. Limitation: fixed illustrative factors rather than a quality claim or simulated training; alpha/r is one, and rank two is not cheaper than the base for this tiny 4x3 example.

## Quantization

- Causal change: actual discrete grid ticks move when the shared clipping range changes. The default small weight 0.3 reconstructs as 0 before clipping and 2/7 after clipping; the outlier error remains explicitly visible in the consequence text.
- Action: **Clip the outlier** narrows the shared range from 5 to 1 while keeping the same bit width and original weights.
- Equation link: highlight `clip_limit / qmax`, then show the selected integer code and reconstruction `scale * q`. Full error tables and packed-byte accounting remain optional.
- Transfer: raising precision from four to eight bits cannot restore an original weight of 5 after clipping at 1. Limitation: symmetric per-tensor quantization with ties toward positive infinity; one unused signed code, theoretical packed storage, no task-quality or throughput prediction.

## Distillation

- Causal change: candidate bars show teacher targets, current student probabilities, and a dashed previous-probability marker on a fixed 0-to-1 scale. The selected logit update and KL change are linked rather than implying logits and probabilities move identically.
- Action: **Train the student once** applies the real gradient of the T-squared-scaled KL. At the default second update, B's logit rises while its probability falls because competing logits also move.
- Equation link: highlight `(student_probability - target_probability)` in `z_new = z_old - 0.5 * T * (q - p)` using the preceding state, not the post-update gradient.
- Transfer: adding the same constant to all student logits preserves probabilities and KL. Limitation: three directly optimized logits rather than a parameterized student network; teacher targets are assumed, not calibrated ground truth.

## Serving Tradeoffs

- Causal change: each request has a serial first-token reference and actual current token times on the same fixed time axis. Selected TTFT is decomposed into queue wait, prefill, and first output round.
- Action: **Serve without batching** switches the same request workload from a batch at 10 ms to immediate serial service. Request A's TTFT changes from 24.5 ms to 8.5 ms even though modeled aggregate token rate falls.
- Equation link: highlight queue wait in `TTFT = wait + prefill + first_output_round`; click other requests to see that the latency effect is not uniform.
- Transfer: four-times-longer prompts increase prefill, TTFT, and KV payload under this model even with fixed output counts. Limitation: deterministic stated costs, no continuous batching or hardware benchmark; the first output is delayed one modeled round after prefill logits, and KV counts processed tokens rather than the newest unconsumed sampled token.

## Verification and Screenshot Review

- `node --check atelier/training-labs.js` and `node --test atelier/tests/training-labs.test.cjs` are the scoped checks. The suite contains 60 passing tests, including finite-difference gradients, exact online/dense attention, all reachable button states, every primary action's numeric effect, selection-to-equation bindings, counterfactual calculations, and readable inactive text styling.
- A separate headless run uses the existing read-only `render-review.cjs` harness for these two chapters only, with output at `/tmp/atelier-causal-training-final`. It tests 320, 390, 720, and 1440px; the parent report at `/tmp/atelier-causal-teaching-final` is preserved.
- All 12 parent desktop and mobile first-view screenshots were opened. They mixed in-progress HTML with older cached CSS; several desktop adaptation screens also predated the new causal markup. These are not treated as final evidence.
- All 12 refreshed desktop and mobile before/after pairs were opened. Post-screenshot owned-file fixes bounded the residual diagram, restored queue/probability legends, showed the normalization method, aligned PEFT cell precision, removed whole-node text opacity, compacted mobile token/probability layouts, and removed the local question-size override in favor of shared typography. All final first views were reopened at both widths, along with eight available bottom screenshots. No shared browser was used or reloaded.
- Additional 320px after-action spot checks covered pretraining/SFT, distillation, and PEFT. PEFT exposed a decimal-wrapping problem despite passing overflow assertions; its vector now uses two columns below 380px, without shrinking the text or reflowing true matrices. The calculation disclosure uses matching precision too. The final four-width render reports 48 checks, 48 changed interactions, and zero errors.
