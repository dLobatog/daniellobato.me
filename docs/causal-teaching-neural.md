# Neural-network teaching pass

Baseline: `1fe1e429ecac7ba898b760730aab5e53c4bfb692`.
Rollback: `codex/atelier-before-causal-teaching-20260925`.

This is an implementation ledger, not a claim of measured learning gains.
Keep the real 9-2-1 image classifier and exact quadratic optimization model.

| Concept | First action and visible mechanism | Formula connection | Transfer check |
| --- | --- | --- | --- |
| Neuron | Toggle a pixel in a combined pixel-times-weight tile. Its contribution disappears from the sum; prior and current scores remain adjacent. | Selected input times its actual weight; sum plus bias; ReLU on that sum. | Does a dark pixel's weight affect this image? |
| Activations | Push the same input from +1 to +4. Compare surviving gradients before exploring curve controls. | Local derivative beside the selected point; incoming gradient 1 times this derivative. | Can a large activation pass a tiny gradient? |
| Output functions | Add evidence for dog while cat's score stays fixed. Cat's sigmoid stays fixed but its softmax share falls. | The actual cat numerator and highlighted shared denominator update with scores. | Would independent image tags need to add to 100%? |
| Forward pass | Compute features, then a prediction, then loss. Each step reveals the next operation with weights fixed. | Current hidden values substituted into the output score, then sigmoid; only the last step uses the label. | Why do equal hidden activations produce a tie with opposing output weights? |
| Chain rule | Compare a lit and dark input connection. The zero factor blocks the path in place; a numerical probe verifies the product. | Four actual local derivatives and their product, not invented gain knobs. | What happens if a different factor, the ReLU slope, becomes zero? |
| Backpropagation | Trace error, prepare an update, run the same image again. Distinguish gradients from parameter changes. | Selected old weight minus learning rate times its exact gradient, with actual new weight. | Does backward alone change the prediction? |
| Gradient descent | Take the next update on a fixed loss landscape, compare the actual loss before and after. | Vertical coordinate update uses actual old weight, rate, direction and result. | Can a downhill direction produce a worse destination? |
| Optimizers | Switch the first update between SGD and Adam on the same gradient; the move, not the problem, changes. | Current direction is identified as slope, remembered velocity, or normalized history. | Does normalization imply Adam wins every problem? |
| LR schedules | At update 13, remove the tenfold rate drop. Compare moves from exactly the same weights and gradient. | Highlight the changed rate in the selected coordinate's update. | Does less movement imply less remaining error? |

## Shared changes

- Chapter entry is shorter. Removed repeated concept numbering and summary before the lesson; the full explanation, formula, code and quiz remain available.
- Advanced images, training labels, full tables and alternate configurations follow the first causal experiment.
- No blanket new quiz boxes. Existing checks remain optional and test reasoning.
- Preserve visible legends when controls move into disclosures.

## Validation

- Pure classifier, finite differences, activations and optimizer tests run during implementation.
- Render review at 1440, 720 and 390 pixels; final all-library report supersedes intermediate screenshots.
- Screenshot review is visual QA, not a learning-outcome study. This tiny classifier is not a realistic handwriting recognizer; synthetic quadratic results do not rank optimizers generally.
