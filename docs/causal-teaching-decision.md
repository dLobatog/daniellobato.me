# Decision Labs: Causal Teaching Pass

Scope: 11 unique renderers, 12 public topics. Reinforcement learning owns MDP,
value functions, TD, Q-learning, linear DQN, and bandits. Generative/RL reuses
bandits and adds diffusion. Alignment adds CFG, DPO, reward hacking, and GRPO.

No shared shell, Pretext integration, public HTML, or other families are changed.
The existing optional quiz slot carries the counterfactuals below; no stage quiz
boxes are added. Parent owns rendered desktop/mobile QA.

## Interaction Contract

Each initial stage has a concrete question and one primary action. Settings,
alternate transitions, full tables, and complete histories stay in native
disclosures. Legends, value types, active equations, and model limitations stay
in the stage. Highlighted terms belong to the selected transition, coordinate,
program, or answer, not to arbitrary sliders.

An applied change retains one immutable before-state, with no recursive history.
The learner sees the change just made rather than immediately losing it to the
next update preview. Before the first update, appropriate lessons explicitly
label the calculation as a preview; seeded observations and policy snapshots are
labeled as comparisons, not newly trained results. Display values are rounded;
all computations use full JavaScript numerical precision.

## MDP

- Causal change: retain the selected edge after acting and move the visible current-state marker from its source to its destination. A zero reward changes fresh to engaged and unlocks the relevant follow-up; it does not magically increase return.
- Action: take the selected recommendation; alternative actions/start states are secondary.
- Equation linkage: the highlighted reward contributes `0.9^episode_step * reward` to the observed return. The two-step trajectory ultimately contributes `0 + 0.9 * 4 = 3.6`.
- Transfer: if the follow-up reward vanished, depth would return zero and lose to the immediate reward of one.
- Limits: deterministic, fully observed, two-decision environment with declared utility, not product measurements.

## Value Functions

- Causal change: clicking a state changes the equation's focus. The first main action propagates engaged value 4 backward, changing fresh value from 1 to 3.6 while its immediate reward remains unchanged.
- Action: perform one synchronous backup sweep.
- Equation linkage: `Q(depth) = 0 + 0.9 * 4 = 3.6`, then `V(fresh) = max(1, 3.6)`. The downstream estimate is highlighted. Fixed-policy mode uses a mean instead of max.
- Transfer: reducing gamma to 0.2 makes depth worth 0.8, below the quick click's 1.
- Limits: exact known model; synchronous sweeps, not sampled updates. Gamma 1 is valid for this finite terminating environment.

## TD Learning

- Causal change: retain the transition just learned. The first click changes only engaged value, 0 to 2. The following click carries that estimate into fresh value, 0 to 0.9.
- Action: learn from one sampled transition under a fixed depth/follow-up policy.
- Equation linkage: `target = 4 + 0.9 * 0`, then `V = 0 + 0.5 * (4 - 0) = 2`. The TD-error term is highlighted. Nonterminal steps substitute the learned next estimate instead.
- Transfer: if the downstream estimate were zero, a zero-reward transition from a zero-valued state would teach nothing yet.
- Limits: TD(0), deterministic repeated episodes, fixed alpha 0.5, no max operator or Monte Carlo return.

## Q-Learning

- Causal change: preserve the changed Q cell and compare Q-learning's target with SARSA on the same evidence. Dim the next action not used by the active backup rule.
- Action: learn from the selected transition; tapping action nodes inspects their terminal transitions.
- Equation linkage: highlight `max(0.5, 2)` in `target = 0 + 0.9 * max(...) = 1.8`; the selected cell changes `0 + 0.5 * 1.8 = 0.9`. SARSA's recorded-ad target is 0.45.
- Transfer: changing the recorded next action changes SARSA's target, not Q-learning's max, when the table is held fixed.
- Limits: tiny tabular control example; off-policy does not establish offline safety or adequate exploration.

## DQN

- Causal change: distinguish a gradient update from copying the target. The first update moves prediction 0.5 to 0.63 and loss 0.845 to 0.68445 while target stays 1.8. A subsequent copy changes target to 1.917 without changing online weights.
- Action: train on one stored replay item; the target node directly copies weights.
- Equation linkage: highlight error 1.3 in `Q_after = 0.5 + 0.1 * 1.3 * 1`; show the same frozen target in the substituted half-squared loss. For engaged features, the feature squared norm is 2, not 1.
- Transfer: changing only online weights cannot change the same replay target in this frozen-network setup.
- Limits: explicitly a two-feature linear approximator, not deep learning; selected sample instead of random minibatch. No stability or convergence guarantee.

## Bandits (Both Public Sections)

- Causal change: retain the selected arm's pre-observation estimate and actual reward. Other arm means are de-emphasized rather than falsely updated. The first new pull chooses C, observes 1, and changes its mean from 0.5 to 2/3.
- Action: try one recommendation using the active policy.
- Equation linkage: show both the chosen UCB index `0.5 + sqrt(2 log(8)/2)` and the update `0.5 + (1 - 0.5)/3`. Counts belong to the selected arm; the logarithm uses prior total pulls.
- Transfer: another zero on B does not reduce the unpulled arms' means, although their UCB time bonuses can change.
- Limits: stationary Bernoulli rewards, three context-free arms, seeded per-arm streams. Hidden means are evaluation-only; one seed is not a statistical policy comparison.

## Diffusion

- Causal change: plot the vector before the selected reverse step against the current vector, not only against the original noisy start. Keep the applied step when a different coordinate is selected; de-emphasize unrelated points.
- Action: remove one schedule step of noise. The first step changes coordinate 3 from approximately 0.107 to 0.242.
- Equation linkage: show the selected coordinate's actual noise prediction in `x0_hat = (x_t - sqrt(1-ab_t) * eps_hat) / sqrt(ab_t)`, followed by the substituted recombination at the previous schedule.
- Transfer: overly positive predicted noise decreases the reconstructed clean coordinate because the prediction is subtracted.
- Limits: known-clean-vector oracle, optionally with fixed bias; deterministic DDIM with eta 0; finite-noise reconstruction, not a learned network or pure-noise generation demonstration.

## Classifier-Free Guidance

- Causal change: retain the prior guided vector and hold noisy input and both model outputs fixed. The first action changes `[0.6, -0.1]` to `[1.4, -0.5]`, demonstrating extrapolation rather than convex averaging. Plot only the three actual prediction vectors; explain their scale-1 coincidence instead of drawing a misleading conditional-colored guide.
- Action: strengthen the prompt direction; arbitrary scale selection stays secondary.
- Equation linkage: both vector coordinates substitute `unconditional + scale * (conditional - unconditional)`, highlighting the shared changed scale.
- Transfer: identical conditional and unconditional predictions make every guidance scale produce the same result.
- Limits: fixed 2D model outputs, not a trained generator; no fabricated fidelity, diversity, or quality measures.

## DPO

- Causal change: retain the probability update and loss decrease, approximately 0.641 to 0.586, while reference ticks stay fixed. Show the opposite chosen/rejected logit changes and de-emphasize the other completion. Large logit values always match the current probability bars; captions explicitly distinguish a preview from an applied change.
- Action: favor the chosen answer to a concrete evaluation claim.
- Equation linkage: substitute the actual chosen and rejected reference-relative log-ratios into the highlighted margin, then `-log sigmoid(beta * margin)` into the loss.
- Transfer: halving both chosen and rejected probabilities leaves their odds and this pair loss unchanged, provided reference probabilities stay fixed.
- Limits: three categorical complete responses, not a tokenized language model. The other logit's direct gradient is zero but its probability can change through normalization. Beta is not the learning rate.

## Reward Hacking

- Causal change: before/after markers now compare adjacent optimization rounds, not a distant untrained baseline. Pair the selected program's rising probability with a concrete held-out failure. The first click raises constant-5 probability from approximately 0.792 to 0.858 despite returning 5 for empty input, whose sum is 0.
- Action: optimize the public-test reward for one more round.
- Equation linkage: highlight `public_accuracy - 0.05 * declared_cost`, then substitute the selected probability, exponent, and actual normalization constant into the policy update.
- Transfer: equalizing costs does not repair the verifier; general sum and constant 5 would still tie on the public tests.
- Limits: finite fixed candidate programs and declared cost units. Held-out tests never enter the update; additional training cases are independent of audit inputs.

## GRPO

- Causal change: retain the prior policy snapshot's selected surrogate term. The first action changes a rewarded answer's ratio from 1.16 to 1.4, but its surrogate rises only from 1.16 to 1.2. Highlight the actual minimum branch, including the opposite clipping direction for negative advantage. Compact the answer group and remove the duplicate advantage card so the clipping equation remains in the first 390x900 screen; keep reward mean and population std visible.
- Action: make rewarded answers likelier; the next snapshot reverses the shift to expose the unfavorable-move branch.
- Equation linkage: substitute reward, group mean/std, both `ratio * advantage` branches, and the reference KL penalty. Equal-reward mode shows the explicit zero-variance guard rather than division by zero.
- Transfer: if all rewards become 1, all relative advantages become zero even though some answers remain wrong; KL may still contribute.
- Limits: fixed one-token categorical snapshots, not claimed optimizer steps; population std; original outcome-supervised form. Old rollout probabilities and reference probabilities serve distinct roles. The finite sample KL mean is not exact full-distribution KL.

## Verification

Run `node --test atelier/tests/decision-labs.test.cjs` and `node --check atelier/decision-labs.js`.
Tests cover exact updates, finite-difference gradients, seeded bandit histories,
applied-event retention, selected-coordinate/transition equation values,
positive/negative GRPO clipping, zero variance, counterfactual mathematics, and
the screenshot-confirmed mobile flex-sizing CSS regression. Secondary calculation
tests also require Bellman operands/results to use the same sweep and DQN's
post-copy calculation to use the newly copied target, without claiming another
gradient step occurred.

The opt-in `DECISION_BROWSER_TEST=1` test launches a separate headless browser,
not a parent browser session. Set `PLAYWRIGHT_MODULE` and `CHROME_PATH` to local
installations. It renders only the 12 owned public topics and checks before/after
content bounds at 320, 390, 720, and 1440px. It also asserts the MDP current-state
marker moves and the GRPO clipping equation stays in the first 390x900 screen.
It writes screenshots and a report to `/tmp/atelier-causal-decision-final`, never
the parent's report directory. The full enabled run passes 53 tests, including
96 browser before/after checks; the non-browser run skips that one opt-in test.

## Rendered Review

Actually viewed the parent's 1440px and 390px initial AND after-action PNGs for
all 12 public topics (48 images):

- `reinforcement-learning/{mdp,value-functions,td-learning,q-learning,dqn,exploration}`.
- `generative-and-rl/{diffusion,bandit}`.
- `alignment-depth/{guidance,dpo,reward-hacking,grpo}`.

Also viewed the parent's bottom captures for TD, Q-learning, DQN, exploration,
bandit, diffusion, DPO, reward hacking, and GRPO at 390px, plus diffusion, CFG,
DPO, and reward hacking at 1440px. These were actual images, not inferred from
the global geometry report. No label overlap remained in those captures.

Post-parent-capture changes are in `atelier/decision-labs.js` and
`atelier/decision-labs.css`: moving MDP state marker; removing CFG's misleading
extrapolation guide; making DPO's current logits match its probability bars;
compacting GRPO without hiding its mean/std or clipping equation. The numerical
review also fixed the Bellman disclosure's old-result mismatch, the DQN copied
target calculation, the SARSA-mode reason, and the reward-hacking hook (reward
increases here, not public-test accuracy). Tests and this document were updated.

Those JS/CSS changes postdate the parent's captures. The separate final rerender
contains the new styles; do not use the parent's old screenshots to assess the
MDP marker or GRPO compact layout. Fresh desktop/mobile initial and after images
for MDP, CFG, DPO, and GRPO were visually rechecked after the corrections.
