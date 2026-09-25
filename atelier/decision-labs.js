(function () {
  'use strict';

  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
  const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
  const softmax = xs => {
    const exps = xs.map(x => Math.exp(x - Math.max(...xs)));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
  };
  const STATES = ['Fresh session', 'Engaged session', 'Terminal'];
  const ACTIONS = [['Quick click', 'Useful depth'], ['Show ad', 'Relevant follow-up']];
  const TRANSITIONS = [
    { s: 0, a: 0, r: 1, next: 2, done: true },
    { s: 0, a: 1, r: 0, next: 1, done: false },
    { s: 1, a: 0, r: 0.5, next: 2, done: true },
    { s: 1, a: 1, r: 4, next: 2, done: true },
  ];
  function transition(s, a) {
    const result = TRANSITIONS.find(t => t.s === s && t.a === a);
    if (!result) throw new RangeError('No action from this state');
    return { ...result };
  }
  function backup(values, gamma = 0.9, policy = 'optimal') {
    return [0, 1].map(s => {
      const qs = [0, 1].map(a => {
        const t = transition(s, a);
        return t.r + (t.done ? 0 : gamma * values[t.next]);
      });
      return policy === 'uniform' ? mean(qs) : Math.max(...qs);
    }).concat(0);
  }
  function valueTrace(steps, gamma = 0.9, policy = 'optimal') {
    const trace = [[0, 0, 0]];
    for (let i = 0; i < steps; i++) trace.push(backup(trace.at(-1), gamma, policy));
    return trace;
  }
  function tdUpdate(values, t, alpha = 0.5, gamma = 0.9) {
    const target = t.r + (t.done ? 0 : gamma * values[t.next]);
    const error = target - values[t.s];
    const after = [...values];
    after[t.s] += alpha * error;
    return { target, error, before: values[t.s], after, updated: after[t.s] };
  }
  function qUpdate(q, t, alpha = 0.5, gamma = 0.9, method = 'q', nextAction = 0) {
    const bootstrap = t.done ? 0 : method === 'sarsa' ? q[t.next][nextAction] : Math.max(...q[t.next]);
    const target = t.r + gamma * bootstrap;
    const error = target - q[t.s][t.a];
    const after = q.map(row => [...row]);
    after[t.s][t.a] += alpha * error;
    return { target, bootstrap, error, before: q[t.s][t.a], after, updated: after[t.s][t.a] };
  }
  const features = s => s === 2 ? [0, 0] : [1, s];
  const linearQ = (weights, s) => weights.map(w => dot(w, features(s)));
  function dqnUpdate(online, frozen, t, alpha = 0.1, gamma = 0.9) {
    const pred = linearQ(online, t.s)[t.a];
    const next = t.done ? 0 : Math.max(...linearQ(frozen, t.next));
    const target = t.r + gamma * next;
    const error = target - pred;
    const after = online.map(row => [...row]);
    after[t.a] = after[t.a].map((w, j) => w + alpha * error * features(t.s)[j]);
    return { pred, next, target, error, loss: 0.5 * error * error, after };
  }
  function random(seed) {
    let x = seed >>> 0;
    return () => { x = (Math.imul(1664525, x) + 1013904223) >>> 0; return (x + 0.5) / 4294967296; };
  }
  const ARM_MEANS = [0.35, 0.5, 0.7];
  function bandit(steps = 8, policy = 'ucb', seed = 19) {
    // Separate per-arm streams keep each arm's nth outcome fixed across policies/horizons.
    const outcomes = ARM_MEANS.map((p, a) => {
      const stream = random(seed + 997 * a);
      return Array.from({ length: steps + 1 }, () => Number(stream() < p));
    });
    const choose = random(seed + 5003);
    const counts = [0, 0, 0], wins = [0, 0, 0], history = [];
    let regret = 0;
    for (let t = 0; t < steps; t++) {
      const estimates = counts.map((n, a) => n ? wins[a] / n : 0);
      const scores = estimates.map((q, a) => policy === 'ucb' ? (counts[a] ? q + Math.sqrt(2 * Math.log(Math.max(1, t)) / counts[a]) : Infinity) : q);
      let reason = 'greedy estimate';
      let a;
      if (t < 3) { a = t; reason = 'initial coverage'; }
      else if (policy === 'epsilon' && choose() < 0.15) { a = Math.floor(choose() * 3); reason = 'epsilon exploration'; }
      else { a = scores.indexOf(Math.max(...scores)); if (policy === 'ucb') reason = 'largest UCB index'; }
      const reward = outcomes[a][counts[a]];
      counts[a]++; wins[a] += reward;
      regret += Math.max(...ARM_MEANS) - ARM_MEANS[a];
      history.push({ t: t + 1, a, reward, reason, estimates, scores, regret });
    }
    return { counts, wins, estimates: counts.map((n, a) => n ? wins[a] / n : 0), history, regret };
  }
  const CLEAN = [-0.9, -0.7, 0.6, 1, 0.8, -0.5, -1, 0.4];
  const BETAS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35];
  const ALPHAS = BETAS.reduce((xs, b) => xs.concat(xs.at(-1) * (1 - b)), [1]);
  function gaussianVector(n, seed = 23) {
    const rng = random(seed);
    return Array.from({ length: n }, () => Math.sqrt(-2 * Math.log(rng())) * Math.cos(2 * Math.PI * rng()));
  }
  function forwardNoise(clean, epsilon, alphaBar) {
    return clean.map((v, i) => Math.sqrt(alphaBar) * v + Math.sqrt(1 - alphaBar) * epsilon[i]);
  }
  function ddimStep(x, eps, alphaBar, previousAlphaBar) {
    const x0 = x.map((v, i) => (v - Math.sqrt(1 - alphaBar) * eps[i]) / Math.sqrt(alphaBar));
    return { x0, previous: forwardNoise(x0, eps, previousAlphaBar) };
  }
  function diffusionTrace(biased = false) {
    const epsilon = gaussianVector(CLEAN.length);
    let x = forwardNoise(CLEAN, epsilon, ALPHAS.at(-1));
    const trace = [];
    for (let t = BETAS.length; t > 0; t--) {
      const oracle = x.map((v, i) => (v - Math.sqrt(ALPHAS[t]) * CLEAN[i]) / Math.sqrt(1 - ALPHAS[t]));
      const eps = oracle.map((v, i) => v + (biased ? (i % 2 ? -0.3 : 0.3) : 0));
      const step = ddimStep(x, eps, ALPHAS[t], ALPHAS[t - 1]);
      trace.push({ t, x, eps, ...step });
      x = step.previous;
    }
    trace.push({ t: 0, x, eps: Array(CLEAN.length).fill(0), x0: x, previous: x });
    return trace;
  }
  const guidedNoise = (unconditional, conditional, scale) => unconditional.map((v, i) => v + scale * (conditional[i] - v));
  function dpo(logits, ref, beta = 0.5) {
    const max = Math.max(...logits);
    const logNormalizer = max + Math.log(logits.reduce((sum, x) => sum + Math.exp(x - max), 0));
    const logProbabilities = logits.map(x => x - logNormalizer);
    const probabilities = logProbabilities.map(Math.exp);
    const logRatios = logProbabilities.map((logp, i) => logp - Math.log(ref[i]));
    const margin = logRatios[0] - logRatios[1];
    const z = beta * margin;
    const loss = Math.max(0, -z) + Math.log1p(Math.exp(-Math.abs(z)));
    const gradient = -beta / (1 + Math.exp(z));
    return { probabilities, logRatios, margin, z, loss, gradient };
  }
  function dpoStep(logits, ref, beta = 0.5, lr = 0.5) {
    const { gradient } = dpo(logits, ref, beta);
    return logits.map((x, i) => x - lr * (i === 0 ? gradient : i === 1 ? -gradient : 0));
  }
  const PROGRAMS = [
    { name: 'General sum', code: 'return sum(xs)', cost: 8, run: xs => xs.reduce((a, b) => a + b, 0) },
    { name: 'Two-item shortcut', code: 'return xs[0] + xs[1]', cost: 5, run: xs => xs.length >= 2 ? xs[0] + xs[1] : NaN },
    { name: 'Constant exploit', code: 'return 5', cost: 1, run: () => 5 },
  ];
  const PUBLIC_CASES = [[2, 3], [1, 4]];
  const HELDOUT_CASES = [[], [1, 2, 3], [-2, 2], [10]];
  const EXTRA_CASES = [[-1, 1], [4, 3, 2], [8]];
  const accuracy = (program, cases) => mean(cases.map(xs => Number(program.run(xs) === xs.reduce((a, b) => a + b, 0))));
  function proxyExperiment(round = 4, repaired = false) {
    const cases = repaired ? PUBLIC_CASES.concat(EXTRA_CASES) : PUBLIC_CASES;
    const rewards = PROGRAMS.map(p => accuracy(p, cases) - 0.05 * p.cost);
    const probabilities = softmax(rewards.map(r => 2 * round * r));
    const heldout = PROGRAMS.map(p => accuracy(p, HELDOUT_CASES));
    return { rewards, probabilities, heldout, proxy: dot(probabilities, rewards), audit: dot(probabilities, heldout) };
  }
  function groupAdvantages(rewards) {
    const avg = mean(rewards);
    const std = Math.sqrt(mean(rewards.map(r => (r - avg) ** 2)));
    return { mean: avg, std, advantages: rewards.map(r => std < 1e-12 ? 0 : (r - avg) / std) };
  }
  function grpoTerm(advantage, current, old, ref, beta = 0.04, epsilon = 0.2) {
    const ratio = current / old;
    const clippedRatio = clamp(ratio, 1 - epsilon, 1 + epsilon);
    const raw = ratio * advantage, clipped = clippedRatio * advantage;
    const surrogate = Math.min(raw, clipped);
    const logRefRatio = Math.log(ref / current);
    const kl = Math.expm1(logRefRatio) - logRefRatio;
    return { ratio, clippedRatio, raw, clipped, surrogate, kl, objective: surrogate - beta * kl, active: clipped < raw };
  }

  const escape = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const f = (x, digits = 3) => Number.isFinite(x) ? x.toFixed(digits) : 'not defined';
  function lesson(title, summary, why, interview, formula, annotations, snippet, prompt, good, bad, details = []) {
    return { title, summary, what: summary, why, interview, details,
      math: { title: 'Mechanism and annotated quantities', formula, note: 'The values in the interactive lesson are computed from these equations.', annotations },
      code: { title: 'Minimal numerical update', lang: 'python', snippet },
      quiz: { prompt, options: [{ text: good[0], correct: true, explanation: good[1] }, { text: bad[0], correct: false, explanation: bad[1] }] },
      controls: [], presets: [], geometry: null };
  }
  const content = {
    mdp: lesson('MDP: inspect what an action changes', 'A quick click ends this session. Useful depth earns nothing now but unlocks a relevant follow-up.',
      'Recommendation actions alter the state distribution, not just immediate engagement labels.',
      'State sufficiency is an assumption: hidden user intent can make an observed session state non-Markov.',
      ['P(s_{t+1},r_{t+1}\\mid s_t,a_t)', 'G_t=r_{t+1}+\\gamma r_{t+2}+\\cdots'],
      [['s', 'Observed session state', 'Fresh or engaged'], ['a', 'Recommendation decision', 'Useful depth'], ['r', 'Specified utility, not a fitted score', '0 now, 4 on relevant follow-up'], ['P', 'Transition kernel', 'This toy has probability 1 per edge']],
      's = 0\na = 1  # useful depth\nr, next_s, done = 0.0, 1, False\n# The next recommendation depends on next_s.',
      'Why is useful depth potentially better than a quick click?', ['It changes which future reward is reachable.', 'At discount 0.9, depth followed by relevance returns 0 + 0.9*4 = 3.6.'], ['Its immediate reward is larger.', 'Its immediate reward is zero; the quick click pays one.'],
      ['Deterministic, fully observed, two-decision toy. Rewards are hand-specified utilities, not real product measurements.']),
    'value-functions': lesson('Value backups: propagate utility through the session', 'Apply synchronous Bellman backups. The first sweep learns the terminal payoff; the second carries it back to the fresh session.',
      'Distinguish evaluating a fixed policy from optimizing a policy; replacing an expectation with max changes the question.',
      'V under a fixed policy averages its actions. Optimal V takes the best action, with terminal bootstrap zero.',
      ['Q_k(s,a)=r(s,a)+\\gamma V_k(s^{\\prime})', 'V_{k+1}(s)=\\max_a Q_k(s,a)', 'V^\\pi_{k+1}(s)=\\sum_a\\pi(a\\mid s)Q_k(s,a)'],
      [['gamma', 'Discount per decision', '0.9'], ['V_0', 'Initial estimate', 'All zero'], ['V_2(fresh)', 'Optimal two-step value', '3.6'], ['V(terminal)', 'No future utility', '0']],
      'q = reward + gamma * (0 if done else old_v[next_s])\nnew_v[s] = max(q_for_actions)  # optimality\n# For a uniform policy: mean(q_for_actions)',
      'Under the optimality backup, why is the fresh-state value only 1 after the first sweep?', ['Synchronous updates use the previous sweep.', 'The engaged-state estimate was still zero when the fresh-state backup was computed.'], ['Discounting removes the delayed reward.', 'The delayed reward appears on the next sweep, discounted by gamma.']),
    'td-learning': lesson('TD(0): update from one sampled transition', 'Replay a depth-follow-up episode. The first transition has no reward; a later episode can bootstrap from what the terminal reward taught us.',
      'TD trades complete-return waiting for a target partly supplied by an imperfect value estimate.',
      'TD(0) is prediction here: the policy always chooses useful depth, then relevant follow-up. No max operator.',
      ['\\delta=r+\\gamma(1-d)V(s^{\\prime})-V(s)', 'V(s)\\leftarrow V(s)+\\alpha\\delta'],
      [['alpha', 'Step size', '0.5'], ['d', 'Terminal mask', '1 for the follow-up transition'], ['delta', 'Target minus current estimate', '4 - 0 = 4 on first terminal update']],
      'target = r + gamma * (0 if done else V[next_s])\ndelta = target - V[s]\nV[s] += 0.5 * delta',
      'Does the first zero-reward transition immediately teach V(fresh)=3.6?', ['No, its bootstrap estimate is initially zero.', 'TD can only propagate the downstream estimate once it has been learned.'], ['Yes, TD observes the entire future.', 'That would use a complete return, not a one-step TD target.']),
    'q-learning': lesson('Q-learning: separate behavior from the backup policy', 'Inspect a sampled useful-depth transition. Behavior selects the ad next, but Q-learning backs up the best next action.',
      'Logged or exploratory behavior does not have to match the greedy policy being learned, although coverage still matters.',
      'SARSA uses the sampled next action; Q-learning uses max. Off-policy does not mean offline data is automatically safe.',
      ['y_Q=r+\\gamma(1-d)\\max_{a^{\\prime}}Q(s^{\\prime},a^{\\prime})', 'y_{SARSA}=r+\\gamma(1-d)Q(s^{\\prime},a^{\\prime}_{sampled})', 'Q(s,a)\\leftarrow Q(s,a)+\\alpha[y-Q(s,a)]'],
      [['Q(engaged, ad)', 'Exploratory next action estimate', '0.5'], ['Q(engaged, relevant)', 'Greedy next action estimate', '2'], ['y_Q / y_SARSA', 'Same transition, different backup', '1.8 / 0.45']],
      'bootstrap = max(Q[next_s])  # Q-learning\n# SARSA: bootstrap = Q[next_s, sampled_next_action]\ntarget = r + gamma * (0 if done else bootstrap)\nQ[s, a] += 0.5 * (target - Q[s, a])',
      'Behavior picks the ad next. What enters the Q-learning target?', ['The maximum next Q, not the sampled ad Q.', 'The target policy is greedy even when behavior explores.'], ['The ad Q because it was sampled.', 'That is the SARSA target.']),
    dqn: lesson('DQN: train online weights against a frozen target', 'Pick a stored transition, take a gradient step, then copy online weights to the target. Observe exactly when the target moves.',
      'Replay reuses nonconsecutive experience; a lagged network prevents every gradient step from redefining its own target.',
      'A target network is not a convergence guarantee. Function approximation, bootstrapping, and off-policy data can still interact badly.',
      ['Q_\\theta(s,a)=w_a^T[1,s]', 'y=r+\\gamma(1-d)\\max_{a^{\\prime}}Q_{\\theta^-}(s^{\\prime},a^{\\prime})', 'L=\\tfrac12(Q_\\theta(s,a)-\\operatorname{stopgrad}(y))^2'],
      [['theta', 'Online action weights', 'Updated by each SGD step'], ['theta^-', 'Frozen copy', 'Changes only on Copy target'], ['phi(s)', 'Shared features', 'Fresh [1,0], engaged [1,1]']],
      '# optimizer = torch.optim.SGD(online_net.parameters(), lr=0.1)\noptimizer.zero_grad()\nwith torch.no_grad():\n    bootstrap = 0.0 if done else target_net(next_s).max()\n    target = r + gamma * bootstrap\nloss = 0.5 * (online_net(s)[a] - target).square()\nloss.backward()\noptimizer.step()\n# Periodically: target_net.load_state_dict(online_net.state_dict())',
      'Why can training one state change the prediction at another?', ['Both states share weights.', 'The bias feature is shared by fresh and engaged states, unlike a Q-table.'], ['Replay rewrites the reward.', 'Stored rewards are unchanged. Function approximation couples predictions.'],
      ['This is a two-feature linear approximator, not a deep network. It isolates the replay/target mechanism; a single selected sample replaces a random minibatch.']),
    bandit: lesson('Bandits: estimates must be earned from selected actions', 'Compare greedy, epsilon-greedy, and UCB on reproducible Bernoulli feedback. The policy only sees rewards from arms it pulls.',
      'A recommender can be confidently wrong if exposure depends on its own early, noisy estimates.',
      'Pseudo-regret uses hidden simulator means; it is not observable from a production reward log alone.',
      ['\\hat\\mu_a=\\frac{\\sum_{t:a_t=a}r_t}{N_a}', 'UCB_a=\\hat\\mu_a+\\sqrt{2\\log t/N_a}', 'R_T=\\sum_{t=1}^T(\\mu_* - \\mu_{a_t})'],
      [['N_a', 'Number of actual pulls', 'One initial pull per arm'], ['mu', 'Hidden Bernoulli rates', '0.35, 0.50, 0.70'], ['epsilon', 'Random-action probability', '0.15, not an uncertainty estimate']],
      'a = argmax(means + sqrt(2 * log(t) / counts))\nr = env.pull(a)\ncounts[a] += 1\nmeans[a] += (r - means[a]) / counts[a]',
      'Can a low empirical mean after one pull prove an arm is bad?', ['No; one Bernoulli outcome is weak evidence.', 'UCB explicitly adds an uncertainty bonus to under-sampled arms.'], ['Yes, greedy estimates are the true means.', 'The true means are hidden; estimates contain sampling noise.'],
      ['Three stationary, context-free arms; no delayed rewards or exposure confounding. Each arm has a seeded outcome stream. A single seed is not a statistical policy comparison.']),
    diffusion: lesson('Diffusion: trace a scheduled reverse update', 'Inspect an eight-coordinate noisy vector and the noise estimate used by each deterministic DDIM step.',
      'A reverse sampler combines a schedule with a noise prediction. It is not local smoothing or a pre-drawn fidelity curve.',
      'Real generation starts near Gaussian noise and uses a trained denoiser. This oracle reconstruction exposes the algebra, not learned generation.',
      ['x_t=\\sqrt{\\bar\\alpha_t}x_0+\\sqrt{1-\\bar\\alpha_t}\\epsilon', '\\hat x_0=(x_t-\\sqrt{1-\\bar\\alpha_t}\\hat\\epsilon)/\\sqrt{\\bar\\alpha_t}', 'x_{t-1}=\\sqrt{\\bar\\alpha_{t-1}}\\hat x_0+\\sqrt{1-\\bar\\alpha_{t-1}}\\hat\\epsilon'],
      [['alpha_bar', 'Product of (1-beta) through t', 'Six explicit noise variances'], ['epsilon_hat', 'Oracle noise or oracle plus fixed bias', 'No neural network is trained'], ['eta', 'DDIM stochasticity', '0 in this deterministic trace']],
      'x0_hat = (x_t - sqrt(1-ab_t)*eps_hat) / sqrt(ab_t)\nx_prev = sqrt(ab_prev)*x0_hat + sqrt(1-ab_prev)*eps_hat\n# In real sampling: eps_hat = network(x_t, t, condition)',
      'Does the oracle reconstruction demonstrate learned image generation?', ['No; it has access to the clean vector.', 'The sampler equations are real, but the noise prediction is supplied by an oracle.'], ['Yes; removing any noise is a trained diffusion model.', 'The denoiser must learn a data distribution; smoothing alone does not do that.'],
      ['Finite-noise reconstruction, not a pure-Gaussian generation claim. Bias mode adds alternating +/-0.3 to the oracle estimate.', '<a href="https://arxiv.org/abs/2010.02502">DDIM paper</a>']),
    guidance: lesson('Classifier-free guidance: inspect the noise-space direction', 'Combine unconditional and conditional predictions at the same noisy input. Scale 1 is conditional; scales above 1 extrapolate.',
      'Guidance is an inference-time prediction change, not a separately measured adherence or diversity score.',
      'State your convention: epsilon_u + s(epsilon_c - epsilon_u), where s=0 is unconditional and s=1 is conditional.',
      ['\\hat\\epsilon_s=\\hat\\epsilon_u+s(\\hat\\epsilon_c-\\hat\\epsilon_u)', '\\hat x_0=(x_t-\\sqrt{1-\\bar\\alpha_t}\\hat\\epsilon_s)/\\sqrt{\\bar\\alpha_t}'],
      [['epsilon_u', 'Unconditional prediction', '[-0.2, 0.3]'], ['epsilon_c', 'Conditional prediction', '[0.6, -0.1]'], ['s', 'Interpolation/extrapolation scale', '2 gives [1.4, -0.5]']],
      'eps_u = model(x_t, t, null_condition)\neps_c = model(x_t, t, prompt)\neps = eps_u + scale * (eps_c - eps_u)',
      'At scale 2, is guided noise a convex average?', ['No, it extrapolates beyond the conditional prediction.', 'The conditional weight is 2 and the unconditional weight is -1.'], ['Yes, guidance always averages two predictions.', 'Only scales between 0 and 1 form a convex interpolation.'],
      ['Fixed two-dimensional model outputs, not a trained image generator. No fabricated quality metrics.', '<a href="https://arxiv.org/abs/2207.12598">Classifier-free guidance paper</a>']),
    dpo: lesson('DPO: optimize a reference-relative preference margin', 'A preferred factual answer competes with a rejected overclaim. Take an actual gradient step on a three-completion categorical policy.',
      'Raw chosen probability is not the DPO objective: the chosen/rejected odds must be compared with reference odds.',
      'DPO uses fixed preference pairs and reference log-probabilities. Beta scales the margin here; it is not a learning rate.',
      ['m=\\log\\frac{\\pi_\\theta(y^+)}{\\pi_{ref}(y^+)}-\\log\\frac{\\pi_\\theta(y^-)}{\\pi_{ref}(y^-)}', 'L=-\\log\\sigma(\\beta m)'],
      [['pi_ref', 'Fixed reference distribution', '[0.2, 0.5, 0.3]'], ['m', 'Reference-adjusted log-odds', '0 when policy equals reference'], ['beta', 'Margin scale / underlying KL coefficient', '0.5 in this example']],
      'logp = logits.log_softmax(-1)\nmargin = (logp[0]-log_ref[0]) - (logp[1]-log_ref[1])\nloss = -F.logsigmoid(beta * margin)\nloss.backward()  # reference stays frozen',
      'The policy equals the reference. What is the pair loss?', ['log(2), regardless of reference preference.', 'Both log-ratios are zero, so the preference logit is zero.'], ['Zero if reference prefers the chosen answer.', 'DPO measures improvement relative to the reference odds, not just their ordering.'],
      ['Three complete responses treated as categorical outcomes. Real sequence log-probabilities sum token log-probabilities.', '<a href="https://arxiv.org/abs/2305.18290">DPO paper</a>']),
    'reward-hacking': lesson('Reward hacking: a program learns the tests, not the task', 'Optimize a policy over three sum programs. Public tests always sum to 5, so a cheap constant answer can win the proxy.',
      'A verifiable reward is only as comprehensive as its verifier. Watch held-out correctness while the training reward improves.',
      'Audit data must remain independent. Adding new training edge cases is different from optimizing on held-out audit results.',
      ['r_i=\\operatorname{testAccuracy}_i-0.05\\operatorname{cost}_i', 'p_{k+1}(i)\\propto p_k(i)\\exp(2r_i)', '\\operatorname{audit}=\\sum_i p(i)\\operatorname{heldoutAccuracy}_i'],
      [['cost', 'Declared toy execution cost units', '8, 5, 1'], ['p', 'Policy over candidate programs', 'Initially uniform'], ['audit', 'Expected held-out correctness', 'Computed on four unseen inputs']],
      'reward = public_test_accuracy - 0.05 * cost\nlogits += 2 * reward\nprobs = softmax(logits)\naudit = probs @ heldout_accuracy  # never used in update',
      'Why does return 5 become attractive?', ['All public tests expect 5, and the cost penalty favors it.', 'The verifier leaves a loophole. Held-out tests expose the failure.'], ['The optimizer is failing to maximize its reward.', 'It is succeeding at the specified proxy, which is the problem.'],
      ['Finite candidate policies and declared cost units. No code is executed from user input. Additional training cases are disjoint from the audit inputs.']),
    grpo: lesson('GRPO: inspect the group-relative policy objective', 'Compare four fixed one-token answers: rewards become group-normalized advantages, then enter clipped policy-ratio terms.',
      'The rollout policy supplies ratios; the reference supplies KL. They are different roles.',
      'Equal rewards give no relative reward signal. A nonzero KL penalty can still contribute.',
      ['A_i=\\begin{cases}(r_i-\\operatorname{mean}(r))/\\operatorname{std}(r),&\\operatorname{std}(r)\\ge 10^{-12}\\\\0,&\\text{otherwise}\\end{cases}', '\\rho_i=\\pi_\\theta(o_i)/\\pi_{old}(o_i)', 'J=\\operatorname{mean}_i[\\min(\\rho_i A_i,\\operatorname{clip}(\\rho_i,0.8,1.2)A_i)-\\beta k_i]', 'k_i=\\pi_{ref}(o_i)/\\pi_\\theta(o_i)-\\log[\\pi_{ref}(o_i)/\\pi_\\theta(o_i)]-1'],
      [['A', 'Population-standardized reward; zero when std=0', '[1,-1,1,-1]'], ['rho', 'Current / rollout probability', '1.4 for a strengthened correct answer'], ['k', 'Per-sample nonnegative KL estimator term', 'Not the exact categorical KL']],
      '# One fixed rollout group; rewards and old/reference models are frozen.\nrewards = rewards.detach()\nstd = rewards.std(unbiased=False)\nadv = torch.zeros_like(rewards) if std < 1e-12 else (rewards-rewards.mean()) / std\nratio = (logp-logp_old.detach()).exp()\nsurrogate = torch.minimum(ratio*adv, ratio.clamp(.8,1.2)*adv)\nd = logp_ref.detach()-logp\nkl_sample = d.exp()-d-1\nloss = -(surrogate-beta*kl_sample).mean()',
      'All four rewards are equal. What remains?', ['Zero reward advantages; the KL term can remain.', 'This implementation handles zero variance explicitly.'], ['A strong positive advantage for every answer.', 'Subtracting the group mean removes a reward shared by the whole group.'],
      ['One-token categorical toy; real outcome-supervised GRPO averages token terms within each completion. This frozen illustrative group uses population std and an explicit zero-variance guard.', '<a href="https://arxiv.org/html/2402.03300v3#S4">DeepSeekMath, equations 3-4</a>']),
  };
  Object.entries(content).forEach(([kind, item]) => { item.viz = kind; });
  const mechanisms = {
    mdp: 'Inspect both available actions before executing one; the chosen edge determines the reward and next state.',
    'value-functions': 'Read an action backup as immediate utility plus discounted old next-state value. Change the operator to compare control with fixed-policy evaluation.',
    'td-learning': 'Preview the next update, then advance the trace. Only the visited state changes; the target is not a full Monte Carlo return.',
    'q-learning': 'Select a replay transition and inspect both targets on the same Q table before applying either update.',
    dqn: 'A gradient step changes shared online weights. The bootstrap is computed with a separate frozen copy and terminal masking.',
    bandit: 'Select any recorded pull to recover the estimates and selection indices that existed before its reward was observed.',
    diffusion: 'Each reverse step subtracts predicted noise, rescales an estimate of the clean vector, then recombines it at the previous noise level.',
    guidance: 'Read the guided vector and its reconstructed clean sample together: noise-space extrapolation changes the reverse estimate.',
    dpo: 'The two log probability ratios form one preference logit. The reference remains frozen while gradients change normalized policy probabilities.',
    'reward-hacking': 'The policy distribution shifts toward programs with higher public-test reward. Inspect each program on the independent audit inputs.',
    grpo: 'Select a completion to inspect its rollout ratio, both clipping branches, and reference penalty. Switch to equal rewards to isolate KL.',
  };
  Object.entries(content).forEach(([kind, item]) => { item.what = mechanisms[kind]; });

  function initial(kind) {
    if (kind === 'mdp') return { state: 0, action: 1, path: [] };
    if (kind === 'value-functions') return { step: 1, gamma: 0.9, policy: 'optimal', selected: 0 };
    if (kind === 'td-learning') return { step: 1 };
    if (kind === 'q-learning') return { q: [[0, 0], [0.5, 2], [0, 0]], selected: 1, method: 'q', updates: 0 };
    if (kind === 'dqn') return { online: [[0.5, 0], [0.5, 1.5]], frozen: [[0.5, 0], [0.5, 1.5]], selected: 1, updates: 0, copies: 0 };
    if (kind === 'bandit') return { step: 8, policy: 'ucb', selected: 7 };
    if (kind === 'diffusion') return { step: 0, selected: 3, biased: false };
    if (kind === 'guidance') return { scale: 1 };
    if (kind === 'dpo') return { logits: [-1.2, -0.5, 0], beta: 0.5, step: 0 };
    if (kind === 'reward-hacking') return { step: 4, repaired: false, selected: 2 };
    if (kind === 'grpo') return { step: 1, selected: 0, rewards: 'mixed', beta: 0.04 };
    return {};
  }
  function reduce(kind, state, action, value) {
    if (action === 'reset') return initial(kind);
    const n = Number(value);
    const s = { ...state };
    if (action === 'disclosure' && ['calculation', 'settings'].includes(value)) s[value + 'Open'] = !s[value + 'Open'];
    if (action === 'select' && Number.isInteger(n)) {
      const max = kind === 'bandit' ? state.step - 1 : kind === 'diffusion' ? 7 : kind === 'reward-hacking' ? 2 : kind === 'value-functions' ? 1 : 3;
      s.selected = clamp(n, 0, max);
    }
    if (kind === 'mdp') {
      if (action === 'state') { s.state = clamp(n, 0, 2); s.action = 1; s.path = []; }
      if (action === 'action') s.action = clamp(n, 0, 1);
      if (action === 'act' && s.state < 2) { const t = transition(s.state, s.action); s.path = s.path.concat(t); s.state = t.next; s.action = 1; }
    }
    if (action === 'next' || action === 'prev') {
      const limits = { 'value-functions': 4, 'td-learning': 12, bandit: 120, diffusion: 6, 'reward-hacking': 12, grpo: 3 };
      if (kind in limits) s.step = clamp(s.step + (action === 'next' ? 1 : -1), kind === 'bandit' ? 3 : 0, limits[kind]);
      if (kind === 'bandit') s.selected = s.step - 1;
    }
    if (kind === 'value-functions') {
      if (action === 'gamma' && [0.2, 0.9, 1].includes(n)) s.gamma = n;
      if (action === 'policy' && ['optimal', 'uniform'].includes(value)) s.policy = value;
    }
    if (kind === 'q-learning') {
      if (action === 'method' && ['q', 'sarsa'].includes(value)) s.method = value;
      if (action === 'update') { s.q = qUpdate(s.q, TRANSITIONS[s.selected], 0.5, 0.9, s.method).after; s.updates++; }
    }
    if (kind === 'dqn') {
      if (action === 'update') { s.online = dqnUpdate(s.online, s.frozen, TRANSITIONS[s.selected]).after; s.updates++; }
      if (action === 'copy') { s.frozen = s.online.map(row => [...row]); s.copies++; }
    }
    if (kind === 'bandit' && action === 'policy' && ['greedy', 'epsilon', 'ucb'].includes(value)) s.policy = value;
    if (kind === 'diffusion' && action === 'bias') { s.biased = value === 'true'; }
    if (kind === 'guidance' && action === 'scale' && [0, 0.5, 1, 2, 3].includes(n)) s.scale = n;
    if ((kind === 'dpo' || kind === 'grpo') && action === 'beta' && [0, 0.04, 0.1, 0.2, 0.5, 1].includes(n)) s.beta = n;
    if (kind === 'dpo' && action === 'update') { s.logits = dpoStep(s.logits, [0.2, 0.5, 0.3], s.beta); s.step++; }
    if (kind === 'dpo' && action === 'reference') { s.logits = [0.2, 0.5, 0.3].map(Math.log); s.step = 0; }
    if (kind === 'reward-hacking' && action === 'repair') s.repaired = value === 'true';
    if (kind === 'grpo' && action === 'rewards' && ['mixed', 'all-correct', 'all-wrong'].includes(value)) s.rewards = value;
    return s;
  }

  const button = (label, action, value = '', selected = false, disabled = false) => `<button type="button" data-action="${escape(action)}" data-value="${escape(value)}"${selected ? ' class="ml-selected" aria-pressed="true"' : ''}${disabled ? ' disabled' : ''}>${escape(label)}</button>`;
  const controls = html => `<div class="ml-controls dl-controls">${html}</div>`;
  const stat = (label, value) => `<div class="ml-stat"><span>${escape(label)}</span><strong>${escape(value)}</strong></div>`;
  const note = text => `<p class="ml-note">${escape(text)}</p>`;
  const equation = text => `<div class="dl-equation">${escape(text)}</div>`;
  const primary = (label, action, value = '') => button(label, action, value).replace('<button ', '<button class="dl-primary" ');
  const advance = (s, max, label) => s.step < max ? primary(label, 'next') : primary('Start again', 'reset');
  const previous = (s, min = 0) => button('Previous', 'prev', '', false, s.step <= min);
  const select = (label, action, current, choices) => `<label class="dl-select">${escape(label)}<select data-action="${escape(action)}">${choices.map(([value, name]) => `<option value="${escape(value)}"${String(current) === String(value) ? ' selected' : ''}>${escape(name)}</option>`).join('')}</select></label>`;
  const table = (headers, rows, caption = '') => `<div class="ml-table dl-table" tabindex="0" role="region" aria-label="${escape(caption || headers.join(', '))}"><table>${caption ? `<caption>${escape(caption)}</caption>` : ''}<thead><tr>${headers.map(h => `<th scope="col">${escape(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(cell => `<td>${cell}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  const stats = items => `<div class="dl-stats">${items.map(([label, value]) => stat(label, value)).join('')}</div>`;
  const disclosure = (s, name, label, html) => `<details class="dl-disclosure"${s[name + 'Open'] ? ' open' : ''}><summary data-action="disclosure" data-value="${escape(name)}">${escape(label)}</summary><div class="dl-disclosure-body">${html}</div></details>`;
  function scene(s, question, action, mechanism, calculation, settings = '', secondary = '') {
    return `<div class="dl-heading"><h3 class="ml-question">${escape(question)}</h3>${controls(action + secondary)}</div><div class="dl-mechanism">${mechanism}</div>` +
      disclosure(s, 'calculation', 'Inspect the calculation', calculation) +
      disclosure(s, 'settings', 'Options and reset', settings + controls(button('Reset', 'reset')));
  }
  function node(label, value, caption = '', action = '', actionValue = '', selected = false) {
    const tag = action ? 'button' : 'div';
    const attrs = action ? ` type="button" data-action="${escape(action)}" data-value="${escape(actionValue)}" aria-pressed="${selected}"` : '';
    return `<${tag} class="dl-flow-node${selected ? ' ml-selected' : ''}"${attrs}><span class="dl-node-label">${escape(label)}</span><strong>${escape(value)}</strong><span class="dl-node-caption">${escape(caption)}</span></${tag}>`;
  }
  const flow = (nodes, arrows = []) => `<div class="dl-flow" role="group" aria-label="Mechanism trace">${nodes.map((n, i) => (i ? `<span class="dl-flow-arrow" aria-hidden="true">${arrows[i - 1] === 'left' ? '&larr;' : '&rarr;'}</span>` : '') + n).join('')}</div>`;
  function probability(label, value, reference, caption = '', percent = true) {
    return `<div class="dl-probability"><div><strong>${escape(label)}</strong><span>${percent ? `${f(value * 100, 1)}%` : f(value)}</span></div><div class="dl-probability-track"><i style="width:${clamp(value, 0, 1) * 100}%"></i><b style="left:${clamp(reference, 0, 1) * 100}%" aria-label="Reference ${f(reference)}"></b></div>${caption ? `<small>${escape(caption)}</small>` : ''}</div>`;
  }
  function bars(items, max = 1) {
    return `<div class="dl-bars">${items.map(([label, value, role = 'blue']) => `<div class="dl-bar-row"><span>${escape(label)}</span><span class="dl-track"><i class="dl-${escape(role)}" style="width:${clamp(Math.abs(value) / max, 0, 1) * 100}%"></i></span><strong>${f(value)}</strong></div>`).join('')}</div>`;
  }
  function transitionChoices(selected) {
    return controls(TRANSITIONS.map((t, i) => button(`${i + 1}. ${STATES[t.s]} / ${ACTIONS[t.s][t.a]}`, 'select', i, i === selected)).join(''));
  }
  function transitionSummary(t) {
    return equation(`${STATES[t.s]} / ${ACTIONS[t.s][t.a]} -> reward ${t.r} -> ${STATES[t.next]}`);
  }
  function mdpView(s) {
    const t = s.state < 2 ? transition(s.state, s.action) : null;
    const total = s.path.reduce((sum, p, i) => sum + 0.9 ** i * p.r, 0);
    const mechanism = t ? flow([
      node('Current state', STATES[s.state], 'Choose an action below'),
      node(ACTIONS[t.s][t.a], `Reward ${t.r}`, 'Deterministic transition'),
      node('Next state', STATES[t.next], t.done ? 'Session ends' : 'Relevant follow-up now reachable'),
    ]) + controls(ACTIONS[s.state].map((name, a) => button(name, 'action', a, a === s.action)).join('')) :
      flow(s.path.length ? s.path.map(p => node(ACTIONS[p.s][p.a], `Reward ${p.r}`, STATES[p.next])) : [node('Terminal', 'No more actions', 'Future value is zero')]);
    return scene(s, 'Can a zero-reward recommendation lead to a better session?',
      t ? primary(`Take ${ACTIONS[t.s][t.a].toLowerCase()}`, 'act') : primary('Start another session', 'reset'),
      mechanism + note(s.path.length ? `Return so far: ${f(total)}. ${s.state === 2 ? 'This episode is finished.' : 'Your action changed which recommendation comes next.'}` : 'Depth then relevant follow-up returns 3.6; a quick click ends the session for 1. Hand-defined utility, not measured engagement.'),
      s.path.map(transitionSummary).join('') + equation(`Return = sum of reward x 0.9^step = ${f(total)}`) + note('Two-decision, fully observed toy. Every displayed transition has probability 1.'),
      controls(STATES.map((name, i) => button(`Start at ${name.toLowerCase()}`, 'state', i, s.state === i)).join('')));
  }
  function valueView(s) {
    const trace = valueTrace(s.step, s.gamma, s.policy), v = trace.at(-1), old = trace[Math.max(0, trace.length - 2)];
    const qs = [0, 1].map(a => { const t = transition(s.selected, a); return t.r + (t.done ? 0 : s.gamma * old[t.next]); });
    const upcoming = backup(v, s.gamma, s.policy);
    return scene(s, 'How does a future reward become valuable now?', advance(s, 4, 'Next: back up values'),
      flow([node('Fresh session value', f(v[0]), `Next sweep: ${f(upcoming[0])}`, 'select', 0, s.selected === 0), node('Engaged session value', f(v[1]), 'Feeds the earlier state', 'select', 1, s.selected === 1), node('Relevant follow-up', 'Reward 4', 'Then terminal value 0')], ['left', 'left']) +
      note(`Sweep ${s.step}: ${s.policy === 'optimal' ? 'take the best action' : 'average a fixed 50/50 policy'}. Each backup reads the previous sweep; discount ${s.gamma}.`),
      (s.step ? table(['Action', 'Reward', 'Old next V', 'Backup'], [0, 1].map(a => { const t = transition(s.selected, a); return [escape(ACTIONS[s.selected][a]), f(t.r), f(t.done ? 0 : old[t.next]), f(qs[a])]; })) + equation(`${s.policy === 'optimal' ? 'max' : 'mean'}(${qs.map(x => f(x)).join(', ')}) = ${f(v[s.selected])}`) : note('Initial estimates are all zero.')) + table(['Sweep', 'Fresh V', 'Engaged V'], trace.map((row, i) => [String(i), f(row[0]), f(row[1])]), 'Exact synchronous history'),
      controls(select('Backup operator', 'policy', s.policy, [['optimal', 'Optimal: max actions'], ['uniform', 'Fixed policy: 50/50 actions']]) + select('Discount gamma', 'gamma', s.gamma, [[0.2, '0.2: short horizon'], [0.9, '0.9: delayed utility'], [1, '1: finite, undiscounted']])), previous(s));
  }
  function tdView(s) {
    let v = [0, 0, 0]; const history = [];
    for (let i = 0; i < s.step; i++) { const t = TRANSITIONS[i % 2 ? 3 : 1]; const result = tdUpdate(v, t); history.push({ t, ...result }); v = result.after; }
    const next = TRANSITIONS[s.step % 2 ? 3 : 1], preview = tdUpdate(v, next);
    return scene(s, 'What can one observed reward teach us before an episode is over?', advance(s, 12, 'Next: apply sampled update'),
      flow([node(ACTIONS[next.s][next.a], `Reward ${next.r}`, next.done ? 'Terminal: no bootstrap' : `Next-state estimate ${f(v[next.next])}`), node('TD target', f(preview.target), 'Reward + estimated future'), node(STATES[next.s], `${f(preview.before)} to ${f(preview.updated)}`, 'Preview: move halfway to target')]) +
      bars([['Fresh value', v[0]], ['Engaged value', v[1]]], 4) + note(`${s.step} updates applied. Only the visited state changes; the policy always chooses depth, then relevant follow-up.`),
      transitionSummary(next) + equation(`target = ${next.r} + 0.9 x ${f(next.done ? 0 : v[next.next])} = ${f(preview.target)}`) + equation(`V: ${f(preview.before)} + 0.5 x ${f(preview.error)} = ${f(preview.updated)}`) + table(['Update', 'State', 'Before', 'Target', 'After'], history.slice(-6).map((h, i) => [String(Math.max(0, history.length - 6) + i + 1), escape(STATES[h.t.s]), f(h.before), f(h.target), f(h.updated)]), 'Last six actual TD updates'), '', previous(s));
  }
  function qView(s) {
    const t = TRANSITIONS[s.selected], result = qUpdate(s.q, t, 0.5, 0.9, s.method), other = qUpdate(s.q, t, 0.5, 0.9, s.method === 'q' ? 'sarsa' : 'q');
    return scene(s, 'Must we learn the same next action that behavior chose?', primary('Apply Q update', 'update'),
      flow([node(t.done ? 'Terminal' : s.method === 'q' ? 'Best next action' : 'Sampled next action', f(result.bootstrap), t.done ? 'No future reward' : s.method === 'q' ? 'Max over both estimates' : 'Behavior selected the ad'), node('Target', f(result.target), `Immediate reward ${t.r}`), node(ACTIONS[t.s][t.a], `${f(result.before)} to ${f(result.updated)}`, 'Preview: update one Q estimate')]) +
      (t.done ? '' : `<div class="dl-choice-nodes">${[0, 1].map(a => node(ACTIONS[1][a], f(s.q[1][a]), a === 0 ? 'Behavior chose this' : 'Greedy choice at initialization', 'select', a + 2, false)).join('')}</div>`) +
      note(`${s.updates} updates applied. ${s.method === 'q' ? 'Q-learning uses the maximum next estimate.' : 'SARSA uses the recorded next action.'} Tap an action node to inspect its terminal transition.`),
      transitionSummary(t) + equation(`target = ${t.r} + 0.9 x ${f(result.bootstrap)} = ${f(result.target)}`) + equation(`Q: ${f(result.before)} + 0.5 x ${f(result.error)} = ${f(result.updated)}`) + stat('Other method target on the same table', f(other.target)) + table(['State', 'Action 0', 'Action 1'], [0, 1].map(i => [escape(STATES[i]), f(s.q[i][0]), f(s.q[i][1])])),
      controls(select('Backup', 'method', s.method, [['q', 'Q-learning: max next'], ['sarsa', 'SARSA: sampled ad next']])) + transitionChoices(s.selected));
  }
  function dqnView(s) {
    const t = TRANSITIONS[s.selected], r = dqnUpdate(s.online, s.frozen, t);
    return scene(s, 'Can the prediction improve while its target stays still?', primary('Apply one SGD step', 'update'),
      flow([node('Online prediction', f(r.pred), `After SGD: ${f(linearQ(r.after, t.s)[t.a])}`), node('Squared-error loss', f(r.loss), `${s.updates} online updates`), node('Frozen target', f(r.target), 'Tap to copy online weights', 'copy')], ['right', 'left']) +
      note(`Replay: ${ACTIONS[t.s][t.a]} from ${STATES[t.s].toLowerCase()}. ${s.copies} target copies. Two shared linear features, not a deep network.`),
      transitionSummary(t) + equation(`y = ${t.r} + 0.9 x ${f(r.next)} = ${f(r.target)}`) + equation(`L = 0.5 x (${f(r.pred)} - ${f(r.target)})^2 = ${f(r.loss)}`) + equation(`w_new = w + 0.1 x ${f(r.error)} x [${features(t.s).join(', ')}]`) + table(['State / features', 'Online a0 / a1', 'Frozen a0 / a1'], [0, 1].map(i => [`${escape(STATES[i])} [1,${i}]`, linearQ(s.online, i).map(x => f(x)).join(' / '), linearQ(s.frozen, i).map(x => f(x)).join(' / ')])) + table(['Action weights', 'Online [bias,slope]', 'Frozen [bias,slope]'], [0, 1].map(i => [String(i), s.online[i].map(x => f(x)).join(', '), s.frozen[i].map(x => f(x)).join(', ')])),
      transitionChoices(s.selected));
  }
  function banditView(s) {
    const run = bandit(s.step, s.policy), item = run.history[Math.min(s.selected, s.step - 1)];
    const observed = run.history.slice(0, item.t).filter(h => h.a === item.a);
    const updated = mean(observed.map(h => h.reward));
    const armNodes = `<div class="dl-choice-nodes dl-three">${ARM_MEANS.map((_, a) => {
      const last = run.history.map(h => h.a).lastIndexOf(a);
      return node(`Arm ${String.fromCharCode(65 + a)}`, `${f(run.estimates[a] * 100, 0)}% estimated`, `${run.counts[a]} pulls; tap to inspect last reward`, 'select', last, item.a === a);
    }).join('')}</div>`;
    return scene(s, 'Which recommendation should we try when estimates are uncertain?', advance(s, 120, 'Next: choose and observe'),
      armNodes + flow([node(`Pull ${item.t}: arm ${String.fromCharCode(65 + item.a)}`, f(item.estimates[item.a]), 'Estimate before observing'), node('Observed feedback', `Reward ${item.reward}`, item.reason), node('Updated estimate', f(updated), 'Only this arm learns')]) +
      note(`${s.step} pulls with ${s.policy === 'ucb' ? 'UCB1' : s.policy === 'epsilon' ? 'epsilon-greedy' : 'greedy'}. An early zero is not proof that an arm is bad.`),
      table(['Arm', 'Pulls', 'Wins', 'Estimate', 'Hidden truth'], ARM_MEANS.map((p, a) => [String.fromCharCode(65 + a), String(run.counts[a]), String(run.wins[a]), f(run.estimates[a]), f(p)])) + stats([['Pseudo-regret', f(run.regret)], ['Observed reward sum', run.wins.reduce((a, b) => a + b, 0)]]) + table(['Arm', 'Estimate before pull', 'Selection index'], item.estimates.map((q, a) => [String.fromCharCode(65 + a), f(q), Number.isFinite(item.scores[a]) ? f(item.scores[a]) : 'unpulled: first priority'])) + `<div class="dl-history" aria-label="Inspect observed pulls">${run.history.map((h, i) => button(`${h.t}: ${String.fromCharCode(65 + h.a)} / r=${h.reward}`, 'select', i, i === item.t - 1)).join('')}</div>` + note('Seed 19; same nth-pull outcome per arm across policies. Selection never sees hidden means. Ties choose the first arm. One seed is not a statistical policy comparison.'),
      controls(select('Selection policy', 'policy', s.policy, [['ucb', 'UCB1'], ['greedy', 'Greedy after initial coverage'], ['epsilon', 'Epsilon-greedy (0.15)']])), previous(s, 3));
  }
  function vectorPlot(clean, noisy, current, selected) {
    const all = clean.concat(noisy, current), low = Math.min(-1.5, ...all), high = Math.max(1.5, ...all);
    const x = i => 25 + i * 36, y = v => 170 - (v - low) / (high - low) * 145;
    const path = values => values.map((v, i) => `${i ? 'L' : 'M'}${x(i)},${y(v)}`).join(' ');
    return `<svg class="dl-vector" viewBox="0 0 302 202" role="group" aria-label="Eight vector coordinates: clean dashed, original noisy dotted, current solid"><path d="M15 ${y(0)}H288" class="dl-axis"/><path d="${path(clean)}" class="dl-clean"/><path d="${path(noisy)}" class="dl-noisy"/><path d="${path(current)}" class="dl-current"/>${current.map((v, i) => `<g role="button" tabindex="0" data-action="select" data-value="${i}" aria-label="Inspect coordinate ${i}, current value ${f(v)}"><circle cx="${x(i)}" cy="${y(v)}" r="15" fill="transparent"/><circle cx="${x(i)}" cy="${y(v)}" r="${i === selected ? 6 : 3}" class="dl-point"/></g><text x="${x(i)}" y="193" text-anchor="middle">${i}</text>`).join('')}</svg>`;
  }
  function diffusionView(s) {
    const trace = diffusionTrace(s.biased), row = trace[s.step], i = s.selected;
    const mse = mean(row.x.map((v, j) => (v - CLEAN[j]) ** 2));
    return scene(s, 'What does one denoising step actually change?', advance(s, 6, 'Next: denoise one step'),
      vectorPlot(CLEAN, trace[0].x, row.x, i) + `<div class="dl-legend"><span class="dl-key-clean">Dashed: clean vector</span><span class="dl-key-noisy">Dotted: noisy start</span><span class="dl-key-current">Solid: current vector</span></div>` +
      flow([node(`Coordinate ${i}, t=${row.t}`, f(row.x[i]), 'Tap a plotted point to inspect'), node('Noise prediction', row.t ? f(row.eps[i]) : 'Finished', s.biased ? 'Oracle + fixed bias' : 'Oracle knows clean vector'), node('Next coordinate', f(row.previous[i]), row.t ? `Schedule moves to t=${row.t - 1}` : 'No further step')]) +
      note('This is oracle reconstruction, not learned generation. The sampler uses an explicit noise schedule, not smoothing.'),
      stats([['alpha-bar(t)', f(ALPHAS[row.t])], ['MSE to known x0', f(mse)]]) + table(['Quantity', 'Value'], [['Clean x0 (oracle access)', f(CLEAN[i])], ['Current x_t', f(row.x[i])], ['Predicted epsilon', f(row.eps[i])], ['Reconstructed x0', f(row.x0[i])], ['Next x_(t-1)', f(row.previous[i])]]) + table(['t', 'beta_t', 'alpha-bar_t'], BETAS.map((b, j) => [String(j + 1), f(b), f(ALPHAS[j + 1])]), 'Explicit forward noise schedule'),
      controls(select('Noise predictor', 'bias', s.biased, [[false, 'Oracle: clean x0 is known'], [true, 'Oracle + fixed prediction bias']])), previous(s));
  }
  function guidanceView(s) {
    const u = [-0.2, 0.3], c = [0.6, -0.1], g = guidedNoise(u, c, s.scale), xt = [0.7, 0.2];
    const clean = ddimStep(xt, g, 0.4, 1).x0;
    const px = v => 65 + v * 65, py = v => 100 - v * 65;
    const line = (v, klass) => `<path d="M65 100L${px(v[0])} ${py(v[1])}" class="${klass}"/><circle cx="${px(v[0])}" cy="${py(v[1])}" r="5" class="${klass}"/>`;
    const scales = [0, 0.5, 1, 2, 3], next = scales[(scales.indexOf(s.scale) + 1) % scales.length];
    return scene(s, 'Does stronger guidance average predictions or go beyond them?', primary(s.scale === 3 ? 'Compare with no guidance' : `Increase guidance to ${next}`, 'scale', next),
      `<svg class="dl-vector" viewBox="0 0 300 210" role="img" aria-label="Unconditional, conditional and guided noise vectors with a shared origin"><path d="M20 100H285M65 15V190" class="dl-axis"/><path d="M${px(u[0])} ${py(u[1])}L${px(2.2)} ${py(-0.9)}" class="dl-noisy"/>${line(u, 'dl-clean')}${line(c, 'dl-noisy')}${line(g, 'dl-current')}<text x="274" y="122">e1</text><text x="75" y="23">e2</text></svg>` +
      `<div class="dl-legend"><span class="dl-key-clean">Dashed: unconditional</span><span class="dl-key-noisy">Dotted: conditional</span><span class="dl-key-current">Solid: guided</span></div>` +
      stats([['Guidance scale', s.scale], ['Guided noise vector', `[${g.map(x => f(x, 1)).join(', ')}]`]]) + note(s.scale > 1 ? 'The guided vector goes beyond the conditional prediction. Extrapolation is not a guarantee of better quality.' : 'Scale 0 is unconditional. Scale 1 is exactly the conditional prediction. Fixed toy model outputs.'),
      table(['Vector', 'Coordinate 0', 'Coordinate 1'], [['Unconditional', ...u.map(x => f(x))], ['Conditional', ...c.map(x => f(x))], ['Guided', ...g.map(x => f(x))], ['Reconstructed x0', ...clean.map(x => f(x))]]) + equation(`epsilon = u + ${s.scale} x (c - u)`) + note('Both predictions use x_t=[0.7,0.2] and alpha-bar=0.4. No network is trained here.'),
      controls(scales.map(v => button(`Scale ${v}`, 'scale', v, s.scale === v)).join('')));
  }
  function dpoView(s) {
    const ref = [0.2, 0.5, 0.3], r = dpo(s.logits, ref, s.beta), next = dpo(dpoStep(s.logits, ref, s.beta), ref, s.beta);
    return scene(s, 'How does one preference change the answers a model favors?', primary('Apply one preference update', 'update'),
      `<p class="dl-context">Claim to assess: better offline NDCG proves user benefit.</p>` +
      probability('Chosen: "No; validate online."', r.probabilities[0], ref[0], 'Increase its odds relative to the rejected answer.') +
      probability('Rejected: "Yes, guaranteed."', r.probabilities[1], ref[1]) +
      probability('Other completions', r.probabilities[2], ref[2]) +
      flow([node('Reference-adjusted loss', f(r.loss), 'Ticks above mark frozen reference'), node('After next update', f(next.loss), `Learning rate 0.5; ${s.step} updates applied`)]) + note('Three-outcome categorical policy. The objective compares chosen/rejected odds against the reference, not chosen probability alone.'),
      table(['Completion', 'Policy p', 'Reference p', 'log(p/ref)'], ['Chosen', 'Rejected', 'Other'].map((name, i) => [name, f(r.probabilities[i]), f(ref[i]), f(r.logRatios[i])])) + equation(`m = ${f(r.logRatios[0])} - (${f(r.logRatios[1])}) = ${f(r.margin)}`) + equation(`z = beta x m = ${f(r.z)}`) + equation(`dL/d(chosen logit) = ${f(r.gradient)}`) + note('Rejected-logit gradient has opposite sign; other-logit gradient is zero. Softmax keeps probabilities normalized.'),
      controls(select('Beta', 'beta', s.beta, [[0.1, '0.1'], [0.5, '0.5'], [1, '1.0']]) + button('Set policy = reference', 'reference')));
  }
  function rewardView(s) {
    const r = proxyExperiment(s.step, s.repaired), base = proxyExperiment(0, s.repaired), selected = PROGRAMS[s.selected];
    const cases = s.repaired ? PUBLIC_CASES.concat(EXTRA_CASES) : PUBLIC_CASES;
    return scene(s, 'Why can a rising test score produce worse code?', advance(s, 12, 'Next: optimize the test reward'),
      `<div class="dl-choice-nodes dl-three">${PROGRAMS.map((p, i) => node(p.name, `${f(r.probabilities[i] * 100, 1)}% selected`, p.code, 'select', i, s.selected === i)).join('')}</div>` +
      flow([node('Training tests passed', `${Math.round(accuracy(selected, cases) * cases.length)}/${cases.length}`, 'Reward also favors lower cost'), node('Selected program', selected.code, `${selected.cost} declared cost units`), node('Unseen tests passed', `${Math.round(r.heldout[s.selected] * 4)}/4`, 'Not used in the update')]) +
      probability('Expected proxy reward', r.proxy, base.proxy, '', false) + probability('Held-out correctness', r.audit, base.audit) + note(`Round ${s.step}. Ticks mark the untrained baseline. ${s.repaired ? 'Additional, independent training inputs now test variable lengths.' : 'Both public tests sum to 5, so the cheap constant exploits the verifier.'}`),
      table(['Program', 'Proxy reward', 'Probability', 'Held-out pass'], PROGRAMS.map((p, i) => [button(p.name, 'select', i, i === s.selected), f(r.rewards[i]), f(r.probabilities[i]), f(r.heldout[i])])) + table(['Input', 'Expected', 'Actual', 'Split'], cases.map(xs => [escape(JSON.stringify(xs)), String(xs.reduce((a, b) => a + b, 0)), Number.isFinite(selected.run(xs)) ? String(selected.run(xs)) : 'error', 'Train']).concat(HELDOUT_CASES.map(xs => [escape(JSON.stringify(xs)), String(xs.reduce((a, b) => a + b, 0)), Number.isFinite(selected.run(xs)) ? String(selected.run(xs)) : 'error', 'Audit']))) + note('Repair recomputes the trace with additional training cases. Audit inputs never enter the reward.'),
      controls(button('Original public tests', 'repair', false, !s.repaired) + button('Add independent training edge cases', 'repair', true, s.repaired)), previous(s));
  }
  function grpoView(s) {
    const answers = ['2', '4', '+2', '-2'];
    const rewards = s.rewards === 'mixed' ? [1, 0, 1, 0] : Array(4).fill(s.rewards === 'all-correct' ? 1 : 0);
    const group = groupAdvantages(rewards), probabilities = [[0.25, 0.25, 0.25, 0.25], [0.29, 0.21, 0.29, 0.21], [0.35, 0.15, 0.35, 0.15], [0.15, 0.35, 0.15, 0.35]][s.step];
    const ref = [0.2, 0.3, 0.2, 0.3];
    const terms = probabilities.map((p, i) => grpoTerm(group.advantages[i], p, 0.25, ref[i], s.beta)), t = terms[s.selected];
    return scene(s, 'When does making a rewarded answer likelier stop helping?', advance(s, 3, s.step === 1 ? 'Next: push past the clipping limit' : 'Next: inspect policy snapshot'),
      `<p class="dl-context">One prompt: positive root of x^2 = 4. Tap an answer.</p><div class="dl-choice-nodes dl-four">${answers.map((answer, i) => node(`Answer ${answer}`, `Reward ${rewards[i]}`, `Relative advantage ${f(group.advantages[i], 1)}`, 'select', i, s.selected === i)).join('')}</div>` +
      flow([node(`Selected answer ${answers[s.selected]}`, f(group.advantages[s.selected], 1), `Group mean ${f(group.mean, 1)}, std ${f(group.std, 1)}`), node('Probability ratio', f(t.ratio, 2), `Current ${f(probabilities[s.selected])} / old 0.250`), node('Clipped reward term', f(t.surrogate), t.active ? `Unclipped: ${f(t.raw)}; limit active` : `Unclipped: ${f(t.raw)}; no clipping`)]) +
      note(group.std === 0 ? 'Equal verifier scores: no reward-advantage signal. Only the KL contribution can change.' : t.active ? 'The reward-improving move is clipped. A still larger ratio no longer improves this sample reward term.' : s.step === 3 ? 'Wrong-direction moves are still penalized. Clipping is not a hard bound on probabilities.' : 'This move is inside the clipping window. Next compares a stronger probability shift.') +
      stats([['Reference KL penalty', f(s.beta * t.kl)], ['Sample objective', f(t.objective)]]) + note('Fixed one-token policy snapshots, not optimizer steps.'),
      table(['Answer', 'Reward', 'Advantage', 'Current p'], answers.map((answer, i) => [button(answer, 'select', i, i === s.selected), f(rewards[i]), f(group.advantages[i]), f(probabilities[i])])) + stats([['Reward mean', f(group.mean)], ['Population std', f(group.std)], ['Mean objective J', f(mean(terms.map(x => x.objective)))]] ) + table(['Quantity', 'Value'], [['Rollout probability (old)', '0.250'], ['Reference probability', f(ref[s.selected])], ['Current / old ratio', f(t.ratio)], ['Unclipped ratio x advantage', f(t.raw)], ['Clamped ratio x advantage', f(t.clipped)], ['Minimum surrogate', f(t.surrogate)], ['Sample KL term', f(t.kl)], ['Surrogate - beta * KL', f(t.objective)]]) + note('Old is uniform; reference is not. Population std with a zero-variance guard. The mean sample KL term is not exact full-distribution KL.'),
      controls(select('Reward group', 'rewards', s.rewards, [['mixed', 'Mixed correctness'], ['all-correct', 'Saturated verifier: all rewards 1'], ['all-wrong', 'Failed verifier: all rewards 0']]) + select('KL coefficient', 'beta', s.beta, [[0, '0: no KL'], [0.04, '0.04'], [0.2, '0.20']])), previous(s));
  }
  const views = { mdp: mdpView, 'value-functions': valueView, 'td-learning': tdView, 'q-learning': qView, dqn: dqnView, bandit: banditView, diffusion: diffusionView, guidance: guidanceView, dpo: dpoView, 'reward-hacking': rewardView, grpo: grpoView };
  function render(kind, state) {
    const view = views[kind];
    return view ? `<div class="dl-root">${view(state)}</div>` : '';
  }
  const api = { transition, backup, valueTrace, tdUpdate, qUpdate, features, linearQ, dqnUpdate, random, bandit, gaussianVector, forwardNoise, ddimStep, diffusionTrace, guidedNoise, softmax, dpo, dpoStep, accuracy, proxyExperiment, groupAdvantages, grpoTerm, content, initial, reduce, render, TRANSITIONS, CLEAN, BETAS, ALPHAS, ARM_MEANS, PROGRAMS, PUBLIC_CASES, HELDOUT_CASES };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window !== 'undefined') {
    if (!window.AtelierLab) throw new Error('Load atelier/lab-core.js before decision-labs.js');
    window.AtelierLab.createModule({ id: 'decision-labs', content, initial, render, reduce });
  }
}());
