'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const lab = require('../decision-labs.js');
const near = (actual, expected, tolerance = 1e-10) => assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);
const sum = xs => xs.reduce((a, b) => a + b, 0);

test('MDP has four exact transitions; terminal has no actions', () => {
  assert.deepEqual(lab.transition(0, 1), { s: 0, a: 1, r: 0, next: 1, done: false });
  assert.equal(lab.transition(1, 1).r, 4);
  assert.throws(() => lab.transition(2, 0), RangeError);
  let s = lab.initial('mdp');
  s = lab.reduce('mdp', s, 'act');
  s = lab.reduce('mdp', s, 'act');
  assert.equal(s.state, 2);
  near(sum(s.path.map((t, i) => 0.9 ** i * t.r)), 3.6);
});

test('Bellman optimality backs up synchronously and reaches exact fixed point', () => {
  const trace = lab.valueTrace(3);
  assert.deepEqual(trace[0], [0, 0, 0]);
  assert.deepEqual(trace[1], [1, 4, 0]);
  assert.deepEqual(trace[2], [3.6, 4, 0]);
  assert.deepEqual(trace[3], trace[2]);
});

test('Fixed uniform policy evaluates an expectation, not max', () => {
  const v = lab.valueTrace(3, 0.9, 'uniform').at(-1);
  near(v[1], 2.25);
  near(v[0], 1.5125);
  near(lab.valueTrace(2, 0.2).at(-1)[0], 1);
  near(lab.valueTrace(2, 1).at(-1)[0], 4);
});

test('TD only updates sampled state and never bootstraps from terminal', () => {
  const before = [0, 0, 999];
  const first = lab.tdUpdate(before, lab.TRANSITIONS[1]);
  near(first.target, 0);
  const terminal = lab.tdUpdate(first.after, lab.TRANSITIONS[3]);
  near(terminal.target, 4); near(terminal.updated, 2);
  const secondEpisode = lab.tdUpdate(terminal.after, lab.TRANSITIONS[1]);
  near(secondEpisode.target, 1.8); near(secondEpisode.updated, 0.9);
  assert.deepEqual(before, [0, 0, 999]);
});

test('Q and SARSA differ on same nonterminal transition', () => {
  const q = [[0, 0], [0.5, 2], [999, 999]];
  const a = lab.qUpdate(q, lab.TRANSITIONS[1]);
  const b = lab.qUpdate(q, lab.TRANSITIONS[1], 0.5, 0.9, 'sarsa', 0);
  near(a.target, 1.8); near(a.updated, 0.9);
  near(b.target, 0.45); near(b.updated, 0.225);
  assert.deepEqual(q[0], [0, 0]);
  near(lab.qUpdate(q, lab.TRANSITIONS[3]).target, 4);
});

test('DQN exact half-squared loss and gradient, target stays frozen', () => {
  const online = [[0.5, 0], [0.5, 1.5]], frozen = online.map(x => [...x]);
  const t = lab.TRANSITIONS[1];
  const first = lab.dqnUpdate(online, frozen, t);
  near(first.pred, 0.5); near(first.target, 1.8); near(first.loss, 0.845);
  near(first.after[1][0], 0.63); near(first.after[1][1], 1.5);
  const second = lab.dqnUpdate(first.after, frozen, t);
  near(second.target, first.target);
  assert.ok(second.loss < first.loss);
  assert.notEqual(lab.linearQ(first.after, 1)[1], lab.linearQ(online, 1)[1]);
  near(lab.dqnUpdate(first.after, first.after, t).target, 1.917);
  assert.deepEqual(frozen, [[0.5, 0], [0.5, 1.5]]);
});

test('DQN terminal mask and analytical gradient match finite difference', () => {
  const w = [[0.5, 0.1], [0.5, 1.5]], frozen = [[100, 100], [100, 100]];
  const t = lab.TRANSITIONS[3], r = lab.dqnUpdate(w, frozen, t);
  near(r.target, 4);
  for (let j = 0; j < 2; j++) {
    const plus = w.map(x => [...x]), minus = w.map(x => [...x]);
    plus[1][j] += 1e-5; minus[1][j] -= 1e-5;
    const derivative = (lab.dqnUpdate(plus, frozen, t).loss - lab.dqnUpdate(minus, frozen, t).loss) / 2e-5;
    near(derivative, -r.error * lab.features(t.s)[j], 1e-7);
  }
});

test('Bandit is reproducible and trace prefixes do not change with horizon', () => {
  for (const policy of ['ucb', 'greedy', 'epsilon']) {
    assert.deepEqual(lab.bandit(8, policy), lab.bandit(8, policy));
    assert.deepEqual(lab.bandit(8, policy).history, lab.bandit(60, policy).history.slice(0, 8));
  }
});

test('Bandit estimates, selected outcomes and regret are reconstructed from history', () => {
  for (const policy of ['ucb', 'greedy', 'epsilon']) {
    const run = lab.bandit(120, policy), counts = [0, 0, 0], wins = [0, 0, 0];
    for (const item of run.history) {
      assert.ok([0, 1].includes(item.reward));
      item.estimates.forEach((x, a) => near(x, counts[a] ? wins[a] / counts[a] : 0));
      counts[item.a]++; wins[item.a] += item.reward;
    }
    assert.deepEqual(counts, run.counts); assert.deepEqual(wins, run.wins);
    run.estimates.forEach((x, a) => near(x, wins[a] / counts[a]));
    near(run.regret, sum(run.history.map(h => 0.7 - lab.ARM_MEANS[h.a])));
    assert.ok(counts.every(x => x >= 1));
  }
});

test('UCB choices use empirical means and count bonuses', () => {
  for (const item of lab.bandit(30).history.slice(3)) assert.equal(item.a, item.scores.indexOf(Math.max(...item.scores)));
});

test('Every policy sees same nth pull outcome for a given arm', () => {
  const a = lab.bandit(120, 'ucb'), b = lab.bandit(120, 'epsilon');
  for (let arm = 0; arm < 3; arm++) {
    const x = a.history.filter(h => h.a === arm).map(h => h.reward);
    const y = b.history.filter(h => h.a === arm).map(h => h.reward);
    const length = Math.min(x.length, y.length);
    assert.deepEqual(x.slice(0, length), y.slice(0, length));
  }
});

test('Forward noising and DDIM oracle reconstruct the known clean vector', () => {
  const eps = lab.gaussianVector(lab.CLEAN.length);
  const x = lab.forwardNoise(lab.CLEAN, eps, lab.ALPHAS[6]);
  const step = lab.ddimStep(x, eps, lab.ALPHAS[6], 1);
  step.x0.forEach((v, i) => near(v, lab.CLEAN[i]));
  step.previous.forEach((v, i) => near(v, lab.CLEAN[i]));
  const trace = lab.diffusionTrace();
  assert.equal(trace.length, 7);
  trace.at(-1).x.forEach((v, i) => near(v, lab.CLEAN[i]));
  trace.slice(0, -1).forEach((row, i) => assert.deepEqual(row.previous, trace[i + 1].x));
  assert.ok(lab.diffusionTrace(true).at(-1).x.some((v, i) => Math.abs(v - lab.CLEAN[i]) > 0.01));
});

test('Diffusion schedule is a product, not a hand-authored quality curve', () => {
  let product = 1;
  lab.BETAS.forEach((b, i) => { product *= 1 - b; near(product, lab.ALPHAS[i + 1]); });
  assert.ok(lab.ALPHAS.slice(1).every((x, i) => x < lab.ALPHAS[i]));
});

test('CFG endpoints and extrapolation use real vector arithmetic', () => {
  const u = [-0.2, 0.3], c = [0.6, -0.1];
  assert.deepEqual(lab.guidedNoise(u, c, 0), u);
  lab.guidedNoise(u, c, 1).forEach((v, i) => near(v, c[i]));
  const g = lab.guidedNoise(u, c, 2); near(g[0], 1.4); near(g[1], -0.5);
});

test('DPO reference cancellation and normalized categorical probabilities', () => {
  const ref = [0.2, 0.5, 0.3], r = lab.dpo(ref.map(Math.log), ref);
  near(r.margin, 0); near(r.loss, Math.log(2)); near(sum(r.probabilities), 1);
  const sameOdds = lab.dpo([Math.log(0.1), Math.log(0.25), Math.log(0.65)], ref);
  near(sameOdds.margin, 0);
});

test('DPO analytic gradient matches finite differences and lowers loss', () => {
  const logits = [-1.2, -0.5, 0], ref = [0.2, 0.5, 0.3], beta = 0.5;
  const r = lab.dpo(logits, ref, beta);
  for (let j = 0; j < 3; j++) {
    const plus = [...logits], minus = [...logits]; plus[j] += 1e-5; minus[j] -= 1e-5;
    const numerical = (lab.dpo(plus, ref, beta).loss - lab.dpo(minus, ref, beta).loss) / 2e-5;
    near(numerical, j === 0 ? r.gradient : j === 1 ? -r.gradient : 0, 1e-7);
  }
  assert.ok(lab.dpo(lab.dpoStep(logits, ref, beta), ref, beta).loss < r.loss);
  near(lab.dpo(logits, ref, 0).loss, Math.log(2));
});

test('DPO stable softplus stays finite for extreme margins', () => {
  const ref = [0.2, 0.5, 0.3];
  assert.ok(Number.isFinite(lab.dpo([1000, -1000, 0], ref).loss));
  assert.ok(Number.isFinite(lab.dpo([-1000, 1000, 0], ref).loss));
});

test('Reward hacking is a computed verifier loophole, not fabricated curves', () => {
  lab.PROGRAMS.forEach(p => near(lab.accuracy(p, lab.PUBLIC_CASES), 1));
  near(lab.accuracy(lab.PROGRAMS[2], lab.HELDOUT_CASES), 0);
  near(lab.accuracy(lab.PROGRAMS[0], lab.HELDOUT_CASES), 1);
  const before = lab.proxyExperiment(0), after = lab.proxyExperiment(12);
  assert.ok(after.proxy > before.proxy); assert.ok(after.audit < before.audit);
  assert.ok(after.probabilities[2] > 0.98);
  near(sum(after.probabilities), 1);
  const repaired = lab.proxyExperiment(12, true);
  assert.ok(repaired.audit > 0.99);
});

test('GRPO population normalization has zero mean and unit variance', () => {
  const g = lab.groupAdvantages([1, 0, 1, 0]);
  near(g.mean, 0.5); near(g.std, 0.5);
  assert.deepEqual(g.advantages, [1, -1, 1, -1]);
  near(sum(g.advantages), 0); near(sum(g.advantages.map(x => x * x)) / 4, 1);
});

test('GRPO handles both all-one and all-zero rewards without NaN', () => {
  for (const rewards of [[1, 1, 1, 1], [0, 0, 0, 0]]) {
    const g = lab.groupAdvantages(rewards);
    assert.equal(g.std, 0); assert.deepEqual(g.advantages, [0, 0, 0, 0]);
    const t = lab.grpoTerm(g.advantages[0], 0.35, 0.25, 0.2);
    near(t.surrogate, 0); assert.ok(t.objective < 0);
    near(lab.grpoTerm(0, 0.35, 0.25, 0.2, 0).objective, 0);
  }
});

test('GRPO clips advantageous changes for either advantage sign', () => {
  const positive = lab.grpoTerm(1, 0.35, 0.25, 0.2);
  near(positive.ratio, 1.4); near(positive.surrogate, 1.2); assert.equal(positive.active, true);
  const negative = lab.grpoTerm(-1, 0.15, 0.25, 0.3);
  near(negative.raw, -0.6); near(negative.surrogate, -0.8); assert.equal(negative.active, true);
  const wrongDirection = lab.grpoTerm(-1, 0.35, 0.25, 0.3);
  near(wrongDirection.surrogate, -1.4); assert.equal(wrongDirection.active, false);
});

test('GRPO KL estimator is nonnegative and vanishes at reference', () => {
  near(lab.grpoTerm(1, 0.2, 0.25, 0.2).kl, 0);
  for (const p of [0.01, 0.1, 0.2, 0.4, 0.9]) assert.ok(lab.grpoTerm(1, p, 0.25, 0.2).kl >= 0);
  const t = lab.grpoTerm(1, 0.35, 0.25, 0.2, 0.2);
  near(t.kl, 0.2 / 0.35 - Math.log(0.2 / 0.35) - 1);
  near(t.objective, t.surrogate - 0.2 * t.kl);
});

test('All eleven kinds have full content, annotated math, code and correct quiz answer', () => {
  assert.equal(Object.keys(lab.content).length, 11);
  for (const [kind, c] of Object.entries(lab.content)) {
    for (const field of ['title', 'summary', 'what', 'why', 'interview']) assert.ok(c[field], `${kind}.${field}`);
    assert.notEqual(c.summary, c.what);
    assert.equal(c.viz, kind); assert.ok(c.math.formula.length); assert.ok(c.math.annotations.length);
    assert.ok(c.math.annotations.every(row => row.length === 3));
    assert.ok(c.code.snippet); assert.equal(c.quiz.options.filter(o => o.correct).length, 1);
    assert.deepEqual(c.controls, []); assert.deepEqual(c.presets, []); assert.equal(c.geometry, null);
  }
});

test('Every default renderer produces inspectable HTML with no invalid numeric values', () => {
  for (const kind of Object.keys(lab.content)) {
    const html = lab.render(kind, lab.initial(kind));
    assert.match(html, /data-action=/);
    assert.doesNotMatch(html, /\bNaN\b|undefined|Infinity/);
    for (const [, svg] of html.matchAll(/<svg[^>]*>([\s\S]*?)<\/svg>/g)) assert.doesNotMatch(svg, /<p[ >]/);
    assert.ok(html.length > 600);
  }
});

test('Reducers are immutable and all exposed buttons survive dispatch/render', () => {
  for (const kind of Object.keys(lab.content)) {
    const original = lab.initial(kind), snapshot = JSON.stringify(original);
    const html = lab.render(kind, original);
    for (const [, action, value] of html.matchAll(/data-action="([^"]+)" data-value="([^"]*)"/g)) {
      const after = lab.reduce(kind, original, action, value);
      assert.equal(JSON.stringify(original), snapshot, `${kind}: ${action} mutated initial state`);
      assert.doesNotMatch(lab.render(kind, after), /\bNaN\b|undefined|Infinity/);
    }
  }
});

test('Previous/Next clamps every bounded trace and reset restores default', () => {
  for (const kind of ['value-functions', 'td-learning', 'bandit', 'diffusion', 'reward-hacking', 'grpo']) {
    let s = lab.initial(kind);
    for (let i = 0; i < 140; i++) s = lab.reduce(kind, s, 'next');
    assert.doesNotMatch(lab.render(kind, s), /\bNaN\b|undefined|Infinity/);
    for (let i = 0; i < 140; i++) s = lab.reduce(kind, s, 'prev');
    assert.doesNotMatch(lab.render(kind, s), /\bNaN\b|undefined|Infinity/);
    assert.deepEqual(lab.reduce(kind, s, 'reset'), lab.initial(kind));
  }
});

test('Every native select option dispatches without invalid HTML computations', () => {
  for (const kind of Object.keys(lab.content)) {
    const original = lab.initial(kind), html = lab.render(kind, original);
    for (const [, action, options] of html.matchAll(/<select data-action="([^"]+)">([\s\S]*?)<\/select>/g)) {
      for (const [, value] of options.matchAll(/<option value="([^"]+)"/g)) {
        const state = lab.reduce(kind, original, action, value);
        assert.doesNotMatch(lab.render(kind, state), /\bNaN\b|undefined|Infinity/);
      }
    }
  }
});

test('Browser registration follows shared core contract; Node needs no window', () => {
  let spec;
  const context = { window: { AtelierLab: { createModule: value => { spec = value; } } } };
  vm.runInNewContext(fs.readFileSync(require.resolve('../decision-labs.js'), 'utf8'), context);
  assert.equal(spec.id, 'decision-labs');
  assert.equal(typeof spec.render, 'function'); assert.equal(typeof spec.reduce, 'function');
  assert.equal(Object.keys(spec.content).length, 11);
  assert.match(spec.render('grpo', spec.initial('grpo')), /data-action="rewards"/);
});

test('Next-state/action math retains LaTeX primes at runtime', () => {
  for (const kind of ['value-functions', 'td-learning', 'q-learning', 'dqn']) {
    const formulas = lab.content[kind].math.formula.join('\n');
    assert.ok(formulas.includes('s^{\\prime}'), `${kind}: next-state prime missing`);
    assert.doesNotMatch(formulas, /sprime|aprime/);
    if (kind === 'q-learning' || kind === 'dqn') assert.ok(formulas.includes('a^{\\prime}'));
  }
});

test('DQN code applies the gradient; GRPO code freezes rollout/reference quantities', () => {
  const dqn = lab.content.dqn.code.snippet;
  assert.ok(dqn.indexOf('optimizer.zero_grad()') < dqn.indexOf('loss.backward()'));
  assert.ok(dqn.indexOf('optimizer.step()') > dqn.indexOf('loss.backward()'));
  const grpo = lab.content.grpo.code.snippet;
  assert.match(grpo, /torch\.zeros_like\(rewards\) if std < 1e-12/);
  assert.match(grpo, /logp_old\.detach\(\)/);
  assert.match(grpo, /logp_ref\.detach\(\)/);
  assert.match(lab.content.grpo.math.formula[0], /otherwise/);
});

test('Mode-specific explanations do not claim nonexistent reward improvement', () => {
  assert.match(lab.content['value-functions'].quiz.prompt, /optimality/);
  for (const rewards of ['all-correct', 'all-wrong']) {
    const html = lab.render('grpo', { ...lab.initial('grpo'), rewards });
    assert.match(html, /no reward-advantage signal/);
    assert.doesNotMatch(html, /reward-improving ratios hit clipping/);
  }
});

test('Every first view has one question, one primary action, and a nonempty mechanism', () => {
  for (const kind of Object.keys(lab.content)) {
    const html = lab.render(kind, lab.initial(kind));
    const firstView = html.split('<details')[0];
    assert.equal((firstView.match(/class="ml-question"/g) || []).length, 1, `${kind}: question`);
    assert.equal((firstView.match(/class="dl-primary"/g) || []).length, 1, `${kind}: primary action`);
    assert.match(firstView, /class="dl-mechanism"/);
    assert.ok(firstView.includes('class="dl-flow"') || firstView.includes('<svg'), `${kind}: missing diagram`);
    assert.doesNotMatch(firstView, /<table|<select|class="dl-equation"/);
    assert.match(html, /<summary data-action="disclosure" data-value="calculation">Inspect the calculation<\/summary>/);
    assert.equal((html.match(/<details[^>]* open/g) || []).length, 0, `${kind}: disclosures should start closed`);
  }
});

test('Every primary action changes the visible mechanism, not just hidden calculations', () => {
  for (const kind of Object.keys(lab.content)) {
    const before = lab.initial(kind), html = lab.render(kind, before);
    const match = html.match(/<button class="dl-primary"[^>]*data-action="([^"]+)" data-value="([^"]*)"/);
    assert.ok(match, kind);
    const after = lab.reduce(kind, before, match[1], match[2]);
    const diagram = output => output.split('<div class="dl-mechanism">')[1].split('<details')[0];
    assert.notEqual(diagram(lab.render(kind, after)), diagram(html), `${kind}: primary action has no visible effect`);
  }
});

test('Native disclosures retain open state through inspection and option changes', () => {
  let s = lab.initial('grpo');
  s = lab.reduce('grpo', s, 'disclosure', 'settings');
  assert.equal(s.settingsOpen, true);
  s = lab.reduce('grpo', s, 'rewards', 'all-wrong');
  assert.equal(s.settingsOpen, true);
  s = lab.reduce('grpo', s, 'disclosure', 'calculation');
  s = lab.reduce('grpo', s, 'select', '1');
  assert.equal(s.calculationOpen, true);
  assert.equal((lab.render('grpo', s).match(/<details[^>]* open/g) || []).length, 2);
  s = lab.reduce('grpo', s, 'disclosure', 'settings');
  assert.equal(s.settingsOpen, false);
  assert.equal(s.calculationOpen, true);
});

test('Direct diagram nodes still dispatch inspection and target-copy actions', () => {
  const diffusion = lab.render('diffusion', lab.initial('diffusion'));
  assert.equal((diffusion.match(/<g role="button" tabindex="0" data-action="select"/g) || []).length, 8);
  const dqn = lab.render('dqn', lab.initial('dqn')).split('<details')[0];
  assert.match(dqn, /data-action="copy"/);
  for (const kind of ['value-functions', 'q-learning', 'bandit', 'reward-hacking', 'grpo']) {
    assert.match(lab.render(kind, lab.initial(kind)).split('<details')[0], /data-action="select"/);
  }
});

// Source-level guard for the screenshot-confirmed mobile collapse, not a layout-engine test.
function assertVerticalFlowSizing(css) {
  const start = /@media\s*\(\s*max-width\s*:\s*520px\s*\)\s*\{/.exec(css);
  assert.ok(start, 'The 520px vertical-flow breakpoint must exist');
  let depth = 1, end = start.index + start[0].length;
  const bodyStart = end;
  for (; end < css.length && depth; end++) {
    if (css[end] === '{') depth++;
    if (css[end] === '}') depth--;
  }
  assert.equal(depth, 0, 'Mobile media block must close');
  const media = css.slice(bodyStart, end - 1);
  assert.match(media, /\.dl-root\s+\.dl-flow\s*\{[^}]*flex-direction\s*:\s*column\s*;/);
  const rules = [...media.matchAll(/\.dl-root\s+\.dl-flow\s+\.dl-flow-node\s*\{([^}]*)\}/g)];
  assert.ok(rules.length, 'Vertical nodes need a flow-specific sizing override');
  const declarations = Object.fromEntries(rules.flatMap(([, body]) => body.split(';').filter(x => x.includes(':')).map(x => {
    const colon = x.indexOf(':');
    return [x.slice(0, colon).trim(), x.slice(colon + 1).trim().replace(/\s+/g, ' ')];
  })));
  assert.equal(declarations.flex, '0 0 auto', 'Vertical nodes must not shrink or use a zero flex basis');
  assert.equal(declarations['min-height'], 'min-content', 'Vertical nodes must fit their label, value, and caption');
}

test('Mobile CSS requires content-sized vertical nodes in the breakpoint covering 320px and 390px', () => {
  const css = fs.readFileSync(require.resolve('../decision-labs.css'), 'utf8');
  assertVerticalFlowSizing(css);
  for (const kind of ['grpo', 'dpo', 'reward-hacking']) {
    const firstView = lab.render(kind, lab.initial(kind)).split('<details')[0];
    assert.match(firstView, /class="dl-flow"/);
    assert.match(firstView, /class="dl-flow-node[^"]*"/);
    assert.match(firstView, /class="dl-node-caption"/);
  }
});

test('Mobile sizing regression guard rejects either original collapsing declaration', () => {
  const css = fs.readFileSync(require.resolve('../decision-labs.css'), 'utf8');
  assert.throws(() => assertVerticalFlowSizing(css.replace(/flex:\s*0\s+0\s+auto\s*;/g, 'flex: 1 1 0;')), /must not shrink/);
  assert.throws(() => assertVerticalFlowSizing(css.replace(/min-height:\s*min-content\s*;/g, 'min-height: 0;')), /must fit/);
});
