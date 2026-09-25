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

function primaryAction(kind, state) {
  const html = lab.render(kind, state);
  const match = html.match(/<button class="dl-primary"[^>]*data-action="([^"]+)" data-value="([^"]*)"/);
  assert.ok(match, kind);
  return lab.reduce(kind, state, match[1], match[2]);
}

function visibleCausal(kind, state) {
  const stage = lab.render(kind, state).split('<details')[0];
  const block = stage.match(/<div class="dl-causal"[\s\S]*?<\/div>/);
  assert.ok(block, `${kind}: substituted equation must remain outside disclosures`);
  return {
    html: block[0],
    before: block[0].match(/<span data-before>(.*?)<\/span>/)[1],
    after: block[0].match(/<span data-after>(.*?)<\/span>/)[1],
  };
}

test('All concepts expose a selected before/after, highlighted numerical term and causal reason', () => {
  for (const kind of Object.keys(lab.content)) {
    const initial = lab.initial(kind);
    for (const state of [initial, primaryAction(kind, initial)]) {
      const c = visibleCausal(kind, state);
      assert.match(c.html, /data-causal-focus="[^"]+"/);
      assert.match(c.html, /<output class="dl-linked-formula"/);
      assert.match(c.html, /<mark>[^<]*[0-9][^<]*<\/mark>/);
      assert.match(c.html, /class="dl-reason">[^<]+/);
    }
    assert.match(visibleCausal(kind, primaryAction(kind, initial)).html, /data-causal-phase="applied"/);
  }
});

test('Last-change snapshots stay bounded, immutable, and survive opening calculations', () => {
  let state = lab.initial('dqn');
  for (let i = 0; i < 100; i++) {
    const old = JSON.stringify(state);
    const next = lab.reduce('dqn', state, 'update');
    assert.equal(JSON.stringify(state), old);
    assert.equal(next.lastChange.action, 'update');
    assert.equal(next.lastChange.before.lastChange, undefined);
    state = next;
  }
  assert.ok(JSON.stringify(state).length < 2000);
  const event = state.lastChange;
  state = lab.reduce('dqn', state, 'disclosure', 'calculation');
  assert.deepEqual(state.lastChange, event);
  state = lab.reduce('dqn', state, 'select', '3');
  assert.equal(state.lastChange, undefined, 'A new replay item must not show the previous item\'s update');
});

test('MDP, Bellman, TD and Q first actions change actual state or learned quantities', () => {
  const mdp = primaryAction('mdp', lab.initial('mdp'));
  assert.equal(mdp.state, 1);
  assert.equal(mdp.path.length, 1);
  const mdpDisplay = visibleCausal('mdp', mdp);
  assert.equal(mdpDisplay.before, 'Fresh session'); assert.equal(mdpDisplay.after, 'Engaged session');

  const value = primaryAction('value-functions', lab.initial('value-functions'));
  near(lab.valueTrace(value.step, value.gamma).at(-1)[0], 3.6);
  assert.equal(visibleCausal('value-functions', value).before, '1.000');
  assert.equal(visibleCausal('value-functions', value).after, '3.600');

  let td = primaryAction('td-learning', lab.initial('td-learning'));
  assert.deepEqual(lab.tdTrace(td.step).values, [0, 2, 0]);
  assert.equal(visibleCausal('td-learning', td).after, '2.000');
  assert.match(visibleCausal('td-learning', td).html, /V\(Engaged session\)/);
  td = primaryAction('td-learning', td);
  near(lab.tdTrace(td.step).values[0], 0.9);
  assert.match(visibleCausal('td-learning', td).html, /V\(Fresh session\)/);

  const q = primaryAction('q-learning', lab.initial('q-learning'));
  near(q.q[0][1], 0.9);
  assert.equal(visibleCausal('q-learning', q).before, '0.000');
  assert.equal(visibleCausal('q-learning', q).after, '0.900');
});

test('DQN renders the applied loss decrease, then separately explains a target copy', () => {
  const before = lab.initial('dqn'), after = primaryAction('dqn', before);
  const oldLoss = lab.dqnUpdate(before.online, before.frozen, lab.TRANSITIONS[1]).loss;
  const newLoss = lab.dqnUpdate(after.online, after.frozen, lab.TRANSITIONS[1]).loss;
  assert.ok(newLoss < oldLoss);
  assert.equal(visibleCausal('dqn', after).before, oldLoss.toFixed(3));
  assert.equal(visibleCausal('dqn', after).after, newLoss.toFixed(3));
  assert.deepEqual(after.frozen, before.frozen);
  const copy = lab.reduce('dqn', after, 'copy');
  assert.deepEqual(copy.online, after.online);
  assert.equal(visibleCausal('dqn', copy).before, '1.800');
  assert.equal(visibleCausal('dqn', copy).after, '1.917');
  assert.match(visibleCausal('dqn', copy).html, /frozen target/);
});

test('Bandit selection and mean-update substitutions use pre-observation counts', () => {
  const state = primaryAction('bandit', lab.initial('bandit'));
  const run = lab.bandit(state.step, state.policy), item = run.history.at(-1);
  const old = item.estimates[item.a];
  near(item.scores[item.a], old + Math.sqrt(2 * Math.log(item.t - 1) / (item.count - 1)));
  near(run.estimates[item.a], old + (item.reward - old) / item.count);
  const view = visibleCausal('bandit', state);
  assert.equal(view.before, old.toFixed(3));
  assert.equal(view.after, run.estimates[item.a].toFixed(3));
  assert.ok(view.html.includes(`sqrt(2 log(${item.t - 1}) / ${item.count - 1})`));
});

test('Diffusion retains the denoising step just applied and follows selected coordinates', () => {
  let state = primaryAction('diffusion', lab.initial('diffusion'));
  const trace = lab.diffusionTrace(false);
  for (const coordinate of [0, 3, 7]) {
    state = lab.reduce('diffusion', state, 'select', String(coordinate));
    const view = visibleCausal('diffusion', state);
    assert.equal(view.before, trace[0].x[coordinate].toFixed(3));
    assert.equal(view.after, trace[1].x[coordinate].toFixed(3));
    assert.ok(view.html.includes(`<mark>${trace[0].eps[coordinate].toFixed(3)}</mark>`));
    assert.match(view.html, /t=6 to 5/);
  }
});

test('Guidance and DPO actions change predictions and normalized policy values', () => {
  const guidance = primaryAction('guidance', lab.initial('guidance'));
  assert.equal(guidance.scale, 2);
  const g = visibleCausal('guidance', guidance);
  assert.equal(g.before, '[0.6, -0.1]'); assert.equal(g.after, '[1.4, -0.5]');
  assert.match(g.html, /<mark>2<\/mark>/);
  const before = lab.initial('dpo'), after = primaryAction('dpo', before), ref = [0.2, 0.5, 0.3];
  const a = lab.dpo(before.logits, ref), b = lab.dpo(after.logits, ref);
  assert.ok(b.probabilities[0] > a.probabilities[0]); assert.ok(b.probabilities[1] < a.probabilities[1]);
  near(sum(b.probabilities), 1);
  assert.equal(visibleCausal('dpo', after).before, a.loss.toFixed(3));
  assert.equal(visibleCausal('dpo', after).after, b.loss.toFixed(3));
});

test('Reward-hacking probability change is an exact normalized exponential update', () => {
  const state = primaryAction('reward-hacking', lab.initial('reward-hacking'));
  for (const repaired of [false, true]) {
    const before = lab.proxyExperiment(state.step - 1, repaired), after = lab.proxyExperiment(state.step, repaired);
    const predicted = lab.exponentiatedStep(before.probabilities, before.rewards);
    predicted.probabilities.forEach((p, i) => near(p, after.probabilities[i]));
  }
  const view = visibleCausal('reward-hacking', state);
  assert.equal(view.before, lab.proxyExperiment(4).probabilities[2].toFixed(3));
  assert.equal(view.after, lab.proxyExperiment(5).probabilities[2].toFixed(3));
  assert.match(view.html, /On unseen \[\], this program gives 5, not 0/);
});

test('GRPO shows the winning clipping branch for positive and negative advantages', () => {
  let state = primaryAction('grpo', lab.initial('grpo'));
  let view = visibleCausal('grpo', state);
  assert.equal(view.before, '1.160'); assert.equal(view.after, '1.200');
  assert.match(view.html, /<mark>1\.200 x 1\.000<\/mark>/);
  state = lab.reduce('grpo', state, 'select', '1');
  view = visibleCausal('grpo', state);
  assert.equal(view.before, '-0.840'); assert.equal(view.after, '-0.800');
  assert.match(view.html, /<mark>0\.800 x -1\.000<\/mark>/);
  state = lab.reduce('grpo', state, 'rewards', 'all-wrong');
  assert.match(visibleCausal('grpo', state).html, /A = 0 \(all rewards equal\)/);
  assert.doesNotMatch(visibleCausal('grpo', state).html, /NaN|Infinity/);
});

test('All existing optional quizzes ask counterfactual transfer rather than UI recall', () => {
  for (const c of Object.values(lab.content)) {
    assert.match(c.quiz.prompt, /\bif\b/i);
    assert.doesNotMatch(c.quiz.prompt, /which button|(?:click|press|tap) (?:the )?(?:button|next|apply)/i);
    assert.equal(c.quiz.options.filter(o => o.correct).length, 1);
  }
  near(lab.valueTrace(2, 0.2).at(-1)[0], 1);
  assert.deepEqual(lab.guidedNoise([0.2, -0.5], [0.2, -0.5], 3), [0.2, -0.5]);
  const ref = [0.2, 0.5, 0.3];
  near(lab.dpo([0.2, 0.4, 0.4].map(Math.log), ref).loss, lab.dpo([0.1, 0.2, 0.7].map(Math.log), ref).loss);
});

test('MDP current-state marker visibly moves instead of only changing preview prose', () => {
  let state = lab.initial('mdp');
  for (const expected of ['Fresh session', 'Engaged session', 'Terminal']) {
    const html = lab.render('mdp', state);
    const markers = [...html.matchAll(/<div class="dl-flow-node ml-selected" aria-current="step">(.*?)<\/div>/g)];
    assert.equal(markers.length, 1);
    assert.ok(markers[0][1].includes(`<strong>${expected}</strong>`));
    assert.match(markers[0][1], /Current state/);
    state = lab.reduce('mdp', state, 'act');
  }
});

test('Guidance legend describes only actual origin-to-prediction vectors', () => {
  const html = lab.render('guidance', primaryAction('guidance', lab.initial('guidance')));
  const svg = html.match(/<svg[\s\S]*?<\/svg>/)[0];
  const conditional = [...svg.matchAll(/<path d="([^"]+)" class="dl-noisy"/g)];
  assert.equal(conditional.length, 1, 'An extrapolation guide must not masquerade as the conditional vector');
  assert.equal(conditional[0][1], 'M65 100L104 106.5');
  assert.match(svg, /M65 100L156 132\.5/);
});

test('DPO displayed logits match current probability bars, not the next update preview', () => {
  const before = lab.initial('dpo'), after = primaryAction('dpo', before);
  for (const state of [before, after]) {
    const html = lab.render('dpo', state);
    assert.ok(html.includes(`Chosen logit</span><strong>${state.logits[0].toFixed(3)}</strong>`));
    assert.ok(html.includes(`Rejected logit</span><strong>${state.logits[1].toFixed(3)}</strong>`));
  }
  assert.match(lab.render('dpo', before), /After update: -1\.082/);
});

test('GRPO keeps its group statistics visible without a duplicate advantage flow card', () => {
  const html = lab.render('grpo', lab.initial('grpo')).split('<details')[0];
  const flow = html.match(/<div class="dl-flow"[\s\S]*?(?=<div class="dl-causal")/)[0];
  assert.equal([...flow.matchAll(/class="dl-flow-node/g)].length, 2);
  assert.match(html, /Reward mean 0\.5/);
  assert.match(html, /Population std 0\.5/);
  assert.equal([...html.matchAll(/Relative advantage/g)].length, 4);
});

test('Secondary calculations use the same backup or copied target as the selected mechanism', () => {
  for (const policy of ['optimal', 'uniform']) for (const selected of [0, 1]) {
    const before = { ...lab.initial('value-functions'), policy, selected };
    for (const state of [before, primaryAction('value-functions', before)]) {
      const source = state.lastChange?.before || state;
      const values = lab.valueTrace(source.step, state.gamma, policy).at(-1);
      const qs = [0, 1].map(action => {
        const t = lab.transition(selected, action);
        return t.r + (t.done ? 0 : state.gamma * values[t.next]);
      });
      const expected = lab.backup(values, state.gamma, policy)[selected];
      const operator = policy === 'optimal' ? 'max' : 'mean';
      assert.ok(lab.render('value-functions', state).includes(`${operator}(${qs.map(x => x.toFixed(3)).join(', ')}) = ${expected.toFixed(3)}`));
    }
  }
  const copied = lab.reduce('dqn', primaryAction('dqn', lab.initial('dqn')), 'copy');
  const html = lab.render('dqn', copied);
  assert.match(html, /y = 0 \+ 0\.9 x 2\.130 = 1\.917/);
  assert.match(html, /previews the next training update; it has not been applied/);
  const sarsa = lab.reduce('q-learning', lab.initial('q-learning'), 'method', 'sarsa');
  assert.match(visibleCausal('q-learning', sarsa).html, /SARSA uses the recorded next action here/);
  assert.doesNotMatch(visibleCausal('q-learning', sarsa).html, /not this max/);
});

// Opt in against the local static server; launches its own headless browser.
test('browser: decision mechanisms fit their content before and after actions at four widths', {
  skip: !process.env.DECISION_BROWSER_TEST,
  timeout: 120000,
}, async () => {
  const path = require('node:path');
  const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
  const browser = await chromium.launch({ headless: true, ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}) });
  const base = process.env.ATELIER_BASE_URL || 'http://127.0.0.1:4174/atelier/';
  const out = process.env.DECISION_SCREENSHOTS || '/tmp/atelier-causal-decision-final';
  const chapters = {
    'reinforcement-learning': ['mdp', 'value-functions', 'td-learning', 'q-learning', 'dqn', 'exploration'],
    'generative-and-rl': ['diffusion', 'bandit'],
    'alignment-depth': ['guidance', 'dpo', 'reward-hacking', 'grpo'],
  };
  const checks = [], errors = [];
  fs.mkdirSync(out, { recursive: true });
  try {
    const page = await browser.newPage({ reducedMotion: 'reduce' });
    page.on('pageerror', error => errors.push(error.message));
    for (const width of [320, 390, 720, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      for (const [chapter, ids] of Object.entries(chapters)) {
        await page.goto(new URL(`${chapter}.html`, base).href, { waitUntil: 'networkidle' });
        await page.evaluate(() => document.fonts.ready);
        for (const id of ids) {
          await page.locator(`.chapter-map a[href="#${id}"]`).click();
          await page.locator(`#${id} .viz-expand-button`).click();
          const dialog = page.locator('.viz-lightbox:not([hidden]) .viz-lightbox__dialog');
          const lesson = dialog.locator('.ml-lab[data-family="decision-labs"]');
          await lesson.waitFor({ state: 'visible' });
          const dir = path.join(out, String(width), chapter);
          if ([390, 1440].includes(width)) fs.mkdirSync(dir, { recursive: true });
          for (const phase of ['before', 'after']) {
            if (phase === 'after') await lesson.locator('.dl-primary').click();
            await dialog.evaluate(el => { el.scrollTop = 0; });
            await page.mouse.move(0, 0);
            await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
            const measured = await lesson.evaluate(el => {
              const nodes = [...el.querySelectorAll('.dl-flow-node')].filter(node => node.getBoundingClientRect().width > 0);
              const failures = nodes.filter(node => {
                const box = node.getBoundingClientRect();
                return node.scrollHeight > node.clientHeight + 2 || [...node.children].some(child => {
                  const rect = child.getBoundingClientRect();
                  return rect.top < box.top - 1 || rect.bottom > box.bottom + 1 || rect.left < box.left - 1 || rect.right > box.right + 1;
                });
              }).map(node => node.textContent.trim());
              return {
                width: el.clientWidth, scrollWidth: el.scrollWidth,
                viewport: innerWidth, documentWidth: document.documentElement.scrollWidth,
                nodeCount: nodes.length, failures,
                formulaBottom: el.querySelector('.dl-linked-formula').getBoundingClientRect().bottom,
              };
            });
            const label = `${width}/${chapter}/${id}/${phase}`;
            checks.push({ label, ...measured });
            assert.deepEqual(measured.failures, [], `${label}: text exceeds its flow node`);
            assert.ok(measured.scrollWidth <= measured.width + 1, `${label}: stage overflow`);
            assert.ok(measured.documentWidth <= width + 1, `${label}: page overflow`);
            if (id === 'grpo' && width === 390) assert.ok(measured.formulaBottom < 900, `${label}: clipping equation below first screen (${measured.formulaBottom})`);
            if (id === 'mdp') assert.match(await lesson.locator('[aria-current="step"]').innerText(), phase === 'before' ? /Fresh session/ : /Engaged session/);
            if (phase === 'after') assert.equal(await lesson.locator('.dl-causal').getAttribute('data-causal-phase'), 'applied');
            if ([390, 1440].includes(width)) await page.screenshot({ path: path.join(dir, `${id}${phase === 'after' ? '-after' : ''}.png`) });
            if (phase === 'before' && [390, 1440].includes(width)) {
              await dialog.evaluate(el => { el.scrollTop = el.scrollHeight; });
              await page.screenshot({ path: path.join(dir, `${id}-bottom.png`) });
              await dialog.evaluate(el => { el.scrollTop = 0; });
            }
          }
          await page.keyboard.press('Escape');
        }
      }
    }
    assert.equal(checks.length, 96);
    assert.deepEqual(errors, []);
  } finally {
    fs.writeFileSync(path.join(out, 'report.json'), JSON.stringify({ checks, errors }, null, 2));
    await browser.close();
  }
});
