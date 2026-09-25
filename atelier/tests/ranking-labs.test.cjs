'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const lab = require('../ranking-labs.js');
const close = (a, b, eps = 1e-10) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

test('Node import exports all eight complete content overrides, independently of window', () => {
  assert.equal(Object.keys(lab.content).length, 8);
  assert.equal(lab.content['rank-objectives'].viz, 'rank-objectives');
  for (const [kind, content] of Object.entries(lab.content)) {
    for (const field of ['title', 'summary', 'what', 'why', 'interview', 'details', 'math', 'code', 'quiz']) assert.ok(content[field], `${kind}.${field}`);
    assert.ok(Array.isArray(content.math.formula));
    assert.ok(content.math.annotations.length >= 3);
    assert.equal(content.quiz.options.filter(o => o.correct).length, 1);
    assert.deepEqual(content.controls, []);
    assert.deepEqual(content.presets, []);
    assert.equal(content.geometry, null);
    assert.ok(lab.render(kind, lab.initial(kind)).includes('data-action="reset"'));
  }
});

test('browser path registers exactly once through the provided core contract', () => {
  const registrations = [];
  const context = { window: { AtelierLab: { createModule: options => registrations.push(options) } } };
  vm.runInNewContext(fs.readFileSync(require.resolve('../ranking-labs.js'), 'utf8'), context);
  assert.equal(registrations.length, 1);
  assert.equal(registrations[0].id, 'ranking-labs');
  assert.equal(typeof registrations[0].reduce, 'function');
});

test('stable logistic math handles extreme logits', () => {
  close(lab.sigmoid(0), .5);
  close(lab.softplus(1000), 1000);
  close(lab.softplus(-1000), 0);
  close(lab.bce(1000, 1), 0);
  close(lab.pairwise(-1000, 1000).loss, 2000);
  close(lab.pairwise(-1000, 1000).gradients[0], -1);
  close(lab.softmax([1000, 1000])[0], .5);
});

test('BCE preserves tiny positive losses without subtractive cancellation', () => {
  const expected = Math.log1p(Math.exp(-40));
  assert.ok(lab.bce(40, 1) > 0);
  close(lab.bce(40, 1) / expected, 1);
  close(lab.bce(-40, 0) / expected, 1);
  close(lab.bce(40, 0), 40);
  close(lab.bce(-40, 1), 40);
});

test('tower encoders and exact retrieval use actual matrix products', () => {
  const q = lab.matvec(lab.QUERY_W, [1, 0, 0]);
  assert.deepEqual(q, [1, .2]);
  assert.deepEqual(lab.catalog[0].vector, [1, .1]);
  close(lab.dot(q, lab.catalog[0].vector), 1.02);
  assert.deepEqual(lab.retrieve(q, 3).map(x => x.id), ['A', 'B', 'F']);
  const rl = lab.matvec(lab.QUERY_W, [0, 0, 1]);
  assert.equal(lab.retrieve(rl, 1)[0].id, 'C');
  for (const row of lab.retrieve(q, 6, true)) assert.ok(row.score >= -1 && row.score <= 1 + 1e-12);
  assert.deepEqual(lab.normalize([0, 0]), [0, 0]);
});

test('factor gradient is simultaneous, analytic, and leaves unexposed pairs untouched', () => {
  const factors = lab.factorInitial(), original = structuredClone(factors);
  const result = lab.factorStep(factors, 0, 0, 1);
  close(result.residual, lab.sigmoid(1.02) - 1);
  close(result.gp[0], result.residual + .02);
  close(result.gq[1], result.residual * .2 + .02 * .1);
  close(result.factors.items[0][1], .1 - .4 * result.gq[1]);
  assert.deepEqual(factors, original);
  assert.ok(result.after < result.before);
  assert.notEqual(lab.dot(factors.users[0], factors.items[3]), lab.dot(result.factors.users[0], result.factors.items[3]));
  assert.deepEqual(result.factors.users[1], factors.users[1]);
  assert.deepEqual(result.factors.items[3], factors.items[3]);
  const skipped = lab.factorStep(factors, 0, 3, null);
  assert.equal(skipped.skipped, true);
  assert.equal(skipped.factors, factors);
});

test('factor gradients match finite differences of regularized BCE', () => {
  const factors = lab.factorInitial(), r = lab.factorStep(factors, 0, 1, 0);
  const p = factors.users[0], q = factors.items[1], eps = 1e-6;
  const loss = (a, b) => lab.bce(lab.dot(a, b), 0) + .01 * (lab.dot(a, a) + lab.dot(b, b));
  for (let j = 0; j < 2; j++) {
    const up = [...p], down = [...p]; up[j] += eps; down[j] -= eps;
    close((loss(up, q) - loss(down, q)) / (2 * eps), r.gp[j], 1e-7);
  }
});

test('an exposure by Lee updates D without inventing a Morgan-D observation', () => {
  assert.equal(lab.observations[0][3], null);
  assert.equal(lab.observations[2][3], 1);
  const factors = lab.factorInitial();
  const result = lab.factorStep(factors, 2, 3, lab.observations[2][3]);
  assert.deepEqual(result.factors.users[0], factors.users[0]);
  assert.notEqual(lab.dot(factors.users[0], factors.items[3]), lab.dot(result.factors.users[0], result.factors.items[3]));
  assert.equal(lab.observations[0][3], null);
});

test('pairwise gradients match finite differences and preferred score rises', () => {
  const eps = 1e-6, r = lab.pairwise(.2, 1.2);
  close(r.loss, Math.log1p(Math.exp(1)));
  close(r.gradients[0], (lab.pairwise(.2 + eps, 1.2).loss - lab.pairwise(.2 - eps, 1.2).loss) / (2 * eps), 1e-7);
  close(r.gradients[0] + r.gradients[1], 0);
  const state = lab.initial('rank-objectives');
  const next = lab.reduce('rank-objectives', state, 'train');
  assert.ok(next.scores[2] > next.scores[1]);
  assert.ok(lab.objective(next.scores, 'pair', 0, 3).ndcg > lab.objective(state.scores, 'pair', 0, 3).ndcg);
  assert.deepEqual(lab.reduce('rank-objectives', next, 'undo').scores, state.scores);
});

test('NDCG and swap delta are exact, truncated, and fixed-denominator', () => {
  const ideal = 7 + 3 / Math.log2(3) + 1 / Math.log2(4);
  close(lab.rankingMetrics([0, 1, 2, 3], 3).idcg, ideal);
  close(lab.rankingMetrics([0, 1, 2, 3], 3).dcg, 1 / Math.log2(3) + 1.5);
  close(lab.rankingMetrics([3, 2, 1, 0], 3).ndcg, 1);
  close(lab.rankingMetrics([0, 1, 2, 3], 3).rr, 1 / 3);
  const rels = [0, 2, 1, 3], swapped = [3, 2, 1, 0];
  close(lab.swapDelta(rels, 0, 3, 3), Math.abs(lab.rankingMetrics(rels, 3).ndcg - lab.rankingMetrics(swapped, 3).ndcg));
  close(lab.swapDelta(rels, 2, 3, 2), 0);
  close(lab.rankingMetrics([0, 0], 2).ndcg, 0);
  assert.ok(lab.rankingMetrics([2, 2], 2, [3, 2, 2, 1]).ndcg < 1);
});

test('Lambda gradient scales the actual selected-pair gradient by swap importance', () => {
  const s = lab.initial('rank-objectives');
  const pair = lab.objective(s.scores, 'pair', 0, 3), lambda = lab.objective(s.scores, 'lambda', 0, 3);
  close(lambda.gradients[2], pair.gradients[2] * lambda.delta);
  close(lambda.loss, pair.loss * lambda.delta);
  close(lambda.delta, Math.abs(lambda.swappedNdcg - lambda.ndcg));
});

test('pointwise and listwise objective gradients match finite differences', () => {
  const scores = [.7, 1.2, .2, .4], eps = 1e-6;
  for (const mode of ['point', 'list']) {
    const result = lab.objective(scores, mode, 0, 3);
    for (let i = 0; i < 4; i++) {
      const up = [...scores], down = [...scores]; up[i] += eps; down[i] -= eps;
      close(result.gradients[i], (lab.objective(up, mode, 0, 3).loss - lab.objective(down, mode, 0, 3).loss) / (2 * eps), 1e-7);
    }
    if (mode === 'list') {
      close(result.targets.reduce((a, b) => a + b), 1);
      close(result.gradients.reduce((a, b) => a + b), 0);
    }
  }
});

test('oracle cannot recover an omitted candidate; added source raises the ceiling', () => {
  const missing = lab.funnel(3, false, true), fixed = lab.funnel(3, true, true);
  assert.ok(!missing.ranked.some(item => item.id === 'D'));
  close(missing.recall, 2 / 3);
  close(missing.metrics.ndcg, missing.ceiling);
  assert.ok(missing.ceiling < 1);
  assert.equal(fixed.ranked[0].id, 'D');
  close(fixed.recall, 1);
  close(fixed.ceiling, 1);
  assert.equal(new Set(lab.funnel(6, true, false).candidates.map(item => item.id)).size, 6);
});

test('cold-start posterior skips unexposed records and counts shown non-clicks', () => {
  close(lab.posterior(.25, []).mean, .25);
  close(lab.posterior(.25, [null]).mean, .25);
  close(lab.posterior(.25, [0]).mean, .2);
  close(lab.posterior(.25, [1, 1]).mean, .5);
  assert.equal(lab.posterior(.25, [null, 0, 1]).n, 2);
  const before = lab.coldRows(2, 'content').find(item => item.id === 'D');
  const after = lab.coldRows(3, 'content').find(item => item.id === 'D');
  assert.deepEqual(after, before);
  const warm = lab.coldRows(8, 'content').find(item => item.id === 'D');
  assert.equal(warm.n, 3);
  assert.equal(warm.clicks, 3);
  assert.ok(warm.mean > before.mean);
  assert.ok(lab.coldRows(0, 'population').every(item => item.mean === .25));
});

test('confusion counts correspond to records; no-selection precision is undefined', () => {
  const c = lab.confusion(lab.impressions, .59);
  assert.deepEqual(c.tp, [1, 3, 4, 6]);
  assert.deepEqual(c.fp, [2, 5]);
  assert.deepEqual(c.fn, [8, 11]);
  assert.deepEqual(c.tn, [7, 9, 10, 12]);
  close(c.precision, 2 / 3); close(c.recall, 2 / 3); close(c.f1, 2 / 3);
  const none = lab.confusion(lab.impressions, 1.01);
  assert.equal(none.precision, null);
  assert.equal(none.recall, 0);
  assert.equal(lab.confusion([{ id: 1, p: .2, y: 0 }], .5).recall, null);
  assert.equal(lab.confusion(lab.impressions, 0).recall, 1);
});

test('recall is monotone under lower cutoffs while precision need not be', () => {
  let previous = 0;
  for (const item of lab.impressions) {
    const c = lab.confusion(lab.impressions, item.p);
    assert.ok(c.recall >= previous); previous = c.recall;
  }
  assert.ok(lab.confusion(lab.impressions, .81).precision > lab.confusion(lab.impressions, .88).precision);
});

test('calibration bins derive means, event rates and ECE from actual outcomes', () => {
  const c = lab.calibration(lab.impressions, 3);
  const upper = c.bins[2];
  assert.equal(upper.n, 5);
  close(upper.mean, (.94 + .88 + .81 + .74 + .68) / 5);
  close(upper.rate, 3 / 5);
  close(upper.contribution, 5 / 12 * Math.abs(upper.mean - .6));
  close(c.ece, c.bins.reduce((total, bin) => total + bin.contribution, 0));
  close(c.brier, lab.impressions.reduce((total, item) => total + (item.p - item.y) ** 2, 0) / 12);
  const extremes = lab.calibration([{ id: 1, p: 0, y: 0 }, { id: 2, p: 1, y: 1 }], 5);
  close(extremes.ece, 0);
  assert.equal(extremes.bins[4].n, 1);
  assert.equal(extremes.bins[2].mean, null);
});

test('positive temperature preserves all ranks but changes probabilities and Brier', () => {
  const raw = lab.calibration(lab.impressions, 3), softened = lab.calibration(lab.impressions, 3, 2);
  assert.deepEqual(raw.transformed.map(x => x.id), softened.transformed.map(x => x.id));
  for (let i = 1; i < softened.transformed.length; i++) assert.ok(softened.transformed[i - 1].p > softened.transformed[i].p);
  assert.notEqual(raw.brier, softened.brier);
  close(lab.temperature(.8, 2), 2 / 3);
  assert.throws(() => lab.temperature(.5, 0), RangeError);
});

test('state transitions are immutable, undoable and resettable', () => {
  for (const kind of Object.keys(lab.content)) {
    const state = lab.initial(kind), copy = structuredClone(state);
    lab.reduce(kind, state, 'invalid', 'not-a-number');
    assert.deepEqual(state, copy);
    assert.deepEqual(lab.reduce(kind, state, 'reset'), copy);
  }
  let state = lab.initial('ranking-metrics');
  const original = structuredClone(state);
  state = lab.reduce('ranking-metrics', state, 'move', '2:-1');
  assert.deepEqual(state.order, [1, 3, 2, 0]);
  assert.deepEqual(original.order, [1, 3, 0, 2]);
  assert.equal(state.selected, 2);
  let cold = lab.initial('cold-start');
  for (let i = 0; i < 100; i++) cold = lab.reduce('cold-start', cold, 'next');
  assert.equal(cold.step, lab.exposureLog.length);
});

test('render every objective, temperature, selection and trace without invalid numbers', () => {
  const check = (kind, state) => {
    const html = lab.render(kind, state);
    assert.ok(!/NaN|undefined|Infinity/.test(html), `${kind}: invalid value`);
    for (const svg of html.matchAll(/<svg[\s\S]*?<\/svg>/g)) assert.ok(!/<p\b/.test(svg[0]));
  };
  for (const kind of Object.keys(lab.content)) check(kind, lab.initial(kind));
  for (const mode of ['pair', 'lambda', 'point', 'list']) {
    let state = { ...lab.initial('rank-objectives'), mode };
    for (let i = 0; i < 30; i++) { check('rank-objectives', state); state = lab.reduce('rank-objectives', state, 'train'); }
  }
  for (const t of [.5, 1, 2]) for (const bins of [3, 5]) for (let bin = 0; bin < bins; bin++) check('calibration', { t, bins, bin });
  for (let cut = 0; cut <= 12; cut++) check('threshold-metrics', { cut, cell: 'all', cost: 5 });
  for (let user = 0; user < 3; user++) for (let item = 0; item < 6; item++) check('matrix-factorization', { ...lab.initial('matrix-factorization'), user, item });
  for (let step = 0; step <= lab.exposureLog.length; step++) check('cold-start', { step, profile: 'content', item: 'D' });
  for (let query = 0; query < 3; query++) for (const item of lab.catalog) for (const cosine of [false, true]) check('two-tower', { query, item: item.id, cosine });
  for (const k of [2, 3, 4, 6]) for (const extra of [false, true]) for (const oracle of [false, true]) check('retrieval-funnel', { k, extra, oracle, item: 'D' });
});

test('every ordering has bounded NDCG and local swap importance equals metric change', () => {
  const permutations = xs => xs.length ? xs.flatMap((x, i) => permutations(xs.filter((_, j) => j !== i)).map(tail => [x, ...tail])) : [[]];
  for (const order of permutations([0, 1, 2, 3])) {
    const rels = order.map(i => lab.rankItems[i].rel);
    for (const k of [1, 2, 3, 4]) {
      const m = lab.rankingMetrics(rels, k);
      assert.ok(m.ndcg >= 0 && m.ndcg <= 1);
      for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) {
        const swapped = [...rels]; [swapped[i], swapped[j]] = [swapped[j], swapped[i]];
        close(lab.swapDelta(rels, i, j, k), Math.abs(lab.rankingMetrics(swapped, k).ndcg - m.ndcg));
      }
    }
  }
});

test('unknown factor cell cannot be trained and exposed updates are undoable', () => {
  const start = lab.initial('matrix-factorization');
  const unknown = lab.reduce('matrix-factorization', start, 'cell', '0:3');
  assert.deepEqual(lab.reduce('matrix-factorization', unknown, 'train'), unknown);
  const trained = lab.reduce('matrix-factorization', start, 'train');
  assert.deepEqual(lab.reduce('matrix-factorization', trained, 'undo').factors, start.factors);
  assert.ok(lab.render('matrix-factorization', unknown).includes('aria-label="Morgan, D ANN indexing: not exposed, unknown label"'));
});

test('every candidate-set oracle is an upper bound for the contextual reranker', () => {
  for (const k of [2, 3, 4, 6]) for (const extra of [false, true]) {
    const actual = lab.funnel(k, extra, false), oracle = lab.funnel(k, extra, true);
    assert.ok(actual.metrics.ndcg <= actual.ceiling + 1e-12);
    close(oracle.metrics.ndcg, oracle.ceiling);
    assert.ok(actual.ranked.every(item => actual.candidates.some(candidate => candidate.id === item.id)));
  }
});

test('dynamic item text is escaped rather than interpreted as markup', () => {
  const previous = lab.catalog[3].name;
  try {
    lab.catalog[3].name = '<img src=x onerror="bad()">';
    for (const kind of ['matrix-factorization', 'two-tower', 'retrieval-funnel', 'cold-start']) {
      const html = lab.render(kind, lab.initial(kind));
      assert.ok(!html.includes('<img'));
      assert.ok(html.includes('&lt;img'));
    }
  } finally {
    lab.catalog[3].name = previous;
  }
});

// This checks the authored first-view contract without using the parent's browser.
function withoutClosedDetails(html) {
  let depth = 0, visible = '', previous = 0;
  for (const match of html.matchAll(/<\/?details\b[^>]*>/g)) {
    if (depth === 0) visible += html.slice(previous, match.index);
    if (match[0].startsWith('</')) depth--;
    else {
      assert.ok(!/\sopen(?:\s|=|>)/.test(match[0]));
      depth++;
    }
    assert.ok(depth >= 0);
    previous = match.index + match[0].length;
  }
  assert.equal(depth, 0);
  return visible + html.slice(previous);
}

test('every first view has one question, one enabled primary action and no calculation table', () => {
  const mechanisms = {
    'matrix-factorization': 'rk-factor-source', 'two-tower': 'rk-tower-flow',
    'rank-objectives': 'rk-objective-list', 'retrieval-funnel': 'rk-pipeline',
    'cold-start': 'rk-evidence-flow', 'threshold-metrics': 'rk-impressions',
    calibration: 'rk-bin-list', 'ranking-metrics': 'rk-ranked-list',
  };
  for (const kind of Object.keys(lab.content)) {
    const state = lab.initial(kind), html = lab.render(kind, state), visible = withoutClosedDetails(html);
    assert.ok(html.includes('<summary>Inspect the calculation</summary>'), kind);
    assert.ok(!/<table\b|class="rk-equation"/.test(visible), `${kind} opens with calculations`);
    const questions = [...visible.matchAll(/<h3\b[^>]*>(.*?)<\/h3>/g)];
    assert.equal(questions.length, 1, kind);
    assert.ok(questions[0][1].endsWith('?'), kind);
    assert.ok(visible.includes(mechanisms[kind]), `${kind} has no visible mechanism`);
    const actions = [...visible.matchAll(/<button class="rk-primary"[^>]*>/g)];
    assert.equal(actions.length, 1, kind);
    assert.ok(!/\sdisabled/.test(actions[0][0]), kind);
    const action = actions[0][0].match(/data-action="([^"]*)"/)[1];
    const value = actions[0][0].match(/data-value="([^"]*)"/)[1];
    const changed = lab.reduce(kind, state, action, value);
    assert.notDeepEqual(changed, state, `${kind} primary action has no effect`);
    assert.notEqual(withoutClosedDetails(lab.render(kind, changed)), visible, `${kind} consequence is hidden`);
  }
});

test('ranking cards show the actual preferred-item jump and unchanged ranks on later steps', () => {
  const start = lab.initial('rank-objectives');
  const next = lab.reduce('rank-objectives', start, 'train');
  const preferred = lab.rankMovement(next.scores, start.scores).find(item => item.index === 2);
  assert.equal(preferred.beforeRank, 4);
  assert.equal(preferred.rank, 1);
  assert.equal(preferred.rankDelta, 3);
  assert.ok(preferred.scoreDelta > 0);
  const visible = withoutClosedDetails(lab.render('rank-objectives', next));
  assert.ok(visible.includes('Before #4 &rarr; Now #1'));
  assert.ok(visible.includes('&uarr; 3 places'));
  assert.ok(visible.includes('Next step: &uarr; score'));
  const again = lab.reduce('rank-objectives', next, 'train');
  const same = lab.rankMovement(again.scores, next.scores).find(item => item.index === 2);
  assert.equal(same.beforeRank, 1);
  assert.equal(same.rank, 1);
  assert.equal(same.rankDelta, 0);
  assert.ok(same.scoreDelta > 0);
  assert.ok(lab.render('rank-objectives', again).includes('Before #1 &rarr; Now #1'));
  assert.ok(lab.render('rank-objectives', again).includes('= same rank'));
});

test('visible tower flow keeps encoders independent and projections secondary', () => {
  const state = lab.initial('two-tower'), visible = withoutClosedDetails(lab.render('two-tower', state));
  assert.ok(visible.includes('Query encoder'));
  assert.ok(visible.includes('Item encoder'));
  assert.ok(visible.includes('Similarity: q dot v'));
  assert.ok(!visible.includes('Wq ='));
  assert.ok(!visible.includes('Wi ='));
  const cached = lab.catalog.map(item => [...item.vector]);
  const next = lab.reduce('two-tower', state, 'query', '2');
  assert.deepEqual(lab.catalog.map(item => item.vector), cached);
  const html = withoutClosedDetails(lab.render('two-tower', next));
  assert.ok(html.includes('Optimize an RL policy'));
  assert.ok(html.includes('C: GRPO training'));
});

test('cold-start Next follows the logged item, including the unexposed event', () => {
  let state = lab.initial('cold-start');
  state = lab.reduce('cold-start', state, 'next');
  assert.equal(state.item, 'A');
  state = lab.reduce('cold-start', state, 'next');
  assert.equal(state.item, 'B');
  state = lab.reduce('cold-start', state, 'next');
  assert.equal(state.item, 'D');
  assert.equal(lab.coldRows(state.step, state.profile).find(item => item.id === 'D').n, 0);
  assert.ok(withoutClosedDetails(lab.render('cold-start', state)).includes('not a zero click rate'));
});

test('table typography stays readable and overflows locally instead of shrinking', () => {
  const css = fs.readFileSync(require.resolve('../ranking-labs.css'), 'utf8');
  assert.ok(css.includes('min-width: 360px'));
  assert.ok(css.includes('overflow-x: auto'));
  assert.ok(css.includes('font-size: max(15px, .94rem)'));
  assert.ok(!/font(?:-size)?:\s*\.(?:75|76|78|8[0145])rem/.test(css));
});
