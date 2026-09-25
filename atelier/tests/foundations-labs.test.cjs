const test = require('node:test');
const assert = require('node:assert/strict');
const lab = require('../foundations-labs.js');

const close = (a, b, eps = 1e-10) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
const vectorClose = (a, b) => a.forEach((v, i) => close(v, b[i]));
const matrixClose = (a, b) => a.forEach((row, i) => vectorClose(row, b[i]));

test('softmax conserves mass, stays shift-invariant, handles large logits', () => {
  const p = lab.softmax([2, 1, 0]);
  close(p.reduce((a, b) => a + b), 1);
  vectorClose(p, lab.softmax([10002, 10001, 10000]));
  close(p[0], .6652409557748218);
  vectorClose(lab.normalize([0, 3, 0]), [0, 1, 0]);
  assert.throws(() => lab.normalize([0, 0]), RangeError);
  assert.throws(() => lab.softmax([Infinity]), RangeError);
});

test('inverse-CDF boundaries and seeded sampling use actual probabilities', () => {
  const p = [.6, .3, .1];
  assert.equal(lab.categorical(p, 0), 0);
  assert.equal(lab.categorical(p, .6), 1);
  assert.equal(lab.categorical(p, .95), 2);
  assert.equal(lab.categorical([0, 1, 0], 0), 1);
  const a = lab.sampleDistribution(p, 100000);
  const b = lab.sampleDistribution(p, 100000);
  assert.deepEqual(a, b);
  assert.equal(a.counts.reduce((x, y) => x + y), 100000);
  a.counts.forEach((c, i) => close(c / 100000, p[i], .005));
  const first = lab.sampleDistribution(p, 10);
  const second = lab.sampleDistribution(p, 20, first.seed);
  const full = lab.sampleDistribution(p, 30);
  vectorClose(first.counts.map((x, i) => x + second.counts[i]), full.counts);
});

test('continuous interval probability conserves area, not density height', () => {
  close(lab.densityMass(0, 1), 1);
  close(lab.densityMass(.5, .75), .3125);
  close([0, 1, 2, 3].reduce((s, i) => s + lab.densityMass(i / 4, (i + 1) / 4), 0), 1);
  assert.throws(() => lab.densityMass(-1, .5), RangeError);
});

test('expected risk reverses model ordering under population shift', () => {
  const A = [.1, .8], B = [.25, .35];
  close(lab.expectedRisk([.8, .2], A).mean, .24);
  close(lab.expectedRisk([.8, .2], B).mean, .27);
  close(lab.expectedRisk([.3, .7], A).mean, .59);
  close(lab.expectedRisk([.3, .7], B).mean, .32);
  close(lab.expectedRisk([.8, .2], A).variance, .0784);
  assert.throws(() => lab.expectedRisk([.8, .3], A), RangeError);
});

test('Bayes conserves joint counts and conditions on the correct subset', () => {
  for (const p of [.01, .1, .5, 0, 1]) {
    const b = lab.bayesCounts(p);
    close(b.tp + b.fp + b.fn + b.tn, 10000);
    close(b.tp + b.fn, b.spam);
    close(b.fp + b.tn, b.ham);
    close(b.posterior, b.tp / (b.tp + b.fp));
    const negative = lab.bayesCounts(p, .9, .02, 10000, false);
    close(negative.posterior, b.fn / (b.fn + b.tn));
  }
  close(lab.bayesCounts(.01).posterior, .3125);
  close(lab.bayesCounts(.1).posterior, 5 / 6);
  assert.equal(lab.bayesCounts(0, 0, 0).posterior, null);
});

test('cross-entropy = entropy + KL, including support edge cases', () => {
  for (const q of [[.9, .09, .01], [.6, .3, .1], [1 / 3, 1 / 3, 1 / 3]]) {
    const r = lab.information([.6, .3, .1], q);
    close(r.crossEntropy, r.entropy + r.kl);
    assert.ok(r.kl > -1e-12);
  }
  close(lab.information([.6, .3, .1], [.6, .3, .1]).kl, 0);
  const deterministic = lab.information([1, 0], [1, 0]);
  close(deterministic.entropy, 0); close(deterministic.crossEntropy, 0);
  assert.equal(lab.information([.5, .5], [1, 0]).crossEntropy, Infinity);
  assert.ok(lab.information([.6, .3, .1], [.9, .09, .01]).terms[0].kl < 0);
});

test('entropy regions encode probability times surprise with exact area and a fixed decomposition scale', () => {
  const initial = lab.initial('entropy');
  assert.equal(initial.model, 'matched');
  for (const [model, q] of Object.entries({ matched: [.6, .3, .1], confident: [.9, .09, .01], uniform: [1 / 3, 1 / 3, 1 / 3] })) {
    const html = lab.render('entropy', { ...initial, model });
    const rectangles = [...html.matchAll(/class="fl-surprise-area" data-probability="([^"]+)" data-surprise="([^"]+)" x="[^"]+" y="[^"]+" width="([^"]+)" height="([^"]+)"/g)];
    assert.equal(rectangles.length, 3);
    let totalArea = 0;
    rectangles.forEach((r, i) => {
      const [, probability, surprise, width, height] = r.map(Number);
      close(probability, [.6, .3, .1][i]);
      close(surprise, -Math.log2(q[i]));
      close(width / 280, probability);
      close(height / 24, surprise);
      totalArea += width * height / (280 * 24);
    });
    const info = lab.information([.6, .3, .1], q);
    close(totalArea, info.crossEntropy);
    const hWidth = Number(html.match(/class="fl-information-entropy" style="width:([^%]+)%/)[1]);
    const klWidth = Number(html.match(/class="fl-information-kl" style="width:([^%]+)%/)[1]);
    close(hWidth / 100 * 2, info.entropy);
    close(klWidth / 100 * 2, info.kl);
    assert.ok(html.includes('She poured the coffee into the'));
  }
  const matched = lab.information([.6, .3, .1], [.6, .3, .1]);
  assert.ok(matched.terms[2].surprise > matched.terms[0].surprise);
  assert.ok(matched.terms[2].crossEntropy < matched.terms[0].crossEntropy);
});

test('binary loss values and logit derivatives match finite differences', () => {
  const result = lab.binaryLoss(1, .01);
  close(result.log, -Math.log(.01)); close(result.square, .9801);
  close(result.logGradient, -.99); close(result.squareGradient, -.019602);
  const h = 1e-5, sigmoid = z => 1 / (1 + Math.exp(-z));
  for (const y of [0, 1]) for (const z of [-5, 0, 5]) {
    const r = lab.binaryLoss(y, sigmoid(z));
    const left = lab.binaryLoss(y, sigmoid(z - h)), right = lab.binaryLoss(y, sigmoid(z + h));
    close((right.log - left.log) / (2 * h), r.logGradient, 1e-7);
    close((right.square - left.square) / (2 * h), r.squareGradient, 1e-7);
  }
  assert.equal(lab.binaryLoss(1, 0).log, Infinity);
  close(lab.binaryLoss(0, 0).log, 0);
  close(lab.binaryRisk(0, 0).log, 0);
});

test('both expected binary losses are minimized at the true probability', () => {
  const optimum = lab.binaryRisk(.1, .1);
  for (const q of [.01, .3, .5, .7, .9, .99]) {
    assert.ok(lab.binaryRisk(.1, q).log > optimum.log);
    assert.ok(lab.binaryRisk(.1, q).square > optimum.square);
  }
  close(optimum.logGradient, 0); close(optimum.squareGradient, 0);
});

test('vector normalization and retrieval ordering preserve the exact geometry', () => {
  const q = [1, .5], A = [.9, .6], B = [2, -.1];
  close(lab.norm(lab.unit(B)), 1);
  close(lab.cosine(B, lab.unit(B)), 1);
  assert.equal(lab.unit([0, 0]), null);
  assert.equal(lab.cosine(q, [0, 0]), null);
  assert.ok(lab.dot(q, B) > lab.dot(q, A));
  assert.ok(lab.cosine(q, A) > lab.cosine(q, B));
  close(lab.dot(q, [-.5, 1]), 0);
});

test('linear transformation equals weighted basis images; area collapse is exact', () => {
  const A = [[1, 1], [0, 1]], x = [1, 2];
  vectorClose(lab.matvec(A, x), [3, 2]);
  vectorClose(lab.matvec(A, x), [A[0][0] * x[0] + A[0][1] * x[1], A[1][0] * x[0] + A[1][1] * x[1]]);
  close(lab.det(A), 1);
  close(lab.det([[1, 1], [1, 1]]), 0);
  vectorClose(lab.matvec([[1, 1], [1, 1]], [1, -1]), [0, 0]);
  vectorClose(lab.matvec([[0, -1], [1, 0]], [1, 2]), [-2, 1]);
});

test('symmetric eigendecomposition reconstructs PSD, indefinite, diagonal and repeated cases', () => {
  for (const A of [[[2, 1], [1, 2]], [[1, 2], [2, 1]], [[2, 0], [0, 2]], [[1, 0], [0, 4]], [[2, -.5], [-.5, 1]], [[0, 0], [0, 0]]]) {
    const e = lab.eigenSymmetric(A);
    e.vectors.forEach((v, i) => {
      close(lab.norm(v), 1);
      vectorClose(lab.matvec(A, v), v.map(x => x * e.values[i]));
    });
    close(lab.dot(...e.vectors), 0);
    matrixClose(lab.reconstructEigen(A), A);
  }
  assert.throws(() => lab.eigenSymmetric([[1, 2], [0, 1]]), RangeError);
});

test('power iteration approaches the dominant eigenline without pretending other starts escape', () => {
  const A = [[2, 1], [1, 2]], e = lab.eigenSymmetric(A);
  assert.ok(Math.abs(lab.cosine(lab.powerStep(A, [1, 0], 5), e.vectors[0])) > .9999);
  close(Math.abs(lab.cosine(lab.powerStep(A, e.vectors[1], 3), e.vectors[1])), 1);
});

test('SVD reconstructs general, reflective, rank-one, zero and equal-spectrum matrices', () => {
  for (const A of [[[3, 1], [0, 1]], [[2, 0], [0, 2]], [[3, 0], [4, 0]], [[0, 0], [0, 0]], [[2, 2], [1, -1]], [[0, 2], [3, 0]], [[3, 1], [1, 3]], [[1, 2], [2, 4]], [[1, 0], [0, 1e-7]]]) {
    const s = lab.svd2(A);
    matrixClose(lab.reconstructSvd(s), A);
    matrixClose(lab.multiply(lab.transpose(s.U), s.U), [[1, 0], [0, 1]]);
    matrixClose(lab.multiply(s.Vt, lab.transpose(s.Vt)), [[1, 0], [0, 1]]);
    const error = lab.frobenius(lab.matrixSubtract(A, lab.reconstructSvd(s, 1)));
    close(error, s.singular[1]);
    close(error ** 2, s.singular[1] ** 2);
    close(lab.frobenius(A) ** 2, s.singular.reduce((v, x) => v + x * x, 0));
  }
});

test('SVD is scale-invariant and keeps orthogonal factors near rank deficiency', () => {
  const identity = [[1, 0], [0, 1]];
  for (const base of [[[-1, 0], [0, 1]], [[1, 2], [0, 1]], [[1, 1], [1, 1 + 1e-10]]]) {
    for (const magnitude of [1e-120, 1e-13, 1, 1e120]) {
      const A = base.map(row => row.map(x => x * magnitude));
      const d = lab.svd2(A), reconstructed = lab.reconstructSvd(d);
      matrixClose(reconstructed.map(row => row.map(x => x / magnitude)), base);
      matrixClose(lab.multiply(lab.transpose(d.U), d.U), identity);
      matrixClose(lab.multiply(d.Vt, lab.transpose(d.Vt)), identity);
      close(Math.abs(lab.det(base)), (d.singular[0] / magnitude) * (d.singular[1] / magnitude));
    }
  }
});

test('all ten concepts have complete overrides, rendered initial scenes and reset behavior', () => {
  assert.equal(Object.keys(lab.content).length, 10);
  for (const [kind, c] of Object.entries(lab.content)) {
    assert.ok(c.title && c.summary && c.what && c.why && c.interview);
    assert.ok(Array.isArray(c.math.formula) && c.math.annotations.length >= 3);
    assert.ok(c.code.snippet && c.quiz.options.some(x => x.correct));
    assert.deepEqual(c.controls, []); assert.deepEqual(c.presets, []);
    assert.equal(c.geometry, null);
    const s = lab.initial(kind), html = lab.render(kind, s);
    assert.ok(html.includes('data-action='));
    assert.ok(!/NaN|undefined|Infinity/.test(html), `${kind} contains invalid computed text`);
    assert.deepEqual(lab.reduce(kind, s, 'reset', ''), s);
    assert.deepEqual(lab.reduce(kind, s, 'unknown', '<script>'), s);
  }
});

test('every initial view has one visible primary action, a question, a mechanism, and secondary tables', () => {
  function outsideDetails(html) {
    let depth = 0, visible = '';
    for (const part of html.split(/(<\/?details\b[^>]*>)/g)) {
      if (/^<details\b/.test(part)) depth++;
      else if (/^<\/details>/.test(part)) depth--;
      else if (depth === 0) visible += part;
    }
    return visible;
  }
  for (const kind of Object.keys(lab.content)) {
    const state = lab.initial(kind), html = lab.render(kind, state), visible = outsideDetails(html);
    const primaries = [...visible.matchAll(/<button class="fl-primary"([^>]*)>/g)];
    assert.equal(primaries.length, 1, `${kind}: one primary action expected`);
    assert.ok(/<h3 class="ml-question">[^<]*\?<\/h3>/.test(visible), `${kind}: question expected`);
    assert.ok(!visible.includes('<table'), `${kind}: initial tables should be secondary`);
    assert.ok(visible.includes('<svg') || visible.includes('fl-cohort') || visible.includes('fl-mass'), `${kind}: visible mechanism required`);
    assert.ok(/<summary[^>]*>Inspect the (?:calculation|underlying data)<\/summary>/.test(html));
    const action = primaries[0][1].match(/data-action="([^"]+)"/)[1];
    const value = primaries[0][1].match(/data-value="([^"]*)"/)?.[1] || '';
    const next = lab.reduce(kind, state, action, value);
    assert.notDeepEqual(next, state, `${kind}: primary must change state`);
    assert.notEqual(outsideDetails(lab.render(kind, next)), visible, `${kind}: primary must change visible scene`);
  }
});

test('native summaries have no action handler or module-owned disclosure state', () => {
  for (const kind of Object.keys(lab.content)) {
    const state = lab.initial(kind);
    const html = lab.render(kind, { ...state, openDetails: [0, 1, 2] });
    const summaries = [...html.matchAll(/<summary([^>]*)>/g)];
    assert.ok(summaries.length);
    assert.ok(summaries.every(match => !match[1].includes('data-action')));
    assert.ok(!/<details[^>]*\sopen(?:\s|>)/.test(html));
    assert.deepEqual(lab.reduce(kind, state, 'disclosure', '0'), state);
    const keys = [...html.matchAll(/data-detail-key="([^"]+)"/g)].map(match => match[1]);
    assert.equal(keys.length, new Set(keys).size, `${kind}: distinct disclosures need distinct core persistence keys`);
  }
});

test('distribution, expectation and every Bayes step consolidate data and arithmetic into one disclosure', () => {
  for (const kind of ['distribution', 'expectation', 'bayes']) {
    const initial = lab.initial(kind);
    const states = kind === 'bayes' ? [0, 1, 2].map(step => ({ ...initial, step })) : [initial];
    for (const state of states) {
      const html = lab.render(kind, state);
      assert.equal([...html.matchAll(/<summary>Inspect the calculation<\/summary>/g)].length, 1);
      assert.ok(html.includes('<table'), `${kind}: preserve the data table`);
    }
  }
});

test('every preset, trace step and selectable state renders finite computed values', () => {
  const choices = {
    distribution: { mode: ['categorical', 'density'], scene: ['peaked', 'flat', 'shifted'], select: [0, 1, 2], sample: [1, 100, 1000] },
    expectation: { population: ['train', 'shifted'], model: ['A', 'B'], select: [0, 1] },
    bayes: { prior: [.01, .1, .5], evidence: ['positive', 'negative'] },
    entropy: { model: ['matched', 'uniform', 'confident'], select: [0, 1, 2] },
    loss: { mode: ['observed', 'expected'], label: [0, 1], q: [.01, .1, .3, .5, .7, .9, .99] },
    vectors: { item: ['A', 'B', 'C', 'zero'], normalize: ['true', 'false'] },
    'dot-products': { item: ['A', 'B', 'C'], normalize: ['true', 'false'], query: ['first', 'second'] },
    'matrix-multiply': { scene: ['shear', 'rotate', 'collapse'], select: [0, 1, 2], row: [0, 1] },
    eigen: { scene: ['covariance', 'saddle', 'isotropic'], select: [0, 1, 2] },
    svd: { scene: ['mixed', 'balanced', 'rankone'], rank: [1, 2], select: [0, 1, 2, 3] },
  };
  for (const [kind, actions] of Object.entries(choices)) {
    let states = [lab.initial(kind)];
    for (const [action, values] of Object.entries(actions)) states = states.flatMap(s => values.map(value => lab.reduce(kind, s, action, String(value))));
    for (let s of states) {
      if (s.step !== undefined) s = { ...s, step: 0 };
      for (let step = 0; step < (s.step === undefined ? 1 : 6); step++) {
      const html = lab.render(kind, s);
      assert.ok(!/NaN|Infinity/.test(html), `${kind} ${JSON.stringify(s)}`);
      assert.ok(!/="(?:undefined|null)"/.test(html));
      s = lab.reduce(kind, s, 'next', '');
      }
    }
  }
});

test('browser registration uses the shared core and state persists across roots/remounts', () => {
  const { readFileSync } = require('node:fs');
  const { resolve } = require('node:path');
  const vm = require('node:vm');
  const context = { window: {}, document: { activeElement: null } };
  vm.createContext(context);
  vm.runInContext(readFileSync(resolve(__dirname, '../lab-core.js'), 'utf8'), context);
  vm.runInContext(readFileSync(resolve(__dirname, '../foundations-labs.js'), 'utf8'), context);
  const module = context.window.AtelierLessonModules[0];
  assert.ok(module.has('svd') && module.has('bayes'));
  function root() {
    return { innerHTML: '', isConnected: true, events: {}, contains: () => false,
      querySelectorAll: () => [],
      addEventListener(name, fn) { this.events[name] = fn; },
      removeEventListener(name) { delete this.events[name]; },
    };
  }
  const inline = root(), modal = root();
  module.render('bayes', inline); module.render('bayes', modal);
  assert.ok(inline.innerHTML.includes('Prior: 100 / 10,000 = 1%'));
  assert.equal(lab.initial('bayes').step, 0);
  const button = { dataset: { action: 'prior', value: '.1' }, tagName: 'BUTTON', disabled: false };
  inline.contains = node => node === button;
  inline.events.click({ target: { closest: () => button }, stopPropagation() {} });
  assert.ok(inline.innerHTML.includes('Prior: 1000 / 10,000 = 10%'));
  assert.equal(inline.innerHTML, modal.innerHTML);
  module.unmount(inline);
  assert.equal(Object.keys(inline.events).length, 0);
  const remount = root(); module.render('bayes', remount);
  assert.equal(remount.innerHTML, modal.innerHTML);
  const point = { dataset: { action: 'prior', value: '.5' } };
  modal.events.keydown({ key: 'Enter', target: { closest: () => point }, preventDefault() {}, stopPropagation() {} });
  assert.ok(remount.innerHTML.includes('Prior: 5000 / 10,000 = 50%'));
  point.dataset = { action: 'next', value: '' };
  modal.events.keydown({ key: 'Enter', target: { closest: () => point }, preventDefault() {}, stopPropagation() {} });
  assert.ok(remount.innerHTML.includes('Filter by evidence'));
  modal.events.keydown({ key: 'Enter', target: { closest: () => point }, preventDefault() {}, stopPropagation() {} });
  assert.ok(remount.innerHTML.includes('97.8%'));
});

test('raising one route score changes the fixed draw, while a shared shift cannot', () => {
  const before = lab.initial('distribution');
  const after = lab.reduce('distribution', before, 'boost', '');
  assert.equal(before.probe, after.probe);
  const p = lab.softmax(lab.distributionLogits(before));
  const boosted = lab.softmax(lab.distributionLogits(after));
  assert.equal(lab.categorical(p, before.probe), 1);
  assert.equal(lab.categorical(boosted, after.probe), 0);
  assert.ok(boosted[0] > p[0] && boosted[1] < p[1] && boosted[2] < p[2]);
  vectorClose(p, lab.softmax(lab.distributionLogits(before).map(x => x + 5)));
  const undo = lab.reduce('distribution', after, 'boost', '');
  vectorClose(lab.distributionLogits(undo), lab.distributionLogits(before));
  const inspected = lab.reduce('distribution', after, 'select', '1');
  assert.ok(!lab.render('distribution', inspected).includes('Same draw selects'));
});

test('cohort shift changes only ten population units and preserves conditional losses', () => {
  const state = lab.reduce('expectation', lab.initial('expectation'), 'population', 'shifted');
  const html = lab.render('expectation', state);
  assert.equal([...html.matchAll(/class="fl-person[^\"]*fl-reassigned/g)].length, 10);
  assert.ok(html.includes('data-before="0.24" data-after="0.59"'));
  for (const p of [[.8, .2], [.3, .7], [0, 1], [1, 0]]) {
    assert.ok(lab.expectedRisk(p, [.1, .2]).mean < lab.expectedRisk(p, [.25, .35]).mean);
  }
});

test('Bayes first action filters actual counts and the second changes the denominator', () => {
  const initial = lab.initial('bayes');
  const filtered = lab.reduce('bayes', initial, 'next', '');
  const final = lab.reduce('bayes', filtered, 'next', '');
  const b = lab.bayesCounts(initial.prior);
  assert.equal(b.numerator, 90); assert.equal(b.other, 198);
  assert.equal(b.evidence, 288);
  const middle = lab.render('bayes', filtered);
  assert.ok(middle.includes('90 survive') && middle.includes('198 survive'));
  assert.ok(middle.includes('10 removed') && middle.includes('9702 removed'));
  assert.ok(lab.render('bayes', final).includes('31.25%'));
  assert.ok(lab.render('bayes', final).includes('width:31.25%'));
});

test('entropy first action inspects a different frequency-weighted region without changing p or q', () => {
  const initial = lab.initial('entropy');
  const next = lab.reduce('entropy', initial, 'select', '0');
  assert.equal(initial.model, 'matched'); assert.equal(next.model, 'matched');
  const before = lab.render('entropy', initial), after = lab.render('entropy', next);
  assert.ok(before.includes('Compare common cup'));
  assert.ok(after.includes('Selected contribution'));
  assert.ok(after.includes('total entropy did not change'));
  const p = [.6, .3, .1], info = lab.information(p, p);
  close(info.terms[2].surprise - info.terms[0].surprise, Math.log2(6));
  assert.ok(info.terms[0].entropy > info.terms[2].entropy);
  const halfGlass = lab.information(p, [.65, .3, .05]);
  close(halfGlass.terms[2].surprise - info.terms[2].surprise, 1);
  close(halfGlass.terms[2].crossEntropy - info.terms[2].crossEntropy, .1);
  assert.notEqual(halfGlass.crossEntropy - info.crossEntropy, .1);
});

test('loss steps apply the same learning rate to actual logit gradients', () => {
  const update = lab.lossStep(1, .01);
  close(update.zLog - update.z, 1.98);
  close(update.zSquare - update.z, .039204);
  close(update.qLog, 1 / (1 + Math.exp(-update.zLog)));
  close(update.qSquare, 1 / (1 + Math.exp(-update.zSquare)));
  assert.ok(update.qLog > update.qSquare && update.qSquare > .01);
  for (const target of [0, 1, .1]) for (const q of [.01, .1, .5, .99]) {
    const u = lab.lossStep(target, q);
    assert.ok(lab.binaryRisk(target, u.qLog).log <= u.log + 1e-12);
    assert.ok(lab.binaryRisk(target, u.qSquare).square <= u.square + 1e-12);
  }
  const optimum = lab.lossStep(.1, .1);
  close(optimum.qLog, .1); close(optimum.qSquare, .1);
  assert.throws(() => lab.lossStep(1, 0), RangeError);
  assert.throws(() => lab.lossStep(1, .1, -1), RangeError);
  const after = lab.reduce('loss', lab.initial('loss'), 'learn', '');
  assert.equal(after.trained, true);
  assert.ok(lab.render('loss', after).includes('Hollow circles are the common start'));
  assert.equal(lab.reduce('loss', after, 'q', '.3').trained, false);
});

test('normalization and query scaling counterfactuals preserve the claimed quantities', () => {
  const v = [2, -.1], q = [1, .5];
  vectorClose(lab.unit(v), lab.unit(v.map(x => 3 * x)));
  vectorClose(lab.unit(v).map(x => -x), lab.unit(v.map(x => -3 * x)));
  for (const item of [[.9, .6], v, [-.5, 1]]) {
    close(lab.dot(q.map(x => 10 * x), item), 10 * lab.dot(q, item));
    close(lab.cosine(q.map(x => 10 * x), item), lab.cosine(q, item));
  }
  const before = lab.render('dot-products', lab.initial('dot-products'));
  const after = lab.render('dot-products', lab.reduce('dot-products', lab.initial('dot-products'), 'normalize', 'true'));
  assert.equal([...before.matchAll(/fl-rank-dot fl-rank-current/g)].length, 3);
  assert.equal([...after.matchAll(/fl-rank-dot fl-rank-current/g)].length, 3);
  assert.ok(after.includes('data-before="B" data-after="A"'));
});

test('matrix first action adds the second column at the first column tip', () => {
  const state = lab.reduce('matrix-multiply', lab.initial('matrix-multiply'), 'next', '');
  const html = lab.render('matrix-multiply', state);
  assert.equal(state.step, 2);
  assert.ok(html.includes('data-before="[1, 0]" data-after="[3, 2]"'));
  assert.ok(html.includes('data-term="2 x [1, 1]"'));
  vectorClose(lab.matvec([[1, 1], [1, 1]], [1, 2]), lab.matvec([[1, 1], [1, 1]], [2, 1]));
});

test('eigen residual is the perpendicular off-line component and shrinks on the default step', () => {
  const A = [[2, 1], [1, 2]];
  const residual = v => {
    const Av = lab.matvec(A, v), rho = lab.dot(v, Av);
    const r = Av.map((x, i) => x - rho * v[i]);
    close(lab.dot(v, r), 0);
    return lab.norm(r);
  };
  close(residual([1, 0]), 1);
  close(residual(lab.powerStep(A, [1, 0], 1)), .6);
  const state = lab.reduce('eigen', lab.initial('eigen'), 'next', '');
  assert.ok(lab.render('eigen', state).includes('data-before="1" data-after="0.6"'));
});

test('SVD restored vector component and matrix error are different exact quantities', () => {
  const A = [[3, 1], [0, 1]], x = [1, 1], d = lab.svd2(A);
  const rank1 = lab.reconstructSvd(d, 1), kept = lab.matvec(rank1, x), full = lab.matvec(A, x);
  const missing = full.map((v, i) => v - kept[i]);
  const component = d.U.map(row => row[1] * d.singular[1] * lab.dot(d.Vt[1], x));
  vectorClose(missing, component);
  close(lab.dot(kept, missing), 0);
  close(lab.norm(missing), d.singular[1] * Math.abs(lab.dot(d.Vt[1], x)));
  assert.ok(Math.abs(lab.norm(missing) - d.singular[1]) > .1);
  const state = lab.reduce('svd', lab.initial('svd'), 'rank', '2');
  assert.ok(lab.render('svd', state).includes('data-before="0.944" data-after="0"'));
  for (const sigma of [1, 2]) {
    const matrix = [[5, 0], [0, sigma]], svd = lab.svd2(matrix);
    close(lab.frobenius(lab.matrixSubtract(matrix, lab.reconstructSvd(svd, 1))) ** 2, sigma ** 2);
  }
});

test('every first action changes drawn geometry, not only prose; history stays one action deep', () => {
  function geometry(html) {
    return [...html.matchAll(/<(?:svg|path|circle|rect|polygon|line|g|div|span|button)\b[^>]*>/g)]
      .map(m => m[0]).filter(tag => /(?:\sx=|\sy=|\sd=|\scx=|\scy=|points=|\sx1=|fl-person|fl-mass|fl-cohort|fl-filter|fl-surprise-region)/.test(tag)).join('');
  }
  for (const kind of Object.keys(lab.content)) {
    const initial = lab.initial(kind), before = lab.render(kind, initial);
    const button = before.match(/<button class="fl-primary"([^>]*)>/)[1];
    const action = button.match(/data-action="([^"]+)"/)[1];
    const value = button.match(/data-value="([^"]*)"/)?.[1] || '';
    const next = lab.reduce(kind, initial, action, value);
    assert.notEqual(geometry(before), geometry(lab.render(kind, next)), kind);
    const twice = lab.reduce(kind, next, action, value);
    assert.ok(!twice.previous.previous, `${kind}: previous state must not recursively grow`);
    assert.ok(before.includes('class="fl-equation"'), `${kind}: selected equation must be visible`);
    assert.ok(!/<summary[^>]*data-action/.test(before));
  }
});

test('semantic highlights and rank nodes are styled without fading required text', () => {
  const css = require('node:fs').readFileSync(require('node:path').join(__dirname, '../foundations-labs.css'), 'utf8');
  assert.match(css, /\.fl-equation-label\s*\{[^}]*display:\s*block/);
  assert.match(css, /\.fl-equation-label::after\s*\{[^}]*content:\s*":"/);
  assert.match(css, /\.fl-term\s*\{[^}]*background:\s*transparent/);
  assert.match(css, /\.fl-rank-dot\s*\{[^}]*fill:\s*var\(--lesson-surface\)/);
  assert.match(css, /\.fl-rank-current\s*\{[^}]*fill:\s*currentColor/);
  assert.match(css, /\.fl-rank-current\s*\+\s*text\s*\{[^}]*fill:\s*var\(--lesson-surface\)/);
  assert.doesNotMatch(css, /\.fl-dim\s*\{[^}]*opacity/);
  assert.match(css, /\.fl-dot-grid \.fl-rank-panel\s*\{[^}]*order:\s*-1/);
});
