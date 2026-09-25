'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const lab = require('../training-labs.js');
const near = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) <= eps, `${a} != ${b}`);
const vectorNear = (a, b, eps) => { assert.equal(a.length, b.length); a.forEach((v, i) => near(v, b[i], eps)); };
const sum = xs => xs.reduce((a, b) => a + b, 0);
const matrixNear = (a, b, eps) => { assert.equal(a.length, b.length); a.forEach((r, i) => vectorNear(r, b[i], eps)); };

test('quadratic updates use the exact Hessian and correct stability threshold', () => {
  const trace = lab.quadraticTrace(.22);
  vectorNear(trace[1].x, [1.56, -.76]);
  near(trace[0].loss, 6);
  for (let i = 1; i < trace.length; i++) {
    vectorNear(trace[i].x, trace[i - 1].x.map((v, j) => v - .22 * trace[i - 1].grad[j]));
    assert.ok(trace[i].loss < trace[i - 1].loss);
    assert.ok(trace[i].x[1] * trace[i - 1].x[1] < 0);
  }
  assert.ok(lab.quadraticTrace(.28).at(-1).loss > 6);
  near(Math.abs(lab.quadraticTrace(.25).at(-1).x[1]), 1);
});

test('initialization trace is deterministic and every activation comes from W h', () => {
  assert.deepEqual(lab.initializationTrace(), lab.initializationTrace());
  for (const mode of ['small', 'xavier', 'he', 'large']) {
    const rows = lab.initializationTrace(mode);
    rows.forEach((r, i) => {
      vectorNear(r.z, lab.mv(r.weights, r.input));
      vectorNear(r.output, r.z.map(v => Math.max(0, v)));
      near(r.rms, Math.sqrt(sum(r.output.map(v => v ** 2)) / 8));
      assert.equal(r.dead, r.output.filter(v => v === 0).length);
      if (i) assert.deepEqual(r.input, rows[i - 1].output);
    });
  }
});

test('initialization choices preserve the random draw and differ only by scale', () => {
  const he = lab.initializationTrace('he'), xavier = lab.initializationTrace('xavier');
  he.forEach((r, i) => {
    matrixNear(r.weights, xavier[i].weights.map(row => row.map(v => v * Math.sqrt(2))));
    near(r.rms, xavier[i].rms * Math.sqrt(2) ** (i + 1));
  });
  assert.ok(lab.initializationTrace('small').at(-1).rms < .001);
  assert.ok(lab.initializationTrace('large').at(-1).rms > 5);
  assert.ok(he.at(-1).rms > .5 && he.at(-1).rms < 2);
});

function chainObjective(input, rows, mode) {
  let x = input;
  for (const row of rows) x = lab.mv(row.weights, x).map(v => mode === 'tanh' ? Math.tanh(v) : mode === 'relu' ? Math.max(0, v) : v);
  return sum(x);
}

test('gradient Jacobian product agrees with finite differences for every activation and gain', () => {
  for (const mode of ['tanh', 'relu', 'linear']) for (const gain of [.7, 1.3]) {
    const rows = lab.gradientTrace(mode, gain), x = rows[0].input, eps = 1e-6;
    for (let j = 0; j < 2; j++) {
      const plus = x.slice(), minus = x.slice(); plus[j] += eps; minus[j] -= eps;
      near(rows[0].inGradient[j], (chainObjective(plus, rows, mode) - chainObjective(minus, rows, mode)) / (2 * eps), 1e-7);
    }
    rows.forEach((r, i) => {
      vectorNear(r.inGradient, lab.mv(lab.transpose(r.jacobian), r.outGradient));
      if (i < 5) assert.deepEqual(r.outGradient, rows[i + 1].inGradient);
    });
  }
});

test('dead ReLU rows eliminate derivative contributions and tanh saturation shrinks gradients', () => {
  const relu = lab.gradientTrace('relu');
  assert.equal(relu[0].derivative[1], 0);
  vectorNear(relu[0].jacobian[1], [0, 0]);
  const tanh = lab.gradientTrace('tanh'), linear = lab.gradientTrace('linear');
  assert.ok(Math.abs(tanh[0].inGradient[0]) < Math.abs(linear[0].inGradient[0]));
});

test('activation comparisons hold both the input and weights fixed', () => {
  const tanh = lab.gradientTrace('tanh');
  for (const mode of ['relu', 'linear']) {
    const rows = lab.gradientTrace(mode);
    assert.deepEqual(rows[0].input, tanh[0].input);
    assert.deepEqual(rows[0].weights, tanh[0].weights);
  }
});

test('normalization computes exact statistics on the selected axis', () => {
  const x = [[1, 2, 6], [2, 4, 8], [3, 6, 10]];
  const b = lab.normalizeMatrix(x, 'batch'), l = lab.normalizeMatrix(x, 'layer'), r = lab.normalizeMatrix(x, 'rms');
  near(b[0][0].mean, 2); near(b[0][0].variance, 2 / 3);
  near(b[0][0].value, -1 / Math.sqrt(2 / 3 + 1e-5));
  near(l[0][0].mean, 3); near(l[0][0].variance, 14 / 3);
  near(r[0][0].mean, 0); near(r[0][0].variance, 41 / 3);
  near(r[0][0].value, 1 / Math.sqrt(41 / 3 + 1e-5));
});

test('only BatchNorm couples examples; LayerNorm is row-shift invariant', () => {
  const x = [[1, 2, 6], [2, 4, 8], [3, 6, 10]], shifted = x.map((r, i) => r.map(v => v + (i === 2 ? 8 : 0)));
  for (const mode of ['batch', 'layer', 'rms']) {
    const a = lab.normalizeMatrix(x, mode), b = lab.normalizeMatrix(shifted, mode);
    if (mode === 'batch') assert.notEqual(a[0][0].value, b[0][0].value);
    else vectorNear(a[0].map(c => c.value), b[0].map(c => c.value));
    if (mode === 'layer') vectorNear(a[2].map(c => c.value), b[2].map(c => c.value));
    if (mode === 'rms') assert.notEqual(a[2][0].value, b[2][0].value);
  }
  vectorNear(lab.normalizeMatrix([[7, 7, 7]], 'layer')[0].map(c => c.value), [0, 0, 0]);
  vectorNear(lab.normalizeMatrix([[0, 0, 0]], 'rms')[0].map(c => c.value), [0, 0, 0]);
});

test('residual forward outputs and Jacobian powers agree', () => {
  for (const mode of ['small', 'positive', 'cancel']) {
    const d = lab.residualTrace(mode);
    d.rows.forEach((r, i) => {
      vectorNear(r.plainX, lab.mv(r.plainJ, [1, -1]));
      vectorNear(r.residualX, lab.mv(r.residualJ, [1, -1]));
      if (i) {
        matrixNear(r.plainJ, lab.mm(d.branch, d.rows[i - 1].plainJ));
        matrixNear(r.residualJ, lab.mm(d.residual, d.rows[i - 1].residualJ));
      }
    });
  }
});

test('residual connections help near-zero branches but can cancel or amplify', () => {
  const small = lab.residualTrace('small').rows.at(-1), positive = lab.residualTrace('positive').rows.at(-1);
  assert.ok(Math.abs(small.residualJ[0][0]) > Math.abs(small.plainJ[0][0]) * 1000);
  assert.ok(positive.residualJ[0][0] > 1);
  for (const row of lab.residualTrace('cancel').rows.slice(1)) {
    matrixNear(row.residualJ, [[0, 0], [0, 0]]);
    vectorNear(row.residualX, [0, 0]);
  }
});

test('online attention equals dense attention for each query and at each prefix', () => {
  const values = [[1, 0], [0, 2], [3, 1], [-1, 2]];
  for (const query of [0, 1]) {
    const a = lab.attentionExample(query);
    vectorNear(a.output, a.dense, 1e-14);
    near(sum(a.probabilities), 1);
    a.trace.forEach(t => {
      const end = t.start + t.scores.length, p = lab.softmax(a.scores.slice(0, end));
      vectorNear(t.output, [0, 1].map(j => sum(p.map((v, i) => v * values[i][j]))));
      near(t.l, t.alpha * t.oldL + sum(t.weights));
      vectorNear(t.numerator, t.oldNumerator.map((v, j) => t.alpha * v + sum(t.weights.map((p, i) => p * values[t.start + i][j]))));
    });
    assert.equal(a.trace[0].alpha, 0);
    assert.ok(a.trace[1].alpha > 0 && a.trace[1].alpha < 1);
  }
});

test('online attention is numerically stable and independent of tile partition', () => {
  const scores = [1000, 999, 1002, 998, 1003], values = [[1, 2], [3, 4], [-2, 5], [7, 0], [1, -4]];
  for (const size of [1, 2, 3, 5]) {
    const d = lab.onlineAttention(scores, values, size);
    vectorNear(d.output, d.dense, 1e-13);
    vectorNear(d.output, lab.onlineAttention(scores.map(v => v - 1000), values, size).output, 1e-13);
    assert.ok(d.output.every(Number.isFinite));
  }
});

test('pretraining and SFT loss masks retain exact targets and denominators', () => {
  const pretrain = lab.tokenLoss('pretrain'), sft = lab.tokenLoss('sft');
  assert.equal(pretrain.count, 9); assert.equal(sft.count, 4);
  assert.deepEqual(sft.rows.map(r => r.mask), [0, 0, 0, 0, 0, 1, 1, 1, 1]);
  near(sft.loss, -Math.log(.6 * .7 * .2 * .8) / 4);
  near(pretrain.loss, sum(pretrain.rows.map(r => -Math.log(r.p))) / 9);
  assert.deepEqual(pretrain.rows.map(r => r.token), sft.rows.map(r => r.token));
  assert.equal(sft.rows[4].token, '<assistant>');
  assert.equal(sft.rows[8].mask, 1);
});

function peftObjective(matrices, adapter) {
  const x = [1, .5, -1, 2];
  const h = lab.mv(matrices.W, x).map((v, i) => v + matrices.b[0][i]);
  const delta = adapter ? lab.mv(matrices.B, lab.mv(matrices.A, x)) : [0, 0, 0, 0];
  const y = lab.mv(matrices.H, h.map((v, i) => v + delta[i])).map((v, i) => v + matrices.c[0][i]);
  return ((y[0] - 1) ** 2 + y[1] ** 2) / 2;
}

test('PEFT partial derivatives match finite differences including frozen parameters', () => {
  const d = lab.peftStep('adapter'), eps = 1e-6;
  for (const key of d.active) d.matrices[key].forEach((row, i) => row.forEach((_, j) => {
    const plus = structuredClone(d.matrices), minus = structuredClone(d.matrices);
    plus[key][i][j] += eps; minus[key][i][j] -= eps;
    near(d.gradients[key][i][j], (peftObjective(plus, true) - peftObjective(minus, true)) / (2 * eps), 1e-8);
  }));
});

test('PEFT respects matrix masks and decreases its actual one-example loss', () => {
  for (const [mode, count] of [['full', 30], ['head', 10], ['bias', 6], ['adapter', 8]]) {
    const d = lab.peftStep(mode, true);
    assert.equal(d.count, count); assert.ok(d.nextLoss < d.loss);
    near(d.nextLoss, peftObjective(d.updated, mode === 'adapter'));
    for (const key of d.active) d.matrices[key].forEach((row, i) => row.forEach((v, j) => {
      near(d.updated[key][i][j], v - .1 * +d.trainable(key) * d.gradients[key][i][j]);
    }));
  }
  const a = lab.peftStep('adapter', true);
  assert.deepEqual(a.matrices.A, a.updated.A);
  assert.notDeepEqual(a.matrices.B, a.updated.B);
  assert.deepEqual(a.matrices.W, a.updated.W);
  assert.deepEqual(a.matrices.H, a.updated.H);
});

test('LoRA BA products, merged paths, parameter counts and zero initialization are exact', () => {
  for (const rank of [1, 2]) for (const zero of [false, true]) {
    const d = lab.loraExample(rank, zero);
    matrixNear(d.delta, lab.mm(d.B, d.A));
    vectorNear(d.output, d.mergedOutput);
    vectorNear(d.output, lab.mv(d.merged, d.x));
    assert.equal(d.parameters, rank * 7);
    if (zero) { matrixNear(d.merged, d.W); vectorNear(d.correction, [0, 0, 0, 0]); }
  }
  near(lab.loraExample(1).merged[0][0], 1.2);
  near(lab.loraExample(2).merged[0][1], -.3);
  assert.ok(lab.loraExample(2).parameters > 12);
});

test('rank-one LoRA correction has zero 2x2 minors', () => {
  const d = lab.loraExample(1).delta;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 2; j++) near(d[i][j] * d[i + 1][j + 1] - d[i][j + 1] * d[i + 1][j], 0);
});

test('quantization uses a discrete bounded codebook and reports real reconstruction errors', () => {
  for (const bits of [2, 4, 8]) for (const limit of [1, 5]) {
    const d = lab.quantize(undefined, bits, limit);
    near(d.scale, limit / (2 ** (bits - 1) - 1));
    for (const r of d.rows) {
      assert.ok(Number.isInteger(r.q)); assert.ok(Math.abs(r.q) <= d.qmax);
      near(r.reconstructed, r.q * d.scale);
      near(r.error, r.reconstructed - r.x);
    }
    near(d.mse, sum(d.rows.map(r => r.error ** 2)) / 8);
    assert.equal(d.packedBytes, bits); assert.equal(d.scaleBytes, 4);
  }
  const kept = lab.quantize(), clipped = lab.quantize(undefined, 4, 1);
  near(clipped.rows.at(-1).error, -4);
  assert.ok(clipped.bulkMse < kept.bulkMse);
  assert.ok(clipped.mse > kept.mse);
});

test('quantizer tie convention matches the annotated Python floor(x + 0.5) expression', () => {
  const d = lab.quantize([-.5, .5, -1.5, 1.5], 4, 7);
  d.rows.forEach(r => assert.equal(r.q || 0, Math.floor(r.x + .5) || 0));
});

test('distillation probabilities and T-squared gradient agree with finite differences', () => {
  const z = [.2, -.3, .7], eps = 1e-6;
  for (const temperature of [1, 2, 4]) for (const objective of ['soft', 'hard']) {
    const d = lab.distillation(z, temperature, objective);
    near(sum(d.teacher), 1); near(sum(d.student), 1); near(sum(d.gradient), 0);
    near(d.loss, d.kl * temperature ** 2);
    z.forEach((_, i) => {
      const plus = z.slice(), minus = z.slice(); plus[i] += eps; minus[i] -= eps;
      near(d.gradient[i], (lab.distillation(plus, temperature, objective).loss - lab.distillation(minus, temperature, objective).loss) / (2 * eps), 1e-8);
    });
  }
});

test('distillation updates lower the exact objective; soft targets preserve the runner-up signal', () => {
  const soft = lab.studentTrace(2, 'soft'), hard = lab.studentTrace(2, 'hard');
  assert.ok(soft.gradient[1] < 0); assert.ok(hard.gradient[1] > 0);
  assert.ok(soft.terms.some(v => v < 0)); assert.ok(soft.kl >= 0);
  for (const t of [1, 2, 4]) for (const mode of ['soft', 'hard']) for (let step = 1; step <= 20; step++) {
    assert.ok(lab.studentTrace(t, mode, step).loss < lab.studentTrace(t, mode, step - 1).loss);
  }
  near(lab.distillation([3, 2, -1], 2).kl, 0);
});

test('distillation text reflects temperature-dependent direction and zero-target KL convention', () => {
  for (const temperature of [1, 2, 4]) {
    const state = { ...lab.initial('distillation'), temperature };
    const d = lab.studentTrace(temperature, 'soft');
    const direction = d.gradient[1] < 0 ? 'raises' : 'lowers';
    assert.ok(lab.render('distillation', state).includes(`therefore ${direction} B&#39;s logit`));
    const hard = lab.render('distillation', { ...state, objective: 'hard', cell: 1 });
    assert.ok(hard.includes('zero-target limit convention'));
    assert.ok(!hard.includes('0 * ln(0/'));
  }
});

test('PEFT inspector identifies whether the head is actually frozen in the selected mode', () => {
  for (const mode of ['full', 'head', 'bias', 'adapter']) {
    const state = lab.reduce('peft', lab.initial('peft'), 'mode', mode);
    const label = mode === 'full' || mode === 'head' ? 'trainable' : 'frozen';
    assert.ok(lab.render('peft', state).includes(`The ${label} head sends initial dh`));
  }
});

test('serving event times follow the stated deterministic cost model', () => {
  const d = lab.servingTimeline('batch');
  near(d.finish, 33.5); near(d.throughput, 9000 / 33.5);
  assert.deepEqual(d.results[0].emitted, [24.5, 28, 31, 33.5]);
  assert.deepEqual(d.results.map(r => r.ttft), [24.5, 20.5, 15.5]);
  assert.deepEqual(d.results.map(r => r.wait), [10, 6, 1]);
  assert.equal(d.peakKVTokens, 21); assert.equal(d.kvBytes, 672);
  d.events.filter(e => e.type === 'decode').forEach(e => near(e.end - e.start, 2 + .5 * e.names.split(',').length));
  assert.equal(d.results[1].emitted.length, 2);
  assert.ok(!d.events.at(-1).names.includes('B'));
});

test('KV payload counts processed tokens rather than the token just sampled', () => {
  for (const policy of ['serial', 'batch']) for (const longContext of [false, true]) {
    const d = lab.servingTimeline(policy, longContext);
    for (const event of d.events.filter(e => e.type === 'decode')) {
      const active = d.results.filter(r => event.names.split(',').includes(r.name));
      const expected = sum(active.map(r => r.prompt + event.step));
      assert.equal(event.cachedTokens, expected);
      assert.equal(event.cachedTokens, sum(active.map(r => r.prompt + r.emitted.filter(t => t <= event.end).length - 1)));
    }
    assert.equal(d.peakKVTokens, Math.max(...d.events.filter(e => e.type === 'decode').map(e => e.cachedTokens)));
  }
  assert.equal(lab.servingTimeline('serial').kvBytes, 352);
});

test('batching trades first-request latency for throughput and larger active KV storage', () => {
  const serial = lab.servingTimeline('serial'), batch = lab.servingTimeline('batch');
  near(serial.finish, 37.5); near(serial.throughput, 240);
  assert.ok(batch.throughput > serial.throughput);
  assert.ok(batch.results[0].ttft > serial.results[0].ttft);
  assert.ok(batch.results[2].ttft < serial.results[2].ttft);
  assert.ok(batch.kvBytes > serial.kvBytes);
  for (const mode of ['serial', 'batch']) {
    const longer = lab.servingTimeline(mode, true), short = lab.servingTimeline(mode);
    assert.ok(longer.finish > short.finish); assert.ok(longer.kvBytes > short.kvBytes);
    longer.results.forEach(r => {
      assert.ok(r.start >= r.arrival);
      assert.ok(r.emitted[0] > r.prefillEnd);
      near(r.latency, r.emitted.at(-1) - r.arrival);
    });
  }
});

test('all 12 content overrides include unique mechanism text, annotated math, code, and a quiz', () => {
  assert.equal(Object.keys(lab.content).length, 12);
  for (const [kind, c] of Object.entries(lab.content)) {
    for (const field of ['title', 'summary', 'what', 'why', 'interview']) assert.ok(c[field].length > 20, `${kind}: ${field}`);
    assert.notEqual(c.summary, c.what);
    assert.ok(c.math.formula.length && c.math.annotations.length);
    c.math.annotations.forEach(a => assert.equal(a.length, 3));
    assert.equal(c.code.lang, 'python'); assert.ok(c.code.snippet.includes('\n'));
    assert.equal(c.quiz.options.filter(o => o.correct).length, 1);
    assert.ok(c.quiz.options.every(o => o.explanation.length > 10));
    assert.deepEqual(c.controls, []); assert.deepEqual(c.presets, []); assert.equal(c.geometry, null);
  }
});

test('browser registration uses the shared core contract without any global access in Node', () => {
  let spec;
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../training-labs.js'), 'utf8'), { window: { AtelierLab: { createModule: s => { spec = s; } } } });
  assert.equal(spec.id, 'training-labs');
  assert.equal(Object.keys(spec.content).length, 12);
  assert.equal(typeof spec.initial, 'function'); assert.equal(typeof spec.render, 'function'); assert.equal(typeof spec.reduce, 'function');
  assert.equal(spec.render('normalization', spec.initial('normalization')), lab.render('normalization', lab.initial('normalization')));
});

test('escaping covers all HTML attribute delimiters and token tags are escaped', () => {
  assert.equal(lab.escape('<x a="b">&\''), '&lt;x a=&quot;b&quot;&gt;&amp;&#39;');
  const html = lab.render('pretrain-finetune', lab.initial('pretrain-finetune'));
  assert.ok(html.includes('&lt;assistant&gt;'));
  assert.ok(!html.includes('<assistant>'));
});

for (const kind of Object.keys(lab.content)) {
  test(`${kind}: every reachable button state renders finite computed values and reset restores default`, () => {
    const queue = [lab.initial(kind)], seen = new Set();
    while (queue.length) {
      const state = queue.shift(), signature = JSON.stringify(state);
      if (seen.has(signature)) continue;
      seen.add(signature);
      assert.ok(seen.size < 2000, 'Unexpected unbounded state space');
      Object.freeze(state);
      const html = lab.render(kind, state);
      assert.ok(html.includes('ml-inspector'), 'Default and alternate states must provide inspection');
      assert.ok(!/NaN|undefined|\bInfinity\b/.test(html));
      assert.ok(!html.includes('<text'), 'Labels stay in HTML, not tiny scaled SVG text');
      assert.ok(!/<(?:script|iframe)/i.test(html));
      assert.deepEqual(lab.reduce(kind, state, 'reset', ''), lab.initial(kind));
      for (const match of html.matchAll(/<button\b([^>]+)>/g)) {
        const attrs = match[1];
        if (/\bdisabled\b/.test(attrs)) continue;
        const action = attrs.match(/data-action="([^"]*)"/)?.[1], value = attrs.match(/data-value="([^"]*)"/)?.[1];
        assert.ok(action, 'Native buttons must use the shared action dispatcher');
        const next = lab.reduce(kind, state, action, value);
        assert.equal(JSON.stringify(state), signature, 'Reducer must not mutate previous state');
        if (!seen.has(JSON.stringify(next))) queue.push(next);
      }
    }
    assert.ok(seen.size > 4, `${kind} should support substantive inspection`);
  });
}

test('numeric selections clamp to valid visible ranges and choice actions reject unexpected input', () => {
  assert.equal(lab.reduce('lora', lab.initial('lora'), 'cell', '999').cell, 11);
  assert.equal(lab.reduce('initialization', lab.initial('initialization'), 'layer', '-20').layer, 0);
  assert.deepEqual(lab.reduce('normalization', lab.initial('normalization'), 'mode', '<script>'), lab.initial('normalization'));
  assert.deepEqual(lab.reduce('lora', lab.initial('lora'), 'cell', 'NaN'), lab.initial('lora'));
});

test('responsive CSS uses scoped flexible grids, 44px controls and no fixed content width', () => {
  const css = fs.readFileSync(path.join(__dirname, '../training-labs.css'), 'utf8');
  assert.ok(css.includes('max-width: 680px'));
  assert.ok(css.includes('minmax(0, 1fr)'));
  assert.ok(css.includes('min-height: 44px'));
  assert.ok(css.includes(':focus-visible'));
  assert.ok(!/\bmin-width:\s*[3-9]\d\dpx/.test(css));
});

test('every first view has one question, one working primary action, and no exposed derivation table', () => {
  for (const kind of Object.keys(lab.content)) {
    const state = lab.initial(kind), html = lab.render(kind, state);
    const visible = html.replace(/<details\b[\s\S]*?<\/details>/g, '');
    assert.equal((visible.match(/class="ml-question"/g) || []).length, 1, kind);
    assert.equal((visible.match(/class="tl-primary"/g) || []).length, 1, kind);
    assert.ok(visible.includes('class="tl-mechanism"'), kind);
    assert.ok(visible.includes('class="tl-outcome"'), kind);
    assert.ok(!visible.includes('<table'), `${kind}: calculation tables must be secondary`);
    assert.ok(!visible.includes('ml-inspector'), `${kind}: full derivations must be secondary`);
    assert.ok(html.includes('<summary>Inspect the calculation</summary>'), kind);
    assert.ok(html.includes('<summary>Change settings</summary>'), kind);
    assert.ok(!/<details[^>]*\sopen\b/.test(html), `${kind}: disclosures start closed`);
    const primary = visible.match(/<button[^>]*class="tl-primary"[^>]*>/)[0];
    const action = primary.match(/data-action="([^"]*)"/)[1], value = primary.match(/data-value="([^"]*)"/)[1];
    const next = lab.reduce(kind, state, action, value);
    assert.notDeepEqual(next, state, `${kind}: primary action changes the experiment`);
    const nextVisible = lab.render(kind, next).replace(/<details\b[\s\S]*?<\/details>/g, '');
    const mechanism = markup => markup.slice(markup.indexOf('<div class="tl-mechanism">'), markup.indexOf('<p class="tl-outcome"'));
    assert.notEqual(mechanism(nextVisible), mechanism(visible), `${kind}: action changes the visible mechanism, not merely the button label`);
  }
});

test('plotted positions remain directly keyboard and pointer inspectable', () => {
  for (const [kind, action] of [['learning-rate', 'step'], ['initialization', 'layer'], ['gradient-flow', 'layer']]) {
    const html = lab.render(kind, lab.initial(kind));
    const nodes = [...html.matchAll(/<g role="button" tabindex="0" data-action="([^"]+)" data-value="(\d+)"/g)];
    assert.ok(nodes.length >= 6, kind);
    for (const node of nodes) {
      assert.equal(node[1], action);
      const state = lab.reduce(kind, lab.initial(kind), node[1], node[2]);
      assert.equal(state[action], Number(node[2]));
      assert.ok(lab.render(kind, state).includes('tl-outcome'));
    }
  }
});
