'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const lab = require('../transformer-labs.js');
const near = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
const vectorNear = (a, b, eps) => { assert.equal(a.length, b.length); a.forEach((x, i) => near(x, b[i], eps)); };

test('BPE counts weighted pairs, picks the real maximum and preserves all text', () => {
  const history = lab.trainBPE();
  assert.equal(history.length, 9);
  assert.deepEqual(history[1].applied, { pair: ['e', 's'], count: 9 });
  history.forEach((h, i) => {
    h.rows.forEach(row => assert.equal(row.tokens.join(''), row.word));
    assert.equal(lab.encodeBPE('lowest', h.merges).join(''), 'lowest');
    if (i) {
      assert.deepEqual(h.applied, history[i - 1].pairs[0]);
      const total = rows => rows.reduce((s, row) => s + row.tokens.length * row.count, 0);
      assert.ok(total(h.rows) < total(history[i - 1].rows));
    }
  });
  assert.deepEqual(lab.mergePair(['a', 'a', 'a'], ['a', 'a']), ['aa', 'a']);
  assert.deepEqual(lab.encodeBPE('lowest', history[3].merges), ['lo', 'w', 'est']);
});

test('dot vs cosine genuinely changes the initial candidate ordering', () => {
  const q = lab.embeddingTrace().query, docs = lab.DOCUMENTS;
  assert.ok(lab.dot(q, docs[1].vector) > lab.dot(q, docs[0].vector));
  assert.ok(lab.cosine(q, docs[1].vector) < lab.cosine(q, docs[0].vector));
  near(lab.cosine(q, docs[0].vector), 1);
  near(lab.cosine([0, 0], [1, 1]), 0);
});

test('contrastive gradient matches finite differences and actual steps reduce loss', () => {
  const q = [0.5, 0.5], analytic = lab.contrastive(q).gradient, eps = 1e-6;
  q.forEach((_, i) => {
    const plus = q.slice(), minus = q.slice(); plus[i] += eps; minus[i] -= eps;
    near(analytic[i], (lab.contrastive(plus).loss - lab.contrastive(minus).loss) / (2 * eps), 1e-7);
  });
  for (let i = 1; i <= 4; i++) assert.ok(lab.embeddingTrace(i).loss < lab.embeddingTrace(i - 1).loss);
});

test('softmax is stable, normalized, shift invariant and masks future keys', () => {
  const p = lab.softmax([1000, 1001, -Infinity]);
  near(p.reduce((s, x) => s + x, 0), 1);
  assert.equal(p[2], 0);
  vectorNear(p, lab.softmax([0, 1, -Infinity]));
  const a = lab.attend([1, 0], [[1, 0], [100, 0]], [[2, 3], [99, 99]], 0);
  vectorNear(a.weights, [1, 0]);
  vectorNear(a.output, [2, 3]);
  assert.throws(() => lab.softmax([-Infinity, -Infinity]), /at least one unmasked score/);
  assert.throws(() => lab.softmax([]), /at least one unmasked score/);
});

test('attention projections and weighted values match hand arithmetic', () => {
  const t = lab.attentionTrace();
  vectorNear(t.q[0], [1.5, 0.5]);
  vectorNear(t.k[0], [0.5, 1.5]);
  vectorNear(t.v[0], [1.5, -0.5]);
  near(t.rows[0].raw[0], 1.5 / Math.sqrt(2));
  vectorNear(t.rows[0].output, t.v[0]);
  t.rows.forEach((row, i) => {
    near(row.weights.reduce((s, x) => s + x, 0), 1);
    row.weights.forEach((w, j) => { if (j > i) assert.equal(w, 0); });
    row.output.forEach((v, j) => near(v, row.weights.reduce((s, w, k) => s + w * t.v[k][j], 0)));
  });
});

test('position-free full attention is equivariant for token identity, not identical across tokens', () => {
  const a = lab.positionalTrace('none'), b = lab.positionalTrace('none', true);
  vectorNear(a.output, b.output);
  vectorNear(a.weights, b.weights.slice().reverse());
  ['rope', 'absolute'].forEach(mode => {
    const x = lab.positionalTrace(mode), y = lab.positionalTrace(mode, true);
    assert.ok(x.output.some((v, i) => Math.abs(v - y.output[i]) > 0.01));
  });
});

test('RoPE preserves norms and its dot product depends on relative position', () => {
  const q = [1.3, -0.7], k = [0.2, 0.9];
  near(lab.norm(lab.rotate(q, 4)), lab.norm(q));
  near(lab.dot(lab.rotate(q, 2), lab.rotate(k, 5)), lab.dot(q, lab.rotate(k, 3)));
  near(lab.dot(lab.rotate(q, 2), lab.rotate(k, 5)), lab.dot(lab.rotate(q, 9), lab.rotate(k, 12)));
});

test('LayerNorm normalizes each row and handles a constant row', () => {
  vectorNear(lab.layerNorm([7, 7, 7]), [0, 0, 0]);
  const n = lab.layerNorm([1, 2, 3]);
  near(n.reduce((s, x) => s + x, 0), 0);
  near(lab.dot(n, n) / 3, (2 / 3) / (2 / 3 + 1e-5));
});

test('pre-norm block follows both residual equations and token-local MLP', () => {
  const t = lab.transformerBlock();
  t.inputs.forEach((x, i) => {
    vectorNear(t.n1[i], lab.layerNorm(x));
    vectorNear(t.residual[i], lab.add(x, t.projected[i]));
    vectorNear(t.n2[i], lab.layerNorm(t.residual[i]));
    vectorNear(t.feedforward[i].output, lab.mlp(t.n2[i]).output);
    vectorNear(t.output[i], lab.add(t.residual[i], t.feedforward[i].output));
  });
  const changed = t.inputs.map(x => x.slice()); changed[0] = [0, 1, 1];
  const next = lab.transformerBlock(changed);
  assert.ok(next.output[3].some((v, j) => Math.abs(v - t.output[3][j]) > 0.001));
  const future = t.inputs.map(x => x.slice()); future[3] = [10, 20, 30];
  vectorNear(lab.transformerBlock(future).output[0], t.output[0]);
  vectorNear(lab.mlp(t.n2[1]).output, lab.mlp(t.n2[1].slice()).output);
});

test('cache append is immutable, preserves prefix rows and equals full causal recomputation', () => {
  for (let step = 0; step <= 4; step++) {
    const c = lab.cacheTrace(step), full = lab.attentionTrace(lab.INPUTS.slice(0, step + 2));
    c.outputs.forEach((row, i) => vectorNear(row, full.rows[i].output));
    assert.equal(c.bytes, 16 * c.length);
    assert.equal(c.projectedRows, c.length);
    let without = 2; for (let n = 3; n <= c.length; n++) without += n;
    assert.equal(c.uncachedRows, without);
    if (step) {
      const prev = lab.cacheTrace(step - 1);
      assert.deepEqual(c.keys.slice(0, -1), prev.keys);
      assert.deepEqual(c.values.slice(0, -1), prev.values);
    }
  }
  const cache = { keys: [[1, 2]], values: [[3, 4]] }, before = JSON.stringify(cache);
  const next = lab.appendCache(cache, [1, 0, 1]);
  assert.equal(JSON.stringify(cache), before);
  assert.notEqual(next.keys, cache.keys);
  assert.equal(next.keys.length, 2);
});

test('RAG term relevance cannot manufacture claim support', () => {
  const results = lab.retrieve('cache memory eviction');
  assert.equal(results[0].id, 'A');
  results.forEach(row => near(row.score, row.dot / (row.qnorm * row.dnorm)));
  assert.deepEqual(lab.claimSupport(lab.CLAIMS[0], ['A']), []);
  assert.deepEqual(lab.claimSupport(lab.CLAIMS[0], ['A', 'B']), ['B']);
  assert.deepEqual(lab.claimSupport(lab.CLAIMS[2], ['A', 'B', 'C', 'D']), []);
  assert.equal(lab.retrieve('unknownword')[0].score, 0);
  assert.equal(lab.retrieve('active request pinned')[0].id, 'C');
  const policy = lab.EVIDENCE.find(doc => doc.id === 'B').text;
  assert.ok(policy.includes(lab.CLAIMS[0].text));
  assert.match(policy, /unpinned cache entry, if one exists/);
  assert.match(policy, /Entries pinned by active requests cannot be evicted/);
});

test('all seven sections provide full content, render safely and obey navigation bounds', () => {
  const kinds = Object.keys(lab.content);
  assert.equal(kinds.length, 7);
  kinds.forEach(kind => {
    const content = lab.content[kind];
    ['title', 'summary', 'what', 'why', 'interview', 'details', 'math', 'code', 'quiz'].forEach(key => assert.ok(content[key], `${kind}: ${key}`));
    assert.deepEqual(content.controls, []);
    assert.deepEqual(content.presets, []);
    assert.equal(content.geometry, null);
    assert.ok(content.math.annotations.length >= 3);
    assert.equal(content.quiz.options.filter(o => o.correct).length, 1);
    let state = lab.initial(kind);
    const before = JSON.stringify(state);
    assert.match(lab.render(kind, state), /tl-lab/);
    assert.doesNotMatch(lab.render(kind, state), /NaN|undefined/);
    lab.reduce(kind, state, 'reset');
    assert.equal(JSON.stringify(state), before);
    for (let i = 0; i < 12; i++) {
      state = lab.reduce(kind, state, 'next');
      assert.doesNotMatch(lab.render(kind, state), /NaN|undefined/);
    }
    for (let i = 0; i < 12; i++) state = lab.reduce(kind, state, 'previous');
    assert.ok(state.step >= 0);
    assert.deepEqual(lab.reduce(kind, state, 'reset'), lab.initial(kind));
  });
  assert.equal(lab.escape('<script>"&'), '&lt;script&gt;&quot;&amp;');
  assert.equal(lab.reduce('tokenization', lab.initial('tokenization'), 'word', '<img>').word, 'lowest');
});

test('browser registration respects the parent module contract without DOM work', () => {
  const vm = require('node:vm'), fs = require('node:fs');
  let registered;
  vm.runInNewContext(fs.readFileSync(require.resolve('../transformer-labs.js'), 'utf8'), { window: { AtelierLab: { createModule: m => { registered = m; } } } });
  assert.equal(registered.id, 'transformer-labs');
  assert.equal(Object.keys(registered.content).length, 7);
  assert.equal(typeof registered.render, 'function');
  assert.equal(typeof registered.reduce, 'function');
});

test('every bounded interaction renders finite inspectable state without mutating its input', () => {
  const cases = {
    tokenization: [['word', 'lower'], ['word', 'newest'], ['word', 'lowest']],
    embeddings: [['metric', 'cosine'], ['metric', 'dot'], ['doc', 0], ['doc', 1], ['doc', 2]],
    positional: [['mode', 'absolute'], ['swap', ''], ['key', 0], ['key', 1], ['mode', 'rope'], ['key', 2], ['swap', '']],
    'transformer-block': [['inspect', ''], ...Array.from({ length: 8 }, (_, i) => ['stage', i]), ['change', ''], ...Array.from({ length: 4 }, (_, i) => ['token', i]), ['inspect', '']],
    attention: [...Array.from({ length: 4 }, (_, i) => ['query', i]), ...Array.from({ length: 16 }, (_, i) => ['cell', `${Math.floor(i / 4)},${i % 4}`]), ['key', 999], ['query', -3]],
    'kv-cache': [['next', ''], ['next', ''], ['next', ''], ['cell', '5,5'], ['key', 0], ['previous', ''], ['cell', '99,99']],
    rag: [['topk', 1], ['claim', 0], ['topk', 2], ['exclude', 'B'], ['exclude', 'B'], ['doc', 'B'], ['query', 'active request pinned'], ['topk', 3], ['claim', 1], ['claim', 2]],
  };
  for (const [kind, actions] of Object.entries(cases)) {
    let state = lab.initial(kind);
    for (const [action, value] of actions) {
      const before = JSON.stringify(state), next = lab.reduce(kind, state, action, value);
      assert.equal(JSON.stringify(state), before, `${kind} / ${action} mutated state`);
      state = next;
      const html = lab.render(kind, state);
      assert.doesNotMatch(html, /NaN|undefined/);
      assert.equal((html.match(/<section\b/g) || []).length, (html.match(/<\/section>/g) || []).length);
      assert.equal((html.match(/<svg\b/g) || []).length, (html.match(/<\/svg>/g) || []).length);
    }
  }
});

test('block degenerate input stays finite and known full trace has the expected final output', () => {
  const zero = lab.transformerBlock([[0, 0, 0], [0, 0, 0]]);
  zero.output.forEach(row => vectorNear(row, [0, 0, 0]));
  // Regression fixture includes normalization, causal mixing, W_O, both residuals and ReLU.
  vectorNear(lab.transformerBlock().output[3], [0.304292523673956, 0.04038370830504595, 1.8345779336062333]);
  const t = lab.transformerBlock();
  assert.ok(t.feedforward.some(row => row.hidden.some(v => v < 0)));
  t.feedforward.forEach(row => row.activated.forEach((v, i) => near(v, Math.max(0, row.hidden[i]))));
});

test('RAG removing the supporting chunk changes support, not the underlying retrieval score', () => {
  const state = lab.initial('rag'), before = lab.retrieve(state.query);
  const without = lab.reduce('rag', state, 'exclude', 'B');
  assert.deepEqual(state.excluded, []);
  assert.deepEqual(without.excluded, ['B']);
  assert.deepEqual(lab.retrieve(without.query), before);
  const claim0 = lab.reduce('rag', without, 'claim', 0);
  assert.match(lab.render('rag', claim0), /evidence-access failure/);
  const restored = lab.reduce('rag', claim0, 'exclude', 'B');
  assert.match(lab.render('rag', restored), /supports this claim: B/);
});

test('block architecture has two residual bypasses, all stages, causal edges and local MLP lanes', () => {
  const initial = lab.initial('transformer-block');
  const html = lab.render('transformer-block', initial);
  assert.equal((html.match(/class="tl-bypass"/g) || []).length, 2);
  for (let i = 0; i < 8; i++) assert.match(html, new RegExp(`data-action="stage" data-value="${i}"`));
  assert.match(html, /carry X unchanged/);
  assert.match(html, /carry H unchanged/);
  assert.match(html, /<details class="tl-details tl-calculation"><summary data-action="inspect"/);
  const weights = [...html.matchAll(/data-query="(\d+)" data-key="(\d+)" data-weight="([^"]+)"/g)];
  assert.equal(weights.length, 10);
  const trace = lab.transformerBlock();
  weights.forEach(([, query, key, weight]) => {
    assert.ok(Number(key) <= Number(query));
    near(Number(weight), trace.attention.rows[Number(query)].weights[Number(key)]);
  });
  assert.equal((html.match(/data-local-token="[0-3]"/g) || []).length, 4);
  const inspected = lab.reduce('transformer-block', initial, 'inspect', '');
  assert.match(lab.render('transformer-block', inspected), /<details class="tl-details tl-calculation" open>/);
  assert.equal(lab.reduce('transformer-block', inspected, 'stage', 6).inspect, true);
});

test('position arrows use exact transformed query/key coordinates in all modes and permutations', () => {
  for (const mode of ['none', 'absolute', 'rope']) for (const swapped of [false, true]) for (const key of [0, 1, 2]) {
    const html = lab.render('positional', { mode, swapped, key });
    const trace = lab.positionalTrace(mode, swapped);
    const vectors = [...html.matchAll(/data-vector="([^\"]+)"/g)].map(m => JSON.parse(m[1]));
    assert.equal(vectors.length, 2);
    vectorNear(vectors[0], trace.query.encoded);
    vectorNear(vectors[1], trace.rows[key].encoded);
    if (mode === 'rope') {
      const angles = [...html.matchAll(/data-angle="([^\"]+)"/g)].map(m => Number(m[1]));
      assert.deepEqual(angles, [trace.query.position, trace.rows[key].position].filter(x => x > 0));
    }
    assert.doesNotMatch(html, /NaN|undefined/);
  }
});

test('every default exposes a mechanism and primary action while keeping tables in native disclosures', () => {
  for (const kind of Object.keys(lab.content)) {
    const html = lab.render(kind, lab.initial(kind));
    assert.match(html, /Inspect the calculation<\/summary>/, kind);
    assert.match(html, /tl-primary|data-action="next"/, kind);
    assert.match(html, /tl-mechanism|<svg|tl-rag-flow/, kind);
    let depth = 0;
    for (const match of html.matchAll(/<details\b[^>]*>|<\/details>|<table\b/g)) {
      if (match[0].startsWith('<details')) depth++;
      else if (match[0] === '</details>') depth--;
      else assert.ok(depth > 0, `${kind}: table must be secondary inside details`);
    }
    assert.equal(depth, 0, `${kind}: disclosures must be balanced`);
  }
});

test('swapping positions tracks the selected key identity as both vectors rotate', () => {
  const before = lab.initial('positional');
  const after = lab.reduce('positional', before, 'swap', '');
  assert.equal(lab.positionalTrace(before.mode, before.swapped).rows[before.key].token,
    lab.positionalTrace(after.mode, after.swapped).rows[after.key].token);
  assert.deepEqual(lab.reduce('positional', after, 'swap', ''), before);
});

test('native calculation and settings disclosures persist while inspecting a mechanism', () => {
  for (const kind of Object.keys(lab.content)) {
    let state = lab.initial(kind);
    assert.doesNotMatch(lab.render(kind, state), /class="tl-details tl-calculation" open/);
    state = lab.reduce(kind, state, 'inspect', '');
    state = lab.reduce(kind, state, 'options', '');
    state = lab.reduce(kind, state, 'next', '');
    assert.match(lab.render(kind, state), /class="tl-details tl-calculation" open/);
    assert.match(lab.render(kind, state), /class="tl-details tl-options" open/);
  }
});
