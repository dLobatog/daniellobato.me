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
  const mlpState = lab.reduce('transformer-block', initial, 'stage', 6);
  assert.equal((lab.render('transformer-block', mlpState).match(/data-local-token="[0-3]"/g) || []).length, 4);
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

test('BPE causal annotations reconcile weighted occurrences with actual token savings', () => {
  const first = lab.bpeEffect(1);
  assert.deepEqual([first.before, first.after, first.count], [79, 70, 9]);
  assert.deepEqual(first.occurrences, [{ word: 'newest', frequency: 6, matches: 1 }, { word: 'widest', frequency: 3, matches: 1 }]);
  for (let step = 1; step <= 8; step++) {
    const effect = lab.bpeEffect(step);
    assert.equal(effect.before - effect.after, effect.count);
    assert.ok(effect.count >= effect.runner.count);
  }
  const ranks = lab.trainBPE()[5].merges;
  const frozen = JSON.stringify(ranks);
  for (let request = 0; request < 20; request++) assert.deepEqual(lab.encodeBPE('lowest', ranks), ['low', 'est']);
  assert.equal(JSON.stringify(ranks), frozen, 'Repeated inference must not retrain or mutate the merge ranks');
});

test('one embedding update improves the labeled margin without claiming Euclidean attraction', () => {
  const before = lab.embeddingTrace(0), after = lab.embeddingTrace(1);
  vectorNear(after.query, lab.add(before.query, lab.scale(before.gradient, -0.2)));
  near(before.scores[0] - before.scores[1], -0.5);
  near(after.scores[0] - after.scores[1], 0.07409699296769467);
  assert.ok(after.probabilities[0] > before.probabilities[0]);
  const doc = lab.DOCUMENTS[1].vector;
  near(lab.dot(before.query, lab.scale(doc, 2)), 2 * lab.dot(before.query, doc));
  near(lab.cosine(before.query, lab.scale(doc, 2)), lab.cosine(before.query, doc));
});

test('value intervention changes exactly one payload while all QK weights remain identical', () => {
  const pristine = JSON.stringify(lab.attentionTrace());
  for (let q = 0; q < 4; q++) for (let k = 0; k < 4; k++) {
    const exp = lab.attentionValueExperiment(q, k, true);
    assert.deepEqual(exp.after.q, exp.before.q);
    assert.deepEqual(exp.after.k, exp.before.k);
    exp.after.rows.forEach((row, i) => assert.deepEqual(row.weights, exp.before.rows[i].weights));
    exp.after.v.forEach((v, i) => vectorNear(v, i === k ? [0, 0] : exp.before.v[i]));
    vectorNear(exp.delta, lab.scale(exp.before.v[k], -exp.before.rows[q].weights[k]));
    if (k > q) vectorNear(exp.delta, [0, 0]);
  }
  assert.equal(JSON.stringify(lab.attentionTrace()), pristine);
});

test('single-coordinate token substitution first reaches later tokens at attention', () => {
  const { before, after } = lab.blockExperiment(true);
  assert.equal(before.inputs.flatMap((row, i) => row.map((v, j) => v !== after.inputs[i][j])).filter(Boolean).length, 1);
  const a = lab.blockStages(before), b = lab.blockStages(after);
  for (let stage = 0; stage < 3; stage++) vectorNear(a[stage][3], b[stage][3]);
  near(a[3][3][0], 0.28028319363206267);
  near(b[3][3][0], -0.36104629453510206);
  for (let stage = 3; stage < 8; stage++) assert.ok(Math.abs(a[stage][3][0] - b[stage][3][0]) > 0.001);
  // Counterfactual: with the attention branch absent, shared MLP weights cannot mix rows.
  const withoutAttention = rows => rows.map(x => lab.add(x, lab.mlp(lab.layerNorm(x)).output));
  vectorNear(withoutAttention(before.inputs)[3], withoutAttention(after.inputs)[3]);
});

test('append action focuses the new KV row and retains all prior pairs exactly', () => {
  const initial = lab.initial('kv-cache'), next = lab.reduce('kv-cache', initial, 'next', '');
  const a = lab.cacheTrace(initial.step), b = lab.cacheTrace(next.step);
  assert.equal(next.key, b.length - 1);
  assert.equal(b.bytes - a.bytes, 16);
  assert.deepEqual(b.keys.slice(0, -1), a.keys);
  assert.deepEqual(b.values.slice(0, -1), a.values);
  vectorNear(b.keys[next.key], lab.linear(lab.INPUTS[next.key], lab.WK));
});

test('highlighted RAG clauses exist verbatim and support remains claim-specific', () => {
  lab.CLAIMS.forEach((claim, index) => lab.EVIDENCE.forEach(doc => {
    const span = lab.supportSpan(index, doc.id);
    assert.equal(Boolean(span), claim.support.includes(doc.id));
    if (span) assert.ok(doc.text.includes(span));
  }));
  assert.deepEqual(lab.claimSupport(lab.CLAIMS[0], ['C']), []);
  assert.deepEqual(lab.claimSupport(lab.CLAIMS[1], ['C']), ['C']);
  assert.deepEqual(lab.claimSupport(lab.CLAIMS[2], ['B', 'C']), []);
});

test('all seven initial actions change a numerical or evidence outcome, not only prose', () => {
  const experiments = {
    tokenization: { action: 'next', value: '', outcome: s => lab.bpeEffect(s.step).after },
    embeddings: { action: 'next', value: '', outcome: s => lab.embeddingTrace(s.step).loss },
    positional: { action: 'swap', value: '', outcome: s => lab.positionalTrace(s.mode, s.swapped).scores[s.key] },
    'transformer-block': { action: 'change', value: '', outcome: s => lab.blockStages(lab.blockExperiment(s.changed).after)[s.step][s.token][0] },
    attention: { action: 'mute-value', value: '', outcome: s => lab.attentionValueExperiment(s.query, s.key, s.muted).after.rows[s.query].output[0] },
    'kv-cache': { action: 'next', value: '', outcome: s => lab.cacheTrace(s.step).bytes },
    rag: { action: 'exclude', value: 'B', outcome: s => lab.claimSupport(lab.CLAIMS[s.claim], lab.retrieve(s.query).slice(0, s.topk).map(d => d.id).filter(id => !s.excluded.includes(id))).length },
  };
  for (const [kind, ex] of Object.entries(experiments)) {
    const state = lab.initial(kind), after = lab.reduce(kind, state, ex.action, ex.value);
    assert.notEqual(ex.outcome(state), ex.outcome(after), `${kind}: first action must change its mechanism`);
    for (const s of [state, after]) {
      const html = lab.render(kind, s), disclosure = html.indexOf('<details');
      assert.ok(html.indexOf('class="tl-equation"') < disclosure, `${kind}: visible equation`);
      assert.ok(html.indexOf('class="tl-comparison"') < disclosure, `${kind}: visible before/after`);
      assert.match(html.slice(0, disclosure), /data-term=/, `${kind}: relevant term is highlighted`);
      assert.doesNotMatch(html, /NaN|undefined/);
    }
  }
  assert.doesNotMatch(lab.render('attention', lab.initial('attention')), /data-action="next"/);
});

test('compact rotary view retains every vector endpoint in all three encoding modes', () => {
  for (const mode of ['none', 'absolute', 'rope']) for (const swapped of [false, true]) {
    const trace = lab.positionalTrace(mode, swapped);
    for (const row of trace.rows) for (const v of [row.x, row.encoded]) {
      const x = 160 + 52 * v[0], y = 140 - 52 * v[1];
      assert.ok(x > 85 && x < 275 && y > 25 && y < 195, `${mode}: ${row.token} arrow stays inside the viewBox`);
    }
  }
});

test('plot legends stay with vectors and the substituted token keeps its correct label', () => {
  const html = lab.render('embeddings', lab.initial('embeddings'));
  const caption = html.match(/<figcaption>([\s\S]*?)<\/figcaption>/)[1];
  for (const label of ['q: cache memory', '0: policy (positive)', '1: catalogue', '2: unrelated']) assert.ok(caption.includes(label));
  const block = lab.render('transformer-block', { ...lab.initial('transformer-block'), step: 0, token: 0, changed: true });
  assert.match(block, /Tracking token 0: buffer/);
  assert.doesNotMatch(block, /Tracking token 0: cache/);
});

test('de-emphasized labels and evidence use theme colors rather than whole-node opacity', () => {
  const css = require('node:fs').readFileSync(require('node:path').join(__dirname, '../transformer-labs.css'), 'utf8');
  for (const selector of ['.tl-lab .tl-dim', '.tl-lab .tl-source-excluded', '.tl-evidence-absent']) {
    const rule = css.slice(css.indexOf(`${selector} {`)).split('}')[0];
    assert.match(rule, /var\(--lesson-muted\)/, selector);
    assert.doesNotMatch(rule, /opacity/, selector);
  }
  assert.match(css, /svg \.tl-dim :is\(path, line, circle\) \{ opacity:/);
});

test('RAG source captions never imply support for a different or unsupported claim', () => {
  for (const claim of [0, 1, 2]) for (const doc of ['A', 'B', 'C', 'D']) {
    const html = lab.render('rag', { ...lab.initial('rag'), claim, doc });
    if (!lab.supportSpan(claim, doc)) {
      assert.match(html, /no clause here supports the selected claim/);
      assert.doesNotMatch(html, /The highlighted clause is available/);
    }
    if (claim === 2) {
      assert.doesNotMatch(html, />Remove the supporting runbook</);
      assert.match(html, /Can the retrieved text justify a latency guarantee/);
    }
  }
});

test('selected architecture stage reports the same real before/after value beside the clicked layer', () => {
  const experiment = lab.blockExperiment(true), before = lab.blockStages(experiment.before), after = lab.blockStages(experiment.after);
  for (let step = 0; step < 8; step++) {
    const html = lab.render('transformer-block', { ...lab.initial('transformer-block'), changed: true, step });
    const actual = html.match(/<span class="tl-stage-result">([^<]+)<\/span>/)[1];
    assert.equal(actual, `t3[0]: ${before[step][3][0].toFixed(3)} &#8594; ${after[step][3][0].toFixed(3)}`);
    assert.equal((html.match(/class="tl-stage-result"/g) || []).length, 1);
  }
});
