/* Inspectable, deliberately small transformer computations. No model-service dependency. */
(() => {
  'use strict';

  const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
  const add = (a, b) => a.map((x, i) => x + b[i]);
  const scale = (a, c) => a.map(x => x * c);
  const norm = a => Math.sqrt(dot(a, a));
  const cosine = (a, b) => norm(a) && norm(b) ? dot(a, b) / (norm(a) * norm(b)) : 0;
  const linear = (x, w) => w[0].map((_, j) => x.reduce((s, v, i) => s + v * w[i][j], 0));
  function softmax(scores) {
    const max = Math.max(...scores);
    if (!scores.length || max === -Infinity) throw new RangeError('Softmax requires at least one unmasked score');
    const exp = scores.map(x => Math.exp(x - max));
    return scale(exp, 1 / exp.reduce((s, x) => s + x, 0));
  }
  function layerNorm(x, epsilon = 1e-5) {
    const mean = x.reduce((s, v) => s + v, 0) / x.length;
    const variance = x.reduce((s, v) => s + (v - mean) ** 2, 0) / x.length;
    return x.map(v => (v - mean) / Math.sqrt(variance + epsilon));
  }

  const CORPUS = [{ word: 'low', count: 5 }, { word: 'lower', count: 2 }, { word: 'newest', count: 6 }, { word: 'widest', count: 3 }];
  function mergePair(tokens, pair) {
    const result = [];
    for (let i = 0; i < tokens.length; i++) {
      if (tokens[i] === pair[0] && tokens[i + 1] === pair[1]) {
        result.push(tokens[i] + tokens[i + 1]);
        i++;
      } else result.push(tokens[i]);
    }
    return result;
  }
  function pairCounts(rows) {
    const counts = new Map();
    rows.forEach(({ tokens, count }) => {
      for (let i = 0; i < tokens.length - 1; i++) {
        const key = JSON.stringify([tokens[i], tokens[i + 1]]);
        counts.set(key, (counts.get(key) || 0) + count);
      }
    });
    return [...counts].map(([key, count]) => ({ pair: JSON.parse(key), count }))
      .sort((a, b) => b.count - a.count || JSON.stringify(a.pair).localeCompare(JSON.stringify(b.pair), 'en'));
  }
  function trainBPE(steps = 8, corpus = CORPUS) {
    let rows = corpus.map(({ word, count }) => ({ word, count, tokens: Array.from(word) }));
    const merges = [];
    const history = [{ rows, pairs: pairCounts(rows), merges: [] }];
    for (let i = 0; i < steps; i++) {
      const best = pairCounts(rows)[0];
      if (!best) break;
      merges.push(best.pair);
      rows = rows.map(row => ({ ...row, tokens: mergePair(row.tokens, best.pair) }));
      history.push({ rows, pairs: pairCounts(rows), merges: merges.slice(), applied: best });
    }
    return history;
  }
  function encodeBPE(word, merges) {
    return merges.reduce((tokens, pair) => mergePair(tokens, pair), Array.from(word));
  }
  function bpeEffect(step) {
    const history = trainBPE(), after = history[step], before = history[Math.max(0, step - 1)];
    const pair = (after.applied || before.pairs[0]).pair;
    const occurrences = before.rows.map(row => ({ word: row.word, frequency: row.count,
      matches: row.tokens.reduce((n, token, i) => n + Number(token === pair[0] && row.tokens[i + 1] === pair[1]), 0) })).filter(row => row.matches);
    const tokenCount = rows => rows.reduce((n, row) => n + row.count * row.tokens.length, 0);
    return { pair, occurrences, before: tokenCount(before.rows), after: tokenCount(after.rows),
      count: occurrences.reduce((n, row) => n + row.frequency * row.matches, 0), runner: before.pairs[1] };
  }

  const LOOKUP = [[1, 0], [0, 1], [0.8, 0.2], [-0.4, 0.9]];
  const LOOKUP_WORDS = ['cache', 'memory', 'latency', 'eviction'];
  const DOCUMENTS = [
    { label: 'cache memory policy', vector: [1, 1] },
    { label: 'cache product catalogue', vector: [3, 0] },
    { label: 'unrelated fixture', vector: [0, -1] },
  ];
  function contrastive(query, candidates = DOCUMENTS.map(d => d.vector), target = 0) {
    const scores = candidates.map(v => dot(query, v));
    const probabilities = softmax(scores);
    const gradient = query.map((_, j) => candidates.reduce((s, v, i) => s + probabilities[i] * v[j], 0) - candidates[target][j]);
    const max = Math.max(...scores);
    const loss = max + Math.log(scores.reduce((s, x) => s + Math.exp(x - max), 0)) - scores[target];
    return { scores, probabilities, gradient, loss };
  }
  function embeddingTrace(steps = 0) {
    let query = scale(add(LOOKUP[0], LOOKUP[1]), 0.5);
    for (let i = 0; i < steps; i++) query = add(query, scale(contrastive(query).gradient, -0.2));
    return { query, ...contrastive(query) };
  }

  function rotate(x, position, frequency = 1) {
    const c = Math.cos(position * frequency), s = Math.sin(position * frequency);
    return [x[0] * c - x[1] * s, x[0] * s + x[1] * c];
  }
  function positionalTrace(mode = 'none', swapped = false) {
    const base = [{ token: 'cache', x: [1, 0] }, { token: 'saves', x: [0.5, 1] }, { token: 'memory', x: [0, 1] }];
    const rows = (swapped ? [base[2], base[1], base[0]] : base).map((row, i) => ({
      ...row, position: i,
      encoded: mode === 'rope' ? rotate(row.x, i) : mode === 'absolute' ? add(row.x, [Math.sin(i), Math.cos(i)]) : row.x.slice(),
    }));
    // Track the same query identity, rather than a fixed slot, to expose equivariance.
    const query = rows.find(row => row.token === 'cache');
    const scores = rows.map(row => dot(query.encoded, row.encoded) / Math.sqrt(2));
    const weights = softmax(scores);
    const values = rows.map(row => mode === 'absolute' ? row.encoded : row.x);
    return { rows, query, scores, weights, output: [0, 1].map(j => values.reduce((s, v, i) => s + weights[i] * v[j], 0)) };
  }

  const TOKENS = ['cache', 'stores', 'past', 'keys', 'for', 'decode'];
  const INPUTS = [[1, 0, 1], [0, 1, 0.5], [1, 1, 0], [-0.5, 1, 1], [0.5, -0.5, 1], [1, 0.5, -0.5]];
  const WQ = [[1, 0.5], [-0.5, 1], [0.5, 0]];
  const WK = [[0.5, 1], [1, 0], [0, 0.5]];
  const WV = [[1, 0], [0, 1], [0.5, -0.5]];
  const WO = [[1, 0, 0.5], [0, 1, -0.5]];
  const W1 = [[1, -1, 0.5, 0], [0, 1, -0.5, 1], [0.5, 0, 1, -1]];
  const W2 = [[0.5, 0, -0.5], [0, 0.5, 0.5], [0.5, -0.5, 0], [-0.5, 0, 0.5]];
  function attend(q, keys, values, queryIndex = keys.length - 1) {
    const raw = keys.map(k => dot(q, k) / Math.sqrt(q.length));
    const scores = raw.map((x, i) => i <= queryIndex ? x : -Infinity);
    const weights = softmax(scores);
    const contributions = values.map((v, i) => scale(v, weights[i]));
    const output = values[0].map((_, j) => contributions.reduce((s, v) => s + v[j], 0));
    return { raw, scores, weights, contributions, output };
  }
  function attentionTrace(inputs = INPUTS.slice(0, 4)) {
    const q = inputs.map(x => linear(x, WQ)), k = inputs.map(x => linear(x, WK)), v = inputs.map(x => linear(x, WV));
    return { q, k, v, rows: q.map((query, i) => attend(query, k, v, i)) };
  }
  function attentionValueExperiment(query, key, muted = false) {
    const before = attentionTrace();
    const v = before.v.map((row, i) => muted && i === key ? row.map(() => 0) : row.slice());
    const after = { q: before.q, k: before.k, v, rows: before.q.map((q, i) => attend(q, before.k, v, i)) };
    return { before, after, delta: add(after.rows[query].output, scale(before.rows[query].output, -1)) };
  }
  function mlp(x) {
    const hidden = linear(x, W1);
    const activated = hidden.map(v => Math.max(0, v));
    return { hidden, activated, output: linear(activated, W2) };
  }
  function transformerBlock(inputs = INPUTS.slice(0, 4)) {
    const n1 = inputs.map(x => layerNorm(x));
    const attention = attentionTrace(n1);
    const projected = attention.rows.map(row => linear(row.output, WO));
    const residual = inputs.map((x, i) => add(x, projected[i]));
    const n2 = residual.map(x => layerNorm(x));
    const feedforward = n2.map(mlp);
    const output = residual.map((x, i) => add(x, feedforward[i].output));
    return { inputs, n1, attention, projected, residual, n2, feedforward, output };
  }
  const BUFFER_INPUT = [0, 0, 1];
  function blockExperiment(changed = false) {
    const inputs = INPUTS.slice(0, 4).map(row => row.slice());
    if (changed) inputs[0] = BUFFER_INPUT.slice();
    return { before: transformerBlock(), after: transformerBlock(inputs) };
  }
  function blockStages(t) {
    return [t.inputs, t.n1, t.attention.q, t.projected, t.residual, t.n2, t.feedforward.map(row => row.output), t.output];
  }
  function appendCache(cache, input) {
    const keys = [...cache.keys, linear(input, WK)], values = [...cache.values, linear(input, WV)];
    const query = linear(input, WQ);
    return { keys, values, query, result: attend(query, keys, values) };
  }
  function cacheTrace(step = 0) {
    const length = 2 + step;
    const prefill = attentionTrace(INPUTS.slice(0, 2));
    let cache = { keys: prefill.k, values: prefill.v };
    const outputs = prefill.rows.map(row => row.output);
    for (let i = 2; i < length; i++) {
      cache = appendCache(cache, INPUTS[i]);
      outputs.push(cache.result.output);
    }
    return { ...cache, outputs, length, projectedRows: length, uncachedRows: 2 + step * (step + 5) / 2, bytes: 2 * length * 2 * 4 };
  }

  const EVIDENCE = [
    { id: 'A', title: 'Dashboard navigation', text: 'Cache memory eviction. Cache memory eviction. These are navigation labels, not operational rules.' },
    { id: 'B', title: 'Serving runbook', text: 'When memory is full, evict the least recently used unpinned cache entry, if one exists. Entries pinned by active requests cannot be evicted.' },
    { id: 'C', title: 'Request isolation', text: 'A cache entry pinned by an active request cannot be evicted until that request finishes.' },
    { id: 'D', title: 'Training notes', text: 'The optimizer updates model parameters after each gradient accumulation step.' },
  ];
  const CLAIMS = [
    { text: 'When memory is full, evict the least recently used unpinned cache entry, if one exists.', support: ['B'] },
    { text: 'Entries pinned by active requests are protected from eviction.', support: ['B', 'C'] },
    { text: 'This eviction policy guarantees p99 latency below 20 ms.', support: [] },
  ];
  function terms(text) {
    return (text.toLowerCase().match(/[a-z]+/g) || []).map(t => ['evict', 'evicted', 'eviction'].includes(t) ? 'eviction' : t);
  }
  function retrieve(query, documents = EVIDENCE) {
    const bag = text => terms(text).reduce((counts, t) => ({ ...counts, [t]: (counts[t] || 0) + 1 }), {});
    const q = bag(query);
    return documents.map(doc => {
      const d = bag(doc.text), vocab = [...new Set([...Object.keys(q), ...Object.keys(d)])];
      const qv = vocab.map(t => q[t] || 0), dv = vocab.map(t => d[t] || 0);
      return { ...doc, score: cosine(qv, dv), dot: dot(qv, dv), qnorm: norm(qv), dnorm: norm(dv) };
    }).sort((a, b) => b.score - a.score || a.id.localeCompare(b.id));
  }
  function claimSupport(claim, packedIds) {
    return claim.support.filter(id => packedIds.includes(id));
  }
  function supportSpan(claimIndex, sourceId) {
    if (claimIndex === 0 && sourceId === 'B') return CLAIMS[0].text;
    if (claimIndex === 1 && sourceId === 'B') return 'Entries pinned by active requests cannot be evicted.';
    if (claimIndex === 1 && sourceId === 'C') return EVIDENCE[2].text;
    return '';
  }

  const escape = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const f = (x, digits = 3) => x === -Infinity ? '-inf' : (Math.abs(x) < 0.5 * 10 ** -digits ? 0 : x).toFixed(digits);
  const vector = x => `[${x.map(v => f(v)).join(', ')}]`;
  const code = value => `<code class="tl-vector">${escape(value)}</code>`;
  const term = (value, name) => `<mark class="tl-term" data-term="${escape(name)}">${escape(value)}</mark>`;
  const equation = (label, body) => `<p class="tl-equation" aria-label="${escape(label)}"><span>${escape(label)}</span><code>${body}</code></p>`;
  const comparison = (label, before, after, reason) => `<div class="tl-comparison" data-before="${escape(before)}" data-after="${escape(after)}"><p><span>${escape(label)}</span> <samp>${escape(before)} <span aria-label="becomes">&#8594;</span> <strong>${escape(after)}</strong></samp></p><p class="ml-note">${escape(reason)}</p></div>`;
  function mergeChips(tokens, pair, merged = false) {
    return `<div class="tl-tokens">${tokens.map((token, i) => {
      const active = merged ? token === pair.join('') : (token === pair[0] && tokens[i + 1] === pair[1]) || (token === pair[1] && tokens[i - 1] === pair[0]);
      return `<span class="${active ? 'tl-token-changed' : 'tl-dim'}">${escape(token)}</span>`;
    }).join('')}</div>`;
  }
  function markedEvidence(text, claimIndex, sourceId) {
    const span = supportSpan(claimIndex, sourceId);
    if (!span) return `<span class="tl-dim">${escape(text)}</span>`;
    const offset = text.indexOf(span);
    return `<span class="tl-dim">${escape(text.slice(0, offset))}</span>${term(span, `source-${sourceId}`)}<span class="tl-dim">${escape(text.slice(offset + span.length))}</span>`;
  }
  const button = (action, value, label, selected = false, disabled = false) => `<button type="button" data-action="${escape(action)}" data-value="${escape(value)}" class="tl-button${selected ? ' ml-selected' : ''}" aria-pressed="${selected}"${disabled ? ' disabled' : ''}>${escape(label)}</button>`;
  const chips = tokens => `<div class="tl-tokens">${tokens.map(token => `<span>${escape(token)}</span>`).join('')}</div>`;
  const table = (headers, rows, caption = '') => `<div class="ml-table tl-table" tabindex="0" role="region" aria-label="${escape(caption || headers.join(', '))}"><table>${caption ? `<caption>${escape(caption)}</caption>` : ''}<thead><tr>${headers.map(h => `<th scope="col">${escape(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(cell => `<td>${cell}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  const panel = (title, body, klass = '') => `<section class="tl-panel ${klass}"><h4>${escape(title)}</h4>${body}</section>`;
  const nav = (step, max, labels, nextLabel = 'Next') => `<div class="ml-step-nav tl-nav">${button('previous', '', 'Previous', false, step === 0)}<span>${escape(labels ? labels[step] : `Step ${step} / ${max}`)}</span>${button('next', '', nextLabel, false, step === max)}</div>`;
  const calculation = body => `<details class="tl-details tl-calculation"><summary data-action="inspect" data-value="">Inspect the calculation</summary>${body}</details>`;
  const options = body => `<details class="tl-details tl-options"><summary data-action="options" data-value="">Try another setting</summary><div class="ml-controls">${body}${button('reset', '', 'Reset')}</div></details>`;
  const primary = (action, value, label) => `<button type="button" class="tl-button tl-primary" data-action="${escape(action)}" data-value="${escape(value)}">${escape(label)}</button>`;
  const heading = (kicker, question) => `<header class="ml-heading"><p class="ml-kicker">${escape(kicker)}</p><h3 class="ml-question">${escape(question)}</h3></header>`;
  const weightsDisclosure = () => `<details class="tl-details"><summary>Inspect fixed projection and MLP matrices</summary>${table(['Matrix', 'Rows (input dimension by output dimension)'], Object.entries({ WQ, WK, WV, WO, W1, W2 }).map(([name, matrix]) => [code(name), matrix.map(row => code(vector(row))).join('<br>')]))}</details>`;

  function renderBPE(s) {
    const history = trainBPE(), h = history[s.step];
    const old = s.step ? history[s.step - 1] : h;
    const total = rows => rows.reduce((n, row) => n + row.tokens.length * row.count, 0);
    const baseVocab = new Set(CORPUS.flatMap(row => Array.from(row.word)));
    h.merges.forEach(pair => baseVocab.add(pair.join('')));
    const encoded = encodeBPE(s.word, h.merges);
    const effect = bpeEffect(s.step);
    const example = old.rows.find(row => row.tokens.some((t, i) => t === effect.pair[0] && row.tokens[i + 1] === effect.pair[1]));
    const after = h.rows.find(row => row.word === example.word);
    return heading('01 / Tokens are reusable pieces', 'Which two pieces become one token?') + nav(s.step, 8, null, 'Apply next merge') +
      `<div class="tl-mechanism"><p class="tl-mechanism-label">${escape(example.word)} appears ${example.count} times in the training corpus</p><div class="tl-merge-flow"><div><span>Before</span>${mergeChips(example.tokens, effect.pair)}</div><div class="tl-flow-connector" aria-hidden="true">&#8595;</div><div><span>After merge ${s.step}</span>${mergeChips(after.tokens, effect.pair, true)}</div></div>${equation(`Why ${effect.pair.join(' + ')} wins`, `C(${escape(effect.pair.join(', '))}) = ${effect.occurrences.map(row => `${term(`${row.frequency} x ${row.matches}`, row.word)} <small>(${escape(row.word)})</small>`).join(' + ')} = ${effect.count}`)}${comparison('Weighted corpus tokens', effect.before, effect.after, s.step ? `Only the highlighted pieces merge. The next-ranked pair occurs ${effect.runner.count} times${effect.runner.count === effect.count ? '; the lexical tie-break selects this pair' : `, versus ${effect.count} for this pair`}.` : 'No merge has been applied yet; frequency chooses the first rule.')}</div>` +
      `<div class="tl-transfer"><p>Apply the same learned rules to ${code(s.word)}${s.word === 'lowest' ? ', an unseen word' : ''}:</p>${chips(encoded)}<p class="ml-note">${s.word.length} characters become ${encoded.length} tokens. Encoding does not learn new merges.</p></div>` +
      options(['lower', 'newest', 'lowest'].map(word => button('word', word, word, s.word === word)).join('')) +
      calculation(table(['Word x frequency', 'Current pieces'], h.rows.map(row => [escape(`${row.word} x ${row.count}`), chips(row.tokens)])) + table(['Next adjacent pair', 'Weighted occurrences'], h.pairs.slice(0, 4).map(p => [code(p.pair.join(' + ')), String(p.count)])) + `<p>${total(h.rows)} corpus tokens; ${baseVocab.size} vocabulary entries. Ties use lexical pair order. No merges cross word boundaries.</p><div class="tl-ranks">${h.merges.map((p, i) => `<span>${i + 1}: ${escape(p.join(' + '))}</span>`).join('')}</div>`);
  }

  function renderEmbeddings(s) {
    const trace = embeddingTrace(s.step), q = trace.query;
    const before = embeddingTrace(Math.max(0, s.step - 1)), d = s.coordinate || 0;
    const update = s.step ? before : trace;
    const nextCoordinate = update.query[d] - .2 * update.gradient[d];
    const margin = state => state.scores[0] - state.scores[1];
    const causalReadout = comparison('Policy minus catalogue: dot-score margin', f(margin(before)), f(margin(trace)), 'Documents stay fixed: q[0] weighs catalogue 3x more than policy; q[1] helps only policy in this pair.') +
      equation(s.step ? `Last update to query coordinate ${d}` : `Next update to query coordinate ${d}`, `q[${d}] = ${f(update.query[d])} - ${term(`0.2 x ${f(update.gradient[d])}`, `gradient-${d}`)} = ${f(nextCoordinate)}`);
    const ranks = DOCUMENTS.map((doc, i) => ({ ...doc, i, score: s.metric === 'cosine' ? cosine(q, doc.vector) : dot(q, doc.vector) })).sort((a, b) => b.score - a.score);
    const points = [...DOCUMENTS.map((d, i) => ({ ...d, label: String(i), klass: i === s.doc ? 'tl-selected-point' : 'tl-point' })), { vector: q, label: 'q', klass: 'tl-query-point' }];
    const svg = `<svg viewBox="0 0 320 220" role="img" aria-label="Actual two-dimensional vectors. Dashed orange is the previous query; solid orange is the current query."><path d="M35 130H300 M55 210V20" class="tl-axis"/>${s.step ? vectorArrow([55, 130], [55 + before.query[0] * 65, 130 - before.query[1] * 65], 'tl-query-point tl-ghost') : ''}${points.map(p => `<g class="${p.klass}${p.label !== 'q' && Number(p.label) !== s.doc ? ' tl-dim' : ''}"><line x1="55" y1="130" x2="${55 + p.vector[0] * 65}" y2="${130 - p.vector[1] * 65}"/><circle cx="${55 + p.vector[0] * 65}" cy="${130 - p.vector[1] * 65}" r="5"/><text x="${55 + p.vector[0] * 65 + (p.label === 'q' ? -19 : 10)}" y="${130 - p.vector[1] * 65 - 10}">${escape(p.label)}</text></g>`).join('')}<text x="265" y="153">dim 0</text><text x="65" y="25">dim 1</text></svg>`;
    return heading('02 / Separate the positive from the distractors', 'Can training make the right result rank first?') +
      nav(s.step, 4, ['Initial query', 'Update 1', 'Update 2', 'Update 3', 'Update 4'], 'Apply training step') +
      `<div class="ml-grid"><figure class="tl-position-figure">${svg}<figcaption><span class="tl-query-label">q: cache memory</span>; 0: policy (positive); 1: catalogue; 2: unrelated. ${s.step ? 'Dashed q: prior; solid q: now.' : 'Toy vectors.'}</figcaption>${causalReadout}</figure><div class="tl-candidate-list">${ranks.map(doc => `<button type="button" class="tl-candidate${s.doc === doc.i ? ' ml-selected' : ' tl-dim'}" data-action="doc" data-value="${doc.i}" aria-pressed="${s.doc === doc.i}"><span>${doc.i}: ${escape(doc.label)}${doc.i === 0 ? ' (positive)' : ''}</span><strong>${f(doc.score)}</strong></button>`).join('')}<p class="ml-note">Best score first (${escape(s.metric)}). IDs match the vector labels. Click a candidate to highlight it.</p><p class="tl-result">Training loss: ${f(trace.loss)}. ${s.step ? 'The query has changed under contrastive supervision.' : 'The large-norm catalogue vector outranks the labeled positive under dot product.'}</p></div></div>` +
      options(button('metric', 'cosine', 'Rank by cosine', s.metric === 'cosine') + button('metric', 'dot', 'Rank by dot product', s.metric === 'dot') + button('coordinate', 0, 'Inspect q[0]', d === 0) + button('coordinate', 1, 'Inspect q[1]', d === 1)) +
      calculation(`<p>Initial query = mean(E[cache], E[memory]) = [0.5, 0.5]. Illustrative coordinates, not measured semantic quality.</p>${table(['Token ID', 'E[id]'], LOOKUP.map((v, i) => [escape(`${i}: ${LOOKUP_WORDS[i]}`), code(vector(v))]))}${table(['Candidate', 'Vector', s.metric], ranks.map(doc => [escape(doc.label), code(vector(doc.vector)), code(f(doc.score))]))}<p>${code(vector(q))} dot ${code(vector(DOCUMENTS[s.doc].vector))} = ${f(dot(q, DOCUMENTS[s.doc].vector))}; cosine = ${f(cosine(q, DOCUMENTS[s.doc].vector))}.</p>${table(['q', 'Gradient dL/dq', 'Loss'], [[code(vector(q)), code(vector(trace.gradient)), code(f(trace.loss))]])}<p>q next = q - 0.2 x gradient. Only q is optimized; this is not encoder training. The loss always uses dot-product logits.</p>`);
  }

  function vectorArrow(from, to, klass, extra = '') {
    const dx = to[0] - from[0], dy = to[1] - from[1], length = Math.hypot(dx, dy);
    if (length < 1e-9) return '';
    const ux = dx / length, uy = dy / length, size = Math.min(7, length / 3);
    return `<g class="${escape(klass)}" ${extra}><path d="M${from[0]},${from[1]} L${to[0]},${to[1]}"/><path class="tl-arrowhead" d="M${to[0]},${to[1]} L${to[0] - size * ux + size * .45 * uy},${to[1] - size * uy - size * .45 * ux} L${to[0] - size * ux - size * .45 * uy},${to[1] - size * uy + size * .45 * ux} Z"/></g>`;
  }

  function positionDiagram(trace, keyIndex, mode) {
    const origin = [160, 140], unit = 52, point = v => [origin[0] + unit * v[0], origin[1] - unit * v[1]];
    const selected = trace.rows[keyIndex];
    const vectors = [{ row: trace.query, role: 'query' }, { row: selected, role: 'key' }];
    const drawings = vectors.map(({ row, role }) => {
      const before = point(row.x), after = point(row.encoded), klass = `tl-position-${role}`;
      let trajectory = '';
      if (mode === 'rope' && row.position > 0) {
        const r = norm(row.x) * unit;
        trajectory = `<path class="${klass} tl-rotation-arc" data-angle="${row.position}" d="M${before[0]},${before[1]} A${r},${r} 0 0 0 ${after[0]},${after[1]}"/>`;
      } else if (mode === 'absolute') {
        trajectory = vectorArrow(before, after, `${klass} tl-position-offset`);
      }
      return vectorArrow(origin, before, `${klass} tl-position-before`) + trajectory +
        vectorArrow(origin, after, `${klass} tl-position-after`, `data-vector="${escape(JSON.stringify(row.encoded))}"`);
    }).join('');
    return `<figure class="tl-position-figure"><div class="tl-vector-legend"><span class="tl-query-label">Query: cache @ ${trace.query.position}</span><span class="tl-key-label">Key: ${escape(selected.token)} @ ${selected.position}</span></div><svg class="tl-position-svg" viewBox="80 20 200 180" role="img" aria-label="${escape(`${mode === 'rope' ? 'Actual rotary trajectories' : mode === 'absolute' ? 'Actual additive position displacements' : 'No position transform'} of the query and selected key. Dashed vectors are original; solid vectors are transformed.`)}"><path class="tl-axis" d="M88 140H272 M160 28V192"/><circle class="tl-origin" cx="160" cy="140" r="3"/>${drawings}</svg><figcaption>${mode === 'rope' ? `Q: ${trace.query.position} rad; K: ${selected.position} rad. Dashed: original; solid: rotated. Lengths unchanged.` : mode === 'absolute' ? 'Add [sin(position), cos(position)]: dashed originals to solid results. Dotted arrows show the offsets.' : 'No transform: dashed originals and solid vectors coincide.'} Equal axis scales.${selected.token === 'cache' ? ' Same token: query/key arrows coincide.' : ''}</figcaption></figure>`;
  }

  function renderPosition(s) {
    const a = positionalTrace(s.mode, false), b = positionalTrace(s.mode, true), active = s.swapped ? b : a;
    const row = active.rows[s.key];
    const originalKey = a.rows.findIndex(r => r.token === row.token), displacement = row.position - active.query.position;
    const matchedKey = s.mode === 'rope' ? rotate(row.x, displacement) : row.encoded;
    return heading('03 / Position changes the match', 'Does swapping token order change what "cache" reads?') +
      `<div class="ml-controls">${primary('swap', '', s.swapped ? 'Restore token order' : 'Swap token order')}</div><div class="ml-controls" aria-label="Sequence order; select a key">${active.rows.map((r, i) => button('key', i, `${i}: ${r.token}`, s.key === i)).join('')}</div>` +
      positionDiagram(active, s.key, s.mode) +
      comparison(`cache -> ${row.token}: original vs displayed score`, f(a.scores[originalKey]), f(active.scores[s.key]), s.mode === 'rope' ? `Content stays fixed; key minus query position = ${displacement}. Relative rotation determines this match.` : s.mode === 'none' ? 'The content vectors did not change and there is no position term, so reordering cannot change this match.' : 'Changing the slots changes the added sin/cos vectors, including the value vectors.') +
      equation(s.mode === 'rope' ? `Relative rotation R(${displacement}) on the selected key` : 'Selected query-key match', `${escape(vector(s.mode === 'rope' ? active.query.x : active.query.encoded))} dot ${term(vector(matchedKey), 'position-key')} / sqrt(2) = ${f(active.scores[s.key])}`) +
      options([['none', 'No position'], ['absolute', 'Add sin/cos'], ['rope', 'RoPE rotation']].map(([v, label]) => button('mode', v, label, s.mode === v)).join('')) +
      calculation(table(['Order', 'Attention output for cache'], [['cache saves memory', code(vector(a.output))], ['memory saves cache', code(vector(b.output))]]) + table(['Position / token', 'Before', 'After'], active.rows.map(r => [escape(`${r.position}: ${r.token}`), code(vector(r.x)), code(vector(r.encoded))])) + `<p>Identity projections, full attention, no causal mask. RoPE changes Q/K only; additive position also changes V. Unmasked, position-free attention is permutation equivariant.</p>${s.mode === 'rope' ? `<p>${code(`R(${row.position}) = [[${f(Math.cos(row.position))}, ${f(-Math.sin(row.position))}], [${f(Math.sin(row.position))}, ${f(Math.cos(row.position))}]]`)}</p><p>One rotary pair, frequency = 1 radian / position.</p>` : s.mode === 'absolute' ? `<p>${code(`p[${row.position}] = ${vector([Math.sin(row.position), Math.cos(row.position)])}`)}</p>` : ''}<p>${code(vector(active.query.encoded))} dot ${code(vector(row.encoded))} / sqrt(2) = ${f(active.scores[s.key])}.</p>`);
  }

  function heatmap(rows, query, key, action = 'cell') {
    const n = rows.length, cell = 42, left = 32, top = 25;
    return `<svg viewBox="0 0 ${left + n * cell + 5} ${top + n * cell + 25}" role="group" aria-label="Causal attention matrix. Rows are query indices, columns key indices. Select a cell.">${Array.from({ length: n }, (_, i) => `<text x="${left + i * cell + 21}" y="16" text-anchor="middle">k${i}</text><text x="3" y="${top + i * cell + 26}">q${i}</text>`).join('')}${rows.map((row, i) => row.weights.map((weight, j) => `<g role="button" tabindex="0" data-action="${escape(action)}" data-value="${i},${j}" aria-label="Query ${i}, key ${j}, ${j > i ? 'masked' : `weight ${f(weight)}`}" aria-pressed="${i === query && j === key}"><rect x="${left + j * cell + 2}" y="${top + i * cell + 2}" width="38" height="38" rx="4" class="tl-heat-cell${i === query && j === key ? ' tl-heat-selected' : ''}" style="fill-opacity:${j > i ? 0.04 : 0.1 + 0.7 * weight}"/><text x="${left + j * cell + 21}" y="${top + i * cell + 26}" text-anchor="middle">${j > i ? 'x' : f(weight, 2)}</text></g>`).join('')).join('')}</svg>`;
  }
  function renderAttention(s) {
    const experiment = attentionValueExperiment(s.query, s.key, s.muted), t = experiment.after, r = t.rows[s.query], k = s.key;
    const other = r.output[0] - r.contributions[k][0];
    return heading('05 / Matching is not the message', 'Can the output change while attention weights stay fixed?') +
      `<div class="ml-controls">${primary('mute-value', '', `${s.muted ? 'Restore' : 'Zero'} the value from "${TOKENS[k]}"`)}</div><figure class="tl-attention-figure"><figcaption>"cache stores past keys". Top: key/value sources. Bottom: query outputs. Select a node.</figcaption><p class="tl-value-change">V from "${escape(TOKENS[k])}": ${code(vector(experiment.before.v[k]))} &#8594; ${term(vector(t.v[k]), 'value-payload')}</p>${tokenFlowDiagram({ attention: t }, s.query, false, true, k)}<div class="tl-vector-legend">${TOKENS.slice(0, 4).map((name, i) => `<span class="${i === k || i === s.query ? '' : 'tl-dim'}">t${i}: ${escape(name)}</span>`).join('')}</div></figure>` +
      comparison(`Output at "${TOKENS[s.query]}", coordinate 0`, f(experiment.before.rows[s.query].output[0]), f(r.output[0]), k > s.query ? `"${TOKENS[k]}" is a future token: its masked weight is zero, so changing its value cannot change this output.` : `Q and K are untouched. The weight on "${TOKENS[k]}" stays ${f(r.weights[k])}; only its value payload ${s.muted ? 'was zeroed' : 'will be zeroed'}.`) +
      equation(`Selected contribution from "${TOKENS[k]}"`, `o[0] = ${f(other)} + ${term(`${f(r.weights[k])} x ${f(t.v[k][0])}`, 'weighted-value')} = ${f(r.output[0])}`) +
      calculation(`<p>Q = ${code(vector(t.q[s.query]))}; K = ${code(vector(t.k[k]))}; V = ${code(vector(t.v[k]))}.</p><p>The intervention is applied after projection: only selected V is replaced with [0, 0]. W_Q, W_K, W_V and X stay fixed.</p><p>Coordinate products = ${code(vector(t.q[s.query].map((x, i) => x * t.k[k][i])))}; sum / sqrt(2) = ${f(r.raw[k])}. Masked score = ${f(r.scores[k])}.</p>${table(['Key', 'Masked score', 'Weight', 'Weighted V'], t.k.map((_, i) => [button('key', i, `${i}: ${TOKENS[i]}`, i === k), code(f(r.scores[i])), code(f(r.weights[i])), code(vector(r.contributions[i]))]))}${heatmap(t.rows, s.query, k)}${table(['Token', 'X', 'Q', 'K', 'V'], t.q.map((q, i) => [escape(TOKENS[i]), code(vector(INPUTS[i])), code(vector(q)), code(vector(t.k[i])), code(vector(t.v[i]))]))}${weightsDisclosure()}`) + options('');
  }

  const BLOCK_STAGES = ['Input residual stream', 'LayerNorm 1', 'Project Q / K / V', 'Causal attention + W_O', 'First residual addition', 'LayerNorm 2', 'Per-token MLP', 'Second residual addition'];
  function blockEquation(t, stage, i) {
    const result = blockStages(t)[stage][i][0];
    if (stage === 0) return equation(`Token ${i}, input coordinate 0`, `X[${i},0] = ${term(f(result), 'input')}`);
    if (stage === 1 || stage === 5) {
      const x = stage === 1 ? t.inputs[i] : t.residual[i], mean = x.reduce((a, b) => a + b, 0) / x.length;
      const variance = x.reduce((sum, v) => sum + (v - mean) ** 2, 0) / x.length;
      return equation(`Token ${i}: normalize features, not tokens`, `(${term(f(x[0]), 'normalized-input')} - ${f(mean)}) / sqrt(${f(variance)} + 0.00001) = ${f(result)}`);
    }
    if (stage === 2) return equation(`Q for token ${i}, coordinate 0`, `${t.n1[i].map((v, j) => j === 0 ? term(`${f(v)} x ${WQ[j][0]}`, 'projection') : `${f(v)} x ${WQ[j][0]}`).join(' + ')} = ${f(result)}`);
    if (stage === 3) return equation(`Attention update at token ${i}, coordinate 0`, `${t.attention.rows[i].weights.map((a, j) => j === 0 ? term(`${f(a)} x ${f(t.attention.v[j][0])}`, 'context-token-0') : `<span class="tl-dim">${f(a)} x ${f(t.attention.v[j][0])}</span>`).join(' + ')} = ${f(result)}`);
    if (stage === 4 || stage === 7) {
      const carry = stage === 4 ? t.inputs[i][0] : t.residual[i][0];
      const update = stage === 4 ? t.projected[i][0] : t.feedforward[i].output[0];
      return equation(`Carry + ${stage === 4 ? 'attention' : 'MLP'} update`, `${f(carry)} + ${term(f(update), 'residual-update')} = ${f(result)}`);
    }
    return equation(`Shared MLP, only token ${i}'s features`, `${t.feedforward[i].activated.map((a, j) => term(`${f(a)} x ${W2[j][0]}`, `mlp-${j}`)).join(' + ')} = ${f(result)}`);
  }
  function tokenFlowDiagram(trace, token, local = false, interactive = false, selectedKey = 0) {
    const xs = [26, 82, 138, 194];
    const edges = local ? xs.map((x, i) => `<g class="${i === token ? 'tl-flow-active' : 'tl-flow-muted'}" data-local-token="${i}">${vectorArrow([x, 26], [x, 43], 'tl-flow-edge')}<rect x="${x - 22}" y="43" width="44" height="31" rx="5"/><text x="${x}" y="63" text-anchor="middle">MLP</text>${vectorArrow([x, 74], [x, 97], 'tl-flow-edge')}</g>`).join('') : trace.attention.rows.map((r, i) => r.weights.map((weight, j) => j > i ? '' : vectorArrow([xs[j], 26], [xs[i], 97], i === token ? `tl-flow-active${j === selectedKey ? ' tl-flow-cause' : ' tl-dim'}` : 'tl-flow-muted', `data-query="${i}" data-key="${j}" data-weight="${weight}" style="stroke-width:${i === token ? 1 + 4 * weight : 1}"`)).join('')).join('');
    return `<svg class="tl-token-flow${interactive ? ' tl-flow-interactive' : ''}" viewBox="0 0 220 124" role="${interactive ? 'group' : 'img'}" aria-label="${escape(local ? `Four independent token-local MLP paths; token ${token} is highlighted. There are no cross-token edges.` : `Causal attention connections. Incoming edges for token ${token} use the actual computed weights.`)}">${edges}${xs.map((x, i) => `<g class="${i === token ? 'tl-flow-active' : 'tl-flow-muted'}"><g${interactive ? ` role="button" tabindex="0" data-action="key" data-value="${i}" aria-pressed="${i === selectedKey}" aria-label="Inspect key ${i}: ${TOKENS[i]}"` : ''}>${interactive ? `<rect class="tl-flow-hit" x="${x - 23}" y="0" width="46" height="46" rx="5"/>` : ''}<text x="${x}" y="14" text-anchor="middle">t${i}</text><circle cx="${x}" cy="26" r="${interactive && i === selectedKey ? 8 : 5}"/></g><g${interactive ? ` role="button" tabindex="0" data-action="query" data-value="${i}" aria-pressed="${i === token}" aria-label="Select query ${i}: ${TOKENS[i]}"` : ''}>${interactive ? `<rect class="tl-flow-hit" x="${x - 23}" y="78" width="46" height="46" rx="5"/>` : ''}<circle cx="${x}" cy="97" r="5"/><text x="${x}" y="118" text-anchor="middle">t${i}</text></g></g>`).join('')}</svg>`;
  }
  function blockDiagram(t, s, baseline) {
    const before = blockStages(baseline), after = blockStages(t);
    const stage = (index, label, subtitle, klass = '') => `<button type="button" class="tl-architecture-stage ${klass}${s.step === index ? ' tl-stage-active' : ' tl-stage-context'}" data-action="stage" data-value="${index}" aria-pressed="${s.step === index}"><strong>${escape(label)}</strong>${s.step === index ? `${subtitle ? `<span>${escape(subtitle)}</span>` : ''}<span class="tl-stage-result">t${s.token}[0]: ${f(before[index][s.token][0])} &#8594; ${f(after[index][s.token][0])}</span>` : ''}</button>`;
    return `<div class="tl-architecture" role="group" aria-label="Pre-norm transformer block; data flows downward. Click a layer to inspect it.">${stage(0, 'Input X', `Tracking token ${s.token}: ${s.token === 0 && s.changed ? 'buffer' : TOKENS[s.token]}`)}<div class="tl-residual-unit"><div class="tl-bypass" aria-hidden="true"><span>carry X unchanged</span></div><div class="tl-stage-stack">${stage(1, 'LayerNorm 1', 'Each token, across its features')}${stage(2, 'Project Q / K / V', 'Shared matrices, separate token rows')}${stage(3, 'Causal attention', 'Mix across tokens, then project W_O')}${stage(4, '+ Add X', '', 'tl-add-stage')}</div></div><div class="tl-residual-unit"><div class="tl-bypass" aria-hidden="true"><span>carry H unchanged</span></div><div class="tl-stage-stack">${stage(5, 'LayerNorm 2', 'Normalize the updated row')}${stage(6, 'Token-local MLP', 'Same weights; no cross-token mixing')}${stage(7, '+ Add H', '', 'tl-add-stage')}</div></div><p class="tl-architecture-output">Output Y: updated residual stream</p></div>`;
  }
  function renderBlock(s) {
    const experiment = blockExperiment(s.changed), t = experiment.after, baseline = experiment.before, i = s.token;
    const inputs = t.inputs, stages = blockStages(t), oldStages = blockStages(baseline);
    const explanations = [
      'Four token rows, three residual coordinates. These are fixed illustrative hidden states, not interpreted semantic axes.',
      'Normalize each row over its three features (epsilon = 1e-5, gain = 1, bias = 0). No cross-token mixing yet.',
      'Three independent projections: model width 3 to head width 2. These weights are shared across token positions.',
      'A causal weighted sum of V crosses token boundaries. W_O maps head width 2 back to residual width 3.',
      'Keep the input and add the attention update. The residual branch bypasses normalization and projection.',
      'Normalize the updated residual row, not the original input. Each token is normalized independently.',
      'Width 3 to 4 to 3, with ReLU between the matrices. The same MLP runs separately on every token.',
      'Add the per-token MLP update to the first residual stream. This is a complete pre-norm block.',
    ];
    let detail;
    if (s.step === 2) detail = `<p>Q = ${code(vector(t.attention.q[i]))}</p><p>K = ${code(vector(t.attention.k[i]))}</p><p>V = ${code(vector(t.attention.v[i]))}</p>`;
    else if (s.step === 3) detail = `<p>Attention weights = ${code(vector(t.attention.rows[i].weights))}</p><p>Mixed V = ${code(vector(t.attention.rows[i].output))}</p><p>Mixed V @ W_O = ${code(vector(t.projected[i]))}</p>`;
    else if (s.step === 4 || s.step === 7) detail = `<p>Carry ${code(vector(s.step === 4 ? inputs[i] : t.residual[i]))}</p><p>+ update ${code(vector(s.step === 4 ? t.projected[i] : t.feedforward[i].output))}</p><p class="tl-result">= ${code(vector(stages[s.step][i]))}</p>`;
    else if (s.step === 6) detail = `<p>LN2 @ W1 = ${code(vector(t.feedforward[i].hidden))}</p><p>ReLU = ${code(vector(t.feedforward[i].activated))}</p><p>ReLU @ W2 = ${code(vector(t.feedforward[i].output))}</p>`;
    else detail = `<p>Input row = ${code(vector(s.step === 5 ? t.residual[i] : inputs[i]))}</p><p>Stage output = ${code(vector(stages[s.step][i]))}</p>`;
    const delta = add(t.output[i], scale(baseline.output[i], -1));
    const localReason = s.step < 3 && i !== 0 ? 'No change yet: this operation reads only the selected token. Token 0 cannot affect a later row until attention.' : s.step === 3 ? 'The highlighted term comes from token 0. Its changed normalized key/value can alter a later token even though that later input stayed fixed.' : s.step === 6 ? 'This MLP reads only its own row. Any effect from token 0 arrived earlier through attention.' : explanations[s.step];
    const routeInput = s.step === 2 ? t.n1[i] : s.step >= 5 ? t.residual[i] : inputs[i];
    const graphic = s.step === 3 || s.step === 6 ? tokenFlowDiagram(t, i, s.step === 6) + `<div class="tl-vector-legend">${TOKENS.slice(0, 4).map((name, j) => `<span class="${j === i || j === 0 ? '' : 'tl-dim'}">t${j}: ${j === 0 && s.changed ? 'buffer' : escape(name)}</span>`).join('')}</div>` : `<div class="tl-vector-route"><span>Selected row ${i}</span>${code(vector(routeInput))}<b aria-hidden="true">&#8595;</b><span>${escape(BLOCK_STAGES[s.step])}</span>${code(vector(stages[s.step][i]))}</div>`;
    return heading('04 / Find the first cross-token effect', 'Where does changing "cache" first affect "keys"?') +
      `<div class="ml-controls">${primary('change', '', s.changed ? 'Restore "cache"' : 'Replace "cache" with "buffer"')}</div><p class="ml-note">${s.changed ? 'buffer' : 'cache'} stores past keys. Toy lookup: cache [1, 0, 1], buffer [0, 0, 1]. Only input coordinate 0 differs; these are not trained semantic axes.</p>` +
      `<div class="tl-block-layout"><div class="tl-block-inspector"><h4>${escape(BLOCK_STAGES[s.step])}: tracking "${i === 0 && s.changed ? 'buffer' : TOKENS[i]}"</h4>${graphic}${comparison(`Selected stage output, coordinate 0`, f(oldStages[s.step][i][0]), f(stages[s.step][i][0]), localReason)}${blockEquation(t, s.step, i)}<p class="ml-note">Select a layer in the architecture to follow the same token. Orange rails bypass each learned branch.</p><details class="tl-details tl-calculation"${s.inspect ? ' open' : ''}><summary data-action="inspect" data-value="">Inspect the calculation</summary><div class="tl-tensor-inspector">${detail}${table(['Token / row', s.step === 2 ? 'Q (head width 2)' : 'Residual-width vector'], stages[s.step].map((v, j) => [escape(`${j}: ${j === 0 && s.changed ? 'buffer' : TOKENS[j]}`), code(vector(v))]))}<p>Final output change at ${i}: ${code(vector(delta))}. Attention coordinate 0 uses W_O[:, 0] = [1, 0], so it equals the mixed V coordinate 0.</p></div>${weightsDisclosure()}</details>${options(TOKENS.slice(0, 4).map((name, j) => button('token', j, `Track ${j}: ${j === 0 && s.changed ? 'buffer' : name}`, i === j)).join('') + nav(s.step, 7, BLOCK_STAGES))}</div>${blockDiagram(t, s, baseline)}</div>`;
  }

  function renderCache(s) {
    const t = cacheTrace(s.step), full = attentionTrace(INPUTS.slice(0, t.length)), selected = Math.min(s.key, t.length - 1), query = Math.min(s.query, t.length - 1);
    const before = s.step ? cacheTrace(s.step - 1) : null;
    const oldKey = before && before.keys[selected];
    const maxError = Math.max(...full.rows.flatMap((r, i) => r.output.map((v, j) => Math.abs(v - t.outputs[i][j]))));
    return heading('06 / Keep the past; compute the new row', 'What changes when one more token arrives?') + nav(s.step, 4, ['Prefill: 2 tokens', 'Decode: past', 'Decode: keys', 'Decode: for', 'Decode: decode'], 'Append next token') +
      `<div class="tl-mechanism"><p class="tl-mechanism-label">Stored keys and values: click a slot</p><div class="tl-cache-slots">${t.keys.map((key, i) => `<button type="button" class="tl-cache-slot${i === selected ? ' ml-selected' : ' tl-dim'}${s.step > 0 && i === t.length - 1 ? ' tl-cache-new' : ''}" data-action="key" data-value="${i}" aria-pressed="${i === selected}"><strong>${escape(TOKENS[i])}</strong><span class="tl-cache-pair"><span>K</span><span>V</span></span><small>${s.step === 0 ? 'prefill' : i === t.length - 1 ? 'new append' : 'reused'}</small></button>`).join('')}</div><div class="tl-cache-read"><span>New query: ${escape(TOKENS[t.length - 1])}</span><strong>reads all ${t.length} cached slots</strong></div></div>` +
      comparison(`Key[0] for "${TOKENS[selected]}"`, oldKey ? f(oldKey[0]) : 'not cached', f(t.keys[selected][0]), `${oldKey ? 'This prefix row is unchanged; causality lets us reuse it.' : 'This row is newly projected, then appended without overwriting the prefix.'} KV storage: ${before ? before.bytes : 0} -> ${t.bytes} bytes; ${s.step ? 'one new pair adds 16 bytes' : 'two prompt pairs use 32 bytes'}.`) +
      equation(`${oldKey ? 'Cached result originally computed as' : 'New key coordinate: x @ W_K'}`, `${INPUTS[selected].map((v, j) => j === 1 ? term(`${f(v)} x ${WK[j][0]}`, 'new-key-projection') : `${f(v)} x ${WK[j][0]}`).join(' + ')} = ${f(t.keys[selected][0])}`) +
      calculation(table(['Position', 'K', 'V'], t.keys.map((key, i) => [escape(`${i}: ${TOKENS[i]}`), code(vector(key)), code(vector(t.values[i]))])) + heatmap(full.rows, query, selected) + `<p>q${query} output = ${code(vector(t.outputs[query]))}. Cached vs full recomputation max difference: ${maxError.toExponential(1)}.</p>` + table(['Cumulative projected K/V row pairs', 'With cache', 'Without cache'], [['Since prefill', String(t.projectedRows), String(t.uncachedRows)]]) + `<p>2 x ${t.length} tokens x 1 layer x 1 KV head x 2 dimensions x 4 bytes = ${t.bytes} bytes. Batch 1; FP32 storage accounting, not latency. The new query still scores every cached key. Prefill and chunked decode need causal masking.</p>`) + options('');
  }

  function renderRag(s) {
    const ranked = retrieve(s.query), packed = ranked.slice(0, s.topk).map(row => row.id).filter(id => !s.excluded.includes(id));
    const selected = ranked.find(row => row.id === s.doc) || ranked[0], claim = CLAIMS[s.claim], support = claimSupport(claim, packed);
    const bRank = ranked.findIndex(doc => doc.id === 'B');
    const runbookLabel = claim.support.includes('B') ? 'supporting runbook' : 'runbook';
    const action = bRank >= s.topk ? primary('topk', bRank + 1, `Retrieve the ${runbookLabel}`) : primary('exclude', 'B', `${packed.includes('B') ? 'Remove' : 'Restore'} the ${runbookLabel}`);
    const selectedNote = supportSpan(s.claim, selected.id)
      ? `Inspecting source ${selected.id}. ${packed.includes(selected.id) ? 'The highlighted clause is available to the answer.' : 'This clause is absent from the prompt, even though it still exists in the collection.'}`
      : `Inspecting source ${selected.id}. ${selected.id === 'A' ? 'Matching query terms is not enough: ' : ''}no clause here supports the selected claim.`;
    const originallyPacked = ranked.slice(0, s.topk).map(doc => doc.id);
    const originalSupport = claimSupport(claim, originallyPacked);
    return heading('07 / A source must support the claim', claim.support.length ? 'What happens when the supporting source leaves the prompt?' : 'Can the retrieved text justify a latency guarantee?') +
      `<div class="ml-controls">${action}</div><div class="tl-rag-flow"><div class="tl-rag-context"><p class="tl-mechanism-label">Retrieved context: ${escape(s.query)}</p><div class="tl-source-cards">${ranked.slice(0, s.topk).map(doc => `<button type="button" class="tl-source-card${packed.includes(doc.id) ? '' : ' tl-source-excluded'}${s.doc === doc.id ? ' ml-selected' : ' tl-dim'}" data-action="doc" data-value="${doc.id}" aria-pressed="${s.doc === doc.id}"><strong>${escape(doc.id)}: ${escape(doc.title)}</strong><span>${packed.includes(doc.id) ? 'In prompt' : 'Removed from prompt'}</span></button>`).join('')}</div><blockquote class="tl-source${packed.includes(selected.id) ? '' : ' tl-evidence-absent'}">${markedEvidence(selected.text, s.claim, selected.id)}</blockquote><p class="ml-note">${escape(selectedNote)}</p>${equation('Claim support, not retrieval confidence', `support = ${term(`{${claim.support.join(', ')}}`, 'supporting-source-ids')} intersect {${escape(packed.join(', '))}} = {${escape(support.join(', '))}}`)}</div><div class="tl-flow-connector" aria-hidden="true">&#8595;</div><div class="tl-rag-answer"><p class="tl-mechanism-label">Candidate answer claim</p><blockquote class="tl-source">${escape(claim.text)}</blockquote>${comparison('Supporting sources before removal vs now', originalSupport.join(', ') || 'none', support.join(', ') || 'none', 'Retrieval scores and the claim stay fixed when a source is removed. Repeating the query in source A does not replace the missing policy clause.')}<p class="tl-result">${support.length ? `Fixture evidence supports this claim: ${escape(support.join(', '))}.` : 'Not supported by the packed evidence.'}</p></div></div>` +
      `<p class="ml-note">${claim.support.length && !support.length ? 'The needed source is absent: an evidence-access failure.' : !claim.support.length ? 'No source here supports the latency guarantee. A citation would not fix that.' : 'Synthetic documents; manually audited support labels, not an automatic factuality guarantee.'}</p>` +
      options(button('query', 'cache memory eviction', 'Query: eviction', s.query === 'cache memory eviction') + button('query', 'active request pinned', 'Query: pinned entries', s.query === 'active request pinned') + [1, 2, 3].map(k => button('topk', k, `Top ${k}`, s.topk === k)).join('') + CLAIMS.map((c, i) => button('claim', i, ['Claim: eviction policy', 'Claim: pinned entries', 'Claim: latency guarantee'][i], s.claim === i)).join('')) +
      calculation(table(['Document', 'Term-count cosine', 'Context'], ranked.map(doc => [button('doc', doc.id, `${doc.id}: ${doc.title}`, s.doc === doc.id), code(f(doc.score)), packed.includes(doc.id) ? 'included' : 'not packed'])) + `<p>Source ${escape(selected.id)}: ${selected.dot} / (${f(selected.qnorm)} x ${f(selected.dnorm)}) = ${f(selected.score)}. evict / evicted / eviction share one term. Cosine is not a probability of truth.</p><p>Support is an explicit intersection of manually audited source IDs with packed IDs. No LLM or automated entailment model runs in this fixture.</p>`);
  }

  function initial(kind) {
    const states = {
      tokenization: { step: 1, word: 'lowest' }, embeddings: { step: 0, metric: 'dot', doc: 0, coordinate: 0 },
      positional: { mode: 'rope', swapped: false, key: 2 }, 'transformer-block': { step: 3, token: 3, changed: false, inspect: false },
      attention: { step: 2, query: 3, key: 0, muted: false }, 'kv-cache': { step: 1, query: 2, key: 2 },
      rag: { query: 'cache memory eviction', topk: 2, excluded: [], doc: 'B', claim: 0 },
    };
    return states[kind] ? { ...states[kind], inspect: false, options: false } : {};
  }
  const limits = { tokenization: 8, embeddings: 4, positional: 0, 'transformer-block': 7, attention: 3, 'kv-cache': 4, rag: 0 };
  const integer = (value, max) => Math.min(max, Math.max(0, Number.isFinite(Number(value)) ? Math.trunc(Number(value)) : 0));
  function reduce(kind, state, action, value) {
    const s = { ...state };
    if (action === 'reset') return initial(kind);
    if (action === 'inspect') return { ...s, inspect: !s.inspect };
    if (action === 'options') return { ...s, options: !s.options };
    if (action === 'next' || action === 'previous') {
      s.step = integer(s.step + (action === 'next' ? 1 : -1), limits[kind]);
      if (kind === 'kv-cache') { s.query = s.step + 1; s.key = s.query; }
    } else if (kind === 'tokenization' && action === 'word' && ['lower', 'newest', 'lowest'].includes(value)) s.word = value;
    else if (kind === 'embeddings' && action === 'metric' && ['dot', 'cosine'].includes(value)) s.metric = value;
    else if (kind === 'embeddings' && action === 'doc') s.doc = integer(value, 2);
    else if (kind === 'embeddings' && action === 'coordinate') s.coordinate = integer(value, 1);
    else if (kind === 'attention' && action === 'mute-value') s.muted = !s.muted;
    else if (kind === 'positional' && action === 'mode' && ['none', 'absolute', 'rope'].includes(value)) s.mode = value;
    else if (kind === 'positional' && action === 'swap') { s.swapped = !s.swapped; s.key = 2 - s.key; }
    else if (kind === 'transformer-block' && action === 'change') s.changed = !s.changed;
    else if (kind === 'transformer-block' && action === 'token') s.token = integer(value, 3);
    else if (kind === 'transformer-block' && action === 'stage') s.step = integer(value, 7);
    else if (['attention', 'kv-cache'].includes(kind) && action === 'cell') {
      const [q, k] = String(value).split(',');
      s.query = integer(q, kind === 'attention' ? 3 : s.step + 1);
      s.key = integer(k, kind === 'attention' ? 3 : s.step + 1);
    } else if (kind === 'attention' && action === 'query') s.query = integer(value, 3);
    else if (['attention', 'positional', 'kv-cache'].includes(kind) && action === 'key') s.key = integer(value, kind === 'positional' ? 2 : kind === 'attention' ? 3 : s.step + 1);
    else if (kind === 'rag') {
      if (action === 'query' && ['cache memory eviction', 'active request pinned'].includes(value)) { s.query = value; s.excluded = []; }
      if (action === 'topk') s.topk = Math.max(1, integer(value, 3));
      if (action === 'doc' && EVIDENCE.some(d => d.id === value)) s.doc = value;
      if (action === 'claim') s.claim = integer(value, 2);
      if (action === 'exclude' && EVIDENCE.some(d => d.id === value)) s.excluded = s.excluded.includes(value) ? s.excluded.filter(id => id !== value) : [...s.excluded, value];
    }
    return s;
  }
  const renderers = { tokenization: renderBPE, embeddings: renderEmbeddings, positional: renderPosition, 'transformer-block': renderBlock, attention: renderAttention, 'kv-cache': renderCache, rag: renderRag };
  function render(kind, state) {
    let html = renderers[kind] ? renderers[kind](state) : '';
    if (state.inspect) html = html.replaceAll('<details class="tl-details tl-calculation">', '<details class="tl-details tl-calculation" open>');
    if (state.options) html = html.replaceAll('<details class="tl-details tl-options">', '<details class="tl-details tl-options" open>');
    return `<div class="tl-lab">${html}</div>`;
  }

  const makeContent = (title, summary, what, why, interview, details, math, snippet, prompt, answers) => ({
    title, summary, what, why, interview, details, math,
    code: { title: 'Annotated implementation sketch', lang: 'python', snippet },
    quiz: { prompt, options: answers.map(([text, correct, explanation]) => ({ text, correct, explanation })) },
    controls: [], presets: [], geometry: null,
  });
  const content = {
    tokenization: makeContent('Tokenization: learn a merge table, then apply it', 'Count adjacent pairs in a weighted corpus. Each actual merge changes sequence length and vocabulary.',
      'The trace separates tokenizer training from inference: corpus statistics choose merge ranks; a new word only applies the frozen ranks.',
      'Sequence length consumes attention and KV-cache budget before any reasoning happens. A changed tokenizer also changes token IDs and the embedding/output vocabulary.',
      'BPE greedily grows a reusable subword vocabulary. Inference applies learned ranks, not new pair-frequency estimates.',
      ['Character-base, word-bounded BPE here; production byte-level tokenizers also define normalization, pretokenization, special tokens, and byte handling.', 'The unseen word uses known characters. This fixture does not claim arbitrary Unicode coverage.'],
      { title: 'Weighted pair counts', formula: ['C(a,b)=\\sum_w f(w)\\sum_i \\mathbf{1}[(t_i,t_{i+1})=(a,b)]', '(a^*,b^*)=\\arg\\max_{a,b} C(a,b)'], note: 'Apply each chosen pair left-to-right without overlap. A merge adds a vocabulary entry; it does not delete base tokens.', annotations: [['f(w)', 'Training word multiplicity', 'newest appears 6 times.'], ['C(e,s)', 'Weighted adjacent occurrences', '6 in newest plus 3 in widest = 9.'], ['t_i', 'Current pieces, not always characters', 'After e + s, the next pair may be es + t.']] },
      'pieces = [list(word) for word in words]\nfor rank in range(8):\n    counts = weighted_adjacent_counts(pieces, frequencies)\n    pair = max_with_lexical_tiebreak(counts)\n    pieces = [merge_nonoverlapping(p, pair) for p in pieces]\n    merges.append(pair)\n\n# Inference: freeze the table; never recount the new word.\ntokens = list("lowest")\nfor pair in merges:\n    tokens = merge_nonoverlapping(tokens, pair)',
      'Suppose "lowest" becomes much more common after deployment, but the tokenizer is not retrained. Does its segmentation change?', [['No: the same frozen merge ranks still apply', true, 'Runtime word frequency does not modify the merge table. A retrained tokenizer could choose different ranks.'], ['Yes: its frequent adjacent pairs automatically become new tokens', false, 'That would silently change the vocabulary and token IDs at inference.'], ['Yes: attention replaces the least useful subword', false, 'Attention consumes the tokenizer output; it does not rewrite the frozen tokenizer.']]),
    embeddings: makeContent('Embeddings: inspect coordinates before assigning meaning', 'Look up token vectors, pool a query, compare dot product with cosine, and take actual contrastive gradient steps.',
      'Embedding lookup selects a parameter row. Retrieval geometry becomes useful through the training objective and data, not through named coordinate axes.',
      'A ranking system must keep encoder, normalization, similarity function, and index conventions consistent. Dot product can reward large norms; cosine removes that factor.',
      'Token embeddings are input parameters, not automatically sentence-level retrieval embeddings. Contrastive supervision can shape representation geometry.',
      ['Two dimensions and hand-specified document vectors make the arithmetic visible; no semantic quality claim is measured.', 'The update directly optimizes q with documents frozen. A trained encoder instead receives these gradients through its parameters.'],
      { title: 'Score and learn', formula: ['q=\\frac{E[0]+E[1]}{2}', 's_j=q^T d_j,\\quad L=-\\log\\frac{e^{s_0}}{\\sum_j e^{s_j}}', '\\nabla_q L=\\sum_j p_jd_j-d_0'], note: 'The update uses dot-product logits regardless of which metric you inspect for ranking. Cosine divides by both vector norms.', annotations: [['E[id]', 'A trainable matrix row selected by token ID', 'E[cache] = [1, 0].'], ['d_0', 'Explicitly labeled positive example', 'Document 0 is [1, 1]; its label is supplied, not inferred.'], ['p_j', 'Softmax over the three candidate logits', 'These are training probabilities within this candidate set, not factual confidence.']] },
      'q = (E[cache_id] + E[memory_id]) / 2\n# Fixture: documents D stay fixed. Target index is 0.\nlogits = D @ q\np = softmax(logits)\ngrad_q = p @ D - D[0]\nq = q - 0.2 * grad_q\n# Retrieval metric is a separate choice.\ncosines = (D @ q) / (norm(D, axis=1) * norm(q))',
      'Keep q fixed and double only the catalogue vector. What changes?', [['Its dot score doubles; its cosine score stays the same', true, 'Positive scaling doubles both the dot product and the candidate norm, which cancel in cosine.'], ['Both scores double', false, 'Cosine divides by the candidate norm as well.'], ['The catalogue becomes a better-supported answer', false, 'Changing vector magnitude says nothing about whether its text supports the claim.']]),
    positional: makeContent('Position: break the right symmetry', 'Swap two tokens while tracking the same query identity. Compare no position, additive sin/cos, and an actual rotary pair.',
      'Unmasked self-attention without position is permutation equivariant: reorder inputs and output rows reorder in the same way. Pooling can make the result invariant.',
      'RoPE changes Q/K matching by relative displacement without rotating V. This affects what information is mixed, not just a label attached to a token.',
      'Distinguish equivariance from invariance, and positional signals from the order structure already exposed by a causal mask.',
      ['The comparison removes the causal mask to isolate positional symmetry.', 'One rotary frequency or two additive coordinates are not a long-context evaluation. Real RoPE uses many frequency pairs; extrapolation is not guaranteed.'],
      { title: 'Actual position transforms', formula: ['p_i=[\\sin i,\\cos i],\\quad x_i^\\prime=x_i+p_i', 'R(\\theta)=\\begin{bmatrix}\\cos\\theta&-\\sin\\theta\\\\\\sin\\theta&\\cos\\theta\\end{bmatrix}', '(R(i)q)^T(R(j)k)=q^TR(j-i)k'], note: 'This fixture uses theta = position radians and identity projections. For RoPE, the value vectors remain unrotated.', annotations: [['R(i)', 'A norm-preserving 2-D rotation', 'At position 1, [1, 0] becomes [cos(1), sin(1)].'], ['j-i', 'Relative key/query displacement', 'Shifting both positions equally leaves their rotary dot product unchanged.'], ['P', 'Permutation of token rows', 'Without position or mask, attention(PX) = P attention(X).']] },
      'import numpy as np\n\ndef rotate(x, pos):\n    c, s = np.cos(pos), np.sin(pos)\n    return [c*x[0] - s*x[1], s*x[0] + c*x[1]]\n\nX = np.asarray(X, dtype=float)  # [tokens, 2]\nQ = np.array([rotate(x, i) for i, x in enumerate(X)])\nK = np.array([rotate(x, i) for i, x in enumerate(X)])\nV = X  # Identity projections; RoPE does not rotate V.\nscores = Q @ K.T / np.sqrt(2)\ne = np.exp(scores - scores.max(axis=-1, keepdims=True))\nY = (e / e.sum(axis=-1, keepdims=True)) @ V',
      'Keep q and k fixed and shift both RoPE positions forward by 5. Does their match score change?', [['No: their relative displacement is unchanged', true, 'R(i+5)^T R(j+5) = R(j-i). This concerns the pair score, not a guarantee about long-context model quality.'], ['Yes: both rotations necessarily reduce the score', false, 'A shared rotation preserves the dot product.'], ['The key value vector is rotated by 5 more radians', false, 'RoPE transforms Q/K, not V.']]),
    'transformer-block': makeContent('Transformer block: trace two residual updates', 'Inspect every small tensor in a full pre-norm block, then change one input token to locate cross-token communication.',
      'The attention branch normalizes, projects, mixes across tokens, and returns to residual width. A second normalized branch runs a shared MLP independently on each token.',
      'Shape and dependency reasoning make block implementation, debugging, and KV-cache behavior concrete. The residual stream carries previous state past both transformations.',
      'Pre-norm means normalization is inside each residual branch. Attention communicates across positions; the MLP rewrites context already present in a row.',
      ['One head, width 3, ReLU hidden width 4, fixed weights, no bias/dropout or positional encoding. LayerNorm uses gain 1, bias 0, epsilon 1e-5.', 'These are real sublayer calculations, not a trained language model; no semantic interpretation is attached to individual coordinates.'],
      { title: 'A complete pre-norm block', formula: ['N=\\operatorname{LN}(X)', 'H=X+\\operatorname{softmax}(NW_Q(NW_K)^T/\\sqrt{2}+M)(NW_V)W_O', 'Y=H+\\operatorname{ReLU}(\\operatorname{LN}(H)W_1)W_2'], note: 'The first residual is X, the second is H. LN operates over features separately for every token.', annotations: [['X,H,Y', 'Residual-width token matrices', 'Four rows, three coordinates each.'], ['W_O', 'Maps the head result back to residual width', '2 x 3 here, required before adding to X.'], ['W_1,W_2', 'Shared per-token MLP parameters', '3 x 4 then 4 x 3; no token-axis multiplication.'], ['M', 'Causal mask', 'Future-key scores become negative infinity.']] },
      'n1 = layer_norm(X, eps=1e-5)\nQ, K, V = n1 @ WQ, n1 @ WK, n1 @ WV\nA = softmax(Q @ K.T / sqrt(2) + causal_mask, axis=-1)\nH = X + (A @ V) @ WO  # first residual\nn2 = layer_norm(H, eps=1e-5)\nY = H + relu(n2 @ W1) @ W2  # second residual\n# MLP multiplies features, never the sequence axis.',
      'Remove this block\'s attention branch entirely. Can changing only token 0 still change token 3 in this block?', [['No: the remaining LayerNorm, MLP and residual paths are token-local', true, 'The cross-token path has been removed. This assumes token 3 enters the block unchanged; earlier blocks could already have mixed context.'], ['Yes: shared MLP weights communicate between rows', false, 'Sharing parameters is not the same as reading another token\'s activations.'], ['Yes: LayerNorm averages all token rows together', false, 'LayerNorm here normalizes features within each row.']]),
    attention: makeContent('Attention: scores are not the payload', 'Select any query/key pair and inspect projections, masked softmax weights, and the weighted value contribution.',
      'Q and K determine weights. V supplies the coordinates being mixed. A large raw future score still contributes exactly zero under a causal mask.',
      'Inspecting both weights and values avoids treating an attention heatmap as a sufficient explanation of the model output.',
      'Scaled dot-product attention forms a row-normalized distribution over allowed keys, then mixes V. These weights are not factual confidence or a causal attribution guarantee.',
      ['One fixed untrained head with model width 3 and head width 2. No dropout, position encoding, or output projection in this isolated attention scene.', 'The block scene adds normalization, output projection, residual connections, and the MLP.'],
      { title: 'Project, score, mask, mix', formula: ['Q=XW_Q,\\quad K=XW_K,\\quad V=XW_V', 'a_{ij}=\\operatorname{softmax}_j(q_i^Tk_j/\\sqrt{d_k}+M_{ij})', 'o_i=\\sum_j a_{ij}v_j'], note: 'M is 0 on/below the diagonal and negative infinity above it. Stable softmax subtracts the maximum finite score.', annotations: [['d_k', 'Key/query width', 'Two coordinates, so scores divide by sqrt(2).'], ['M_{ij}', 'Disallowed future keys', 'q2 cannot read k3 even if their raw dot product is large.'], ['a_{ij}v_j', 'The actual vector contribution', 'Multiply both value coordinates by the selected weight.']] },
      'Q, K, V = X @ WQ, X @ WK, X @ WV\nscores = Q @ K.T / sqrt(K.shape[-1])\nscores[future_mask] = -inf\nweights = softmax(scores, axis=-1)\noutput = weights @ V\n# Inspect one contribution, not only a heatmap color.\ncontribution = weights[i, j] * V[j]',
      'Double one allowed value vector while keeping every query and key fixed. What changes?', [['Its weighted contribution doubles; attention weights do not change', true, 'QK determines weights. Only a_ij V_j changes; the total output need not double because other contributions stay fixed.'], ['Its attention weight doubles', false, 'Changing V does not alter the QK scores.'], ['The entire output necessarily doubles', false, 'Only one summand doubles, not the other weighted values.']]),
    'kv-cache': makeContent('KV cache: reuse exact rows, append one token', 'Step from causal prefill to incremental decode, inspect retained K/V vectors, and compare with full recomputation.',
      'A fixed causal prefix has unchanged past states at each layer. Store its keys and values; project the new token and attend over the extended prefix.',
      'Caching removes repeated prefix projections but not the growing key scan. Memory scales with sequence length, batch size, layers, KV heads, head dimension, and element size.',
      'Prefill is parallel causal processing; single-token decode appends one KV slice per layer. Changed prefixes, weights, or position conventions invalidate reuse.',
      ['One layer/head, width 2 K/V, batch 1, FP32. Counts are projected token rows, not FLOPs, benchmark latency, or a promised speedup.', 'Fixed hidden-state inputs and fixed decode tokens isolate cache arithmetic. No sampling, multi-layer execution, GQA, paging, or quantization.'],
      { title: 'Append and account', formula: ['K_{1:t}=[K_{1:t-1};x_tW_K],\\quad V_{1:t}=[V_{1:t-1};x_tW_V]', 'o_t=\\operatorname{softmax}(q_tK_{1:t}^T/\\sqrt{d_k})V_{1:t}', '\\text{KV bytes}=2BTLH_{kv}d_hb'], note: 'The cache contains only allowed positions for one-token decode. Prefill and multi-token chunks still require causal masking.', annotations: [['2', 'Store both keys and values', 'Two arrays, not a query cache.'], ['T', 'Current cached sequence length', 'Prompt 2 plus one decode token gives 3.'], ['B,L,H_{kv},d_h,b', 'Batch, layers, KV heads, head width, bytes per element', '1 x 1 x 1 x 2 x 4; total at T=3 is 48 bytes.']] },
      'import numpy as np\n\n# prompt: [T, 3]; matrices: [3, 2].\n# Prefill masks attention scores, not K/V projections.\ncache_k, cache_v = prompt @ WK, prompt @ WV\n# Preserve the token axis for append and row-wise softmax.\nx = np.asarray(new_x, dtype=float).reshape(1, 3)\nq = x @ WQ  # [1, 2]\ncache_k = np.concatenate([cache_k, x @ WK], axis=0)\ncache_v = np.concatenate([cache_v, x @ WV], axis=0)\nscores = q @ cache_k.T / np.sqrt(2)\ne = np.exp(scores - scores.max(axis=-1, keepdims=True))\noutput = (e / e.sum(axis=-1, keepdims=True)) @ cache_v\n# No future rows exist in this single-token decode cache.',
      'Keep token count, head count and FP32 fixed, but double the K/V head width. What happens to cache storage?', [['It doubles: each cached key and value has twice as many coordinates', true, 'The factor d_h doubles in 2 B T L H_kv d_h b. This does not by itself establish a latency multiplier.'], ['It stays fixed because old tokens are reused', false, 'Reuse avoids computation; the retained vectors still consume memory.'], ['It quadruples because both K and V are stored', false, 'The factor of two for K and V was already included before the width change.']]),
    rag: makeContent('RAG: audit evidence, not a confidence dial', 'Retrieve concrete chunks, choose which enter context, and check each candidate answer claim against exact source text.',
      'Relevance ranking selects context. Claim support is a separate relationship between a statement and the supplied evidence. A relevant or cited document can fail to support an answer.',
      'Diagnose evidence access separately from context use: did the supporting chunk enter the prompt, and does the answer stay within what it says?',
      'Retrieval quality, evidence coverage, and claim-level grounding need separate evaluation. Retrieval and citations alone do not guarantee correctness.',
      ['Synthetic local documents; transparent term-count cosine rather than a production embedding encoder or reranker.', 'Support labels are manually specified for these three claims. No generator or automatic entailment detector is being simulated.'],
      { title: 'Rank first; support separately', formula: ['s(q,d)=\\frac{\\operatorname{tf}(q)^T\\operatorname{tf}(d)}{\\lVert\\operatorname{tf}(q)\\rVert\\lVert\\operatorname{tf}(d)\\rVert}', 'C=\\operatorname{pack}(\\operatorname{topk}_d s(q,d))'], note: 'Neither s nor membership in C is a probability that an answer claim is true. The fixture support check is an explicit source-ID intersection.', annotations: [['tf', 'Term-count vector after minimal normalization', 'evict, evicted, and eviction map to one term.'], ['C', 'Only the chunks actually included in context', 'Removing B can remove the only evidence for the LRU claim.'], ['support', 'Manually audited claim/source relation', 'No document supports the p99 < 20 ms guarantee.']] },
      'ranked = sorted(docs, key=lambda d: cosine(tf(query), tf(d)), reverse=True)\ncontext = pack(ranked[:k])\n# Production: generate conditioned on context, then evaluate.\n# This lesson instead exposes three fixed candidate claims.\nfor claim in fixture_claims:\n    support = manual_support_ids[claim] & context.ids\n    show_claim_and_exact_evidence(claim, support)\n# A citation marker alone is not an entailment check.',
      'B and C are both in context, then B is removed. Which claim still has evidence?', [['Pinned entries are protected; C states that rule explicitly', true, 'The LRU eviction policy loses its only supporting source B. Support must be evaluated claim by claim.'], ['Every claim loses support together', false, 'C independently supports the pinned-entry claim.'], ['The latency guarantee becomes supported by C', false, 'C states an eviction restriction, not a p99 latency bound.']]),
  };

  const api = { dot, add, scale, norm, cosine, linear, softmax, layerNorm, CORPUS, mergePair, pairCounts, trainBPE, encodeBPE, bpeEffect, LOOKUP, DOCUMENTS, contrastive, embeddingTrace, rotate, positionalTrace, INPUTS, WQ, WK, WV, WO, W1, W2, attend, attentionTrace, attentionValueExperiment, mlp, transformerBlock, BUFFER_INPUT, blockExperiment, blockStages, appendCache, cacheTrace, EVIDENCE, CLAIMS, terms, retrieve, claimSupport, supportSpan, initial, reduce, render, content, escape };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window !== 'undefined') {
    if (!window.AtelierLab) throw new Error('Load atelier/lab-core.js before transformer-labs.js');
    window.AtelierLab.createModule({ id: 'transformer-labs', content, initial, render, reduce });
  }
})();
