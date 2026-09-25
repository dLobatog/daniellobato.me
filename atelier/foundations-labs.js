/* Exact, inspectable probability and linear-algebra lessons. No DOM dependency in Node. */
(() => {
  'use strict';

  const sum = xs => xs.reduce((a, b) => a + b, 0);
  const dot = (a, b) => sum(a.map((x, i) => x * b[i]));
  const norm = a => Math.sqrt(dot(a, a));
  const scale = (a, k) => a.map(x => x * k);
  const add = (a, b) => a.map((x, i) => x + b[i]);
  const subtract = (a, b) => a.map((x, i) => x - b[i]);
  const unit = a => norm(a) === 0 ? null : scale(a, 1 / norm(a));
  const cosine = (a, b) => norm(a) * norm(b) === 0 ? null : dot(a, b) / (norm(a) * norm(b));
  const transpose = A => A[0].map((_, j) => A.map(row => row[j]));
  const matvec = (A, x) => A.map(row => dot(row, x));
  const multiply = (A, B) => A.map(row => transpose(B).map(col => dot(row, col)));
  const outer = (a, b) => a.map(x => b.map(y => x * y));
  const matrixAdd = (A, B) => A.map((row, i) => add(row, B[i]));
  const matrixSubtract = (A, B) => A.map((row, i) => subtract(row, B[i]));
  const frobenius = A => norm(A.flat());
  const det = A => A[0][0] * A[1][1] - A[0][1] * A[1][0];

  function normalize(weights) {
    if (!weights.length || weights.some(x => !Number.isFinite(x) || x < 0) || sum(weights) <= 0) throw new RangeError('Weights must be finite, nonnegative, with positive total.');
    const total = sum(weights);
    return weights.map(x => x / total);
  }
  function softmax(logits) {
    if (!logits.length || logits.some(x => !Number.isFinite(x))) throw new RangeError('Finite logits required.');
    const max = Math.max(...logits);
    return normalize(logits.map(z => Math.exp(z - max)));
  }
  function probability(p) {
    if (!p.length || p.some(x => !Number.isFinite(x) || x < 0 || x > 1) || Math.abs(sum(p) - 1) > 1e-10) throw new RangeError('Probabilities must sum to one.');
  }
  function expectedRisk(p, losses) {
    probability(p);
    if (p.length !== losses.length || losses.some(x => !Number.isFinite(x))) throw new RangeError('One finite loss per outcome required.');
    const contributions = p.map((x, i) => x * losses[i]);
    const mean = sum(contributions);
    return { mean, contributions, variance: dot(p, losses.map(x => (x - mean) ** 2)) };
  }
  function information(p, q) {
    probability(p); probability(q);
    if (p.length !== q.length) throw new RangeError('Distribution lengths must match.');
    const terms = p.map((pi, i) => ({
      surprise: q[i] === 0 ? Infinity : -Math.log2(q[i]),
      entropy: pi === 0 ? 0 : -pi * Math.log2(pi),
      crossEntropy: pi === 0 ? 0 : -pi * Math.log2(q[i]),
      kl: pi === 0 ? 0 : pi * Math.log2(pi / q[i]),
    }));
    return { terms, entropy: sum(terms.map(t => t.entropy)), crossEntropy: sum(terms.map(t => t.crossEntropy)), kl: sum(terms.map(t => t.kl)) };
  }
  function bayesCounts(prior, recall = .9, fpr = .02, population = 10000, positive = true) {
    if ([prior, recall, fpr].some(x => !Number.isFinite(x) || x < 0 || x > 1) || !Number.isFinite(population) || population <= 0) throw new RangeError('Invalid population or probability.');
    const spam = population * prior, ham = population - spam;
    const tp = spam * recall, fp = ham * fpr, fn = spam - tp, tn = ham - fp;
    const numerator = positive ? tp : fn, other = positive ? fp : tn;
    return { spam, ham, tp, fp, fn, tn, numerator, other, evidence: numerator + other, posterior: numerator + other === 0 ? null : numerator / (numerator + other) };
  }
  function binaryLoss(y, q) {
    if (![0, 1].includes(y) || !Number.isFinite(q) || q < 0 || q > 1) throw new RangeError('Binary label and probability required.');
    return { log: -Math.log(y ? q : 1 - q), square: (q - y) ** 2, logGradient: q - y, squareGradient: 2 * (q - y) * q * (1 - q) };
  }
  function binaryRisk(p, q) {
    if (!Number.isFinite(p) || p < 0 || p > 1) throw new RangeError('Invalid true probability.');
    const a = binaryLoss(1, q), b = binaryLoss(0, q);
    return { log: (p === 0 ? 0 : p * a.log) + (p === 1 ? 0 : (1 - p) * b.log), square: p * a.square + (1 - p) * b.square, logGradient: q - p, squareGradient: 2 * (q - p) * q * (1 - q) };
  }
  function lossStep(target, q, rate = 2) {
    if (!(q > 0 && q < 1) || !Number.isFinite(rate) || rate < 0) throw new RangeError('Finite nonnegative rate and interior probability required.');
    const loss = binaryRisk(target, q), z = Math.log(q / (1 - q));
    const zLog = z - rate * loss.logGradient, zSquare = z - rate * loss.squareGradient;
    const sigmoid = x => 1 / (1 + Math.exp(-x));
    return { z, zLog, zSquare, qLog: sigmoid(zLog), qSquare: sigmoid(zSquare), rate, ...loss };
  }
  function categorical(p, u) {
    probability(p);
    if (!Number.isFinite(u) || u < 0 || u >= 1) throw new RangeError('Uniform draw must lie in [0,1).');
    let cumulative = 0;
    for (let i = 0; i < p.length; i++) { cumulative += p[i]; if (u < cumulative) return i; }
    return p.length - 1;
  }
  function sampleDistribution(p, n, seed = 42) {
    probability(p);
    const counts = p.map(() => 0);
    let last = null;
    for (let i = 0; i < n; i++) {
      seed = (Math.imul(1664525, seed) + 1013904223) >>> 0;
      const u = seed / 4294967296;
      last = { u, outcome: categorical(p, u) };
      counts[last.outcome]++;
    }
    return { counts, seed, last };
  }
  function densityMass(a, b) {
    if (a < 0 || b > 1 || a > b) throw new RangeError('Interval must lie in [0,1].');
    return b * b - a * a;
  }
  function eigenSymmetric(A) {
    if (A.length !== 2 || A.some(r => r.length !== 2 || r.some(x => !Number.isFinite(x))) || Math.abs(A[0][1] - A[1][0]) > 1e-10) throw new RangeError('A real symmetric 2 by 2 matrix is required.');
    const a = A[0][0], b = A[0][1], d = A[1][1];
    const angle = .5 * Math.atan2(2 * b, a - d);
    const gap = Math.hypot(a - d, 2 * b);
    return { values: [(a + d + gap) / 2, (a + d - gap) / 2], vectors: [[Math.cos(angle), Math.sin(angle)], [-Math.sin(angle), Math.cos(angle)]] };
  }
  function reconstructEigen(A) {
    const e = eigenSymmetric(A);
    return matrixAdd(outer(scale(e.vectors[0], e.values[0]), e.vectors[0]), outer(scale(e.vectors[1], e.values[1]), e.vectors[1]));
  }
  function svd2(A) {
    if (A.length !== 2 || A.some(row => row.length !== 2 || row.some(x => !Number.isFinite(x)))) throw new RangeError('A finite 2 by 2 matrix is required.');
    const magnitude = Math.max(...A.flat().map(Math.abs));
    if (magnitude === 0) return { U: [[1, 0], [0, 1]], singular: [0, 0], Vt: [[1, 0], [0, 1]] };
    // Scale before forming A^T A; rank must not depend on the matrix's units.
    const B = A.map(row => row.map(x => x / magnitude));
    const e = eigenSymmetric(multiply(transpose(B), B));
    const first = Math.sqrt(Math.max(0, e.values[0])), determinant = det(B);
    // sigma1*sigma2 = |det B| avoids cancellation in the smaller eigenvalue.
    const singular = [first * magnitude, Math.abs(determinant) / first * magnitude];
    const u0 = unit(matvec(B, e.vectors[0]));
    // V has positive orientation. Preserve det(B)'s sign without dividing by tiny sigma2.
    const u1 = scale([-u0[1], u0[0]], determinant < 0 ? -1 : 1);
    return { U: transpose([u0, u1]), singular, Vt: e.vectors };
  }
  function reconstructSvd(decomposition, rank = 2) {
    const { U, singular, Vt } = decomposition;
    return multiply(U.map(row => row.map((x, j) => j < rank ? x * singular[j] : 0)), Vt);
  }
  function powerStep(A, x, n) {
    let v = unit(x);
    for (let i = 0; i < n && v; i++) v = unit(matvec(A, v));
    return v;
  }

  const esc = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = (x, n = 3) => x === null ? 'undefined' : !Number.isFinite(x) ? 'infinity' : Number(x.toFixed(n)).toString();
  const pct = x => `${fmt(100 * x, 1)}%`;
  const vec = x => `[${x.map(v => fmt(v)).join(', ')}]`;
  const btn = (action, value, label, selected = false) => `<button type="button" data-action="${esc(action)}" data-value="${esc(value)}" aria-pressed="${selected}">${esc(label)}</button>`;
  const primary = (action, value, label) => `<div class="ml-controls"><button class="fl-primary" type="button" data-action="${esc(action)}" data-value="${esc(value)}">${esc(label)}</button></div>`;
  const calculation = (html, key = 'calculation') => `<details class="fl-calculation" data-detail-key="${esc(key)}"><summary>${key === 'inspector-calculation' ? 'Inspect the calculation' : 'Inspect the underlying data'}</summary>${html}</details>`;
  const settings = html => `<details class="fl-settings"><summary>Change the example</summary>${html}</details>`;
  const controls = (label, html) => `<div class="ml-controls" role="group" aria-label="${esc(label)}">${html}</div>`;
  const stat = (label, value) => `<div class="ml-stat"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`;
  const note = text => `<p class="ml-note">${esc(text)}</p>`;
  const equation = (label, terms, result) => `<p class="fl-equation" data-equation="${esc(label)}"><span class="fl-equation-label">${esc(label)}</span><span>${terms.map(t => `<${t.active ? 'mark' : 'span'} class="${t.active ? 'fl-term' : 'fl-term-context'}" data-term="${esc(t.text)}">${esc(t.text)}</${t.active ? 'mark' : 'span'}>`).join(' ')} = <strong>${esc(result)}</strong></span></p>`;
  const consequence = (label, before, after, reason) => `<p class="fl-consequence" data-before="${esc(before)}" data-after="${esc(after)}"><span>${esc(label)}: <span class="fl-before">${esc(before)}</span> <span aria-hidden="true">&rarr;</span> <strong>${esc(after)}</strong>.</span> ${esc(reason)}</p>`;
  const heading = (kicker, question) => `<header class="ml-heading"><p class="ml-kicker">${esc(kicker)}</p><h3 class="ml-question">${esc(question)}</h3></header>`;
  const inspector = (title, lines, data = '') => `<aside class="ml-inspector" aria-live="polite"><h4>${esc(title)}</h4><p>${esc(lines[0])}</p>${lines.length > 1 || data ? calculation(data + lines.slice(1).map(line => `<p>${esc(line)}</p>`).join(''), 'inspector-calculation') : ''}</aside>`;
  function table(headers, rows, caption = '') {
    return `<div class="ml-table"><table>${caption ? `<caption>${esc(caption)}</caption>` : ''}<thead><tr>${headers.map(h => `<th scope="col">${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map((x, i) => i === 0 ? `<th scope="row">${x}</th>` : `<td>${x}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  }
  function matrix(A, label) {
    return `<div class="fl-matrix"><span>${esc(label)}</span><div role="img" aria-label="${esc(`${label}: ${A.map(vec).join('; ')}`)}">${A.flat().map(x => `<span>${esc(fmt(x))}</span>`).join('')}</div></div>`;
  }
  function navigation(step, labels, actions = []) {
    return `<nav class="ml-step-nav fl-nav" aria-label="Computation trace"><button type="button" data-action="previous" ${step === 0 ? 'disabled' : ''}>Previous</button><span>${step + 1}/${labels.length}: ${esc(labels[step])}</span><button class="fl-primary" type="button" data-action="${step === labels.length - 1 ? 'restart' : 'next'}">${step === labels.length - 1 ? 'Start again' : esc(actions[step] || 'Next')}</button></nav>`;
  }
  const colors = ['blue', 'orange', 'green'];
  // Fixed plot coordinates and HTML legends keep SVG labels out of crowded geometry.
  function plane({ arrows = [], points = [], shape = [], ghostShape = [], range = 4, label = 'Vector geometry', unitCircle = false, gridMap = null }) {
    const c = v => [150 + v[0] * 112 / range, 150 - v[1] * 112 / range];
    const line = (a, b, cls = 'fl-axis') => { const p = c(a), q = c(b); return `<line class="${cls}" x1="${p[0]}" y1="${p[1]}" x2="${q[0]}" y2="${q[1]}"/>`; };
    let grid = '';
    if (gridMap) for (let k = -2; k <= 2; k++) {
      grid += line(matvec(gridMap, [-2, k]), matvec(gridMap, [2, k]), 'fl-grid-line');
      grid += line(matvec(gridMap, [k, -2]), matvec(gridMap, [k, 2]), 'fl-grid-line');
    }
    return `<svg class="fl-plane" viewBox="0 0 300 300" role="group" aria-label="${esc(label)}">
      ${grid}${line([-range, 0], [range, 0])}${line([0, -range], [0, range])}
      ${unitCircle ? `<circle class="fl-unit" cx="150" cy="150" r="${112 / range}"/>` : ''}
      ${ghostShape.length ? `<polygon class="fl-ghost-shape" points="${ghostShape.map(v => c(v).join(',')).join(' ')}"/>` : ''}
      ${shape.length ? `<polygon class="fl-shape" points="${shape.map(v => c(v).join(',')).join(' ')}"/>` : ''}
      ${arrows.map((a, i) => { const from = c(a.from || [0, 0]), to = c(a.v), angle = Math.atan2(to[1] - from[1], to[0] - from[0]); const back = [to[0] - 9 * Math.cos(angle), to[1] - 9 * Math.sin(angle)]; return `<g class="fl-${a.color || colors[i % 3]}"><line class="fl-vector ${a.dashed ? 'fl-dashed' : ''}" x1="${from[0]}" y1="${from[1]}" x2="${to[0]}" y2="${to[1]}"/>${norm(subtract(a.v, a.from || [0, 0])) > 1e-10 ? `<path class="fl-arrowhead" d="M ${to[0]},${to[1]} L ${back[0] + 4 * Math.sin(angle)},${back[1] - 4 * Math.cos(angle)} L ${back[0] - 4 * Math.sin(angle)},${back[1] + 4 * Math.cos(angle)} Z"/>` : ''}</g>`; }).join('')}
      ${points.map(p => { const xy = c(p.v); return `<g role="button" tabindex="0" data-action="${esc(p.action)}" data-value="${esc(p.value)}" aria-label="${esc(`${p.label}: ${vec(p.v)}`)}" aria-pressed="${Boolean(p.selected)}" class="fl-point fl-${p.color || 'blue'} ${p.selected ? 'fl-point-selected' : ''}"><circle class="fl-hit" cx="${xy[0]}" cy="${xy[1]}" r="22"/><circle class="fl-point-dot" cx="${xy[0]}" cy="${xy[1]}" r="6"/></g>`; }).join('')}
    </svg><p class="fl-axis-caption">Horizontal: coordinate 1. Vertical: coordinate 2. Both axes: -${range} to ${range}.</p>`;
  }
  const legend = items => `<div class="fl-legend">${items.map(([color, text]) => `<span class="fl-${esc(color)}"><i></i>${esc(text)}</span>`).join('')}</div>`;

  const distributionCases = { peaked: [2, 1, 0], flat: [0, 0, 0], shifted: [0, 1, 2] };
  const tokenNames = ['retrieval', 'ranking', 'generation'];
  const entropyNames = ['cup', 'mug', 'glass'];
  const continuationCounts = [60, 30, 10];
  const trueTokens = normalize(continuationCounts);
  const tokenModels = { matched: [.6, .3, .1], uniform: [1 / 3, 1 / 3, 1 / 3], confident: [.9, .09, .01] };
  const populations = { train: [.8, .2], shifted: [.3, .7] };
  const modelLosses = { A: [.1, .8], B: [.25, .35] };
  const embeddings = { A: [.9, .6], B: [2, -.1], C: [-.5, 1], zero: [0, 0] };
  const maps = { shear: [[1, 1], [0, 1]], rotate: [[0, -1], [1, 0]], collapse: [[1, 1], [1, 1]] };
  const eigenCases = { covariance: [[2, 1], [1, 2]], saddle: [[1, 2], [2, 1]], isotropic: [[2, 0], [0, 2]] };
  const svdCases = { mixed: [[3, 1], [0, 1]], balanced: [[2, 0], [0, 2]], rankone: [[3, 0], [4, 0]] };
  const distributionLogits = s => distributionCases[s.scene].map((z, i) => z + (s.boosts?.[i] || 0));

  function initial(kind) {
    const states = {
      distribution: { mode: 'categorical', scene: 'peaked', selected: 0, boosts: [0, 0, 0], probe: .8, samples: 0, counts: [0, 0, 0], seed: 42, last: null },
      expectation: { population: 'train', model: 'A', selected: 1 },
      bayes: { prior: .01, positive: true, step: 0 },
      entropy: { model: 'matched', selected: 2 },
      loss: { q: .01, y: 1, mode: 'observed', trained: false },
      vectors: { item: 'B', normalized: false },
      'dot-products': { item: 'A', normalized: false, query: 'first' },
      'matrix-multiply': { scene: 'shear', selected: 2, step: 1, row: 0 },
      eigen: { scene: 'covariance', selected: 2, step: 0 },
      svd: { scene: 'mixed', rank: 1, step: 3, selected: 0 },
    };
    return states[kind];
  }
  function reduce(kind, state, action, value) {
    if (action === 'reset') return initial(kind);
    const s = { ...state };
    const n = Number(value);
    if (action === 'restart' && s.step !== undefined) s.step = 0;
    if ((action === 'next' || action === 'previous') && Number.isInteger(s.step)) s.step = Math.max(0, Math.min(kind === 'bayes' ? 2 : kind === 'eigen' ? 5 : 3, s.step + (action === 'next' ? 1 : -1)));
    if (kind === 'distribution') {
      if (action === 'mode' && ['categorical', 'density'].includes(value)) { s.mode = value; s.selected = 0; }
      if (action === 'scene' && distributionCases[value]) Object.assign(s, initial(kind), { scene: value });
      if (action === 'select' && Number.isInteger(n) && n >= 0 && n < (s.mode === 'density' ? 4 : 3)) s.selected = n;
      if (action === 'boost') {
        s.boosts = (s.boosts || [0, 0, 0]).map((v, i) => i === s.selected ? 1 - v : v);
        s.samples = 0; s.counts = [0, 0, 0]; s.last = null;
      }
      if (action === 'sample' && [1, 100, 1000].includes(n)) {
        const result = sampleDistribution(softmax(distributionLogits(s)), n, s.seed);
        s.counts = add(s.counts, result.counts); s.samples += n; s.seed = result.seed; s.last = result.last; s.probe = result.last.u;
      }
    }
    if (kind === 'expectation') {
      if (action === 'population' && populations[value]) s.population = value;
      if (action === 'model' && modelLosses[value]) s.model = value;
      if (action === 'select' && [0, 1].includes(n)) s.selected = n;
    }
    if (kind === 'bayes') {
      if (action === 'prior' && [.01, .1, .5].includes(n)) s.prior = n;
      if (action === 'evidence') s.positive = value !== 'negative';
    }
    if (kind === 'entropy') {
      if (action === 'model' && tokenModels[value]) s.model = value;
      if (action === 'select' && [0, 1, 2].includes(n)) s.selected = n;
    }
    if (kind === 'loss') {
      if (action === 'learn') s.trained = !s.trained;
      if (action === 'q' && [.01, .1, .3, .5, .7, .9, .99].includes(n)) { s.q = n; s.trained = false; }
      if (action === 'label' && [0, 1].includes(n)) { s.y = n; s.trained = false; }
      if (action === 'mode' && ['observed', 'expected'].includes(value)) { s.mode = value; s.trained = false; }
    }
    if (kind === 'vectors' || kind === 'dot-products') {
      if (action === 'item' && embeddings[value] && (kind === 'vectors' || value !== 'zero')) s.item = value;
      if (action === 'normalize') s.normalized = value === 'true';
      if (action === 'query' && ['first', 'second'].includes(value)) s.query = value;
    }
    if (kind === 'matrix-multiply') {
      if (action === 'scene' && maps[value]) s.scene = value;
      if (action === 'select' && [0, 1, 2].includes(n)) s.selected = n;
      if (action === 'row' && [0, 1].includes(n)) s.row = n;
    }
    if (kind === 'eigen') {
      if (action === 'scene' && eigenCases[value]) { s.scene = value; s.step = 0; }
      if (action === 'select' && [0, 1, 2].includes(n)) { s.selected = n; s.step = 0; }
    }
    if (kind === 'svd') {
      if (action === 'scene' && svdCases[value]) s.scene = value;
      if (action === 'rank' && [1, 2].includes(n)) { s.rank = n; s.step = 3; }
      if (action === 'select' && [0, 1, 2, 3].includes(n)) s.selected = n;
    }
    // A bounded snapshot records the actual last change, without an ever-growing history.
    const { previous: ignoredBefore, ...before } = state;
    const { previous: ignoredAfter, ...after } = s;
    if (JSON.stringify(before) !== JSON.stringify(after)) s.previous = before;
    return s;
  }

  function distributionView(s) {
    const switches = controls('Distribution type', btn('mode', 'categorical', 'Token probabilities', s.mode === 'categorical') + btn('mode', 'density', 'Continuous score density', s.mode === 'density'));
    if (s.mode === 'density') {
      const a = s.selected / 4, b = a + .25, mass = densityMass(a, b);
      return heading('Distribution / density is not probability', 'Same width, different probability?') + primary('select', (s.selected + 1) % 4, 'Move the interval') + `<div class="ml-grid"><div class="ml-stage">${equation('Selected area', [{ text: `${b}^2 - ${a}^2`, active: true }], `${fmt(mass)} probability`)}<svg viewBox="0 0 300 220" class="fl-density" role="group" aria-label="Density f(x)=2x on [0,1]; choose a quarter interval">${[0, 1, 2, 3].map(i => { const x = i / 4, y = x + .25; return `<g role="button" tabindex="0" data-action="select" data-value="${i}" aria-pressed="${s.selected === i}" aria-label="Interval ${x} to ${y}: probability ${pct(densityMass(x, y))}" class="fl-density-region ${s.selected === i ? 'fl-density-selected' : ''}"><rect class="fl-hit" x="${35 + 230 * x}" y="25" width="57.5" height="155"/><path d="M ${35 + 230 * x},180 L ${35 + 230 * x},${180 - 145 * x} L ${35 + 230 * y},${180 - 145 * y} L ${35 + 230 * y},180 Z"/></g>`; }).join('')}<path class="fl-axis" d="M35 25V180H270"/><text x="12" y="40">2</text><text x="28" y="204">0</text><text x="261" y="204">1</text><text x="55" y="30">density f(x) = 2x</text></svg>${s.previous?.mode === 'density' ? consequence('Captured mass', pct(densityMass(s.previous.selected / 4, (s.previous.selected + 1) / 4)), pct(mass), 'Width stays 0.25; density changes across the interval.') : note('Equal-width intervals capture different area, because the density slopes upward.')}</div>${inspector(`[${a}, ${b}] captures ${pct(mass)}`, ['Area counts probability, not the curve height.', `At x = ${b}, density = ${2 * b}; P(X = ${b}) = 0 for this continuous distribution.`, 'This is a normalized toy score density, not a calibration claim.'])}</div>${settings(switches)}`;
    }
    const z = distributionLogits(s), p = softmax(z), i = s.selected;
    const previous = s.previous?.mode === 'categorical' ? s.previous : null;
    const prior = previous && previous.probe === s.probe && distributionLogits(previous).some((z, j) => z !== distributionLogits(s)[j]) ? previous : null;
    const oldP = prior ? softmax(distributionLogits(prior)) : p;
    const probe = s.probe ?? .8, chosen = categorical(p, probe), oldChosen = categorical(oldP, probe);
    const before = sum(p.slice(0, i)), after = before + p[i];
    let cursor = 0;
    const intervals = p.map((pi, j) => { const left = cursor; cursor += pi; return `<g role="button" tabindex="0" data-action="select" data-value="${j}" aria-pressed="${i === j}" aria-label="${tokenNames[j]}, probability ${pct(pi)}" class="fl-${colors[j]} fl-interval ${i === j ? 'fl-interval-selected' : ''}"><rect x="${20 + 260 * left}" y="30" width="${260 * pi}" height="48"/></g>`; }).join('');
    return heading('Distribution / scores compete for probability', 'Can one score change the same draw?')
      + note('A model routes a request to retrieval, ranking, or generation. Softmax turns its three scores into route probabilities.')
      + primary('boost', '', s.boosts?.[i] ? 'Undo the score boost' : 'Raise this score')
      + `<div class="ml-grid"><div class="ml-stage">${equation(`p(${tokenNames[i]})`, [{ text: `exp(${z[i]})`, active: true }, { text: `/ (${z.map(v => `exp(${v})`).join(' + ')})` }], fmt(p[i], 4))}<svg viewBox="0 0 300 125" role="group" aria-label="Categorical inverse-CDF sampler; interval width is probability">${intervals}${prior ? [0, 1].map(j => `<path class="fl-old-boundary" d="M${20 + 260 * sum(oldP.slice(0, j + 1))} 20V85"/>`).join('') : ''}<text x="20" y="105">0</text><text x="280" y="105" text-anchor="end">1</text><path class="fl-sample-marker" d="M${20 + 260 * probe} 17v68"/></svg>${legend(tokenNames.map((name, j) => [colors[j], `${name}: ${pct(p[j])}`]))}${prior ? consequence('Same draw selects', tokenNames[oldChosen], tokenNames[chosen], `u=${fmt(probe, 4)} stays fixed; the score changes probability boundaries. Dashed lines are the previous boundaries.`) : note(`Hold sampler draw u=${fmt(probe, 4)} fixed: it selects ${tokenNames[chosen]}. Boosting one logit widens its region and shrinks the others.`)}</div>${inspector(`Selected: ${tokenNames[i]}`, [`${prior ? `Probability ${pct(oldP[i])} to ${pct(p[i])}` : `Probability ${pct(p[i])}`}; all regions sum to 100%.`, `Score ${z[i]} produces exp(score)=${fmt(Math.exp(z[i]))}; every probability shares denominator ${fmt(sum(z.map(Math.exp)))}.`, `Selected interval: [${fmt(before, 4)}, ${fmt(after, 4)}).`, s.samples ? `${s.samples} actual seeded draws: observed share ${pct(s.counts[i] / s.samples)}.` : 'The fixed draw is a controlled sampling example, not a claim that every draw equals 0.8.'], table(['Token', 'Logit', 'p', 'Draws'], p.map((pi, j) => [btn('select', j, tokenNames[j], i === j), esc(fmt(z[j])), esc(pct(pi)), esc(s.counts[j])])) )}</div>`
      + settings(switches + controls('Logit cases', Object.entries({ peaked: 'Prefer retrieval', flat: 'Equal logits', shifted: 'Prefer generation' }).map(([key, text]) => btn('scene', key, text, s.scene === key)).join('')) + controls('Sampling experiment', btn('sample', 1, 'Draw a new point') + btn('sample', 100, 'Draw 100') + btn('sample', 1000, 'Draw 1,000') + btn('reset', '', 'Reset')) + note('Seeded draws are reproducible. Changing logits resets empirical counts; a score boost holds the current sampler point fixed.'));
  }

  function expectationView(s) {
    const p = populations[s.population], losses = modelLosses[s.model], risk = expectedRisk(p, losses), other = expectedRisk(p, modelLosses[s.model === 'A' ? 'B' : 'A']);
    const names = ['Frequent users', 'New users'];
    const oldP = s.previous ? populations[s.previous.population] : p;
    const oldRisk = expectedRisk(oldP, losses).mean;
    return heading('Expectation / population-weighted model risk', 'Will the same model win when the users change?')
      + primary('population', s.population === 'train' ? 'shifted' : 'train', s.population === 'train' ? 'Send more new users' : 'Restore the user mix')
      + `<div class="ml-grid"><div class="ml-stage">${equation(`Model ${s.model} risk`, [{ text: `${p[0]} x ${losses[0]}`, active: s.selected === 0 }, { text: '+' }, { text: `${p[1]} x ${losses[1]}`, active: s.selected === 1 }], `${fmt(risk.mean)} nats / arrival`)}<div class="fl-cohort" role="group" aria-label="20 population units; each is 5 percent">${Array.from({ length: 20 }, (_, i) => { const group = i < Math.round(p[0] * 20) ? 0 : 1, moved = (i < Math.round(oldP[0] * 20)) !== (group === 0); return `<button type="button" class="fl-person fl-${colors[group]} ${group === s.selected ? '' : 'fl-dim'} ${moved ? 'fl-reassigned' : ''}" data-action="select" data-value="${group}" aria-label="Inspect ${names[group]}${moved ? '; changed segment' : ''}" aria-pressed="${group === s.selected}"><span></span></button>`; }).join('')}</div>${legend([['blue', `${pct(p[0])} frequent; loss ${losses[0]}`], ['orange', `${pct(p[1])} new; loss ${losses[1]}`]])}${s.previous?.population !== undefined && s.previous.population !== s.population ? consequence(`Model ${s.model} risk`, fmt(oldRisk), fmt(risk.mean), 'Only the user mix changed. Outlined dots changed segment; per-segment losses stayed fixed.') : note(`Each dot is 5% of arrivals. ${names[s.selected]} add ${fmt(risk.contributions[s.selected])} nats to the total. Tap either group to highlight its term.`)}</div>${inspector(`Model ${s.model} ${risk.mean < other.mean ? 'wins' : 'loses'} on this population`, [`Model ${s.model}: ${fmt(risk.mean)}; model ${s.model === 'A' ? 'B' : 'A'}: ${fmt(other.mean)}. Lower expected loss wins.`, `${names[s.selected]}: ${fmt(p[s.selected])} x ${fmt(losses[s.selected])} = ${fmt(risk.contributions[s.selected])} nats per arrival.`, `Total = ${risk.contributions.map(x => fmt(x)).join(' + ')} = ${fmt(risk.mean)}.`, `Unweighted average = ${fmt(sum(losses) / 2)}: it silently assumes a 50/50 population.`], table(['Segment', 'p', 'Loss', 'p x loss'], p.map((pi, i) => [btn('select', i, names[i], s.selected === i), esc(fmt(pi)), esc(fmt(losses[i])), esc(fmt(risk.contributions[i]))])) )}</div>`
      + settings(controls('Model to inspect', ['A', 'B'].map(m => btn('model', m, `Model ${m}`, s.model === m)).join('')) + note('Losses are fixed segment-conditional mean log-losses in nats. Within-segment variance is omitted.'));
  }

  function bayesView(s) {
    const b = bayesCounts(s.prior, .9, .02, 10000, s.positive), titles = ['Population', 'Filter by evidence', 'Renormalize within subset'];
    const share = s.step === 2 ? b.posterior : s.prior;
    const populationTable = table(['Actual', 'Count'], [['Spam', fmt(b.spam)], ['Not spam', fmt(b.ham)]], '10,000 emails before observing classifier evidence');
    const evidenceTable = table(['Actual', 'Flagged', 'Not flagged'], [
      ['Spam', `<span class="${s.positive ? 'fl-conditioned' : 'fl-evidence-excluded'}">${fmt(b.tp)}</span>`, `<span class="${!s.positive ? 'fl-conditioned' : 'fl-evidence-excluded'}">${fmt(b.fn)}</span>`],
      ['Not spam', `<span class="${s.positive ? 'fl-conditioned' : 'fl-evidence-excluded'}">${fmt(b.fp)}</span>`, `<span class="${!s.positive ? 'fl-conditioned' : 'fl-evidence-excluded'}">${fmt(b.tn)}</span>`],
    ], 'Fixed classifier: 90% recall, 2% false-positive rate');
    const strip = `<p class="ml-kicker">${s.step === 2 ? `Only ${fmt(b.evidence)} ${s.positive ? 'flagged' : 'unflagged'} emails` : 'All 10,000 emails'}</p><div class="fl-mass" role="img" aria-label="${esc(`${pct(share)} spam in the selected population`)}"><span class="fl-mass-spam" style="width:${share * 100}%"></span><span class="fl-mass-other" style="width:${(1 - share) * 100}%"></span></div>${legend([['blue', 'Spam'], ['orange', 'Not spam']])}`;
    const survivors = `<div class="fl-filter-flow" aria-label="${esc(`Keep only ${s.positive ? 'flagged' : 'unflagged'} emails`)}">${[[b.spam, b.numerator, 'spam'], [b.ham, b.other, 'non-spam']].map(([source, kept, label]) => `<div class="fl-filter-row"><span>${source} ${label}</span><strong>${kept} survive<br><small>${pct(kept / source)} kept</small></strong><span class="fl-discarded">${source - kept} removed</span></div>`).join('')}</div>`;
    const linked = s.step === 0
      ? equation('Spam among all emails', [{ text: `${b.spam}`, active: true }, { text: `/ (${b.spam} + ${b.ham})` }], pct(s.prior))
      : s.step === 1 ? equation('Emails matching the evidence', [{ text: `${b.numerator} spam`, active: true }, { text: `+ ${b.other} non-spam` }], `${b.evidence} survivors`)
        : equation('Spam among the survivors', [{ text: `${b.numerator}`, active: true }, { text: `/ (${b.numerator} + ${b.other})` }], `${fmt(100 * b.posterior, 2)}%`);
    return heading('Bayes / change who counts', 'Who is left after the spam filter?')
      + navigation(s.step, titles, ['Apply the filter', 'Use the surviving group'])
      + `<div class="ml-grid"><div class="ml-stage">${linked}${s.step === 1 ? survivors : strip}${s.step === 0 ? note(`${b.spam} spam and ${b.ham} non-spam emails. The filter flags 90% of spam and 2% of non-spam.`) : s.step === 1 ? consequence('Population size', '10,000 emails', `${b.evidence} survivors`, `Only ${s.positive ? 'flagged' : 'unflagged'} emails match the evidence. The grey counts no longer belong in the denominator.`) : consequence('Spam share', pct(s.prior), `${fmt(100 * b.posterior, 2)}%`, `${b.numerator} of the ${b.evidence} survivors are spam. No new emails appeared; the denominator changed.`)}</div>${inspector(titles[s.step], s.step === 0 ? [`Prior: ${fmt(b.spam)} / 10,000 = ${pct(s.prior)}.`, 'The starting share counts spam among every email, before observing a flag.', 'The filter flags 90% of spam and 2% of non-spam.'] : s.step === 1 ? [`Keep ${fmt(b.numerator)} spam and ${fmt(b.other)} non-spam emails. These ${fmt(b.evidence)} emails will become the new whole.`, `Spam survivors: ${b.spam} x ${s.positive ? .9 : .1} = ${b.numerator}.`, `Non-spam survivors: ${b.ham} x ${s.positive ? .02 : .98} = ${b.other}.`] : [`${b.numerator} surviving spam emails / ${b.evidence} survivors = ${pct(b.posterior)} spam (rounded).`, 'The denominator counts all evidence-matching emails, including non-spam.', s.positive ? 'Same recall and false-positive rate, different base rate: the surviving mixture changes.' : 'No flag is evidence too: false negatives remain in the surviving subset.'], s.step === 0 ? populationTable : evidenceTable)}</div>`
      + settings(controls('Inbox spam base rate', [.01, .1, .5].map(p => btn('prior', p, `${pct(p)} spam`, s.prior === p)).join('')) + controls('Observed evidence', btn('evidence', 'positive', 'Flagged email', s.positive) + btn('evidence', 'negative', 'Not flagged', !s.positive)) + note('Synthetic expected counts; fixed 90% recall and 2% false-positive rate. Rectangle widths are exact fractions, without minimum-width inflation.'));
  }

  function entropyView(s) {
    const p = trueTokens, q = tokenModels[s.model], info = information(p, q), i = s.selected, t = info.terms[i];
    const oldQ = tokenModels[s.previous?.model] || q;
    const changed = oldQ.some((v, j) => v !== q[j]);
    const oldInfo = information(p, oldQ);
    let left = 40;
    const regions = p.map((pi, j) => {
      const width = 280 * pi, height = 24 * info.terms[j].surprise;
      const rect = `<g role="button" tabindex="0" data-action="select" data-value="${j}" aria-pressed="${i === j}" class="fl-surprise-region ${i === j ? 'fl-surprise-selected' : 'fl-dim'}" aria-label="${esc(`${entropyNames[j]}: ${continuationCounts[j]} of 100 continuations, ${fmt(info.terms[j].surprise)} bits each, ${fmt(info.terms[j].crossEntropy)} bits toward the average`)}"><rect class="fl-hit" x="${left}" y="30" width="${width}" height="175"/><rect class="fl-surprise-area" data-probability="${pi}" data-surprise="${info.terms[j].surprise}" x="${left}" y="${205 - height}" width="${width}" height="${height}"/>${changed && i === j ? `<rect class="fl-ghost-shape" x="${left}" y="${205 - 24 * oldInfo.terms[j].surprise}" width="${width}" height="${24 * oldInfo.terms[j].surprise}"/>` : ''}</g>`;
      left += width;
      return rect;
    }).join('');
    const matched = s.model === 'matched';
    return heading('Entropy / average surprise', 'Does the most surprising token dominate the average?')
      + primary('select', i === 2 ? 0 : 2, i === 2 ? 'Compare common cup' : 'Compare rare glass')
      + `<p class="fl-token-context">Toy next-token context: "She poured the coffee into the <strong>...</strong>"</p>`
      + controls('Inspect a reference outcome', entropyNames.map((name, j) => btn('select', j, `${name} ${pct(p[j])}`, j === i)).join(''))
      + `<div class="ml-grid"><div class="ml-stage">${equation(`${entropyNames[i]} area`, [{ text: `${p[i]} x` }, { text: `[-log2(${fmt(q[i], 4)})]`, active: true }], `${fmt(t.crossEntropy)} bits / continuation`)}<p class="ml-note">${matched ? 'Entropy' : 'Cross-entropy'}: <strong>${fmt(info.crossEntropy)} bits / continuation</strong>, the sum of all three areas. Height: bits per occurrence.</p><svg class="fl-surprise-plot" viewBox="0 0 340 240" role="group" aria-label="Outcome regions: width is true frequency, height is surprise under the model, total area is average surprise">${[0, 2, 4, 6].map(v => `<path class="fl-grid-line" d="M40 ${205 - 24 * v}H320"/><text x="32" y="${210 - 24 * v}" text-anchor="end">${v}</text>`).join('')}${regions}<path class="fl-axis" d="M40 30V205H320"/><text x="40" y="231">0</text><text x="320" y="231" text-anchor="end">100%</text></svg><p class="fl-axis-caption">Width: reference frequency. Orange: ${esc(entropyNames[i])}. Dashed outline, when present: its previous surprise. Axes do not rescale.</p>${changed ? consequence(`${entropyNames[i]} contribution`, `${fmt(oldInfo.terms[i].crossEntropy)} bits / continuation`, `${fmt(t.crossEntropy)} bits / continuation`, `Model probability ${pct(oldQ[i])} to ${pct(q[i])}; reference frequency stays ${continuationCounts[i]}/100.`) : s.previous && s.previous.selected !== i ? consequence('Selected contribution', `${entropyNames[s.previous.selected]}: ${fmt(info.terms[s.previous.selected].crossEntropy)}`, `${entropyNames[i]}: ${fmt(t.crossEntropy)} bits / continuation`, `${continuationCounts[i]}/100 occurrences x ${fmt(t.surprise)} bits. ${matched ? 'Cup is less surprising than glass, yet its wider region contributes more. The total entropy did not change.' : 'Only the inspected token changed; the total cross-entropy did not change.'}`) : note(`${entropyNames[i]} occurs ${continuationCounts[i]} times per 100 continuations. Its surprise is ${fmt(t.surprise)} bits per occurrence, but its area adds ${fmt(t.crossEntropy)} bits per continuation to the average.`)}</div>
      <aside class="ml-inspector fl-surprise-inspector" aria-live="polite"><h4>One ${esc(entropyNames[i])}: ${fmt(t.surprise)} bits</h4><p>The model gives this token ${pct(q[i])} probability.</p><svg viewBox="0 0 300 86" role="img" aria-label="${esc(`Surprise ${fmt(t.surprise)} bits is ${fmt(t.surprise)} halvings of probability from 1`)}"><path class="fl-axis" d="M20 34H280"/>${Array.from({ length: 8 }, (_, bit) => `<path class="fl-axis" d="M${20 + bit * 260 / 7} 28V40"/><text x="${20 + bit * 260 / 7}" y="65" text-anchor="middle">${bit}</text>`).join('')}<circle class="fl-surprise-marker" cx="${20 + t.surprise * 260 / 7}" cy="34" r="6"/></svg><p class="ml-note">Each halving adds one bit: probability 1/2 costs 1 bit; 1/4 costs 2.</p><div class="fl-surprise-average"><strong>${continuationCounts[i]} / 100 occurrences x ${fmt(t.surprise)} bits = ${fmt(t.crossEntropy)} bits toward the average.</strong></div><p>${matched ? 'Rare means tall but narrow. Average surprise adds the areas, not the heights.' : 'Its real frequency stays fixed. Only the model probability, and therefore surprise, changed.'}</p></aside></div>
      <div class="fl-entropy-decomposition"><h4>${matched ? 'Entropy: model and reference agree' : 'Cross-entropy: task uncertainty plus mismatch'}</h4><p><strong>${fmt(info.entropy)}</strong> H(p) + <strong>${fmt(info.kl)}</strong> KL(p || q) = <strong>${fmt(info.crossEntropy)}</strong> H(p,q) bits / continuation</p><div class="fl-information-mass" role="img" aria-label="${esc(`${fmt(info.entropy)} bits entropy plus ${fmt(info.kl)} bits KL equals ${fmt(info.crossEntropy)} bits cross-entropy; common scale 0 to 2 bits`)}"><span class="fl-information-entropy" style="width:${info.entropy / 2 * 100}%"></span><span class="fl-information-kl" style="width:${Math.max(0, info.kl) / 2 * 100}%"></span></div>${legend([['blue', 'Reference entropy: unchanged'], ['orange', 'Extra expected loss: KL']])}<p class="fl-axis-caption">Fixed scale: full track = 2 bits per continuation. No minimum-width inflation; matching p makes the orange segment disappear.</p></div>
      <details class="fl-entropy-terms"><summary>Inspect the calculation</summary>${note(`Selected surprise: -log2(${fmt(q[i])}) = ${fmt(t.surprise)} bits. ${continuationCounts[i]} occurrences cost ${fmt(continuationCounts[i] * t.surprise)} bits per 100 continuations, or ${fmt(t.crossEntropy)} bits per continuation toward the average.`)}${table(['Token', 'p', 'q', 'p x surprise', 'KL term'], p.map((pi, j) => [esc(entropyNames[j]), fmt(pi), fmt(q[j]), fmt(info.terms[j].crossEntropy), fmt(info.terms[j].kl)]))}${note('Individual KL terms can be negative; their sum cannot. For one observed token, training scores -ln q(token). Bits convert to nats by multiplying by ln 2.')}</details>`
      + `<details class="fl-settings"><summary>What if the model overpredicts cup?</summary>${controls('Model probabilities', Object.entries({ matched: 'Match p', confident: 'Overpredict cup', uniform: 'Uniform predictions' }).map(([key, name]) => btn('model', key, name, key === s.model)).join(''))}${note('Reference counts [60,30,10] are illustrative, not measured language frequencies. Only q changes; p stays fixed.')}</details>`;
  }

  function lossView(s) {
    const target = s.mode === 'observed' ? s.y : .1, step = lossStep(target, s.q);
    const x = z => 35 + (z + 6) / 12 * 230;
    const endpoints = [step.zLog, step.zSquare];
    const rows = endpoints.map((z, i) => {
      const y = 45 + i * 58, start = x(step.z), end = x(z);
      return `<g class="fl-${colors[i]}" data-before-z="${step.z}" data-after-z="${s.trained ? z : step.z}"><path class="fl-axis" d="M35 ${y}H265"/><line class="fl-vector ${s.trained ? '' : 'fl-dashed'}" x1="${start}" x2="${end}" y1="${y}" y2="${y}"/><circle class="fl-prior-point" cx="${start}" cy="${y}" r="5"/><circle class="fl-update-point" cx="${s.trained ? end : start}" cy="${y}" r="6"/></g>`;
    }).join('');
    const qLog = s.trained ? step.qLog : s.q, qSquare = s.trained ? step.qSquare : s.q;
    return heading('Loss / same start, same learning rate', 'Why can a confident mistake learn so slowly?')
      + primary('learn', '', s.trained ? 'Rewind the step' : 'Apply one learning step')
      + `<div class="ml-grid"><div class="ml-stage">${equation('Squared-loss gradient', [{ text: `${fmt(s.q - target)} x` }, { text: `2 x ${s.q} x ${fmt(1 - s.q)}`, active: true }], fmt(step.squareGradient, 6))}<svg class="fl-loss-step" viewBox="0 0 300 165" role="img" aria-label="Two independent logit updates from the same start; blue is log-loss and orange is squared error">${rows}${[-6, -3, 0, 3, 6].map(z => `<text x="${x(z)}" y="145" text-anchor="middle">${z}</text>`).join('')}</svg><p class="fl-axis-caption">Horizontal: logit z. Same scale and learning rate 2. ${s.trained ? 'Hollow circles are the common start; solid circles are the two updated predictions.' : 'Dashed segments are proposed updates; both solid points still share the start.'}</p>${legend([['blue', `Log-loss q: ${fmt(100 * qLog, 2)}%`], ['orange', `Squared-loss q: ${fmt(100 * qSquare, 2)}%`]])}${s.trained ? consequence('Predicted click chance', `${pct(s.q)} for both`, `${fmt(100 * qLog, 2)}% vs ${fmt(100 * qSquare, 2)}%`, 'Only the loss differs. The extra sigmoid factor suppresses the squared-loss update near probability 0 or 1.') : note(`${s.mode === 'observed' ? (s.y ? 'Clicked impression' : 'No click') : 'True click rate 10%'}; starting prediction ${pct(s.q)}. Log-loss gradient ${fmt(step.logGradient)} is not multiplied by the highlighted saturation factor.`)}</div>${inspector(s.mode === 'observed' ? `${s.y ? 'Clicked' : 'Not clicked'}: two independent updates` : 'Expected update at a 10% click rate', [`After ${s.trained ? 'applying' : 'proposing'} the step: log-loss z=${fmt(step.zLog)}, squared-loss z=${fmt(step.zSquare)}.`, `Starting z=ln(${s.q}/${fmt(1 - s.q)})=${fmt(step.z)}.`, `Log-loss: z'=${fmt(step.z)} - 2 x (${fmt(step.logGradient)})=${fmt(step.zLog)}.`, `Squared loss: z'=${fmt(step.z)} - 2 x (${fmt(step.squareGradient, 6)})=${fmt(step.zSquare)}.`, 'Convert back with q=sigmoid(z). This isolates two scalar objectives, not a full network training run.', 'Both losses are proper probability scores; numeric loss scales and a shared illustrative step size do not prove that one is always better.'])}</div>`
      + settings(controls('Loss view', btn('mode', 'observed', 'One impression', s.mode === 'observed') + btn('mode', 'expected', 'Expected: 10% click rate', s.mode === 'expected')) + (s.mode === 'observed' ? controls('Observed label', btn('label', 1, 'Clicked: y = 1', s.y === 1) + btn('label', 0, 'No click: y = 0', s.y === 0)) : '') + controls('Predicted click probability', [.01, .1, .3, .5, .7, .9, .99].map(q => btn('q', q, `q = ${q}`, s.q === q)).join('')));
  }

  function vectorView(s) {
    const x = embeddings[s.item], u = unit(x), shown = s.normalized && u ? u : x;
    const points = Object.entries(embeddings).map(([name, v]) => ({ v: s.normalized ? unit(v) || v : v, action: 'item', value: name, label: `Item ${name}`, selected: s.item === name }));
    const changed = s.previous?.item === s.item && s.previous.normalized !== s.normalized;
    return heading('Vectors / embedding norm and direction', 'What does normalizing an embedding remove?')
      + primary('normalize', !s.normalized, s.normalized ? 'Restore lengths' : 'Remove length')
      + `<div class="ml-grid"><div class="ml-stage">${u ? equation(`Unit direction for ${s.item}`, [{ text: vec(x) }, { text: `/ ${fmt(norm(x))}`, active: true }], vec(u)) : equation('Zero embedding', [{ text: 'length 0', active: true }], 'no unit direction')}${plane({ arrows: [{ v: s.normalized ? x : (u || x), color: 'ghost', dashed: true }, { v: shown, color: 'blue' }], points, range: 2.5, unitCircle: true, label: 'Embedding vectors; dashed arrow is the alternate representation, dotted circle is unit length.' })}${legend([['blue', 'Current representation'], ['ghost', s.normalized ? 'Original vector' : 'Unit-vector target']])}${changed && u ? consequence('Selected length', fmt(s.previous.normalized ? 1 : norm(x)), fmt(norm(shown)), `Direction is unchanged: cosine(original, unit)=${fmt(cosine(x, u))}.`) : note(u ? 'Both arrows lie on the same ray. Dividing both coordinates by one norm removes length, not direction.' : 'The zero vector stays at the origin: there is no ray or unit direction to preserve.')}</div>${inspector(`Item ${s.item}: ${vec(shown)}`, [u ? `Length ${fmt(norm(shown))}. ${s.normalized ? 'Same direction; original magnitude removed.' : 'The dashed target has length 1.'}` : 'Zero has no direction: unit normalization is undefined.', `Length = sqrt(${fmt(x[0] ** 2)} + ${fmt(x[1] ** 2)}) = ${fmt(norm(x))}.`, u ? `Unit vector = ${vec(x)} / ${fmt(norm(x))} = ${vec(u)}.` : 'A production retrieval system must handle zero embeddings explicitly.', u ? `Reconstruction: ${fmt(norm(x))} x ${vec(u)} = ${vec(x)}.` : 'The origin shown is not a valid unit embedding.', 'Magnitude can influence inner-product retrieval. It is not guaranteed to encode popularity.'])}</div>`
      + settings(controls('Embedding', Object.keys(embeddings).map(k => btn('item', k, k === 'zero' ? 'Zero vector' : `Item ${k}`, s.item === k)).join('')));
  }

  function dotView(s) {
    const q = s.query === 'first' ? [1, .5] : [.2, 1], v = embeddings[s.item];
    const shownQ = s.normalized ? unit(q) : q, shownV = s.normalized ? unit(v) : v;
    const projection = scale(shownQ, dot(shownQ, shownV) / dot(shownQ, shownQ));
    const ranking = ['A', 'B', 'C'].map(k => ({ k, score: s.normalized ? cosine(q, embeddings[k]) : dot(q, embeddings[k]) })).sort((a, b) => b.score - a.score);
    const rawRank = ['A', 'B', 'C'].sort((a, b) => dot(q, embeddings[b]) - dot(q, embeddings[a]));
    const cosineRank = ['A', 'B', 'C'].sort((a, b) => cosine(q, embeddings[b]) - cosine(q, embeddings[a]));
    const rankTrace = `<svg class="fl-rank-trace" viewBox="0 0 300 190" role="group" aria-label="Rank positions before and after removing embedding norms"><text x="72" y="18" text-anchor="middle">Dot</text><text x="226" y="18" text-anchor="middle">Cosine</text>${[0, 1, 2].map(r => `<text x="8" y="${50 + r * 55}">${r + 1}</text>`).join('')}${['A', 'B', 'C'].map(k => { const y1 = 45 + 55 * rawRank.indexOf(k), y2 = 45 + 55 * cosineRank.indexOf(k); return `<g role="button" tabindex="0" data-action="item" data-value="${k}" aria-label="Item ${k}: dot rank ${rawRank.indexOf(k) + 1}, cosine rank ${cosineRank.indexOf(k) + 1}" class="fl-rank-item ${k === s.item ? 'fl-orange' : 'fl-ghost fl-dim'}"><line x1="72" y1="${y1}" x2="226" y2="${y2}"/>${[[72, y1, !s.normalized], [226, y2, s.normalized]].map(([cx, cy, current]) => `<circle class="fl-hit" cx="${cx}" cy="${cy}" r="22"/><circle class="fl-rank-dot ${current ? 'fl-rank-current' : ''}" cx="${cx}" cy="${cy}" r="15"/><text x="${cx}" y="${cy + 5}" text-anchor="middle">${k}</text>`).join('')}</g>`; }).join('')}</svg>`;
    return heading('Dot products / two-tower retrieval scores', 'Can normalization change the top retrieval result?')
      + primary('normalize', !s.normalized, s.normalized ? 'Restore norm advantage' : 'Remove norm advantage')
      + equation(`Item ${s.item} ${s.normalized ? 'cosine' : 'dot'}`, s.normalized ? [{ text: `${fmt(dot(q, v))} /` }, { text: `(${fmt(norm(q))} x ${fmt(norm(v))})`, active: true }] : [{ text: `${q[0]} x ${v[0]} + ${q[1]} x ${v[1]}`, active: true }], fmt(s.normalized ? cosine(q, v) : dot(q, v)))
      + `<div class="ml-grid fl-dot-grid"><div class="ml-stage">${plane({ arrows: [{ v: shownQ }, { v: shownV, color: 'orange' }, { from: shownV, v: projection, color: 'green', dashed: true }], points: ['A', 'B', 'C'].map(k => ({ v: s.normalized ? unit(embeddings[k]) : embeddings[k], label: `Item ${k}`, action: 'item', value: k, selected: k === s.item, color: 'orange' })), range: 2.5, label: 'Query, candidate embedding, and perpendicular projection onto query' })}${legend([['blue', 'Query'], ['orange', `Item ${s.item}`], ['green', 'Perpendicular to query']])}${note('Blue is the query; orange is the selected item. Tap a point to inspect its projection and score.')}</div><div class="fl-rank-panel">${rankTrace}${s.previous?.query === s.query && s.previous.normalized !== s.normalized ? consequence('Top result', s.previous.normalized ? cosineRank[0] : rawRank[0], ranking[0].k, 'Same embeddings and query. Only the norm factors were removed or restored.') : note('Tap an item to connect its projection, score, and rank. Filled rank nodes mark the active scoring rule.')}${inspector(`Top result: item ${ranking[0].k}`, [`${s.normalized ? 'Cosine' : 'Raw dot'} score ${fmt(ranking[0].score)}. ${s.normalized ? 'Only direction matters.' : 'Length contributes as well as direction.'}`, `Selected item ${s.item}: dot = ${fmt(q[0])} x ${fmt(v[0])} + ${fmt(q[1])} x ${fmt(v[1])} = ${fmt(dot(q, v))}.`, `Cosine = ${fmt(dot(q, v))} / (${fmt(norm(q))} x ${fmt(norm(v))}) = ${fmt(cosine(q, v))}.`, `Current projection of item onto query = ${vec(projection)}.`])}</div></div>`
      + calculation(table(['Rank', 'Item', 'Score'], ranking.map((r, i) => [esc(i + 1), btn('item', r.k, `Item ${r.k}`, s.item === r.k), esc(fmt(r.score))])))
      + settings(controls('Query vector', btn('query', 'first', 'Query [1, 0.5]', s.query === 'first') + btn('query', 'second', 'Query [0.2, 1]', s.query === 'second')) + note('Train, index, and serve with a consistent scoring rule. Removing norms changes the retrieval objective.'));
  }

  function matrixView(s) {
    const A = maps[s.scene], inputs = [[1, 0], [0, 1], [1, 2]], names = ['e1', 'e2', 'x'], x = inputs[s.selected], y = matvec(A, x);
    const c1 = scale([A[0][0], A[1][0]], x[0]), c2 = scale([A[0][1], A[1][1]], x[1]);
    const arrows = s.step === 0 ? [] : s.step === 1 ? [{ v: c1, color: 'orange' }] : s.step === 2 ? [{ v: c1, color: 'ghost' }, { from: c1, v: y, color: 'orange' }] : [{ v: y, color: 'orange' }];
    const trace = ['Input coordinates', 'First weighted column', 'Add second weighted column', 'Output / row products'];
    const accumulated = step => step === 0 ? [0, 0] : step === 1 ? c1 : y;
    const linked = s.step === 0 ? equation('Input decomposition', [{ text: `${x[0]} x [1, 0]`, active: true }, { text: `+ ${x[1]} x [0, 1]` }], vec(x))
      : s.step === 1 ? equation('Place first column', [{ text: `${x[0]} x ${vec([A[0][0], A[1][0]])}`, active: true }], vec(c1))
        : equation('Add second column', [{ text: vec(c1) }, { text: '+' }, { text: `${x[1]} x ${vec([A[0][1], A[1][1]])}`, active: true }], vec(y));
    return heading('Matrix multiplication / one map, every input', 'How do two input coordinates become one output vector?')
      + navigation(s.step, trace, ['Place first column', 'Add second column', 'Show their sum'])
      + `<div class="ml-grid"><div class="ml-stage fl-active-map"><p class="ml-kicker">Input ${esc(vec(x))}; output so far ${esc(vec(accumulated(s.step)))}</p>${linked}${plane({ arrows, gridMap: A, range: 4, shape: [[0, 0], [1, 0], [1, 1], [0, 1]].map(v => matvec(A, v)), label: 'Transformed grid and the selected column contribution' })}${s.previous?.scene === s.scene && s.previous.selected === s.selected && s.previous.step !== s.step && s.step !== 3 ? consequence('Accumulated output', vec(accumulated(s.previous.step)), vec(accumulated(s.step)), s.step === 0 || s.previous.step === 0 ? 'The accumulator starts at zero, then places the weighted first column; A and x stay fixed.' : 'Only the highlighted second-column contribution was added or removed; A and x stayed fixed.') : note(s.step === 1 ? `The orange arrow is ${x[0]} times column 1. Add ${x[1]} times column 2 at its tip.` : s.step === 2 ? 'Grey is the first contribution; orange is the second. Their shared endpoint is Ax.' : 'The output is the sum of the same two weighted columns.')}</div><div class="ml-stage"><p class="ml-kicker">Input coordinates: ${esc(vec(x))}</p>${plane({ arrows: [{ v: x }], points: inputs.map((v, i) => ({ v, label: names[i], action: 'select', value: i, selected: s.selected === i })), gridMap: [[1, 0], [0, 1]], range: 4, label: 'Input grid and basis vectors', shape: [[0, 0], [1, 0], [1, 1], [0, 1]] })}${note('Tap an input point. The same matrix must map every input, not just this example.')}</div></div>`
      + calculation(`<div class="fl-matrices">${matrix(A, 'A')}${matrix([[x[0], y[0]], [x[1], y[1]]], 'columns: x, Ax')}</div>${controls('Inspect output coordinate', [0, 1].map(i => btn('row', i, `Output y${i + 1}`, s.row === i)).join(''))}${note(`Column sum: ${vec(c1)} + ${vec(c2)} = ${vec(y)}.`)}${note(`Row ${s.row + 1}: ${A[s.row][0]} x ${x[0]} + ${A[s.row][1]} x ${x[1]} = ${y[s.row]}.`)}${note(`det(A) = ${fmt(det(A))}: ${det(A) === 0 ? 'the unit square collapses to a line; one input direction is lost.' : `signed area scales by ${fmt(det(A))}.`}`)}`)
      + settings(controls('Linear map', Object.entries({ shear: 'Mix coordinates', rotate: 'Quarter-turn', collapse: 'Rank-one collapse' }).map(([k, text]) => btn('scene', k, text, s.scene === k)).join('')) + controls('Input vector', inputs.map((v, i) => btn('select', i, `${names[i]} = ${vec(v)}`, s.selected === i)).join('')));
  }

  function eigenView(s) {
    const A = eigenCases[s.scene], eig = eigenSymmetric(A), start = s.selected === 2 ? [1, 0] : eig.vectors[s.selected];
    const v = powerStep(A, start, s.step), Av = matvec(A, v), rayleigh = dot(v, Av), residual = norm(subtract(Av, scale(v, rayleigh)));
    const onLine = scale(v, rayleigh), offLine = subtract(Av, onLine);
    const prior = s.previous?.scene === s.scene && s.previous.selected === s.selected && s.previous.step !== s.step ? s.previous : null;
    const oldV = prior ? powerStep(A, start, prior.step) : v, oldAv = matvec(A, oldV);
    const oldResidual = norm(subtract(oldAv, scale(oldV, dot(oldV, oldAv))));
    return heading('Eigenvectors / covariance and curvature', 'Which directions survive this matrix without turning?')
      + navigation(s.step, Array.from({ length: 6 }, (_, i) => `Applied ${i} times`), Array(5).fill('Feed output back'))
      + `<div class="ml-grid"><div class="ml-stage">${equation('Off-line component', [{ text: vec(Av) }, { text: '-' }, { text: `${fmt(rayleigh)} x ${vec(v)}`, active: true }], vec(offLine))}${plane({ arrows: [...(prior ? [{ v: oldV, color: 'ghost', dashed: true }] : []), { v: onLine, color: 'ghost', dashed: true }, { v, color: 'blue' }, { v: Av, color: 'orange' }, { from: onLine, v: Av, color: 'green' }], range: 3.5, unitCircle: true, points: eig.vectors.map((v, i) => ({ v, action: 'select', value: i, label: `Eigenvector ${i + 1}`, selected: s.selected === i })), label: 'Input v, output Av, and green off-line residual from the projected output to Av' })}${legend([['blue', 'Unit input v'], ['orange', 'Mapped output Av'], ['green', 'Off-line component'], ['ghost', 'Projection / prior input']])}${prior ? consequence('Off-line length', fmt(oldResidual, 5), fmt(residual, 5), s.scene === 'isotropic' ? 'This matrix scales every direction equally; there is no unique leading eigenline.' : 'The previous output becomes the next unit input. An ordinary start approaches the leading eigenline; an exact eigenvector stays on its own line.') : note(`The green segment has length ${fmt(residual, 5)}. It disappears when Av stays on the input line. Tap an eigenvector point to test that directly.`)}</div>${inspector(residual < 1e-10 ? 'These arrows share one line' : 'The matrix turns this direction', [`Input ${vec(v)} maps to ${vec(Av)}. Normalize that output before feeding it back.`, `Rayleigh quotient v^T A v = ${fmt(rayleigh)}.`, `Residual ||Av - (v^T A v)v|| = ${fmt(residual, 5)}. Zero identifies an eigenvector.`, s.scene === 'saddle' ? 'Eigenvalues 3 and -1: the second direction flips sign. This is a Hessian example, not a covariance.' : s.scene === 'isotropic' ? 'Equal eigenvalues: every nonzero direction is an eigenvector; no unique leading axis.' : `PCA explained-variance fraction for v1: ${fmt(eig.values[0])} / ${fmt(sum(eig.values))} = ${pct(eig.values[0] / sum(eig.values))}.`])}</div>`
      + calculation(`<div class="fl-matrices">${matrix(A, 'A')}${matrix(reconstructEigen(A), 'Q diag(lambda) Q^T')}</div>${table(['Eigenpair', 'Unit vector', 'Eigenvalue'], eig.vectors.map((v, i) => [btn('select', i, `v${i + 1}`, s.selected === i), esc(vec(v)), esc(fmt(eig.values[i]))]))}${note('Power iteration needs a dominant absolute eigenvalue and a starting component along its eigenvector. Starting exactly along another eigenvector stays there.')}`)
      + settings(controls('Symmetric matrix', Object.entries({ covariance: 'Correlated covariance', saddle: 'Saddle Hessian', isotropic: 'Isotropic covariance' }).map(([k, text]) => btn('scene', k, text, s.scene === k)).join('')) + controls('Starting direction', btn('select', 2, 'Ordinary [1, 0]', s.selected === 2) + btn('select', 0, 'Leading eigenvector', s.selected === 0) + btn('select', 1, 'Second eigenvector', s.selected === 1)));
  }

  function svdView(s) {
    const A = svdCases[s.scene], d = svd2(A), approx = reconstructSvd(d, s.rank), residual = matrixSubtract(A, approx);
    const x = [1, 1], aligned = matvec(d.Vt, x), stretched = aligned.map((v, i) => i < s.rank ? v * d.singular[i] : 0), output = matvec(d.U, stretched);
    const stages = [x, aligned, stretched, output], labels = ['Input x', 'V^T: change input basis', 'Sigma: stretch / truncate', 'U: map to output basis'];
    const circle = Array.from({ length: 65 }, (_, i) => [Math.cos(i * Math.PI / 32), Math.sin(i * Math.PI / 32)]);
    const transformStage = (v, rank = s.rank) => { if (s.step === 0) return v; const a = matvec(d.Vt, v); if (s.step === 1) return a; const b = a.map((t, i) => i < rank ? t * d.singular[i] : 0); return s.step === 2 ? b : matvec(d.U, b); };
    const fullOutput = transformStage(x, 2), discarded = subtract(fullOutput, stages[s.step]);
    const components = d.singular.map((sigma, j) => scale(d.U.map(row => row[j]), sigma * aligned[j]));
    const prior = s.previous?.scene === s.scene && s.previous.rank !== s.rank ? s.previous : null;
    const oldError = prior ? frobenius(matrixSubtract(A, reconstructSvd(d, prior.rank))) : frobenius(residual);
    const row = Math.floor(s.selected / 2), col = s.selected % 2;
    const contributions = d.singular.map((sigma, j) => sigma * d.U[row][j] * d.Vt[j][col]);
    const retained = sum(d.singular.slice(0, s.rank).map(v => v * v)) / sum(d.singular.map(v => v * v));
    return heading('SVD / exact factorization and rank truncation', 'What disappears when you keep only one direction?')
      + primary('rank', s.rank === 1 ? 2 : 1, s.rank === 1 ? 'Restore direction' : 'Drop direction')
      + `<div class="ml-grid"><div class="ml-stage"><p class="ml-kicker">${esc(labels[s.step])}; input x = [1, 1]</p>${s.step === 3 ? equation('Kept output components', [{ text: vec(components[0]) }, { text: '+' }, { text: s.rank === 1 ? '[0, 0]' : vec(components[1]), active: true }], vec(output)) : equation('Current transformed input', [{ text: labels[s.step], active: true }], vec(stages[s.step]))}${plane({ arrows: [{ v: stages[s.step] }, ...(norm(discarded) > 1e-10 ? [{ from: stages[s.step], v: fullOutput, color: 'orange' }] : [])], shape: circle.map(v => transformStage(v)), ghostShape: circle.map(v => transformStage(v, 2)), range: 5, label: 'Kept map of the unit circle, dashed full map, and the exact discarded output component for x=[1,1]' })}${legend([['blue', 'Kept map / output'], ['ghost', 'Full map of unit circle'], ['orange', 'Missing output component']])}${prior ? consequence('Matrix reconstruction error', fmt(oldError), fmt(frobenius(residual)), `Discarded singular value ${fmt(d.singular[1])} determines rank-one Frobenius error; this is not a ranking-quality score.`) : note(s.step < 2 ? 'No component has been stretched or dropped yet.' : `For this input, the missing vector is ${vec(discarded)} (length ${fmt(norm(discarded))}). ${d.singular[1] === 0 ? 'The full map already has rank one; restoring a zero component adds nothing.' : s.rank === 1 ? 'Dropping a direction collapses the ellipse to a line.' : 'Both components are present.'}`)}</div>${inspector(`Keep ${s.rank} component${s.rank === 1 ? '' : 's'}: ${pct(retained)} matrix energy retained`, [`Whole-matrix error: ${fmt(frobenius(residual))}. This differs from the missing-vector length for one input.`, `Singular values: ${vec(d.singular)}.`, `||A - A_k||_F = ${fmt(frobenius(residual))}; squared error = ${fmt(frobenius(residual) ** 2)}.`, `Optimal rank-one error equals discarded sigma2 = ${fmt(d.singular[1])}.`, 'Retained matrix energy is not a guarantee of ranking quality.'])}</div>`
      + calculation(`<div class="fl-matrices">${matrix(d.U, 'U')}${matrix([[d.singular[0], 0], [0, s.rank === 2 ? d.singular[1] : 0]], 'Sigma kept')}${matrix(d.Vt, 'V^T')}</div><div class="fl-matrices">${matrix(A, 'Original A')}${matrix(approx, `A_k: keep ${s.rank}`)}${matrix(residual, 'Residual')}</div>${controls('Inspect reconstructed entry', [0, 1, 2, 3].map(i => btn('select', i, `Entry ${Math.floor(i / 2) + 1},${i % 2 + 1}`, i === s.selected)).join(''))}${note(`Entry ${row + 1},${col + 1}: ${fmt(contributions[0])} + ${fmt(contributions[1])} = ${fmt(A[row][col])}. Keeping ${s.rank} gives ${fmt(approx[row][col])}.`)}${note(`Each term is sigma_i * U[row,i] * Vt[i,column]. Tracked x=${vec(x)} maps to ${vec(output)}; full Ax=${vec(matvec(A, x))}.`)}`)
      + `<details class="fl-settings"><summary>Follow the factorization step by step</summary>${navigation(s.step, labels)}${note('U and V are orthogonal and may include reflections. The singular values are computed from A.')}</details>`
      + settings(controls('Matrix example', Object.entries({ mixed: 'Mixed linear map', balanced: 'Equal singular values', rankone: 'Already rank one' }).map(([k, label]) => btn('scene', k, label, s.scene === k)).join('')));
  }

  const views = { distribution: distributionView, expectation: expectationView, bayes: bayesView, entropy: entropyView, loss: lossView, vectors: vectorView, 'dot-products': dotView, 'matrix-multiply': matrixView, eigen: eigenView, svd: svdView };
  function render(kind, state) {
    return `<div class="fl-lab">${views[kind](state)}</div>`;
  }

  function contentEntry(title, summary, why, interview, details, formulas, annotations, snippet, prompt, correct, wrong, explanation) {
    return { title, summary, what: '', why, interview, details, math: { title: 'Read the computation', formula: formulas, note: 'Symbols below refer to the worked examples; the inspector recomputes values for each selection.', annotations }, code: { title: 'Reproduce the mechanism', lang: 'python', snippet }, quiz: { prompt, options: [{ text: correct, correct: true, explanation }, { text: wrong, correct: false, explanation }] }, controls: [], presets: [], geometry: null };
  }
  const content = {
    distribution: contentEntry('From model scores to possible outcomes.', 'Inspect a softmax distribution, then sample from its actual probability intervals.', 'Sampling and argmax are different decoding policies. A density is a different mathematical object from a discrete class probability.', 'Distinguish logits, normalized probability mass, empirical frequencies, and continuous density.', ['A shared shift of all logits leaves softmax unchanged.', 'Categorical mass sums to one; continuous density integrates to one. A density height can exceed one.'],
      ['p_i = \\frac{e^{z_i}}{\\sum_j e^{z_j}}', 'P(a \\leq X \\leq b) = \\int_a^b 2x\\,dx = b^2-a^2'],
      [['z_i', 'Logit for candidate i; j indexes all candidates.', 'z=[2,1,0] gives p=[0.6652,0.2447,0.0900].'], ['p_i', 'Normalized probability assigned to token i.', 'retrieval occupies [0,0.6652) in a uniform draw.'], ['X, a, b', 'Continuous score and selected interval endpoints.', 'For f(x)=2x on [0,1], P(0.5<=X<=0.75)=0.3125.']],
      'import numpy as np\nz = np.array([2., 1., 0.])\nw = np.exp(z - z.max())  # stable, same softmax\np = w / w.sum()         # [0.66524, 0.24473, 0.09003]\nrng = np.random.default_rng(42)\nu = rng.random(1000)\ntokens = np.searchsorted(np.cumsum(p), u, side="right")\ncounts = np.bincount(tokens, minlength=3)\n# A separate continuous example: integral of 2x on [a,b].\na, b = .5, .75\nmass = b*b - a*a        # 0.3125',
      'Add 5 to ALL three route scores, holding the sampler draw fixed. What changes?', 'Neither the probabilities nor the selected route.', 'Every route becomes more likely, so the selected route must change.', 'The common exp(5) factor cancels in softmax. All interval boundaries stay fixed; the same draw selects the same route.'),
    expectation: contentEntry('Expected risk depends on the serving population.', 'Hold segment losses fixed. Change the user mix and watch the winning model change.', 'Offline averages can hide population weighting assumptions and fail under serving shift.', 'Expected risk averages loss under a specified data distribution, not an unweighted list of segments.', ['The example uses segment-conditional mean losses; within-segment variation is not represented.', 'Expectation of a sum is the sum of expectations without an independence assumption. Variance of a sum must also account for covariance terms.'],
      ['R(f)=\\mathbb{E}_{(x,y)\\sim P}[\\ell(f(x),y)]=\\sum_g P(g)L_g(f)'],
      [['P(g)', 'Share of arrivals in segment g under population P.', 'Training: frequent 0.8, new 0.2.'], ['f, x, y, ell', 'Model, user context, observed outcome, and per-example loss.', 'Log-loss on impressions, summarized into segment means.'], ['L_g(f)', 'Mean loss of model f within segment g.', 'Model A: [0.1,0.8]; B: [0.25,0.35] nats.'], ['R(f)', 'Population-weighted risk.', 'A: 0.8*0.1+0.2*0.8=0.24; B: 0.27. Serving [0.3,0.7]: A=0.59, B=0.32.']],
      'import numpy as np\nloss_A = np.array([.1, .8])\nloss_B = np.array([.25, .35])\ntrain = np.array([.8, .2])\nserve = np.array([.3, .7])\nprint(train @ loss_A, train @ loss_B)  # 0.24, 0.27\nprint(serve @ loss_A, serve @ loss_B)  # 0.59, 0.32\n# Same models, different population weighting.',
      'Suppose A instead has lower loss than B in BOTH segments. Can changing only the user mix make B win?', 'No: every weighted loss difference still favors A.', 'Yes: population shift can always reverse the winner.', 'A reversal requires a tradeoff between segments. Nonnegative weights summing to one cannot reverse a model that is strictly better in every segment.'),
    bayes: contentEntry('A spam flag changes the denominator.', 'Count true and false flags inside the evidence-matching subset.', 'Precision changes with prevalence even when recall and false-positive rate do not.', 'Recall is P(flag|spam); precision is P(spam|flag). They condition on different populations.', ['Unflagged messages also update the posterior.', 'Holding class-conditional rates fixed models prior-probability shift, not arbitrary distribution shift.'],
      ['P(S\\mid F)=\\frac{P(F\\mid S)P(S)}{P(F\\mid S)P(S)+P(F\\mid \\neg S)P(\\neg S)}'],
      [['S, F', 'Spam status and observed flag.', '10,000 emails; 100 spam at a 1% prior.'], ['P(F|S)', 'Recall: fraction of spam messages flagged.', '0.9*100=90 true flags.'], ['P(F|not S)', 'False-positive rate among non-spam.', '0.02*9900=198 false flags.'], ['P(S|F)', 'Spam fraction among all flags.', '90/(90+198)=0.3125, not 0.90.']],
      'n, prior, recall, fpr = 10_000, .01, .9, .02\ntp = n * prior * recall       # 90\nfp = n * (1-prior) * fpr      # 198\nfn = n * prior - tp           # 10\ntn = n * (1-prior) - fp       # 9702\nprecision = tp / (tp + fp)    # 0.3125\nspam_if_no_flag = fn / (fn + tn)  # about 0.00103',
      'Spam rises from 1% to 10%, but recall and false-positive rate stay fixed. Among flags, does the spam share change?', 'It rises from 31.25% to 83.33%.', 'It stays 31.25% because the filter did not change.', 'In 10,000 emails, 900 spam and 180 non-spam now survive. Spam share is 900/1080. The filter rates are unchanged; the incoming population is different.'),
    entropy: contentEntry('Entropy is average surprise, not maximum surprise.', 'Complete one sentence: inspect how often each next token occurs, how surprising it is, and why its region area contributes to the average.', 'Rare events can be individually surprising without dominating average uncertainty. A mismatched model adds avoidable expected surprise.', 'H(p,q)=H(p)+KL(p||q): only the mismatch term can be eliminated by matching a fixed p.', ['The reference counts are an illustrative 100-continuation dataset for one context, treated as p. They are not measured language frequencies or a claim about a real model.', 'The distribution for a context is not the one-hot label of a single sample.', 'Individual KL terms may be negative, although the sum is nonnegative. Positive p with zero q gives infinite cross-entropy.'],
      ['I_q(i)=-\\log_2 q_i', 'H(p,q)=-\\sum_i p_i\\log_2 q_i=H(p)+D_{KL}(p\\Vert q)', 'D_{KL}(p\\Vert q)=\\sum_i p_i\\log_2(p_i/q_i)'],
      [['p_i, q_i', 'Reference next-token frequency and predicted probability.', 'After "She poured the coffee into the ...": cup/mug/glass counts [60,30,10] give p=[0.6,0.3,0.1]; overconfident q=[0.9,0.09,0.01].'], ['I_q(i)', 'Surprise in bits when token i occurs.', 'With q=p, glass costs -log2(0.1)=3.322 bits, but occurs only 10/100 times: contribution=0.3322.'], ['H(p)', 'Mean surprise under the reference probabilities.', 'H([0.6,0.3,0.1])=1.2955 bits: sum the three probability-times-surprise areas.'], ['D_KL(p||q)', 'Extra expected code length caused by mismatched probabilities.', 'For overconfident q: H(p,q)=1.7978, KL=0.5023 bits (rounded).']],
      'from math import log2\n# Illustrative next tokens: cup, mug, glass.\ncounts = [60, 30, 10]\np = [n / sum(counts) for n in counts]\nq = [.9, .09, .01]  # contrast with q = p\nh = sum(-pi*log2(pi) for pi in p if pi)\nce = sum(-pi*log2(qi) for pi, qi in zip(p, q) if pi)\nkl = sum(pi*log2(pi/qi) for pi, qi in zip(p, q) if pi)\nassert abs(ce - h - kl) < 1e-12\n# Natural-log training uses nats: multiply bits by ln(2).',
      'Keep the reference glass frequency at 10%, but halve its model probability. How does the GLASS contribution change?', 'One extra bit per glass, so +0.1 bits per continuation.', 'One extra bit per continuation, regardless of glass frequency.', 'Halving q adds one bit to -log2(q). Weighting by p=0.1 adds 0.1 to this token contribution. Other probabilities must also change to keep q normalized, so this is not necessarily the change in total cross-entropy.'),
    loss: contentEntry('The loss changes the gradient on a confident miss.', 'Compare log-loss and squared probability error on the same clicked impression.', 'Sigmoid plus squared loss can attenuate gradients on confident errors; sigmoid cross-entropy avoids that additional factor.', 'Both Brier score and log-loss are proper probability losses, but their curvature and gradients differ.', ['Expected losses use a fixed true click probability p=0.1.', 'Log-loss values are in nats; Brier values are unitless. Raw numerical scales are not interchangeable.'],
      ['q=\\sigma(z),\\quad \\ell_{log}=-y\\ln q-(1-y)\\ln(1-q)', '\\ell_{sq}=(q-y)^2', '\\frac{\\partial\\ell_{log}}{\\partial z}=q-y,\\quad \\frac{\\partial\\ell_{sq}}{\\partial z}=2(q-y)q(1-q)'],
      [['y, q', 'Binary observed label and predicted click probability.', 'y=1, q=0.01: log-loss=4.6052; squared loss=0.9801.'], ['z, sigma', 'Logit and sigmoid function.', 'z=ln(0.01/0.99)=-4.5951; sigmoid(z)=0.01.'], ['partial ell / partial z', 'Training signal before the sigmoid.', 'Log-loss gradient=-0.99; squared-loss gradient=-0.019602.']],
      'from math import log\ny, q = 1, .01\nlog_loss = -log(q if y else 1-q)  # 4.60517\nsquared = (q-y)**2               # 0.9801\ndlog_dz = q-y                    # -0.99\ndsq_dz = 2*(q-y)*q*(1-q)         # -0.019602\n# In production, evaluate BCE directly from logits for stability.',
      'For a clicked impression, make the wrong prediction even closer to zero. What happens to the two logit gradients?', 'Log-loss approaches -1; squared-loss approaches 0.', 'Both grow without bound because the mistake is more confident.', 'As q tends to zero with y=1, q-y tends to -1, but 2(q-y)q(1-q) tends to zero. The log-loss VALUE diverges; its logit gradient does not.'),
    vectors: contentEntry('An embedding has a direction and a norm.', 'Inspect real coordinates before and after unit normalization.', 'Normalization can change retrieval rankings by discarding norm information.', 'L2 normalization preserves a nonzero vector direction and forces its norm to one; zero needs explicit handling.', ['Learned latent axes need not have human-readable semantic meanings.', 'Normalization is a modeling choice, not an automatically correct cleanup step.'],
      ['\\|v\\|_2=\\sqrt{\\sum_i v_i^2},\\quad \\hat v=v/\\|v\\|_2'],
      [['v, v_i', 'Item embedding and its coordinate i.', 'Item B=[2,-0.1].'], ['||v||_2', 'Euclidean length.', 'sqrt(4+0.01)=2.0025.'], ['v-hat', 'Unit direction, defined only when v is nonzero.', 'B/2.0025=[0.9988,-0.04994]; its length is 1.']],
      'import numpy as np\nv = np.array([2., -.1])\nn = np.linalg.norm(v)  # 2.002498...\nif n == 0:\n    raise ValueError("zero embedding has no direction")\nu = v / n\nassert np.allclose(np.linalg.norm(u), 1)\nassert np.allclose(n * u, v)',
      'Multiply a nonzero embedding by 3 before normalizing it. Does its unit direction change?', 'No: both its coordinates and its norm grow by 3.', 'Yes: the normalized vector becomes three times longer.', 'For positive c, cv / ||cv|| = v / ||v||. A negative multiplier would reverse the direction; a zero vector still has no unit direction.'),
    'dot-products': contentEntry('Raw retrieval and cosine can choose different items.', 'Inspect coordinate products, geometric projection, and the resulting ranking.', 'Two-tower systems must align training, indexing, and serving similarity metrics.', 'Dot product combines norms and angle; cosine removes norm factors.', ['Orthogonal vectors have zero inner product. Negative inner products correspond to obtuse angles.', 'Norm effects are learned, not inherently desirable or undesirable.'],
      ['q^Tv=\\sum_i q_i v_i=\\|q\\|_2\\|v\\|_2\\cos\\theta', '\\operatorname{proj}_q(v)=\\frac{q^Tv}{q^Tq}q'],
      [['q, v', 'Query and item embeddings.', 'q=[1,0.5], A=[0.9,0.6], B=[2,-0.1].'], ['q^T v', 'Raw retrieval score.', 'A=1.2; B=1.95, so raw dot ranks B first.'], ['theta, cos theta', 'Angle and normalized directional similarity.', 'A cosine=0.9923; B cosine=0.8710, so cosine ranks A first.'], ['proj_q(v)', 'Component of v parallel to q.', 'For A: (1.2/1.25)*[1,0.5]=[0.96,0.48].']],
      'import numpy as np\nq = np.array([1., .5])\nitems = np.array([[.9, .6], [2., -.1], [-.5, 1.]])\nraw = items @ q                       # [1.2, 1.95, 0]\ncosine = raw / (np.linalg.norm(items, axis=1)*np.linalg.norm(q))\nprint(np.argmax(raw), np.argmax(cosine))  # 1, 0\nv = items[0]\nprojection = (q @ v)/(q @ q)*q         # [0.96, 0.48]',
      'Scale only the query by 10, leaving every item fixed. Must either retrieval ranking change?', 'No: all dot scores scale by 10; cosine scores stay unchanged.', 'Yes: the highest-norm item necessarily becomes the winner.', 'A common positive scale preserves dot-score order, and cosine divides out the query norm. Changing individual item norms is different: those factors are not shared.'),
    'matrix-multiply': contentEntry('A linear layer is a weighted sum of columns.', 'Trace input coordinates into transformed basis vectors and inspect the output row products.', 'The same linear map acts on every input; a singular map loses information.', 'Column j is A e_j. Ax sums those mapped basis vectors, weighted by input coordinates.', ['The determinant scales signed area; zero determinant means the map is not invertible.', 'A linear layer may change dimension; this exact 2D example keeps the geometry visible.'],
      ['y=Ax=x_1 A e_1+x_2 A e_2', 'y_i=\\sum_j A_{ij}x_j'],
      [['A, A_ij', 'Linear map and entry at output row i, input column j.', 'A=[[1,1],[0,1]].'], ['x_j, e_j', 'Input coefficient and coordinate basis vector.', 'x=[1,2]; e1=[1,0], e2=[0,1].'], ['y_i', 'Output coordinate i.', 'Ax=1*[1,0]+2*[1,1]=[3,2].']],
      'import numpy as np\nA = np.array([[1., 1.], [0., 1.]])\nx = np.array([1., 2.])\ncol_sum = x[0]*A[:, 0] + x[1]*A[:, 1]\ny = A @ x                         # [3,2]\nassert np.allclose(col_sum, y)\nassert np.isclose(np.linalg.det(A), 1)  # shear preserves area',
      'Use A=[[1,1],[1,1]]. Can its output distinguish inputs [1,2] and [2,1]?', 'No: both map to [3,3].', 'Yes: different input coordinates always produce different outputs.', 'The input difference [1,-1] maps to zero. A loses that direction, so the map cannot be inverted even if the output is known exactly.'),
    eigen: contentEntry('Find the directions preserved by a matrix.', 'Compare an ordinary direction with exact eigenvectors; follow normalized power iteration.', 'Covariance eigenvectors define PCA axes. Hessian eigenvalues describe local curvature, including negative directions.', 'For symmetric matrices an orthonormal eigenbasis reconstructs A; equal eigenvalues need not define a unique axis.', ['A negative eigenvalue reverses orientation but preserves the eigenvector line.', 'Power iteration needs a dominant absolute eigenvalue and an initial component along its eigendirection.'],
      ['Av_i=\\lambda_i v_i,\\quad A=Q\\Lambda Q^T', '\\rho(v)=\\frac{v^TAv}{v^Tv},\\quad v_{t+1}=\\frac{Av_t}{\\|Av_t\\|_2}'],
      [['A, v_i, lambda_i', 'Symmetric map, eigenvector, and signed scale.', 'A=[[2,1],[1,2]]; v1=[1,1]/sqrt(2), lambda1=3; lambda2=1.'], ['Q, Lambda', 'Columns of unit eigenvectors and diagonal eigenvalues.', 'Q diag(3,1) Q^T reconstructs [[2,1],[1,2]].'], ['rho(v)', 'Rayleigh quotient: directional quadratic form per squared norm.', 'v=[1,0]: rho=2 but Av=[2,1] is not parallel to v.'], ['v_t', 'Normalized iterate after t multiplications.', 'v0=[1,0]; v1=[2,1]/sqrt(5)=[0.8944,0.4472].']],
      'import numpy as np\nA = np.array([[2., 1.], [1., 2.]])\nlam, Q = np.linalg.eigh(A)  # ascending eigenvalues [1,3]\nassert np.allclose(A, Q @ np.diag(lam) @ Q.T)\nv = np.array([1., 0.])\nfor _ in range(5):\n    v = A @ v\n    v /= np.linalg.norm(v)\nrho = v @ A @ v\nresidual = np.linalg.norm(A @ v - rho * v)',
      'Start power iteration exactly on the weaker eigenvector. Will repeated steps discover the leading direction?', 'No, not in exact arithmetic: that line is preserved.', 'Yes: enough steps always find the largest eigenvalue.', 'A zero starting component along the leading eigenvector remains zero. Floating-point noise can eventually introduce a component; that is not a guarantee of the exact iteration.'),
    svd: contentEntry('Measure exactly what rank truncation discards.', 'Trace one vector through the computed factors and inspect reconstructed matrix entries.', 'Low-rank structure supports compression and latent factor models, but matrix energy is not task quality.', 'Truncated SVD is optimal for Frobenius error; discarded squared singular values give the exact squared error.', ['Orthogonal factors may include reflections, not only rotations.', 'A full SVD of a 2x2 example is pedagogical, not a storage saving. For a large m-by-n matrix, factors can reduce storage to k(m+n+1).'],
      ['A=U\\Sigma V^T=\\sum_i\\sigma_i u_i v_i^T', 'A_k=\\sum_{i=1}^k\\sigma_i u_i v_i^T,\\quad \\|A-A_k\\|_F^2=\\sum_{i>k}\\sigma_i^2'],
      [['A, U, V', 'Original map and orthonormal output/input bases.', 'A=[[3,1],[0,1]]; computed U and V reconstruct A.'], ['sigma_i, Sigma', 'Nonnegative singular values and their diagonal matrix.', 'sigma=[3.1796,0.9435] (rounded).'], ['u_i, v_i, k', 'Corresponding basis columns and number of kept components.', 'Rank k=1 keeps sigma1*u1*v1^T only.'], ['||A-A_k||_F^2', 'Sum of squared entrywise reconstruction errors.', 'At k=1, error squared=sigma2^2=0.8902; retained energy=91.9%.']],
      'import numpy as np\nA = np.array([[3., 1.], [0., 1.]])\nU, s, Vt = np.linalg.svd(A, full_matrices=False)\nfull = (U * s) @ Vt\nrank1 = s[0] * np.outer(U[:, 0], Vt[0])\nassert np.allclose(full, A)\nerror_sq = np.linalg.norm(A-rank1, "fro")**2\nassert np.isclose(error_sq, s[1]**2)\nretained = s[0]**2 / (s @ s)  # about 91.9%',
      'Change singular values from [5,1] to [5,2], still keeping rank one. What happens to squared reconstruction error?', 'It quadruples, from 1 to 4.', 'It doubles, from 1 to 2.', 'Rank-one squared Frobenius error is the square of the discarded singular value. This is an exact matrix-error statement, not a prediction of retrieval quality.'),
  };

  const mechanisms = {
    distribution: 'A uniform random number selects the token whose cumulative-probability interval contains it. Interval widths are exactly softmax probabilities.',
    expectation: 'Each segment contributes its probability of arrival times its conditional mean loss. A small segment with large errors can outweigh a large segment with small errors.',
    bayes: 'After observing a flag, discard the unflagged column. The remaining spam count divided by the remaining total is the posterior.',
    entropy: 'In the diagram, width is reference frequency p and height is surprise -log2(q). Each region area is p times surprise. When q=p, the sum is entropy; using a different q gives cross-entropy.',
    loss: 'Different loss derivatives pass different signals through the sigmoid. The relevant comparison is the gradient with respect to the logit, not only the reported loss.',
    vectors: 'Dividing every coordinate by the same positive norm moves a nonzero embedding onto the unit sphere without turning its direction.',
    'dot-products': 'Project an item onto the query direction. The dot product is this signed projected length times query length; normalization removes both length factors.',
    'matrix-multiply': 'Multiply each matrix column by the corresponding input coordinate, then add the resulting vectors. This is the same computation as taking each row dot input.',
    eigen: 'The residual Av minus the Rayleigh-scaled vector measures how much the map pushes off the current line. It vanishes at an eigenvector.',
    svd: 'Express an input in the V basis, scale its coordinates by singular values, then combine the output basis columns U. Dropping a singular value removes that entire input-to-output component.',
  };
  for (const [kind, what] of Object.entries(mechanisms)) content[kind].what = what;

  const api = { normalize, softmax, expectedRisk, information, bayesCounts, binaryLoss, binaryRisk, lossStep, categorical, sampleDistribution, densityMass, dot, norm, unit, cosine, matvec, multiply, transpose, det, frobenius, matrixSubtract, eigenSymmetric, reconstructEigen, svd2, reconstructSvd, powerStep, distributionLogits, initial, reduce, render, content };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window !== 'undefined') {
    window.AtelierLab.createModule({ id: 'foundations-labs', content, initial, render, reduce });
  }
})();
