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
  const heading = (kicker, question) => `<header class="ml-heading"><p class="ml-kicker">${esc(kicker)}</p><h3 class="ml-question">${esc(question)}</h3></header>`;
  const inspector = (title, lines, data = '') => `<aside class="ml-inspector" aria-live="polite"><h4>${esc(title)}</h4><p>${esc(lines[0])}</p>${lines.length > 1 || data ? calculation(data + lines.slice(1).map(line => `<p>${esc(line)}</p>`).join(''), 'inspector-calculation') : ''}</aside>`;
  function table(headers, rows, caption = '') {
    return `<div class="ml-table"><table>${caption ? `<caption>${esc(caption)}</caption>` : ''}<thead><tr>${headers.map(h => `<th scope="col">${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map((x, i) => i === 0 ? `<th scope="row">${x}</th>` : `<td>${x}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  }
  function matrix(A, label) {
    return `<div class="fl-matrix"><span>${esc(label)}</span><div role="img" aria-label="${esc(`${label}: ${A.map(vec).join('; ')}`)}">${A.flat().map(x => `<span>${esc(fmt(x))}</span>`).join('')}</div></div>`;
  }
  function navigation(step, labels) {
    return `<nav class="ml-step-nav fl-nav" aria-label="Computation trace"><button type="button" data-action="previous" ${step === 0 ? 'disabled' : ''}>Previous</button><span>${step + 1}/${labels.length}: ${esc(labels[step])}</span><button class="fl-primary" type="button" data-action="${step === labels.length - 1 ? 'restart' : 'next'}">${step === labels.length - 1 ? 'Start again' : 'Next'}</button></nav>`;
  }
  const colors = ['blue', 'orange', 'green'];
  // Fixed plot coordinates and HTML legends keep SVG labels out of crowded geometry.
  function plane({ arrows = [], points = [], shape = [], range = 4, label = 'Vector geometry', unitCircle = false, gridMap = null }) {
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

  function initial(kind) {
    const states = {
      distribution: { mode: 'categorical', scene: 'peaked', selected: 0, samples: 0, counts: [0, 0, 0], seed: 42, last: null },
      expectation: { population: 'train', model: 'A', selected: 1 },
      bayes: { prior: .01, positive: true, step: 0 },
      entropy: { model: 'matched', selected: 2 },
      loss: { q: .01, y: 1, mode: 'observed' },
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
    if (action === 'next' || action === 'previous') s.step = Math.max(0, Math.min(kind === 'bayes' ? 2 : kind === 'eigen' ? 5 : 3, s.step + (action === 'next' ? 1 : -1)));
    if (kind === 'distribution') {
      if (action === 'mode' && ['categorical', 'density'].includes(value)) { s.mode = value; s.selected = 0; }
      if (action === 'scene' && distributionCases[value]) Object.assign(s, initial(kind), { scene: value });
      if (action === 'select' && Number.isInteger(n) && n >= 0 && n < (s.mode === 'density' ? 4 : 3)) s.selected = n;
      if (action === 'sample' && [1, 100, 1000].includes(n)) {
        const result = sampleDistribution(softmax(distributionCases[s.scene]), n, s.seed);
        s.counts = add(s.counts, result.counts); s.samples += n; s.seed = result.seed; s.last = result.last;
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
      if (action === 'q' && [.01, .1, .3, .5, .7, .9, .99].includes(n)) s.q = n;
      if (action === 'label' && [0, 1].includes(n)) s.y = n;
      if (action === 'mode' && ['observed', 'expected'].includes(value)) s.mode = value;
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
    return s;
  }

  function distributionView(s) {
    const switches = controls('Distribution type', btn('mode', 'categorical', 'Token probabilities', s.mode === 'categorical') + btn('mode', 'density', 'Continuous score density', s.mode === 'density'));
    if (s.mode === 'density') {
      const a = s.selected / 4, b = a + .25, mass = densityMass(a, b);
      return heading('Distribution / density is not probability', 'Can a density be greater than one?') + primary('select', (s.selected + 1) % 4, 'Inspect the next interval') + `<div class="ml-grid"><div class="ml-stage"><svg viewBox="0 0 300 220" class="fl-density" role="group" aria-label="Density f(x)=2x on [0,1]; choose a quarter interval">${[0, 1, 2, 3].map(i => { const x = i / 4, y = x + .25; return `<g role="button" tabindex="0" data-action="select" data-value="${i}" aria-pressed="${s.selected === i}" aria-label="Interval ${x} to ${y}: probability ${pct(densityMass(x, y))}" class="fl-density-region ${s.selected === i ? 'fl-density-selected' : ''}"><rect class="fl-hit" x="${35 + 230 * x}" y="25" width="57.5" height="155"/><path d="M ${35 + 230 * x},180 L ${35 + 230 * x},${180 - 145 * x} L ${35 + 230 * y},${180 - 145 * y} L ${35 + 230 * y},180 Z"/></g>`; }).join('')}<path class="fl-axis" d="M35 25V180H270"/><text x="12" y="40">2</text><text x="28" y="204">0</text><text x="261" y="204">1</text><text x="55" y="30">density f(x) = 2x</text></svg>${note('Area is probability. Tap an interval to compare equal widths.')}</div>${inspector(`[${a}, ${b}] captures ${pct(mass)}`, ['The area counts probability; the curve height does not.', `Integral of 2x = b^2 - a^2 = ${fmt(b * b)} - ${fmt(a * a)} = ${fmt(mass)}.`, `At x = ${b}, density = ${2 * b}; P(X = ${b}) = 0 for this continuous distribution.`, 'This is a normalized toy score distribution, not a calibrated click-probability claim.'])}</div>${settings(switches)}`;
    }
    const z = distributionCases[s.scene], p = softmax(z), i = s.selected;
    const before = sum(p.slice(0, i)), after = before + p[i];
    let cursor = 0;
    const intervals = p.map((pi, j) => { const left = cursor; cursor += pi; return `<g role="button" tabindex="0" data-action="select" data-value="${j}" aria-pressed="${i === j}" aria-label="${tokenNames[j]}, probability ${pct(pi)}" class="fl-${colors[j]} fl-interval ${i === j ? 'fl-interval-selected' : ''}"><rect x="${20 + 260 * left}" y="30" width="${260 * pi}" height="48"/></g>`; }).join('');
    return heading('Distribution / next-token sampling', 'Must the most likely token win every draw?')
      + primary('sample', 1, s.samples ? 'Draw another token' : 'Draw a token')
      + `<div class="ml-grid"><div class="ml-stage"><svg viewBox="0 0 300 125" role="group" aria-label="Categorical inverse-CDF sampler; interval width is probability">${intervals}<text x="20" y="105">0</text><text x="280" y="105" text-anchor="end">1</text>${s.last ? `<path class="fl-sample-marker" d="M${20 + 260 * s.last.u} 17v68"/>` : ''}</svg>${legend(tokenNames.map((name, j) => [colors[j], `${name}: ${pct(p[j])}`]))}${note(s.last ? `Draw ${s.samples}: the marker lands in ${tokenNames[s.last.outcome]}.` : 'A uniform random point picks the region it lands in. Width is probability.')}</div>${inspector(`Inspect ${tokenNames[i]}`, [`${pct(p[i])} chance on each draw. Less likely is not impossible.`, `exp(${z[i]}) / sum(exp(logits)) = ${fmt(p[i], 4)}.`, `A uniform draw u in [${fmt(before, 4)}, ${fmt(after, 4)}) selects this token.`, s.last ? `Last actual draw: u = ${fmt(s.last.u, 4)} selects ${tokenNames[s.last.outcome]}.` : 'Sampling is not argmax.', s.samples ? `${s.samples} draws: observed share ${pct(s.counts[i] / s.samples)}, model share ${pct(p[i])}.` : 'Draw tokens to compare empirical frequencies with the model.'], table(['Token', 'Logit', 'p', 'Draws'], p.map((pi, j) => [btn('select', j, tokenNames[j], i === j), esc(fmt(z[j])), esc(pct(pi)), esc(s.counts[j])])) )}</div>`
      + settings(switches + controls('Logit cases', Object.entries({ peaked: 'Prefer retrieval', flat: 'Equal logits', shifted: 'Prefer generation' }).map(([key, text]) => btn('scene', key, text, s.scene === key)).join('')) + controls('Sampling experiment', btn('sample', 100, 'Draw 100') + btn('sample', 1000, 'Draw 1,000') + btn('reset', '', 'Reset')) + note('Seeded pseudorandom draws are reproducible, not hand-built frequencies. Changing logits starts a fresh sample.'));
  }

  function expectationView(s) {
    const p = populations[s.population], losses = modelLosses[s.model], risk = expectedRisk(p, losses), other = expectedRisk(p, modelLosses[s.model === 'A' ? 'B' : 'A']);
    const names = ['Frequent users', 'New users'];
    return heading('Expectation / population-weighted model risk', 'Will the same model win when the users change?')
      + primary('population', s.population === 'train' ? 'shifted' : 'train', s.population === 'train' ? 'Apply the serving user mix' : 'Restore the training user mix')
      + note(`${names[s.selected]} contribute ${fmt(risk.contributions[s.selected])} of the ${fmt(risk.mean)} total expected loss. Tap a group to inspect its share.`)
      + `<div class="ml-grid"><div class="ml-stage"><div class="fl-cohort" role="group" aria-label="20 population units; each is 5 percent">${Array.from({ length: 20 }, (_, i) => { const group = i < Math.round(p[0] * 20) ? 0 : 1; return `<button type="button" class="fl-person fl-${colors[group]}" data-action="select" data-value="${group}" aria-label="Inspect ${names[group]}" aria-pressed="${group === s.selected}"><span></span></button>`; }).join('')}</div>${legend([['blue', `${pct(p[0])} frequent`], ['orange', `${pct(p[1])} new`]])}${note('Each dot is 5% of arrivals. Models and their segment losses stay fixed.')}</div>${inspector(`Model ${s.model} ${risk.mean < other.mean ? 'wins' : 'loses'} on this population`, [`Expected loss: ${fmt(risk.mean)} versus ${fmt(other.mean)} for the other model. Lower is better.`, `${names[s.selected]}: ${fmt(p[s.selected])} x ${fmt(losses[s.selected])} = ${fmt(risk.contributions[s.selected])} nats per arrival.`, `Total = ${risk.contributions.map(x => fmt(x)).join(' + ')} = ${fmt(risk.mean)}.`, `Unweighted average = ${fmt(sum(losses) / 2)}: it silently assumes a 50/50 population.`], table(['Segment', 'p', 'Loss', 'p x loss'], p.map((pi, i) => [btn('select', i, names[i], s.selected === i), esc(fmt(pi)), esc(fmt(losses[i])), esc(fmt(risk.contributions[i]))])) )}</div>`
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
    return heading('Bayes / same classifier, different base rate', 'Does a spam flag mean the email is probably spam?')
      + navigation(s.step, titles)
      + `<div class="ml-grid"><div class="ml-stage">${s.step === 1 ? evidenceTable + note(`Keep the outlined ${s.positive ? 'flagged' : 'not-flagged'} column: ${fmt(b.numerator)} spam + ${fmt(b.other)} non-spam.`) : strip}</div>${inspector(titles[s.step], s.step === 0 ? [`Prior: ${fmt(b.spam)} / 10,000 = ${pct(s.prior)}.`, 'The prior counts spam among every email, before observing a flag.', 'Next: apply the same classifier to both groups, then retain only evidence-matching emails.'] : s.step === 1 ? [`Keep ${fmt(b.numerator)} spam and ${fmt(b.other)} non-spam emails. Next, this subset becomes the new whole.`, `Spam: ${fmt(b.spam)} x .90 = ${fmt(b.tp)} flags; ${fmt(b.fn)} remain unflagged.`, `Not spam: ${fmt(b.ham)} x .02 = ${fmt(b.fp)} false alarms; ${fmt(b.tn)} remain unflagged.`] : [`Posterior: ${fmt(b.numerator)} / (${fmt(b.numerator)} + ${fmt(b.other)}) = ${pct(b.posterior)} spam.`, `The denominator is ${fmt(b.evidence)} evidence-matching emails, not 10,000.`, s.positive ? 'Change the base rate: recall and false-positive rate stay fixed; precision changes.' : 'No flag is evidence too: false negatives remain in the unflagged subset.'], s.step === 0 ? populationTable : s.step === 2 ? evidenceTable : '')}</div>`
      + settings(controls('Inbox spam base rate', [.01, .1, .5].map(p => btn('prior', p, `${pct(p)} spam`, s.prior === p)).join('')) + controls('Observed evidence', btn('evidence', 'positive', 'Flagged email', s.positive) + btn('evidence', 'negative', 'Not flagged', !s.positive)) + note('Synthetic expected counts; fixed 90% recall and 2% false-positive rate. Rectangle widths are exact fractions, without minimum-width inflation.'));
  }

  function entropyView(s) {
    const p = trueTokens, q = tokenModels[s.model], info = information(p, q), i = s.selected, t = info.terms[i];
    let left = 40;
    const regions = p.map((pi, j) => {
      const width = 280 * pi, height = 24 * info.terms[j].surprise;
      const rect = `<g role="button" tabindex="0" data-action="select" data-value="${j}" aria-pressed="${i === j}" class="fl-surprise-region ${i === j ? 'fl-surprise-selected' : ''}" aria-label="${esc(`${entropyNames[j]}: ${continuationCounts[j]} of 100 continuations, ${fmt(info.terms[j].surprise)} bits each, ${fmt(info.terms[j].crossEntropy)} bits toward the average`)}"><rect class="fl-hit" x="${left}" y="30" width="${width}" height="175"/><rect class="fl-surprise-area" data-probability="${pi}" data-surprise="${info.terms[j].surprise}" x="${left}" y="${205 - height}" width="${width}" height="${height}"/></g>`;
      left += width;
      return rect;
    }).join('');
    const matched = s.model === 'matched';
    return heading('Entropy / average surprise', 'Does the most surprising token dominate the average?')
      + primary('model', matched ? 'confident' : 'matched', matched ? 'Make the model overconfident' : 'Match the reference again')
      + `<div class="fl-token-context"><p>She poured the coffee into the <strong>...</strong></p><p class="ml-note">Illustrative set of 100 continuations, not measured language frequencies.</p></div>`
      + controls('Inspect a reference outcome', entropyNames.map((name, j) => btn('select', j, `${name} ${pct(p[j])}`, j === i)).join(''))
      + `<div class="ml-grid"><div class="ml-stage"><h4>Area is the contribution to average surprise</h4><p class="ml-note">Height: surprise in bits per occurrence</p><svg class="fl-surprise-plot" viewBox="0 0 340 240" role="group" aria-label="Outcome regions: width is true frequency, height is surprise under the model, total area is average surprise">${[0, 2, 4, 6].map(v => `<path class="fl-grid-line" d="M40 ${205 - 24 * v}H320"/><text x="32" y="${210 - 24 * v}" text-anchor="end">${v}</text>`).join('')}${regions}<path class="fl-axis" d="M40 30V205H320"/><text x="40" y="231">0</text><text x="320" y="231" text-anchor="end">100%</text></svg><p class="fl-axis-caption">Width: fraction of reference continuations. Selected region: ${esc(entropyNames[i])}. Both axes keep the same scale when q changes.</p>${note(matched ? 'Glass is tallest, but narrow: only 10 of 100 outcomes incur that much surprise. Add all three areas, not their heights.' : 'The widths do not change: reality p is fixed. Changing model q changes the heights, so the total area is cross-entropy.')}</div>
      <aside class="ml-inspector fl-surprise-inspector" aria-live="polite"><h4>One ${esc(entropyNames[i])}: ${fmt(t.surprise)} bits</h4><p>The model gives this token ${pct(q[i])} probability.</p><svg viewBox="0 0 300 86" role="img" aria-label="${esc(`Surprise ${fmt(t.surprise)} bits is ${fmt(t.surprise)} halvings of probability from 1`)}"><path class="fl-axis" d="M20 34H280"/>${Array.from({ length: 8 }, (_, bit) => `<path class="fl-axis" d="M${20 + bit * 260 / 7} 28V40"/><text x="${20 + bit * 260 / 7}" y="65" text-anchor="middle">${bit}</text>`).join('')}<circle class="fl-surprise-marker" cx="${20 + t.surprise * 260 / 7}" cy="34" r="6"/></svg><p class="ml-note">Each halving adds one bit: probability 1/2 costs 1 bit; 1/4 costs 2.</p><div class="fl-surprise-average"><strong>${continuationCounts[i]} / 100 occurrences x ${fmt(t.surprise)} bits = ${fmt(t.crossEntropy)} bits toward the average.</strong></div><p>${matched ? 'Rare means tall but narrow. Average surprise adds the areas, not the heights.' : 'Its real frequency stays fixed. Only the model probability, and therefore surprise, changed.'}</p></aside></div>
      <div class="fl-entropy-decomposition"><h4>${matched ? 'Entropy: model and reference agree' : 'Cross-entropy: task uncertainty plus mismatch'}</h4><p><strong>${fmt(info.entropy)}</strong> H(p) + <strong>${fmt(info.kl)}</strong> KL(p || q) = <strong>${fmt(info.crossEntropy)}</strong> H(p,q) bits / continuation</p><div class="fl-information-mass" role="img" aria-label="${esc(`${fmt(info.entropy)} bits entropy plus ${fmt(info.kl)} bits KL equals ${fmt(info.crossEntropy)} bits cross-entropy; common scale 0 to 2 bits`)}"><span class="fl-information-entropy" style="width:${info.entropy / 2 * 100}%"></span><span class="fl-information-kl" style="width:${Math.max(0, info.kl) / 2 * 100}%"></span></div>${legend([['blue', 'Reference entropy: unchanged'], ['orange', 'Extra expected loss: KL']])}<p class="fl-axis-caption">Fixed scale: full track = 2 bits per continuation. No minimum-width inflation; matching p makes the orange segment disappear.</p></div>
      <details class="fl-entropy-terms"><summary>Inspect the calculation</summary>${note(`Selected surprise: -log2(${fmt(q[i])}) = ${fmt(t.surprise)} bits. ${continuationCounts[i]} occurrences cost ${fmt(continuationCounts[i] * t.surprise)} bits per 100 continuations, or ${fmt(t.crossEntropy)} bits each on average.`)}${table(['Token', 'p', 'q', 'p x surprise', 'KL term'], p.map((pi, j) => [esc(entropyNames[j]), fmt(pi), fmt(q[j]), fmt(info.terms[j].crossEntropy), fmt(info.terms[j].kl)]))}${note('Individual KL terms can be negative; their sum cannot. For one observed token, training scores -ln q(token). Bits convert to nats by multiplying by ln 2.')}</details>`
      + settings(controls('Model probabilities', Object.entries({ matched: 'Match p', confident: 'Overconfident in cup', uniform: 'Uniform predictions' }).map(([key, name]) => btn('model', key, name, key === s.model)).join('')));
  }

  function lossView(s) {
    const evaluate = q => s.mode === 'observed' ? binaryLoss(s.y, q) : binaryRisk(.1, q);
    const loss = evaluate(s.q), path = key => Array.from({ length: 99 }, (_, i) => { const q = (i + 1) / 100, v = evaluate(q)[key]; return `${i ? 'L' : 'M'}${35 + q * 230},${180 - v * 30}`; }).join(' ');
    const nextQ = s.mode === 'expected' ? (s.q === .1 ? .99 : .1) : (s.q < .5 ? .99 : .01);
    return heading('Loss / binary click prediction', 'How does a confident mistake change learning?')
      + primary('q', nextQ, s.mode === 'expected' && nextQ === .1 ? 'Predict the real click rate' : nextQ > .5 ? 'Predict a 99% chance of clicking' : 'Predict a 1% chance of clicking')
      + `<div class="ml-grid"><div class="ml-stage"><svg class="fl-loss" viewBox="0 0 300 220" role="img" aria-label="${esc(`Log-loss ${fmt(loss.log)} and squared loss ${fmt(loss.square)} at probability ${s.q}`)}"><path class="fl-axis" d="M35 25V180H272"/>${[0, 2, 4].map(v => `<text x="25" y="${184 - v * 30}" text-anchor="end">${v}</text>`).join('')}<text x="35" y="205">0</text><text x="260" y="205">1</text><text x="150" y="205" text-anchor="middle">prediction q</text><path class="fl-curve fl-blue" d="${path('log')}"/><path class="fl-curve fl-orange" d="${path('square')}"/><line class="fl-guide" x1="${35 + s.q * 230}" x2="${35 + s.q * 230}" y1="25" y2="180"/><circle class="fl-dot-blue" cx="${35 + s.q * 230}" cy="${180 - loss.log * 30}" r="5"/><circle class="fl-dot-orange" cx="${35 + s.q * 230}" cy="${180 - loss.square * 30}" r="5"/></svg>${legend([['blue', `Log-loss: ${fmt(loss.log)} nats`], ['orange', `Squared error: ${fmt(loss.square)}`]])}</div>${inspector(s.mode === 'observed' ? `${s.y ? 'Clicked' : 'Not clicked'}; model predicts ${pct(s.q)} click chance` : `Real click rate 10%; model predicts ${pct(s.q)}`, [`Logit gradient: log-loss ${fmt(loss.logGradient, 4)}, squared error ${fmt(loss.squareGradient, 4)}.`, 'q = sigmoid(z). Squared loss multiplies the residual by 2q(1-q), suppressing gradients near 0 and 1.', s.mode === 'observed' ? 'On a confident wrong prediction, log-loss avoids this extra sigmoid saturation factor.' : 'Both expected losses are minimized at q = p = 0.1. They are proper probability scores, with different curvature.', 'Log-loss diverges at a wrong endpoint; squared probability error is bounded by 1. Raw loss scales are not interchangeable.'])}</div>`
      + settings(controls('Loss view', btn('mode', 'observed', 'One impression', s.mode === 'observed') + btn('mode', 'expected', 'Expected: 10% click rate', s.mode === 'expected')) + (s.mode === 'observed' ? controls('Observed label', btn('label', 1, 'Clicked: y = 1', s.y === 1) + btn('label', 0, 'No click: y = 0', s.y === 0)) : '') + controls('Predicted click probability', [.01, .1, .3, .5, .7, .9, .99].map(q => btn('q', q, `q = ${q}`, s.q === q)).join('')));
  }

  function vectorView(s) {
    const x = embeddings[s.item], u = unit(x), shown = s.normalized && u ? u : x;
    const points = Object.entries(embeddings).map(([name, v]) => ({ v: s.normalized ? unit(v) || v : v, action: 'item', value: name, label: `Item ${name}`, selected: s.item === name }));
    return heading('Vectors / embedding norm and direction', 'What does normalizing an embedding remove?')
      + primary('normalize', !s.normalized, s.normalized ? 'Restore the original lengths' : 'Normalize the embeddings')
      + `<div class="ml-grid"><div class="ml-stage">${plane({ arrows: [{ v: shown }], points, range: 2.5, unitCircle: true, label: 'Embedding vectors; dotted circle is unit length. Select an item point.' })}${note('Tap an item point. The dotted circle has length 1; axes are latent dimensions.')}</div>${inspector(`Item ${s.item}: ${vec(shown)}`, [u ? `Length ${fmt(norm(shown))}. ${s.normalized ? 'Same direction; original magnitude removed.' : 'Normalize to remove length without changing direction.'}` : 'Zero has no direction: unit normalization is undefined.', `Length = sqrt(${fmt(x[0] ** 2)} + ${fmt(x[1] ** 2)}) = ${fmt(norm(x))}.`, u ? `Unit vector = ${vec(x)} / ${fmt(norm(x))} = ${vec(u)}.` : 'A production retrieval system must handle zero embeddings explicitly.', u ? `Reconstruction: ${fmt(norm(x))} x ${vec(u)} = ${vec(x)}.` : 'The origin shown is not a valid unit embedding.', 'Magnitude can influence inner-product retrieval. It is not guaranteed to encode popularity.'])}</div>`
      + settings(controls('Embedding', Object.keys(embeddings).map(k => btn('item', k, k === 'zero' ? 'Zero vector' : `Item ${k}`, s.item === k)).join('')));
  }

  function dotView(s) {
    const q = s.query === 'first' ? [1, .5] : [.2, 1], v = embeddings[s.item];
    const shownQ = s.normalized ? unit(q) : q, shownV = s.normalized ? unit(v) : v;
    const projection = scale(shownQ, dot(shownQ, shownV) / dot(shownQ, shownQ));
    const ranking = ['A', 'B', 'C'].map(k => ({ k, score: s.normalized ? cosine(q, embeddings[k]) : dot(q, embeddings[k]) })).sort((a, b) => b.score - a.score);
    return heading('Dot products / two-tower retrieval scores', 'Can normalization change the top retrieval result?')
      + primary('normalize', !s.normalized, s.normalized ? 'Rank by raw dot product' : 'Normalize and rank by cosine')
      + `<div class="ml-grid"><div class="ml-stage">${plane({ arrows: [{ v: shownQ }, { v: shownV, color: 'orange' }, { from: shownV, v: projection, color: 'green', dashed: true }], points: ['A', 'B', 'C'].map(k => ({ v: s.normalized ? unit(embeddings[k]) : embeddings[k], label: `Item ${k}`, action: 'item', value: k, selected: k === s.item, color: 'orange' })), range: 2.5, label: 'Query, candidate embedding, and perpendicular projection onto query' })}${legend([['blue', 'Query'], ['orange', `Item ${s.item}`], ['green', 'Perpendicular to query']])}</div>${inspector(`Top result: item ${ranking[0].k}`, [`${s.normalized ? 'Cosine' : 'Raw dot'} score ${fmt(ranking[0].score)}. ${s.normalized ? 'Only direction matters.' : 'Length contributes as well as direction.'}`, `Selected item ${s.item}: dot = ${fmt(q[0])} x ${fmt(v[0])} + ${fmt(q[1])} x ${fmt(v[1])} = ${fmt(dot(q, v))}.`, `Cosine = ${fmt(dot(q, v))} / (${fmt(norm(q))} x ${fmt(norm(v))}) = ${fmt(cosine(q, v))}.`, `Current projection of item onto query = ${vec(projection)}.`])}</div>`
      + calculation(table(['Rank', 'Item', 'Score'], ranking.map((r, i) => [esc(i + 1), btn('item', r.k, `Item ${r.k}`, s.item === r.k), esc(fmt(r.score))])))
      + settings(controls('Query vector', btn('query', 'first', 'Query [1, 0.5]', s.query === 'first') + btn('query', 'second', 'Query [0.2, 1]', s.query === 'second')) + note('Train, index, and serve with a consistent scoring rule. Removing norms changes the retrieval objective.'));
  }

  function matrixView(s) {
    const A = maps[s.scene], inputs = [[1, 0], [0, 1], [1, 2]], names = ['e1', 'e2', 'x'], x = inputs[s.selected], y = matvec(A, x);
    const c1 = scale([A[0][0], A[1][0]], x[0]), c2 = scale([A[0][1], A[1][1]], x[1]);
    const arrows = s.step === 0 ? [] : s.step === 1 ? [{ v: c1 }] : s.step === 2 ? [{ v: c1 }, { from: c1, v: y, color: 'orange' }] : [{ v: y, color: 'green' }];
    const trace = ['Input coordinates', 'First weighted column', 'Add second weighted column', 'Output / row products'];
    return heading('Matrix multiplication / one map, every input', 'How do two input coordinates become one output vector?')
      + navigation(s.step, trace)
      + `<div class="ml-grid"><div class="ml-stage"><p class="ml-kicker">Input ${esc(vec(x))}</p>${plane({ arrows: [{ v: x }], points: inputs.map((v, i) => ({ v, label: names[i], action: 'select', value: i, selected: s.selected === i })), gridMap: [[1, 0], [0, 1]], range: 4, label: 'Input grid and basis vectors', shape: [[0, 0], [1, 0], [1, 1], [0, 1]] })}</div><div class="ml-stage"><p class="ml-kicker">${s.step === 0 ? 'Mapped basis grid' : s.step === 1 ? `First contribution ${esc(vec(c1))}` : `Output ${esc(vec(y))}`}</p>${plane({ arrows, gridMap: A, range: 4, shape: [[0, 0], [1, 0], [1, 1], [0, 1]].map(v => matvec(A, v)), label: 'Transformed grid and unit square' })}</div></div>${note(s.step === 0 ? 'Next places the first weighted column in the output space.' : s.step === 1 ? 'Next adds the second weighted column to the first.' : 'The two weighted columns add tip-to-tail. Their sum is the output arrow.')}`
      + calculation(`<div class="fl-matrices">${matrix(A, 'A')}${matrix([[x[0], y[0]], [x[1], y[1]]], 'columns: x, Ax')}</div>${controls('Inspect output coordinate', [0, 1].map(i => btn('row', i, `Output y${i + 1}`, s.row === i)).join(''))}${note(`Column sum: ${vec(c1)} + ${vec(c2)} = ${vec(y)}.`)}${note(`Row ${s.row + 1}: ${A[s.row][0]} x ${x[0]} + ${A[s.row][1]} x ${x[1]} = ${y[s.row]}.`)}${note(`det(A) = ${fmt(det(A))}: ${det(A) === 0 ? 'the unit square collapses to a line; one input direction is lost.' : `signed area scales by ${fmt(det(A))}.`}`)}`)
      + settings(controls('Linear map', Object.entries({ shear: 'Mix coordinates', rotate: 'Quarter-turn', collapse: 'Rank-one collapse' }).map(([k, text]) => btn('scene', k, text, s.scene === k)).join('')) + controls('Input vector', inputs.map((v, i) => btn('select', i, `${names[i]} = ${vec(v)}`, s.selected === i)).join('')));
  }

  function eigenView(s) {
    const A = eigenCases[s.scene], eig = eigenSymmetric(A), start = s.selected === 2 ? [1, 0] : eig.vectors[s.selected];
    const v = powerStep(A, start, s.step), Av = matvec(A, v), rayleigh = dot(v, Av), residual = norm(subtract(Av, scale(v, rayleigh)));
    return heading('Eigenvectors / covariance and curvature', 'Which directions survive this matrix without turning?')
      + navigation(s.step, Array.from({ length: 6 }, (_, i) => `Multiply and normalize: ${i} times`))
      + `<div class="ml-grid"><div class="ml-stage">${plane({ arrows: [{ v }, { v: Av, color: 'orange' }], range: 3.5, unitCircle: true, points: eig.vectors.map((v, i) => ({ v, action: 'select', value: i, label: `Eigenvector ${i + 1}`, selected: s.selected === i })), label: 'Unit input v and mapped Av; select either eigenvector point' })}${legend([['blue', 'Input direction'], ['orange', 'After multiplying']])}${note('Tap either point to inspect an exact eigenvector.')}</div>${inspector(residual < 1e-10 ? 'These arrows share one line' : 'The matrix turns this direction', [`Off-line residual: ${fmt(residual, 5)}. Next feeds the normalized output back in.`, `v = ${vec(v)}; Av = ${vec(Av)}.`, `Rayleigh quotient v^T A v = ${fmt(rayleigh)}.`, `Residual ||Av - (v^T A v)v|| = ${fmt(residual, 5)}. Zero identifies an eigenvector.`, s.scene === 'saddle' ? 'Eigenvalues 3 and -1: the second direction flips sign. This is a Hessian example, not a covariance.' : s.scene === 'isotropic' ? 'Equal eigenvalues: every nonzero direction is an eigenvector; no unique leading axis.' : `PCA explained-variance fraction for v1: ${fmt(eig.values[0])} / ${fmt(sum(eig.values))} = ${pct(eig.values[0] / sum(eig.values))}.`])}</div>`
      + calculation(`<div class="fl-matrices">${matrix(A, 'A')}${matrix(reconstructEigen(A), 'Q diag(lambda) Q^T')}</div>${table(['Eigenpair', 'Unit vector', 'Eigenvalue'], eig.vectors.map((v, i) => [btn('select', i, `v${i + 1}`, s.selected === i), esc(vec(v)), esc(fmt(eig.values[i]))]))}${note('Power iteration needs a dominant absolute eigenvalue and a starting component along its eigenvector. Starting exactly along another eigenvector stays there.')}`)
      + settings(controls('Symmetric matrix', Object.entries({ covariance: 'Correlated covariance', saddle: 'Saddle Hessian', isotropic: 'Isotropic covariance' }).map(([k, text]) => btn('scene', k, text, s.scene === k)).join('')) + controls('Starting direction', btn('select', 2, 'Ordinary [1, 0]', s.selected === 2) + btn('select', 0, 'Leading eigenvector', s.selected === 0) + btn('select', 1, 'Second eigenvector', s.selected === 1)));
  }

  function svdView(s) {
    const A = svdCases[s.scene], d = svd2(A), approx = reconstructSvd(d, s.rank), residual = matrixSubtract(A, approx);
    const x = [1, 1], aligned = matvec(d.Vt, x), stretched = aligned.map((v, i) => i < s.rank ? v * d.singular[i] : 0), output = matvec(d.U, stretched);
    const stages = [x, aligned, stretched, output], labels = ['Input x', 'V^T: change input basis', 'Sigma: stretch / truncate', 'U: map to output basis'];
    const circle = Array.from({ length: 65 }, (_, i) => [Math.cos(i * Math.PI / 32), Math.sin(i * Math.PI / 32)]);
    const transformStage = v => { if (s.step === 0) return v; const a = matvec(d.Vt, v); if (s.step === 1) return a; const b = a.map((t, i) => i < s.rank ? t * d.singular[i] : 0); return s.step === 2 ? b : matvec(d.U, b); };
    const row = Math.floor(s.selected / 2), col = s.selected % 2;
    const contributions = d.singular.map((sigma, j) => sigma * d.U[row][j] * d.Vt[j][col]);
    const retained = sum(d.singular.slice(0, s.rank).map(v => v * v)) / sum(d.singular.map(v => v * v));
    return heading('SVD / exact factorization and rank truncation', 'What disappears when you keep only one direction?')
      + primary('rank', s.rank === 1 ? 2 : 1, s.rank === 1 ? 'Restore the second direction' : 'Discard the weaker direction')
      + `<div class="ml-grid"><div class="ml-stage"><p class="ml-kicker">${esc(labels[s.step])}</p>${plane({ arrows: [{ v: stages[s.step] }], shape: circle.map(transformStage), range: 5, label: 'Unit circle transformed through the selected SVD stage; arrow tracks x=[1,1]' })}${note(s.step < 2 ? 'This basis-change step comes before stretching or truncation.' : d.singular[1] === 0 ? 'This matrix already has rank one: both versions map the circle to a line.' : s.rank === 1 ? 'The transformed circle collapses to a line: one input direction is discarded.' : 'Keeping both directions restores the full transformed circle.')}</div>${inspector(`Keep ${s.rank} component${s.rank === 1 ? '' : 's'}: ${pct(retained)} matrix energy retained`, [`Reconstruction error: ${fmt(frobenius(residual))}. ${frobenius(residual) > 1e-10 ? 'Restoring the other direction removes this error.' : 'The kept components already reconstruct the matrix.'}`, `Singular values: ${vec(d.singular)}.`, `||A - A_k||_F = ${fmt(frobenius(residual))}; squared error = ${fmt(frobenius(residual) ** 2)}.`, `Optimal rank-one error equals discarded sigma2 = ${fmt(d.singular[1])}.`, 'Retained matrix energy is not a guarantee of ranking quality.'])}</div>`
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
      'The density f(0.75) is 1.5. What is P(0.5 <= X <= 0.75)?', '0.3125: integrate the density over the interval.', '1.5: use the density at the right endpoint.', 'Probability is area: 0.75 squared minus 0.5 squared = 0.3125.'),
    expectation: contentEntry('Expected risk depends on the serving population.', 'Hold segment losses fixed. Change the user mix and watch the winning model change.', 'Offline averages can hide population weighting assumptions and fail under serving shift.', 'Expected risk averages loss under a specified data distribution, not an unweighted list of segments.', ['The example uses segment-conditional mean losses; within-segment variation is not represented.', 'Expectation of a sum is the sum of expectations without an independence assumption. Variance of a sum must also account for covariance terms.'],
      ['R(f)=\\mathbb{E}_{(x,y)\\sim P}[\\ell(f(x),y)]=\\sum_g P(g)L_g(f)'],
      [['P(g)', 'Share of arrivals in segment g under population P.', 'Training: frequent 0.8, new 0.2.'], ['f, x, y, ell', 'Model, user context, observed outcome, and per-example loss.', 'Log-loss on impressions, summarized into segment means.'], ['L_g(f)', 'Mean loss of model f within segment g.', 'Model A: [0.1,0.8]; B: [0.25,0.35] nats.'], ['R(f)', 'Population-weighted risk.', 'A: 0.8*0.1+0.2*0.8=0.24; B: 0.27. Serving [0.3,0.7]: A=0.59, B=0.32.']],
      'import numpy as np\nloss_A = np.array([.1, .8])\nloss_B = np.array([.25, .35])\ntrain = np.array([.8, .2])\nserve = np.array([.3, .7])\nprint(train @ loss_A, train @ loss_B)  # 0.24, 0.27\nprint(serve @ loss_A, serve @ loss_B)  # 0.59, 0.32\n# Same models, different population weighting.',
      'A wins on the training mix. Must it win after population shift?', 'No: expected risk changes with segment weights.', 'Yes: each segment loss stayed fixed.', 'A wins 0.24 vs 0.27 at training, but loses 0.59 vs 0.32 at serving.'),
    bayes: contentEntry('A spam flag changes the denominator.', 'Count true and false flags inside the evidence-matching subset.', 'Precision changes with prevalence even when recall and false-positive rate do not.', 'Recall is P(flag|spam); precision is P(spam|flag). They condition on different populations.', ['Unflagged messages also update the posterior.', 'Holding class-conditional rates fixed models prior-probability shift, not arbitrary distribution shift.'],
      ['P(S\\mid F)=\\frac{P(F\\mid S)P(S)}{P(F\\mid S)P(S)+P(F\\mid \\neg S)P(\\neg S)}'],
      [['S, F', 'Spam status and observed flag.', '10,000 emails; 100 spam at a 1% prior.'], ['P(F|S)', 'Recall: fraction of spam messages flagged.', '0.9*100=90 true flags.'], ['P(F|not S)', 'False-positive rate among non-spam.', '0.02*9900=198 false flags.'], ['P(S|F)', 'Spam fraction among all flags.', '90/(90+198)=0.3125, not 0.90.']],
      'n, prior, recall, fpr = 10_000, .01, .9, .02\ntp = n * prior * recall       # 90\nfp = n * (1-prior) * fpr      # 198\nfn = n * prior - tp           # 10\ntn = n * (1-prior) - fp       # 9702\nprecision = tp / (tp + fp)    # 0.3125\nspam_if_no_flag = fn / (fn + tn)  # about 0.00103',
      'At 1% prevalence, 90% recall, 2% false positives, what fraction of flags are spam?', '31.25%: 90 true flags out of 288 flags.', '90%: the classifier recall.', 'Recall counts among spam. The flagged subset also contains 198 non-spam emails.'),
    entropy: contentEntry('Entropy is average surprise, not maximum surprise.', 'Complete one sentence: inspect how often each next token occurs, how surprising it is, and why its region area contributes to the average.', 'Rare events can be individually surprising without dominating average uncertainty. A mismatched model adds avoidable expected surprise.', 'H(p,q)=H(p)+KL(p||q): only the mismatch term can be eliminated by matching a fixed p.', ['The reference counts are an illustrative 100-continuation dataset for one context, treated as p. They are not measured language frequencies or a claim about a real model.', 'The distribution for a context is not the one-hot label of a single sample.', 'Individual KL terms may be negative, although the sum is nonnegative. Positive p with zero q gives infinite cross-entropy.'],
      ['I_q(i)=-\\log_2 q_i', 'H(p,q)=-\\sum_i p_i\\log_2 q_i=H(p)+D_{KL}(p\\Vert q)', 'D_{KL}(p\\Vert q)=\\sum_i p_i\\log_2(p_i/q_i)'],
      [['p_i, q_i', 'Reference next-token frequency and predicted probability.', 'After "She poured the coffee into the ...": cup/mug/glass counts [60,30,10] give p=[0.6,0.3,0.1]; overconfident q=[0.9,0.09,0.01].'], ['I_q(i)', 'Surprise in bits when token i occurs.', 'With q=p, glass costs -log2(0.1)=3.322 bits, but occurs only 10/100 times: contribution=0.3322.'], ['H(p)', 'Mean surprise under the reference probabilities.', 'H([0.6,0.3,0.1])=1.2955 bits: sum the three probability-times-surprise areas.'], ['D_KL(p||q)', 'Extra expected code length caused by mismatched probabilities.', 'For overconfident q: H(p,q)=1.7978, KL=0.5023 bits (rounded).']],
      'from math import log2\n# Illustrative next tokens: cup, mug, glass.\ncounts = [60, 30, 10]\np = [n / sum(counts) for n in counts]\nq = [.9, .09, .01]  # contrast with q = p\nh = sum(-pi*log2(pi) for pi in p if pi)\nce = sum(-pi*log2(qi) for pi, qi in zip(p, q) if pi)\nkl = sum(pi*log2(pi/qi) for pi, qi in zip(p, q) if pi)\nassert abs(ce - h - kl) < 1e-12\n# Natural-log training uses nats: multiply bits by ln(2).',
      'If q is changed to equal p exactly, which quantity becomes zero?', 'KL(p||q), not necessarily entropy.', 'Both entropy and cross-entropy.', 'The real task retains 1.2955 bits of uncertainty. Matching q eliminates only mismatch.'),
    loss: contentEntry('The loss changes the gradient on a confident miss.', 'Compare log-loss and squared probability error on the same clicked impression.', 'Sigmoid plus squared loss can attenuate gradients on confident errors; sigmoid cross-entropy avoids that additional factor.', 'Both Brier score and log-loss are proper probability losses, but their curvature and gradients differ.', ['Expected losses use a fixed true click probability p=0.1.', 'Log-loss values are in nats; Brier values are unitless. Raw numerical scales are not interchangeable.'],
      ['q=\\sigma(z),\\quad \\ell_{log}=-y\\ln q-(1-y)\\ln(1-q)', '\\ell_{sq}=(q-y)^2', '\\frac{\\partial\\ell_{log}}{\\partial z}=q-y,\\quad \\frac{\\partial\\ell_{sq}}{\\partial z}=2(q-y)q(1-q)'],
      [['y, q', 'Binary observed label and predicted click probability.', 'y=1, q=0.01: log-loss=4.6052; squared loss=0.9801.'], ['z, sigma', 'Logit and sigmoid function.', 'z=ln(0.01/0.99)=-4.5951; sigmoid(z)=0.01.'], ['partial ell / partial z', 'Training signal before the sigmoid.', 'Log-loss gradient=-0.99; squared-loss gradient=-0.019602.']],
      'from math import log\ny, q = 1, .01\nlog_loss = -log(q if y else 1-q)  # 4.60517\nsquared = (q-y)**2               # 0.9801\ndlog_dz = q-y                    # -0.99\ndsq_dz = 2*(q-y)*q*(1-q)         # -0.019602\n# In production, evaluate BCE directly from logits for stability.',
      'Why is the squared-loss logit gradient small when y=1 and q=0.01?', 'The sigmoid derivative q(1-q) suppresses it.', 'Squared loss thinks the prediction is already correct.', 'Its residual is large, but multiplying by 2q(1-q)=0.0198 attenuates the logit gradient.'),
    vectors: contentEntry('An embedding has a direction and a norm.', 'Inspect real coordinates before and after unit normalization.', 'Normalization can change retrieval rankings by discarding norm information.', 'L2 normalization preserves a nonzero vector direction and forces its norm to one; zero needs explicit handling.', ['Learned latent axes need not have human-readable semantic meanings.', 'Normalization is a modeling choice, not an automatically correct cleanup step.'],
      ['\\|v\\|_2=\\sqrt{\\sum_i v_i^2},\\quad \\hat v=v/\\|v\\|_2'],
      [['v, v_i', 'Item embedding and its coordinate i.', 'Item B=[2,-0.1].'], ['||v||_2', 'Euclidean length.', 'sqrt(4+0.01)=2.0025.'], ['v-hat', 'Unit direction, defined only when v is nonzero.', 'B/2.0025=[0.9988,-0.04994]; its length is 1.']],
      'import numpy as np\nv = np.array([2., -.1])\nn = np.linalg.norm(v)  # 2.002498...\nif n == 0:\n    raise ValueError("zero embedding has no direction")\nu = v / n\nassert np.allclose(np.linalg.norm(u), 1)\nassert np.allclose(n * u, v)',
      'What happens to a zero embedding under exact unit normalization?', 'It is undefined; the zero vector has no direction.', 'It becomes a valid arbitrary unit direction.', 'Division by a zero norm is undefined. Applications need an explicit fallback or exclusion policy.'),
    'dot-products': contentEntry('Raw retrieval and cosine can choose different items.', 'Inspect coordinate products, geometric projection, and the resulting ranking.', 'Two-tower systems must align training, indexing, and serving similarity metrics.', 'Dot product combines norms and angle; cosine removes norm factors.', ['Orthogonal vectors have zero inner product. Negative inner products correspond to obtuse angles.', 'Norm effects are learned, not inherently desirable or undesirable.'],
      ['q^Tv=\\sum_i q_i v_i=\\|q\\|_2\\|v\\|_2\\cos\\theta', '\\operatorname{proj}_q(v)=\\frac{q^Tv}{q^Tq}q'],
      [['q, v', 'Query and item embeddings.', 'q=[1,0.5], A=[0.9,0.6], B=[2,-0.1].'], ['q^T v', 'Raw retrieval score.', 'A=1.2; B=1.95, so raw dot ranks B first.'], ['theta, cos theta', 'Angle and normalized directional similarity.', 'A cosine=0.9923; B cosine=0.8710, so cosine ranks A first.'], ['proj_q(v)', 'Component of v parallel to q.', 'For A: (1.2/1.25)*[1,0.5]=[0.96,0.48].']],
      'import numpy as np\nq = np.array([1., .5])\nitems = np.array([[.9, .6], [2., -.1], [-.5, 1.]])\nraw = items @ q                       # [1.2, 1.95, 0]\ncosine = raw / (np.linalg.norm(items, axis=1)*np.linalg.norm(q))\nprint(np.argmax(raw), np.argmax(cosine))  # 1, 0\nv = items[0]\nprojection = (q @ v)/(q @ q)*q         # [0.96, 0.48]',
      'Why does B beat A with raw dot product but lose with cosine?', 'Its larger norm outweighs its weaker alignment under raw scoring.', 'Cosine reverses every raw-dot ranking.', 'The default query gives A better angular alignment, while B gains from its larger norm.'),
    'matrix-multiply': contentEntry('A linear layer is a weighted sum of columns.', 'Trace input coordinates into transformed basis vectors and inspect the output row products.', 'The same linear map acts on every input; a singular map loses information.', 'Column j is A e_j. Ax sums those mapped basis vectors, weighted by input coordinates.', ['The determinant scales signed area; zero determinant means the map is not invertible.', 'A linear layer may change dimension; this exact 2D example keeps the geometry visible.'],
      ['y=Ax=x_1 A e_1+x_2 A e_2', 'y_i=\\sum_j A_{ij}x_j'],
      [['A, A_ij', 'Linear map and entry at output row i, input column j.', 'A=[[1,1],[0,1]].'], ['x_j, e_j', 'Input coefficient and coordinate basis vector.', 'x=[1,2]; e1=[1,0], e2=[0,1].'], ['y_i', 'Output coordinate i.', 'Ax=1*[1,0]+2*[1,1]=[3,2].']],
      'import numpy as np\nA = np.array([[1., 1.], [0., 1.]])\nx = np.array([1., 2.])\ncol_sum = x[0]*A[:, 0] + x[1]*A[:, 1]\ny = A @ x                         # [3,2]\nassert np.allclose(col_sum, y)\nassert np.isclose(np.linalg.det(A), 1)  # shear preserves area',
      'If a 2D matrix maps the unit square to a line, what follows?', 'Its determinant is zero and some input information is lost.', 'It is merely a rotation with no information loss.', 'A line has zero area; a rank-deficient map cannot distinguish all input vectors.'),
    eigen: contentEntry('Find the directions preserved by a matrix.', 'Compare an ordinary direction with exact eigenvectors; follow normalized power iteration.', 'Covariance eigenvectors define PCA axes. Hessian eigenvalues describe local curvature, including negative directions.', 'For symmetric matrices an orthonormal eigenbasis reconstructs A; equal eigenvalues need not define a unique axis.', ['A negative eigenvalue reverses orientation but preserves the eigenvector line.', 'Power iteration needs a dominant absolute eigenvalue and an initial component along its eigendirection.'],
      ['Av_i=\\lambda_i v_i,\\quad A=Q\\Lambda Q^T', '\\rho(v)=\\frac{v^TAv}{v^Tv},\\quad v_{t+1}=\\frac{Av_t}{\\|Av_t\\|_2}'],
      [['A, v_i, lambda_i', 'Symmetric map, eigenvector, and signed scale.', 'A=[[2,1],[1,2]]; v1=[1,1]/sqrt(2), lambda1=3; lambda2=1.'], ['Q, Lambda', 'Columns of unit eigenvectors and diagonal eigenvalues.', 'Q diag(3,1) Q^T reconstructs [[2,1],[1,2]].'], ['rho(v)', 'Rayleigh quotient: directional quadratic form per squared norm.', 'v=[1,0]: rho=2 but Av=[2,1] is not parallel to v.'], ['v_t', 'Normalized iterate after t multiplications.', 'v0=[1,0]; v1=[2,1]/sqrt(5)=[0.8944,0.4472].']],
      'import numpy as np\nA = np.array([[2., 1.], [1., 2.]])\nlam, Q = np.linalg.eigh(A)  # ascending eigenvalues [1,3]\nassert np.allclose(A, Q @ np.diag(lam) @ Q.T)\nv = np.array([1., 0.])\nfor _ in range(5):\n    v = A @ v\n    v /= np.linalg.norm(v)\nrho = v @ A @ v\nresidual = np.linalg.norm(A @ v - rho * v)',
      'Does a negative eigenvalue mean the eigenvector leaves its line?', 'No: the vector flips orientation along the same line.', 'Yes: every negative eigenvalue creates a new direction.', 'Av=lambda*v stays collinear for any real eigenvalue, including negative ones.'),
    svd: contentEntry('Measure exactly what rank truncation discards.', 'Trace one vector through the computed factors and inspect reconstructed matrix entries.', 'Low-rank structure supports compression and latent factor models, but matrix energy is not task quality.', 'Truncated SVD is optimal for Frobenius error; discarded squared singular values give the exact squared error.', ['Orthogonal factors may include reflections, not only rotations.', 'A full SVD of a 2x2 example is pedagogical, not a storage saving. For a large m-by-n matrix, factors can reduce storage to k(m+n+1).'],
      ['A=U\\Sigma V^T=\\sum_i\\sigma_i u_i v_i^T', 'A_k=\\sum_{i=1}^k\\sigma_i u_i v_i^T,\\quad \\|A-A_k\\|_F^2=\\sum_{i>k}\\sigma_i^2'],
      [['A, U, V', 'Original map and orthonormal output/input bases.', 'A=[[3,1],[0,1]]; computed U and V reconstruct A.'], ['sigma_i, Sigma', 'Nonnegative singular values and their diagonal matrix.', 'sigma=[3.1796,0.9435] (rounded).'], ['u_i, v_i, k', 'Corresponding basis columns and number of kept components.', 'Rank k=1 keeps sigma1*u1*v1^T only.'], ['||A-A_k||_F^2', 'Sum of squared entrywise reconstruction errors.', 'At k=1, error squared=sigma2^2=0.8902; retained energy=91.9%.']],
      'import numpy as np\nA = np.array([[3., 1.], [0., 1.]])\nU, s, Vt = np.linalg.svd(A, full_matrices=False)\nfull = (U * s) @ Vt\nrank1 = s[0] * np.outer(U[:, 0], Vt[0])\nassert np.allclose(full, A)\nerror_sq = np.linalg.norm(A-rank1, "fro")**2\nassert np.isclose(error_sq, s[1]**2)\nretained = s[0]**2 / (s @ s)  # about 91.9%',
      'With singular values 4 and 2, what is the optimal rank-one squared Frobenius error?', '4: the square of the discarded singular value 2.', '2: the discarded singular value without squaring.', 'Frobenius error is 2; squared Frobenius error is 2 squared, or 4.'),
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

  const api = { normalize, softmax, expectedRisk, information, bayesCounts, binaryLoss, binaryRisk, categorical, sampleDistribution, densityMass, dot, norm, unit, cosine, matvec, multiply, transpose, det, frobenius, matrixSubtract, eigenSymmetric, reconstructEigen, svd2, reconstructSvd, powerStep, initial, reduce, render, content };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window !== 'undefined') {
    window.AtelierLab.createModule({ id: 'foundations-labs', content, initial, render, reduce });
  }
})();
