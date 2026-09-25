/* Ranking labs: deterministic examples, shared inline/modal state via lab-core. */
(() => {
  'use strict';

  const escape = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const sum = xs => xs.reduce((a, b) => a + b, 0);
  const dot = (a, b) => sum(a.map((v, i) => v * b[i]));
  const matvec = (matrix, vector) => matrix.map(row => dot(row, vector));
  const sigmoid = x => x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
  const softplus = x => Math.max(0, x) + Math.log1p(Math.exp(-Math.abs(x)));
  const softmax = xs => {
    const exps = xs.map(x => Math.exp(x - Math.max(...xs)));
    return exps.map(x => x / sum(exps));
  };
  const bce = (score, y) => Math.max(score, 0) - y * score + Math.log1p(Math.exp(-Math.abs(score)));
  const normalize = v => {
    const norm = Math.sqrt(dot(v, v));
    return v.map(x => norm ? x / norm : 0);
  };
  const f = (x, digits = 3) => Number.isFinite(x) ? x.toFixed(digits) : 'n/a';
  const pct = x => x === null ? 'n/a' : `${f(100 * x, 1)}%`;
  const vec = v => `[${v.map(x => f(x, 2)).join(', ')}]`;
  const gain = rel => 2 ** rel - 1;
  const discount = (rank, k) => rank < k ? 1 / Math.log2(rank + 2) : 0;
  const dcg = (relevances, k = relevances.length) => sum(relevances.map((rel, rank) => gain(rel) * discount(rank, k)));
  function rankingMetrics(relevances, k, allRelevances = relevances) {
    const ideal = dcg([...allRelevances].sort((a, b) => b - a), k);
    const value = dcg(relevances, k);
    const first = relevances.slice(0, k).findIndex(r => r >= 2);
    return { dcg: value, idcg: ideal, ndcg: ideal ? value / ideal : 0, rr: first < 0 ? 0 : 1 / (first + 1) };
  }
  function swapDelta(relevances, a, b, k) {
    const ideal = dcg([...relevances].sort((x, y) => y - x), k);
    return ideal ? Math.abs((gain(relevances[a]) - gain(relevances[b])) * (discount(a, k) - discount(b, k))) / ideal : 0;
  }
  function pairwise(preferred, other, weight = 1) {
    const margin = preferred - other;
    const magnitude = sigmoid(-margin);
    return { margin, probability: sigmoid(margin), loss: weight * softplus(-margin), gradients: [-weight * magnitude, weight * magnitude] };
  }

  const ITEM_W = [[1, 0.7, 0], [0.1, 0.3, 1]];
  const QUERY_W = [[1, 0.65, 0.05], [0.2, 0.65, 1]];
  const catalog = [
    { id: 'A', name: 'Embedding retrieval', features: [1, 0, 0], rel: 2, intent: 0.3 },
    { id: 'B', name: 'CTR features', features: [0.5, 0.5, 0.15], rel: 1, intent: 0 },
    { id: 'C', name: 'GRPO training', features: [0.05, 0, 1], rel: 0, intent: 0 },
    { id: 'D', name: 'ANN indexing', features: [0.3, 0.2, 0.25], rel: 3, intent: 1 },
    { id: 'E', name: 'Macro forecasting', features: [0.1, 0, 0.65], rel: 0, intent: 0 },
    { id: 'F', name: 'Ranking evaluation', features: [0, 1, 0], rel: 2, intent: 0.6 },
  ].map(item => ({ ...item, vector: matvec(ITEM_W, item.features) }));
  const queries = [
    { name: 'Retrieve ML documents', features: [1, 0, 0] },
    { name: 'Improve ranking', features: [0, 1, 0] },
    { name: 'Optimize an RL policy', features: [0, 0, 1] },
  ];
  const lookup = id => catalog.find(item => item.id === id);
  function retrieve(query, count, cosine = false) {
    const q = cosine ? normalize(query) : query;
    return catalog.map(item => ({ ...item, score: dot(q, cosine ? normalize(item.vector) : item.vector) }))
      .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id)).slice(0, count);
  }
  function funnel(count, extraSource, oracle) {
    const q = matvec(QUERY_W, queries[0].features);
    const candidates = retrieve(q, count);
    if (extraSource && !candidates.some(item => item.id === 'D')) candidates.push({ ...lookup('D'), score: dot(q, lookup('D').vector) });
    const ranked = candidates.map(item => ({ ...item, rankScore: oracle ? item.rel : item.score + 2 * item.intent }))
      .sort((a, b) => b.rankScore - a.rankScore || a.id.localeCompare(b.id));
    const relevant = catalog.filter(item => item.rel >= 2).length;
    const recall = candidates.filter(item => item.rel >= 2).length / relevant;
    const all = catalog.map(item => item.rel);
    const ceiling = rankingMetrics(candidates.map(item => item.rel).sort((a, b) => b - a), 2, all).ndcg;
    return { candidates, ranked, recall, ceiling, metrics: rankingMetrics(ranked.map(item => item.rel), 2, all) };
  }

  const observations = [[1, 0, null, null, null, 1], [null, 0, 1, null, 1, null], [1, null, 0, 1, null, 1]];
  const users = ['Morgan', 'Sam', 'Lee'];
  const factorInitial = () => ({ users: [[1, 0.2], [0.2, 1], [0.7, 0.7]], items: catalog.map(item => [...item.vector]) });
  function factorStep(factors, user, item, label, learningRate = 0.4, regularization = 0.02) {
    if (label === null) return { factors, skipped: true };
    const p = factors.users[user], q = factors.items[item];
    const score = dot(p, q), residual = sigmoid(score) - label;
    const gp = q.map((x, j) => residual * x + regularization * p[j]);
    const gq = p.map((x, j) => residual * x + regularization * q[j]);
    const next = { users: factors.users.map(v => [...v]), items: factors.items.map(v => [...v]) };
    next.users[user] = p.map((x, j) => x - learningRate * gp[j]);
    next.items[item] = q.map((x, j) => x - learningRate * gq[j]);
    return { factors: next, skipped: false, score, residual, gp, gq, before: bce(score, label), after: bce(dot(next.users[user], next.items[item]), label) };
  }

  const rankItems = [
    { id: 'A', name: 'Embedding retrieval', rel: 2 },
    { id: 'B', name: 'Off-topic overview', rel: 0 },
    { id: 'D', name: 'Exact ANN answer', rel: 3 },
    { id: 'F', name: 'Related ranking note', rel: 1 },
  ];
  const pairs = [[2, 1], [0, 3], [2, 0]];
  const scoreOrder = scores => scores.map((score, i) => ({ score, i })).sort((a, b) => b.score - a.score || a.i - b.i).map(row => row.i);
  function rankMovement(scores, beforeScores = scores) {
    const beforeOrder = scoreOrder(beforeScores);
    return scoreOrder(scores).map((index, rank) => ({
      index, rank: rank + 1, beforeRank: beforeOrder.indexOf(index) + 1,
      rankDelta: beforeOrder.indexOf(index) - rank,
      score: scores[index], beforeScore: beforeScores[index], scoreDelta: scores[index] - beforeScores[index],
    }));
  }
  function objective(scores, mode, pairIndex, k) {
    const order = scoreOrder(scores), rels = order.map(i => rankItems[i].rel);
    const [a, b] = pairs[pairIndex], pa = order.indexOf(a), pb = order.indexOf(b);
    const delta = swapDelta(rels, pa, pb, k);
    const pair = pairwise(scores[a], scores[b], mode === 'lambda' ? delta : 1);
    let gradients = scores.map(() => 0), targets = scores.map(() => null), loss = pair.loss;
    const probabilities = mode === 'list' ? softmax(scores) : scores.map(sigmoid);
    if (mode === 'point') {
      targets = rankItems.map(item => Number(item.rel >= 2));
      gradients = scores.map((s, i) => (sigmoid(s) - targets[i]) / scores.length);
      loss = sum(scores.map((s, i) => bce(s, targets[i]))) / scores.length;
    } else if (mode === 'list') {
      const gains = rankItems.map(item => gain(item.rel));
      targets = gains.map(g => g / sum(gains));
      gradients = probabilities.map((p, i) => p - targets[i]);
      const max = Math.max(...scores), logZ = max + Math.log(sum(scores.map(s => Math.exp(s - max))));
      loss = sum(targets.map((t, i) => t * (logZ - scores[i])));
    } else {
      [gradients[a], gradients[b]] = pair.gradients;
    }
    const swapped = [...rels];
    [swapped[pa], swapped[pb]] = [swapped[pb], swapped[pa]];
    return { order, pair, delta, gradients, targets, probabilities, loss, ndcg: rankingMetrics(rels, k).ndcg, swappedNdcg: rankingMetrics(swapped, k).ndcg };
  }

  const impressions = [
    [0.94, 1], [0.88, 0], [0.81, 1], [0.74, 1], [0.68, 0], [0.59, 1],
    [0.52, 0], [0.43, 1], [0.34, 0], [0.26, 0], [0.17, 1], [0.08, 0],
  ].map(([p, y], i) => ({ id: i + 1, p, y }));
  const thresholds = [1.01, ...impressions.map(row => row.p)];
  function confusion(rows, threshold) {
    const cells = { tp: [], fp: [], fn: [], tn: [] };
    rows.forEach(row => cells[row.p >= threshold ? (row.y ? 'tp' : 'fp') : (row.y ? 'fn' : 'tn')].push(row.id));
    const tp = cells.tp.length, fp = cells.fp.length, fn = cells.fn.length, tn = cells.tn.length;
    return { ...cells, counts: { tp, fp, fn, tn }, precision: tp + fp ? tp / (tp + fp) : null, recall: tp + fn ? tp / (tp + fn) : null, f1: 2 * tp + fp + fn ? 2 * tp / (2 * tp + fp + fn) : 0 };
  }
  function temperature(p, t) {
    if (!(t > 0)) throw new RangeError('Temperature must be positive');
    if (p === 0 || p === 1) return p;
    return sigmoid(Math.log(p / (1 - p)) / t);
  }
  function calibration(rows, count, t = 1) {
    const transformed = rows.map(row => ({ ...row, p: temperature(row.p, t) }));
    const bins = Array.from({ length: count }, (_, i) => ({ index: i, low: i / count, high: (i + 1) / count, rows: [] }));
    transformed.forEach(row => bins[Math.min(count - 1, Math.floor(row.p * count))].rows.push(row));
    bins.forEach(bin => {
      bin.n = bin.rows.length;
      bin.mean = bin.n ? sum(bin.rows.map(row => row.p)) / bin.n : null;
      bin.rate = bin.n ? sum(bin.rows.map(row => row.y)) / bin.n : null;
      bin.contribution = bin.n && rows.length ? bin.n / rows.length * Math.abs(bin.mean - bin.rate) : 0;
    });
    return { bins, transformed, ece: sum(bins.map(bin => bin.contribution)), brier: rows.length ? sum(transformed.map(row => (row.p - row.y) ** 2)) / rows.length : null };
  }

  const exposureLog = [
    { id: 'A', y: 1 }, { id: 'B', y: 0 }, { id: 'D', y: null }, { id: 'C', y: 0 },
    { id: 'D', y: 1 }, { id: 'D', y: 1 }, { id: 'F', y: 0 }, { id: 'D', y: 1 }, { id: 'A', y: 1 },
  ];
  function posterior(prior, outcomes, strength = 4) {
    const exposed = outcomes.filter(y => y !== null);
    const clicks = sum(exposed), n = exposed.length;
    const alpha = strength * prior + clicks, beta = strength * (1 - prior) + n - clicks;
    const mean = alpha / (alpha + beta);
    return { mean, alpha, beta, n, clicks, weight: n / (n + strength), sd: Math.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))) };
  }
  function coldRows(step, profile) {
    const q = matvec(QUERY_W, queries[0].features);
    return catalog.map(item => {
      const prior = profile === 'content' ? sigmoid(2 * dot(q, item.vector) - 2) : 0.25;
      const outcomes = exposureLog.slice(0, step).filter(event => event.id === item.id).map(event => event.y);
      return { ...item, prior, ...posterior(prior, outcomes) };
    });
  }

  const button = (action, value, text, selected, disabled = false, label = '') => `<button type="button" data-action="${escape(action)}" data-value="${escape(value)}"${selected === undefined ? '' : ` aria-pressed="${Boolean(selected)}"`}${disabled ? ' disabled' : ''}${label ? ` aria-label="${escape(label)}"` : ''}>${escape(text)}</button>`;
  const primary = (action, value, text, disabled = false) => button(action, value, text, undefined, disabled).replace('<button ', '<button class="rk-primary" ');
  const controls = html => `<div class="ml-controls rk-controls">${html}</div>`;
  const stat = (label, value) => `<div class="ml-stat"><span>${escape(label)}</span><strong>${escape(value)}</strong></div>`;
  const stats = entries => `<div class="rk-stats">${entries.map(([label, value]) => stat(label, value)).join('')}</div>`;
  const heading = (kicker, question) => `<header class="ml-heading"><p class="ml-kicker">${escape(kicker)}</p><h3 class="ml-question">${escape(question)}</h3></header>`;
  const note = html => `<p class="ml-note rk-note">${html}</p>`;
  const table = (caption, heads, rows) => `<div class="ml-table rk-table" tabindex="0" role="region" aria-label="${escape(caption)}"><table><caption>${escape(caption)}</caption><thead><tr>${heads.map(h => `<th scope="col">${escape(h)}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table></div>`;
  const row = cells => `<tr>${cells.map(cell => `<td>${cell}</td>`).join('')}</tr>`;
  const disclose = (title, html) => `<details class="rk-details"><summary>${escape(title)}</summary>${html}</details>`;
  const calculation = html => disclose('Inspect the calculation', html);
  const reset = () => disclose('Start over', controls(button('reset', '', 'Reset example')));
  const inspector = (title, html) => `<aside class="ml-inspector rk-inspector"><h4>${escape(title)}</h4>${html}</aside>`;
  const select = (label, action, choices, value) => `<label class="rk-select">${escape(label)}<select data-action="${escape(action)}" aria-label="${escape(label)}">${choices.map(([v, text]) => `<option value="${escape(v)}"${String(value) === String(v) ? ' selected' : ''}>${escape(text)}</option>`).join('')}</select></label>`;
  const nav = (step, max, label = 'events') => `<nav class="ml-step-nav rk-controls" aria-label="Trace steps">${primary('next', '', 'Next exposure record', step >= max)}${button('previous', '', 'Previous', undefined, step === 0)}<span>${step} / ${max} ${escape(label)}</span></nav>`;

  function initial(kind) {
    if (kind === 'matrix-factorization') return { factors: factorInitial(), user: 0, item: 0, history: [], last: null };
    if (kind === 'two-tower') return { query: 0, item: 'D', cosine: false };
    if (kind === 'rank-objectives') return { scores: [0.7, 1.2, 0.2, 0.4], mode: 'pair', pair: 0, k: 3, history: [] };
    if (kind === 'retrieval-funnel') return { k: 3, extra: false, oracle: false, item: 'D' };
    if (kind === 'cold-start') return { step: 0, profile: 'content', item: 'D' };
    if (kind === 'threshold-metrics') return { cut: 6, cell: 'all', cost: 1 };
    if (kind === 'calibration') return { t: 1, bins: 3, bin: 2 };
    if (kind === 'ranking-metrics') return { order: [1, 3, 0, 2], selected: 2, k: 3 };
    return {};
  }
  const memberNumber = (value, values) => values.includes(Number(value)) ? Number(value) : null;
  function reduce(kind, state, action, value) {
    if (action === 'reset') return initial(kind);
    const s = { ...state }, n = Number(value);
    if (kind === 'matrix-factorization') {
      if (action === 'cell' && /^\d:\d$/.test(value)) {
        const [u, i] = value.split(':').map(Number);
        if (u < 3 && i < 6) { s.user = u; s.item = i; s.last = null; }
      }
      if (action === 'train' && s.history.length < 20) {
        const result = factorStep(s.factors, s.user, s.item, observations[s.user][s.item]);
        if (!result.skipped) { s.history = [...s.history, s.factors]; s.factors = result.factors; s.last = { before: result.before, after: result.after }; }
      }
      if (action === 'undo' && s.history.length) { s.factors = s.history[s.history.length - 1]; s.history = s.history.slice(0, -1); s.last = null; }
    }
    if (kind === 'two-tower') {
      if (action === 'query' && memberNumber(value, [0, 1, 2]) !== null) s.query = n;
      if (action === 'similarity') s.cosine = value === 'cosine';
    }
    if (['two-tower', 'retrieval-funnel', 'cold-start'].includes(kind) && action === 'item' && lookup(value)) s.item = value;
    if (kind === 'rank-objectives') {
      if (action === 'mode' && ['pair', 'lambda', 'point', 'list'].includes(value)) s.mode = value;
      if (action === 'pair' && memberNumber(value, [0, 1, 2]) !== null) s.pair = n;
      if (action === 'k' && memberNumber(value, [2, 3, 4]) !== null) s.k = n;
      if (action === 'train' && s.history.length < 30) {
        const result = objective(s.scores, s.mode, s.pair, s.k);
        s.history = [...s.history, [...s.scores]];
        s.scores = s.scores.map((score, i) => score - 0.7 * result.gradients[i]);
      }
      if (action === 'swap' && s.history.length < 30) {
        const [a, b] = pairs[s.pair];
        s.history = [...s.history, [...s.scores]]; s.scores = [...s.scores];
        [s.scores[a], s.scores[b]] = [s.scores[b], s.scores[a]];
      }
      if (action === 'undo' && s.history.length) { s.scores = s.history[s.history.length - 1]; s.history = s.history.slice(0, -1); }
    }
    if (kind === 'retrieval-funnel') {
      if (action === 'k' && memberNumber(value, [2, 3, 4, 6]) !== null) s.k = n;
      if (action === 'source') s.extra = !s.extra;
      if (action === 'oracle') s.oracle = !s.oracle;
    }
    if (kind === 'cold-start') {
      if (action === 'next') { s.step = Math.min(exposureLog.length, s.step + 1); s.item = exposureLog[s.step - 1].id; }
      if (action === 'previous') { s.step = Math.max(0, s.step - 1); s.item = s.step ? exposureLog[s.step - 1].id : 'D'; }
      if (action === 'profile' && ['content', 'population'].includes(value)) s.profile = value;
    }
    if (kind === 'threshold-metrics') {
      if (action === 'cut' && Number.isInteger(n) && n >= 0 && n < thresholds.length) s.cut = n;
      if (action === 'looser') s.cut = Math.min(thresholds.length - 1, s.cut + 1);
      if (action === 'stricter') s.cut = Math.max(0, s.cut - 1);
      if (action === 'cell' && ['all', 'tp', 'fp', 'fn', 'tn'].includes(value)) s.cell = value;
      if (action === 'cost' && memberNumber(value, [1, 5]) !== null) s.cost = n;
    }
    if (kind === 'calibration') {
      if (action === 'temperature' && memberNumber(value, [0.5, 1, 2]) !== null) s.t = n;
      if (action === 'bins' && memberNumber(value, [3, 5]) !== null) { s.bins = n; s.bin = Math.min(s.bin, n - 1); }
      if (action === 'bin' && Number.isInteger(n) && n >= 0 && n < s.bins) s.bin = n;
    }
    if (kind === 'ranking-metrics') {
      if (action === 'inspect' && memberNumber(value, [0, 1, 2, 3]) !== null) s.selected = n;
      if (action === 'k' && memberNumber(value, [1, 3, 4]) !== null) s.k = n;
      if (action === 'ideal') s.order = [2, 0, 3, 1];
      if (action === 'move' && /^\d:-?1$/.test(value)) {
        const [i, direction] = value.split(':').map(Number), index = s.order.indexOf(i), next = index + direction;
        if (index >= 0 && next >= 0 && next < s.order.length) {
          s.order = [...s.order]; [s.order[index], s.order[next]] = [s.order[next], s.order[index]]; s.selected = i;
        }
      }
    }
    return s;
  }

  function factorMarkup(s) {
    const p = s.factors.users[s.user], q = s.factors.items[s.item], y = observations[s.user][s.item];
    const logit = dot(p, q), update = factorStep(s.factors, s.user, s.item, y);
    const before = s.history.length ? s.history[s.history.length - 1] : s.factors;
    const matrix = table('Observed exposures: 1 = clicked, 0 = shown without click, ? = unexposed', ['User', ...catalog.map(item => item.id)], users.map((name, u) => row([
      escape(name), ...catalog.map((item, i) => button('cell', `${u}:${i}`, observations[u][i] === null ? '?' : observations[u][i], s.user === u && s.item === i, false, `${name}, ${item.id} ${item.name}: ${observations[u][i] === null ? 'not exposed, unknown label' : observations[u][i] ? 'exposed and clicked' : 'exposed without click'}`)),
    ])));
    const predictions = table(`${users[s.user]}: every item shares the same user vector`, ['Item', 'Logit', 'P(click | shown)'], catalog.map((item, i) => row([
      button('cell', `${s.user}:${i}`, `${item.id} ${item.name}`, i === s.item), f(dot(p, s.factors.items[i])), pct(sigmoid(dot(p, s.factors.items[i]))),
    ])));
    return heading('01 / Shared factors', 'Can one click change scores for items never shown?') +
      controls(primary('train', '', 'Apply this observed-pair update', y === null || s.history.length >= 20) + button('undo', '', 'Previous update', undefined, !s.history.length)) +
      `<div class="rk-factor-source"><div><span>Training evidence</span><strong>${escape(users[s.user])} + ${escape(catalog[s.item].id)}</strong><p>${y === null ? 'Not exposed: no label to train on' : y ? 'Shown and clicked' : 'Shown, no click'}</p></div><span class="rk-flow-arrow" aria-hidden="true">&rarr;</span><div><span>Shared user vector</span><strong>${escape(vec(p))}</strong><p>Used in every score below</p></div></div>
      <div class="rk-flow-arrow rk-broadcast" aria-hidden="true">&darr;</div>
      <div class="rk-candidate-grid" role="group" aria-label="Inspect the shared user vector's item predictions">${catalog.map((item, i) => {
        const oldP = sigmoid(dot(before.users[s.user], before.items[i])), nowP = sigmoid(dot(p, s.factors.items[i]));
        const label = observations[s.user][i];
        return `<button type="button" class="rk-candidate" data-action="cell" data-value="${s.user}:${i}" aria-pressed="${s.item === i}"><strong>${escape(item.id)}: ${escape(item.name)}</strong><span>${label === null ? 'Never shown to this user' : label ? 'Observed click' : 'Observed non-click'}</span><b>${s.history.length ? `${pct(oldP)} &rarr; ` : ''}${pct(nowP)}</b><span>predicted click if shown</span></button>`;
      }).join('')}</div>` +
      note(s.last ? `One pair updated the shared factors. Its BCE changed from ${f(s.last.before)} to ${f(s.last.after)}; even unexposed scores moved.` : 'Select any item. An unknown interaction is not a negative; only observed pairs supply labels.') +
      disclose('Choose another user', controls(users.map((name, u) => button('cell', `${u}:${s.item}`, name, s.user === u)).join(''))) +
      calculation(`${matrix}${inspector(`${users[s.user]} + ${catalog[s.item].id}`, `<div class="rk-equation">p = ${escape(vec(p))}<br>q = ${escape(vec(q))}<br>p &middot; q = ${p.map((x, i) => `${f(x, 2)} &times; ${f(q[i], 2)}`).join(' + ')} = ${f(logit)}</div>${y === null ? note('No direct likelihood term exists for this pair. Other observed pairs can still update its shared factors.') : `<div class="rk-equation">residual = ${f(sigmoid(logit))} - ${y} = ${f(update.residual)}<br>&nabla;p = ${escape(vec(update.gp))}<br>&nabla;q = ${escape(vec(update.gq))}</div>${note('Simultaneous update from old vectors: residual times other vector + 0.02 times own vector. Learning rate = 0.4.')}`}`)}${predictions}`) + reset();
  }

  function vectorPlot(q, item, cosine) {
    const v = cosine ? normalize(item.vector) : item.vector;
    const query = cosine ? normalize(q) : q;
    const xy = vector => [35 + vector[0] * 145, 192 - vector[1] * 145];
    const [qx, qy] = xy(query), [ix, iy] = xy(v);
    return `<svg class="rk-vector-plot" viewBox="0 0 230 225" role="img" aria-label="Query and selected item vectors, from the origin. Values are in the adjacent inspector."><path d="M35 18 V192 H215" class="rk-axis"/><path d="M35 192 L${qx} ${qy}" class="rk-query-line"/><path d="M35 192 L${ix} ${iy}" class="rk-item-line"/><circle cx="${qx}" cy="${qy}" r="5" class="rk-query-dot"/><rect x="${ix - 5}" y="${iy - 5}" width="10" height="10" class="rk-item-dot"/><text x="20" y="210">0</text><text x="180" y="216">axis 1</text><text x="5" y="15">axis 2</text></svg>`;
  }
  function towerMarkup(s) {
    const request = queries[s.query], q = matvec(QUERY_W, request.features), item = lookup(s.item);
    const qUsed = s.cosine ? normalize(q) : q, vUsed = s.cosine ? normalize(item.vector) : item.vector;
    const ranked = retrieve(q, 6, s.cosine), score = dot(qUsed, vUsed);
    const tower = (side, inputLabel, input, encoder, cadence, output) => `<div class="rk-tower-lane rk-tower-${escape(side)}">
      <div class="rk-tower-input"><span>${escape(inputLabel)}</span><strong>${escape(input)}</strong></div>
      <div class="rk-flow-arrow" aria-hidden="true">&darr;</div>
      <div class="rk-encoder-box"><span>${escape(cadence)}</span><strong>${escape(encoder)}</strong></div>
      <div class="rk-flow-arrow" aria-hidden="true">&darr;</div>
      <div class="rk-tower-output">${escape(output)}</div>
    </div>`;
    return heading('02 / Independent encoders, one score', 'Why can a new request reuse the same item vectors?') +
      controls(primary('query', (s.query + 1) % queries.length, `Try request: ${queries[(s.query + 1) % queries.length].name}`)) +
      `<div class="rk-tower-flow" role="group" aria-label="Independent query and item encoders feed similarity">
        ${tower('query', 'Live request', request.name, 'Query encoder', 'Online: once per request', 'q: fresh query vector')}
        ${tower('item', 'Selected catalog item', `${item.id}: ${item.name}`, 'Item encoder', 'Offline: encode, then cache', 'v: reused item vector')}
        <svg class="rk-tower-merge" viewBox="0 0 100 24" preserveAspectRatio="none" aria-hidden="true"><path d="M25 0 V10 H50 V23 M75 0 V10 H50 M48 19 L50 23 L52 19"/></svg>
        <div class="rk-similarity"><span>Combine the two independent outputs</span><h4>${s.cosine ? 'Normalize q and v, then dot product' : 'Similarity: q dot v'}</h4><strong>${f(score)}</strong><p>Selected item ${escape(item.id)} is #${ranked.findIndex(entry => entry.id === item.id) + 1} of ${catalog.length}.</p></div>
      </div><div class="rk-candidate-grid" role="group" aria-label="Top three retrieved items">${ranked.slice(0, 3).map((entry, i) => `<button type="button" class="rk-candidate" data-action="item" data-value="${escape(entry.id)}" aria-pressed="${s.item === entry.id}"><span>Rank ${i + 1}</span><strong>${escape(entry.id)}: ${escape(entry.name)}</strong><span>score ${f(entry.score)}</span></button>`).join('')}</div>` +
      note('Only the query branch runs for a new request. The cached item vectors stay unchanged.') +
      disclose('Change request or similarity', controls(select('Online request', 'query', queries.map((query, i) => [i, query.name]), s.query) + button('similarity', 'dot', 'Dot product', !s.cosine) + button('similarity', 'cosine', 'L2-normalized cosine', s.cosine))) +
      calculation(`<div class="ml-grid"><div>${table('Exact search over the cached catalog; select an item to inspect', ['Rank', 'Item', 'Score'], ranked.map((entry, i) => row([String(i + 1), button('item', entry.id, `${entry.id} ${entry.name}`, s.item === entry.id), f(entry.score)])))}</div>${inspector(`${item.id}: computed vectors`, `<div class="rk-equation">query q = ${escape(vec(q))}<br>cached v = ${escape(vec(item.vector))}${s.cosine ? `<br>unit q = ${escape(vec(qUsed))}<br>unit v = ${escape(vec(vUsed))}` : ''}<br>${qUsed.map((x, i) => `${f(x)} &times; ${f(vUsed[i])}`).join(' + ')}<br>= ${f(score)}</div>${note(s.cosine ? 'Both scoring vectors have unit norm. Only direction affects this score.' : 'Dot product depends on direction and norm; it is not a probability.')}<div class="rk-equation">query features = ${escape(vec(request.features))}<br>Wq = [[1, .65, .05], [.2, .65, 1]]<br>q = Wq &times; features<br>item features = ${escape(vec(item.features))}<br>Wi = [[1, .7, 0], [.1, .3, 1]]<br>v = Wi &times; features</div>${note('Toy metadata features: retrieval, ranking, policy. Fixed linear encoders stand in for trained towers.')}${vectorPlot(q, item, s.cosine)}<p class="rk-legend"><span class="rk-query-key">Circle: query</span><span class="rk-item-key">Square: item</span></p>`)}</div>${note('This is exact search, not a simulated ANN index. Item-encoder changes require refreshing cached vectors and the index. A compatible query-only update can reuse them; a cross-encoder cannot cache the same query-independent item representation.')}`) + reset();
  }

  const modes = { pair: 'Pairwise logistic', lambda: 'Lambda-weighted pair', point: 'Pointwise BCE', list: 'Listwise softmax CE' };
  function objectiveList(s, result, preferred, other) {
    const beforeScores = s.history.length ? s.history[s.history.length - 1] : s.scores;
    const movement = rankMovement(s.scores, beforeScores);
    const min = Math.min(...s.scores, ...beforeScores), max = Math.max(...s.scores, ...beforeScores);
    const padding = (max - min || 1) * 0.12, low = min - padding, high = max + padding;
    const position = value => 100 * (value - low) / (high - low);
    const pairMoves = [preferred, other].map(i => {
      const item = movement.find(entry => entry.index === i);
      return `${rankItems[i].id}: #${item.beforeRank} to #${item.rank}${item.rankDelta === 0 ? ' (same rank)' : ''}`;
    }).join('; ');
    return `<div class="rk-ranking-stage"><h4>${s.history.length ? 'After the latest change' : 'Current ranking: before training'}</h4>
      <p class="rk-change-summary" role="status">${s.history.length ? escape(pairMoves) : `Preferred ${escape(rankItems[preferred].id)} is #${movement.find(item => item.index === preferred).rank}; other ${escape(rankItems[other].id)} is #${movement.find(item => item.index === other).rank}. Apply changes their scores, not their labels.`}</p>
      <ol class="rk-objective-list" aria-label="Items ordered by current score">${movement.map(entry => {
        const i = entry.index, isPreferred = i === preferred, isOther = i === other;
        const oldPosition = position(entry.beforeScore), newPosition = position(entry.score);
        const update = -0.7 * result.gradients[i];
        const rankDirection = entry.rankDelta > 0 ? `&uarr; ${entry.rankDelta} ${entry.rankDelta === 1 ? 'place' : 'places'}` : entry.rankDelta < 0 ? `&darr; ${-entry.rankDelta} ${entry.rankDelta === -1 ? 'place' : 'places'}` : '= same rank';
        const scoreDirection = update > 0 ? '&uarr; score' : update < 0 ? '&darr; score' : 'no score change';
        return `<li class="rk-objective-card${isPreferred ? ' rk-pair-preferred' : isOther ? ' rk-pair-other' : ''}" data-ranked-item="${escape(rankItems[i].id)}">
          <div class="rk-objective-item"><span class="rk-position-number">#${entry.rank}</span><div><strong>${escape(rankItems[i].id)}: ${escape(rankItems[i].name)}</strong><span>${isPreferred ? 'Preferred item' : isOther ? 'Other item' : 'Other candidate'} &middot; grade ${rankItems[i].rel}</span></div></div>
          <div class="rk-movement-label"><span>Before #${entry.beforeRank} &rarr; Now #${entry.rank}</span><strong>${rankDirection}</strong></div>
          <div class="rk-score-track" aria-hidden="true"><span class="rk-score-travel" style="left:${Math.min(oldPosition, newPosition)}%;width:${Math.abs(newPosition - oldPosition)}%"></span><span class="rk-score-before" style="left:${oldPosition}%"></span><span class="rk-score-now" style="left:${newPosition}%"></span></div>
          <div class="rk-score-change"><span>Score ${f(entry.beforeScore)} &rarr; ${f(entry.score)}</span><span class="rk-next-direction">Next step: ${scoreDirection}</span></div>
        </li>`;
      }).join('')}</ol>
      <p class="rk-score-legend">Score rail: ${f(low, 2)} to ${f(high, 2)}, shared by all items. Open marker = before; filled = now. Rank can stay unchanged while the score moves.</p>
    </div>`;
  }
  function objectiveMarkup(s) {
    const r = objective(s.scores, s.mode, s.pair, s.k), [a, b] = pairs[s.pair];
    const isPair = s.mode === 'pair' || s.mode === 'lambda';
    const pairLabel = `${rankItems[a].id} preferred to ${rankItems[b].id}`;
    const gradientNote = s.mode === 'lambda' ? `Multiply the logistic pair gradients by |swap &Delta;NDCG@${s.k}| = ${f(r.delta)}. Hold this weight fixed for this update; recompute ranks afterward.` : s.mode === 'pair' ? 'A negative score gradient raises the preferred score under gradient descent. The other score moves down by the same amount.' : s.mode === 'point' ? 'Each item receives its own binary target (grade >= 2). Mean BCE gradient = (sigmoid(score) - target) / 4.' : 'The target distribution is normalized relevance gain, not binary clicks. All four scores compete through one softmax denominator.';
    return heading('03 / Optimize scores; evaluate the resulting order', 'Which score receives the learning signal?') +
      controls(primary('train', '', 'Apply gradient step', s.history.length >= 30) + button('undo', '', 'Previous update', undefined, !s.history.length)) +
      stats([[modes[s.mode], f(r.loss)], [`NDCG@${s.k}`, f(r.ndcg)]]) + objectiveList(s, r, a, b) +
      disclose('Change objective, pair, or cutoff', controls(select('Training objective', 'mode', Object.entries(modes), s.mode) + select('Inspected preference', 'pair', pairs.map(([i, j], p) => [p, `${rankItems[i].id} > ${rankItems[j].id}`]), s.pair) + select('NDCG cutoff', 'k', [2, 3, 4].map(k => [k, `@${k}`]), s.k) + button('swap', '', 'Swap selected scores', undefined, s.history.length >= 30))) +
      calculation(inspector(pairLabel, `<div class="rk-equation">margin = ${f(s.scores[a])} - ${f(s.scores[b])} = ${f(r.pair.margin)}<br>Pair-model P(${escape(rankItems[a].id)} &gt; ${escape(rankItems[b].id)}) = ${f(r.pair.probability)}<br>unweighted pair loss = ${f(pairwise(s.scores[a], s.scores[b]).loss)}<br>NDCG if pair swapped = ${f(r.swappedNdcg)}</div>${note(gradientNote)}${isPair ? `<div class="rk-equation">dL/ds(${escape(rankItems[a].id)}) = ${f(r.gradients[a])}<br>dL/ds(${escape(rankItems[b].id)}) = ${f(r.gradients[b])}</div>` : `${table('Next-step targets, probabilities and gradients', ['Item', 'Target', 'Model', 'Gradient'], rankItems.map((item, i) => row([escape(item.id), f(r.targets[i]), f(r.probabilities[i]), f(r.gradients[i])])))}${note('The selected pair is an inspection view; this objective updates all items, not only this pair.')}`}${note('Next step: score minus 0.7 times gradient. List arrows describe the next score update, not a guaranteed rank change. These are independent logits, not a full neural ranker.')}${note('NDCG is constant between rank crossings. Lambda weighting is a local metric-weighted rule with detached swap weights, not a derivative through sorting.')}`)) + reset();
  }

  function funnelMarkup(s) {
    const r = funnel(s.k, s.extra, s.oracle), chosen = lookup(s.item), present = r.candidates.some(item => item.id === s.item);
    const chip = (item, label, missing = false) => `<button type="button" class="rk-pipeline-item${missing ? ' rk-omitted' : ''}" data-action="item" data-value="${escape(item.id)}" aria-pressed="${s.item === item.id}"><strong>${escape(item.id)}: ${escape(item.name)}</strong><span>${escape(label)}</span></button>`;
    return heading('04 / Candidate-set ceiling', 'Can the ranker rescue an item that was never retrieved?') +
      controls(primary('source', '', s.extra ? 'Remove lexical source D' : 'Recover D with lexical retrieval')) +
      stats([['Relevant candidate recall', pct(r.recall)], ['Served NDCG@2', f(r.metrics.ndcg)]]) +
      `<div class="rk-pipeline"><section><h4>Catalog: 6 items</h4>${catalog.map(item => chip(item, `grade ${item.rel}${r.candidates.some(c => c.id === item.id) ? '' : ' / omitted'}`, !r.candidates.some(c => c.id === item.id))).join('')}</section><span class="rk-pipeline-arrow" aria-hidden="true">&rarr;</span><section><h4>Retrieved: ${r.candidates.length}</h4>${r.candidates.map(item => chip(item, 'Available to reranker')).join('')}</section><span class="rk-pipeline-arrow" aria-hidden="true">&rarr;</span><section><h4>Served: top 2</h4>${r.ranked.slice(0, 2).map((item, i) => chip(item, `#${i + 1} / grade ${item.rel}`)).join('')}</section></div>` +
      note(present ? `${escape(chosen.id)} is available to rank. Candidate membership changed; its grade ${chosen.rel} did not.` : `${escape(chosen.id)} has grade ${chosen.rel} but was omitted. The ranker cannot score an item it never receives.`) +
      disclose('Change retrieval size or try an oracle', controls(select('Vector retrieval K', 'k', [2, 3, 4, 6].map(k => [k, String(k)]), s.k) + button('oracle', '', 'Use oracle reranker', s.oracle))) +
      calculation(`${stats([['Oracle ceiling NDCG@2', f(r.ceiling)]])}${table('Retrieval scores and candidate membership', ['Item / grade', 'Retrieval score', 'Candidate?'], retrieve(matvec(QUERY_W, queries[0].features), 6).map(item => row([escape(`${item.id} ${item.name} / ${item.rel}`), f(item.score), r.candidates.some(c => c.id === item.id) ? 'Included' : 'Omitted'])))}${table(`${s.oracle ? 'Oracle' : 'Contextual'} reranking`, ['Position', 'Item', 'Rank score'], r.ranked.map((item, i) => row([String(i + 1), escape(item.id), f(item.rankScore)])))}${note('Relevant means grade >= 2: A, D, F. Recall and ideal DCG use the full catalog. The oracle sorts by evaluation labels as a diagnostic upper bound, not a deployable model.')}${note('Contextual score = dot product + 2 times exact-intent feature. D has intent 1, F 0.6, A 0.3, others 0. The feature is not the grade. The lexical source is a fixed complementary candidate D, not a simulated search engine.')}`) + reset();
  }

  function coldMarkup(s) {
    const rows = coldRows(s.step, s.profile), selected = rows.find(item => item.id === s.item), event = exposureLog[s.step - 1];
    const eventText = !event ? 'No events observed. Every behavioral rate is unknown.' : event.y === null ? `Event ${s.step}: D was not exposed. No success or failure enters the likelihood.` : `Event ${s.step}: ${event.id} was exposed and ${event.y ? 'clicked' : 'not clicked'}. Add one ${event.y ? 'success' : 'failure'}.`;
    return heading('05 / New item, no invented negatives', 'What changes when an item is actually shown?') + nav(s.step, exposureLog.length) +
      `<p class="rk-change-summary" role="status">${escape(eventText)}</p><h4>${escape(selected.id)}: ${escape(selected.name)}</h4>
      <div class="rk-evidence-flow"><div class="rk-mechanism-node"><span>${s.profile === 'content' ? 'Content prior' : 'Population prior'}</span><strong>${pct(selected.prior)}</strong><span>before any exposure</span></div><span class="rk-flow-arrow" aria-hidden="true">+</span><div class="rk-mechanism-node"><span>Observed evidence</span><strong>${selected.n ? `${selected.clicks} / ${selected.n}` : 'Unknown'}</strong><span>${selected.n ? 'clicks / shown' : 'not a zero click rate'}</span></div><span class="rk-flow-arrow" aria-hidden="true">&rarr;</span><div class="rk-mechanism-node rk-posterior"><span>Updated estimate</span><strong>${pct(selected.mean)}</strong><span>${selected.n ? 'prior plus real outcomes' : 'equals the prior'}</span></div></div>
      <div class="rk-candidate-grid" role="group" aria-label="Inspect item evidence">${rows.map(item => `<button type="button" class="rk-candidate" data-action="item" data-value="${escape(item.id)}" aria-pressed="${s.item === item.id}"><strong>${escape(item.id)}: ${escape(item.name)}</strong><b>${pct(item.mean)}</b><span>${item.n ? `${item.clicks} clicks / ${item.n} shown` : 'No exposure yet'}</span></button>`).join('')}</div>` +
      disclose('Change available user information', controls(button('profile', 'content', 'Onboarded retrieval interest', s.profile === 'content') + button('profile', 'population', 'No profile: population prior', s.profile === 'population'))) +
      calculation(`${inspector(`${selected.id}: prior + evidence`, `<div class="rk-equation">prior p0 = ${f(selected.prior)}<br>prior strength m = 4<br>posterior = (4 &times; ${f(selected.prior)} + ${selected.clicks}) / (4 + ${selected.n})<br>= ${f(selected.mean)}</div>${stats([['Data weight n/(n+4)', pct(selected.weight)], ['Posterior standard deviation', f(selected.sd)]])}${note(s.profile === 'content' ? 'Toy content prior: sigmoid(2 times query-item dot - 2). Onboarding provides a query without behavioral history.' : 'With no profile, all items start at a 25% population mean.')}`)}${table('A fixed user segment', ['Item', 'Clicks / shown', 'Prior', 'Posterior'], rows.map(item => row([escape(item.id), item.n ? `${item.clicks} / ${item.n}` : 'Unknown', pct(item.prior), pct(item.mean)])))}${note('This Beta-Bernoulli baseline shrinks exposed click rates toward the prior. Unexposed records contribute neither successes nor failures. It is not collaborative factorization, and logged outcomes remain conditional on exposure and position.')}`) + reset();
  }

  function thresholdMarkup(s) {
    const t = thresholds[s.cut], c = confusion(impressions, t), counts = c.counts;
    const choices = thresholds.map(threshold => {
      const result = confusion(impressions, threshold);
      return { threshold, cost: result.counts.fp + s.cost * result.counts.fn };
    });
    const best = Math.min(...choices.map(choice => choice.cost));
    const names = { tp: 'TP: clicked, selected', fp: 'FP: no click, selected', fn: 'FN: clicked, rejected', tn: 'TN: no click, rejected' };
    const selectedIds = s.cell === 'all' ? impressions.map(r => r.id) : c[s.cell];
    return heading('06 / Same scores, different decisions', 'Who changes sides when the cutoff moves?') +
      controls(primary('looser', '', 'Admit the next impression', s.cut === thresholds.length - 1) + button('stricter', '', 'Stricter cutoff', undefined, s.cut === 0)) +
      stats([['Select when p >=', f(t, 2)], ['Precision', pct(c.precision)], ['Recall', pct(c.recall)]]) +
      `<div class="ml-grid"><div><h4>Scored impressions</h4><div class="rk-impressions" role="group" aria-label="Select a record to place the cutoff">${impressions.map((item, i) => `<button type="button" class="rk-impression ${item.p >= t ? 'rk-admitted' : 'rk-rejected'}${selectedIds.includes(item.id) ? '' : ' rk-dim'}" data-action="cut" data-value="${i + 1}" aria-pressed="${s.cut === i + 1}"><span>#${item.id} / p=${f(item.p, 2)}</span><strong>${item.y ? 'Clicked' : 'No click'}</strong><span>${item.p >= t ? 'Selected' : 'Rejected'}</span></button>`).join('')}</div></div><div><h4>Where those records land</h4><div class="rk-confusion">${Object.entries(names).map(([key, label]) => `<button type="button" data-action="cell" data-value="${key}" aria-pressed="${s.cell === key}"><span>${escape(label)}</span><strong>${counts[key]}</strong><small>IDs ${c[key].join(', ') || 'none'}</small></button>`).join('')}</div>${s.cell !== 'all' ? controls(button('cell', 'all', 'Clear highlight')) : ''}${note('Tap a confusion cell to locate its records. Scores and click labels never change; only the selection cutoff moves.')}</div></div>` +
      disclose('Change mistake costs', controls(button('cost', 1, 'FP = FN = 1', s.cost === 1) + button('cost', 5, 'FP = 1, FN = 5', s.cost === 5))) +
      calculation(`${stats([[`Precision ${counts.tp}/${counts.tp + counts.fp}`, pct(c.precision)], [`Recall ${counts.tp}/${counts.tp + counts.fn}`, pct(c.recall)], ['F1', f(c.f1)], ['Cost FP + cFN', String(counts.fp + s.cost * counts.fn)], ['Best cost on this set', String(best)]])}${table('Held-out impression decisions', ['ID', 'Score', 'Clicked?', 'Decision'], impressions.map(item => row([String(item.id), f(item.p, 2), String(item.y), item.p >= t ? 'Select' : 'Reject'])))}${note('All 12 examples were exposed. Rejection is a hypothetical downstream decision, not a missing label. Cost changes do not alter precision or recall at a fixed cutoff. Choose thresholds on validation data, not this evaluation sample.')}`) + reset();
  }

  function calibrationPlot(result) {
    const x = p => 35 + 165 * p, y = p => 200 - 165 * p;
    return `<svg class="rk-calibration-plot" viewBox="0 0 240 245" role="img" aria-label="Reliability diagram. Horizontal axis: mean predicted click probability. Vertical axis: observed click fraction. Dashed diagonal is equality."><path d="M35 30 V200 H205" class="rk-axis"/><path d="M35 200 L200 35" class="rk-reference"/>${result.bins.filter(bin => bin.n).map(bin => `<circle cx="${x(bin.mean)}" cy="${y(bin.rate)}" r="${4 + Math.sqrt(bin.n)}" class="rk-query-dot"><title>${escape(`Bin ${bin.index + 1}: ${bin.n} impressions; predicted ${pct(bin.mean)}, observed ${pct(bin.rate)}`)}</title></circle>`).join('')}<text x="18" y="205">0</text><text x="18" y="40">1</text><text x="197" y="218">1</text><text x="5" y="16">observed click fraction</text><text x="38" y="239">predicted probability</text></svg>`;
  }
  function calibrationMarkup(s) {
    const r = calibration(impressions, s.bins, s.t), bin = r.bins[s.bin];
    const nextT = s.t === 1 ? 2 : s.t === 2 ? .5 : 1;
    const binBar = (label, value, observed = false) => `<span class="rk-reliability-row"><span>${escape(label)}</span><span class="rk-reliability-rail"><span class="${observed ? 'rk-outcome-bar' : 'rk-probability-bar'}" style="width:${100 * value}%"></span></span><b>${pct(value)}</b></span>`;
    return heading('07 / Probabilities are a separate contract', 'Do predicted click rates match observed click rates?') +
      controls(primary('temperature', nextT, nextT === 2 ? 'Soften the probabilities (T=2)' : nextT === .5 ? 'Sharpen the probabilities (T=0.5)' : 'Restore raw probabilities (T=1)')) +
      stats([['Empirical ECE', pct(r.ece)], ['Score order', 'Unchanged']]) +
      `<div class="rk-bin-list" role="group" aria-label="Predicted versus observed click rates">${r.bins.map(b => `<button type="button" class="rk-bin-card" data-action="bin" data-value="${b.index}" aria-pressed="${s.bin === b.index}"><strong>Bin ${b.index + 1}: ${pct(b.low)} to ${pct(b.high)}</strong><span>${b.n ? `${sum(b.rows.map(item => item.y))} clicks / ${b.n} shown` : 'No examples: rate unknown'}</span>${b.n ? binBar('Predicted', b.mean) + binBar('Observed', b.rate, true) : ''}</button>`).join('')}</div>
      <div class="rk-observed-outcomes"><h4>Actual outcomes in bin ${s.bin + 1}</h4><div>${bin.rows.map(item => `<span class="rk-outcome-chip">#${item.id}: ${item.y ? 'click' : 'no click'}</span>`).join('') || '<span>No records in this bin.</span>'}</div></div>` +
      note('The labels stay fixed. Temperature changes probabilities and bin membership, not the ranked order.') +
      disclose('Change temperature or binning', controls(select('Temperature T', 'temperature', [[1, '1: raw scores'], [2, '2: soften logits'], [0.5, '0.5: sharpen logits']], s.t) + select('Equal-width bins', 'bins', [[3, '3 bins'], [5, '5 bins']], s.bins))) +
      calculation(`${stats([['Brier score', f(r.brier)]])}${calibrationPlot(r)}${inspector(`Bin ${s.bin + 1}: [${f(bin.low, 2)}, ${f(bin.high, 2)}${s.bin === s.bins - 1 ? ']' : ')'}`, bin.n ? `<div class="rk-equation">mean p = (${bin.rows.map(item => f(item.p, 2)).join(' + ')}) / ${bin.n} = ${f(bin.mean)}<br>observed = ${sum(bin.rows.map(item => item.y))}/${bin.n} = ${f(bin.rate)}<br>ECE contribution = ${bin.n}/12 &times; |${f(bin.mean)} - ${f(bin.rate)}| = ${f(bin.contribution)}</div>${table('Actual outcomes in this bin', ['ID', 'Predicted p', 'Clicked y'], bin.rows.map(item => row([String(item.id), f(item.p), String(item.y)])))}` : note('No examples land here; the ECE weight is zero.'))}${note('p(T) = sigmoid(logit(raw p) / T). This is positive-event calibration, not classification accuracy. Temperature is not fitted here; fit it on a separate calibration split. This tiny bin-sensitive ECE does not establish population calibration; Brier also measures discrimination.')}`) + reset();
  }

  function metricMarkup(s) {
    const rels = s.order.map(i => rankItems[i].rel), m = rankingMetrics(rels, s.k);
    const position = s.order.indexOf(s.selected), selected = rankItems[s.selected], contribution = gain(selected.rel) * discount(position, s.k);
    return heading('08 / Fixed judgments, movable ranks', 'How much does moving a useful result upward matter?') +
      controls(primary('move', `${s.selected}:-1`, `Move ${selected.id} up one rank`, position === 0) + button('move', `${s.selected}:1`, `Move ${selected.id} down`, undefined, position === s.order.length - 1)) +
      stats([[`NDCG@${s.k}`, f(m.ndcg)], [`RR@${s.k} (grade >= 2)`, f(m.rr)]]) +
      `<ol class="rk-ranked-list">${s.order.map((i, rank) => {
        const value = gain(rankItems[i].rel) * discount(rank, s.k);
        return `<li class="${i === s.selected ? 'rk-selected' : ''}${rank >= s.k ? ' rk-outside' : ''}"><div class="rk-rank-title"><span>#${rank + 1}</span>${button('inspect', i, `${rankItems[i].id} ${rankItems[i].name}`, i === s.selected)}</div><p>grade ${rankItems[i].rel} &middot; DCG contribution ${f(value)}${rank >= s.k ? ' (outside cutoff)' : ''}</p><div class="rk-contribution-rail" role="img" aria-label="${escape(rankItems[i].id)} contributes ${pct(value / m.idcg)} of ideal DCG"><span style="width:${100 * value / m.idcg}%"></span></div></li>`;
      }).join('')}</ol>` +
      note('Select a result, then move it. Each bar is its contribution to the fixed ideal score; the grades never change.') +
      disclose('Change cutoff or show ideal order', controls(select('Evaluation cutoff', 'k', [1, 3, 4].map(k => [k, `@${k}`]), s.k) + button('ideal', '', 'Order by true grade'))) +
      calculation(inspector(`${selected.id} at rank ${position + 1}`, `<div class="rk-equation">gain = 2^${selected.rel} - 1 = ${gain(selected.rel)}<br>discount = ${position < s.k ? `1 / log2(${position + 2}) = ${f(discount(position, s.k))}` : '0 (outside cutoff)'}<br>contribution = ${f(contribution)}<br>NDCG = ${f(m.dcg)} / ${f(m.idcg)} = ${f(m.ndcg)}</div>${note('IDCG sorts the complete judgment set: grades [3, 2, 1, 0]. It stays fixed when items move. RR uses the first grade >= 2 within K; MRR averages over queries.')}${note('These are held-out relevance judgments, not raw clicks. Click-derived labels need attention to exposure, position bias, and missing judgments.')}`)) + reset();
  }

  const renderers = {
    'matrix-factorization': factorMarkup, 'two-tower': towerMarkup, 'rank-objectives': objectiveMarkup,
    'retrieval-funnel': funnelMarkup, 'cold-start': coldMarkup, 'threshold-metrics': thresholdMarkup,
    calibration: calibrationMarkup, 'ranking-metrics': metricMarkup,
  };
  const render = (kind, state) => `<div class="rk-lab">${renderers[kind] ? renderers[kind](state) : ''}</div>`;

  function lesson(title, summary, what, why, interview, details, math, code, quiz) {
    return { title, summary, what, why, interview, details, math, code: { title: 'Executable mechanism', lang: 'python', snippet: code }, quiz, controls: [], presets: [], geometry: null };
  }
  const math = (title, formula, noteText, annotations) => ({ title, formula, note: noteText, annotations });
  const quiz = (prompt, correct, wrong, explanation, misconception) => ({ prompt, options: [
    { text: correct, correct: true, explanation }, { text: wrong, correct: false, explanation: misconception },
  ] });
  const content = {
    'matrix-factorization': lesson(
      'Matrix factorization: train shared factors on observed evidence',
      'Inspect a sparse exposure matrix, then apply an actual logistic factorization update.',
      'A user vector participates in every item score for that user. Updating one observed pair therefore changes predictions for unseen pairs too.',
      'The low-rank assumption transfers statistical strength without pretending that an unexposed item was disliked.',
      'Separate the interaction model from the observation process: exposure and negative sampling assumptions determine what the fitted score means.',
      ['The example uses logistic matrix factorization on exposed binary clicks, not least-squares explicit ratings.', 'A factor axis is not identifiable as a named preference. Joint rotations can preserve dot products.', 'A production implicit-feedback objective may sample unobserved pairs, but those are assumed or weighted negatives, not known non-clicks.'],
      math('An exposed-pair gradient', ['s_{ui}=p_u^Tq_i,\\quad \\hat y_{ui}=\\sigma(s_{ui})', '\\ell=\\operatorname{softplus}(s_{ui})-y_{ui}s_{ui}+\\frac{\\lambda}{2}(\\|p_u\\|^2+\\|q_i\\|^2)', '\\nabla_{p_u}\\ell=(\\hat y-y)q_i+\\lambda p_u'], 'Apply simultaneous updates using the old vectors. No likelihood term is added for a question-mark cell.', [['p_u, q_i', 'Shared two-dimensional factors', 'Morgan p=[1, .2], A q=[1, .1]'], ['y', 'Observed click after exposure', 'A clicked: y=1; unexposed D: unknown'], ['lambda', 'L2 gradient coefficient', '0.02 in each displayed update']]),
      'import numpy as np\n\ndef step(p, q, y, eta=0.4, reg=0.02):\n    if y is None:  # unexposed is not a negative\n        return p, q\n    prob = 1 / (1 + np.exp(-(p @ q)))\n    residual = prob - y\n    gp = residual * q + reg * p\n    gq = residual * p + reg * q\n    return p - eta * gp, q - eta * gq\n\np = np.array([1., .2]); q = np.array([1., .1])\np, q = step(p, q, y=1)',
      quiz('Morgan was never shown D. Which update is justified by this dataset?', 'No direct loss term for Morgan-D; train on observed pairs, including Lee-D.', 'Set Morgan-D to zero and treat it as a measured dislike.', 'D was shown to Lee. That observation can update D\'s shared vector and indirectly change Morgan-D, but Morgan\'s missing exposure supplies no label.', 'A sampled negative would require a separate modeling assumption and appropriate weighting.')
    ),
    'two-tower': lesson(
      'Two towers: independent encoders make item vectors cacheable',
      'Encode three request types and inspect exact scores over a fixed six-item catalog.',
      'The item encoder never sees the live request. Its output is reusable across requests; the online query vector supplies the changing side of the dot product.',
      'Independent encoding makes vector indexing possible, but limits how much pair-specific interaction happens before candidate selection.',
      'Discuss encoder compatibility, index freshness, ANN recall, and dot-product versus cosine semantics separately from downstream ranking quality.',
      ['Linear projections make every number inspectable; production towers are learned from retrieval objectives and may be nonlinear.', 'The table performs exact search. It does not invent latency numbers or claim to model ANN search errors.', 'Contrastive training uses sampled alternatives. False negatives, sampling distributions, and popularity bias affect the learned retrieval scores.'],
      math('Factorized serving score', ['q=W_qx_q,\\quad v_i=W_ix_i,\\quad s_i=q^Tv_i', '\\cos(q,v_i)=\\frac{q^Tv_i}{\\|q\\|\\|v_i\\|}'], 'Cosine normalizes away vector magnitudes. Dot-product retrieval need not produce the same order.', [['x_i', 'Toy item metadata features', 'A: [1, 0, 0]'], ['v_i', 'Cached item representation', 'A: [1, .1]'], ['q^T v_i', 'Compatibility, not a probability', 'Retrieval query [1, .2] scores A at 1.02']]),
      'import numpy as np\n\nWq = np.array([[1, .65, .05], [.2, .65, 1]])\nWi = np.array([[1, .7, 0], [.1, .3, 1]])\nfeatures = np.array([[1,0,0], [.5,.5,.15], [.05,0,1],\n                     [.3,.2,.25], [.1,0,.65], [0,1,0]])\ncached_items = features @ Wi.T  # offline\nq = Wq @ np.array([1,0,0])       # per request\nscores = cached_items @ q\nprint(np.argsort(-scores)[:3])   # A, B, F',
      quiz('Why can the same item vectors serve a different live query?', 'Each item vector depends on item features, not the query.', 'A cross-encoder produces one universal item vector for all pairs.', 'The query changes while the cached item side stays fixed.', 'A cross-encoder uses query-item interactions; its pair representation cannot generally be cached once per item.')
    ),
    'rank-objectives': {
      ...lesson(
        'Ranking objectives: inspect gradients, not just a final metric',
        'Compare pointwise BCE, pairwise logistic, a Lambda-weighted pair, and listwise softmax cross-entropy on one fixed slate.',
        'Pairwise training moves a score difference. Lambda weighting scales that update by the NDCG consequence of swapping the current positions.',
        'A scalar ranking metric hides the update mechanism. Inspecting per-score gradients reveals which pairs receive learning capacity and why.',
        'A ranking surrogate is not the metric itself. Explain sampled preferences, pair weighting, the cutoff, and how selection bias enters labels.',
        ['Pointwise BCE here binarizes grade >= 2; listwise CE instead uses normalized exponential gains as its target distribution.', 'Lambda weights are recomputed from the current ranks, then held fixed during a local update. This is not differentiating through sort.', 'Independent logits isolate the objective. A shared neural ranker can couple updates across items and queries.'],
        math('Pair logistic loss and metric-aware weighting', ['m=s^+-s^-,\\quad L_{pair}=\\log(1+e^{-m})', '\\frac{\\partial L}{\\partial s^+}=-\\sigma(-m),\\quad \\frac{\\partial L}{\\partial s^-}=\\sigma(-m)', 'w=|\\Delta NDCG@K|=\\frac{|(G_i-G_j)(D_i-D_j)|}{IDCG@K}', '\\lambda_i=w\\,\\frac{\\partial L}{\\partial s_i}'], 'D is zero beyond the cutoff. A pair entirely outside K can receive zero weight in this local swap rule.', [['m', 'Preferred minus other score', 'D .2 minus B 1.2 gives -1'], ['G_i', 'Exponential relevance gain', 'Grade 3 gives 7'], ['w', 'Current-position swap importance', 'Swapping D and B changes the actual NDCG numerator']]),
        'import math\n\ndef pair_step(preferred, other, weight=1., eta=.7):\n    margin = preferred - other\n    g = 1 / (1 + math.exp(margin))\n    # weight=abs(delta_ndcg) for the local Lambda rule\n    return preferred + eta*weight*g, other - eta*weight*g\n\nprint(pair_step(.2, 1.2))  # D rises; B falls\n# Pointwise: grad_i = (sigmoid(s_i)-y_i)/n\n# Listwise: grad_i = softmax(s)_i - gain_i/sum(gains)',
        quiz('Why can the logistic loss improve while NDCG stays unchanged?', 'Scores can improve their margin without changing the sorted order.', 'NDCG uses the numeric score margin in its numerator.', 'NDCG depends on judged relevance and positions, so it is constant between rank crossings.', 'The scores induce order but are not the gains in NDCG.')
      ), viz: 'rank-objectives',
    },
    'retrieval-funnel': lesson(
      'Retrieval sets a hard ceiling on downstream ranking',
      'Inspect which real catalog items survive, then compare a contextual reranker with an oracle.',
      'D is the best judged answer but is absent from the default vector top-3. Neither a better score nor an oracle can rank it without changing the candidate set.',
      'Candidate recall, final ordering, and end-to-end quality diagnose different stages. Evaluate the ceiling against the whole judged corpus.',
      'Use an oracle reranking ceiling to separate retrieval failures from ranker failures, then examine source complementarity and marginal recall per unit cost.',
      ['Increasing K raises the amount of downstream scoring work; no synthetic latency estimate is attached to this toy catalog.', 'A lexical source can recover the exact ANN document missing from vector retrieval; unioning sources is not merely reshuffling scores.', 'Relevant-candidate recall binarizes grade >= 2, while NDCG keeps all graded relevance.'],
      math('Candidate coverage and an oracle ceiling', ['Recall(C)=\\frac{|C\\cap R|}{|R|}', 'Ceiling@K=\\frac{DCG@K(\\operatorname{sort}_{rel}(C))}{IDCG@K(\\text{catalog})}'], 'The denominator never shrinks to the retrieved set. Otherwise a bad retriever could report a misleading perfect score.', [['C', 'Items available to the reranker', 'Default top-3: A, B, F'], ['R', 'Catalog items with grade >= 2', 'A, D, F'], ['K', 'Final display cutoff', '2 served results, independent of retrieval K']]),
      'import math\n\ndef dcg(grades, k=2):\n    return sum((2**r-1)/math.log2(i+2)\n               for i,r in enumerate(grades[:k]))\n\nall_grades = [2, 1, 0, 3, 0, 2]\ncandidate_grades = [2, 1, 2]  # A, B, F; no D\nrecall = sum(r >= 2 for r in candidate_grades) / 3\nceiling = dcg(sorted(candidate_grades, reverse=True)) / dcg(\n    sorted(all_grades, reverse=True))\nprint(recall, ceiling)',
      quiz('An oracle achieves poor end-to-end NDCG on a shortlist. What should you inspect first?', 'Whether high-grade catalog items are missing from the shortlist.', 'Whether the oracle needs a more expressive scoring network.', 'The oracle already sorts surviving items ideally; the candidate set is the remaining constraint.', 'The oracle is a diagnostic upper bound, not a trainable network.')
    ),
    'cold-start': lesson(
      'Cold start: unknown evidence is not negative evidence',
      'Replay exposure records and watch a content prior yield gradually to observed clicks.',
      'Before exposure, an item has no measured click rate. A content-conditioned prior provides a starting estimate and uncertainty, then actual exposures update it.',
      'Without exploration and exposure logging, newly introduced items can remain unknown indefinitely, regardless of ranker sophistication.',
      'Specify how new users get features, how new items get representations, and how the exposure policy creates the evidence required to leave cold start.',
      ['Onboarding supplies content preference for a new user. Without it, the example falls back to a population prior.', 'The Beta-Bernoulli posterior is an interpretable shrinkage baseline for a fixed segment, not a replacement for collaborative learning.', 'Unclicked exposed impressions are noisy relevance evidence: examination, position, and delayed outcomes matter in real logs.'],
      math('A prior with four pseudo-observations', ['p\\sim Beta(mp_0,m(1-p_0))', '\\mathbb{E}[p\\mid c,n]=\\frac{mp_0+c}{m+n}=(1-\\alpha)p_0+\\alpha\\frac{c}{n}', '\\alpha=\\frac{n}{m+n}'], 'For n=0, use the first fraction: the posterior mean is p0 and the empirical rate c/n is undefined.', [['p0', 'Content or population prior mean', 'Population fallback = .25'], ['m', 'Prior strength, not real impressions', '4 pseudo-observations'], ['n, c', 'Observed exposures and clicks', 'D non-exposure record adds neither']]),
      'def posterior(prior, outcomes, strength=4):\n    shown = [y for y in outcomes if y is not None]\n    clicks, n = sum(shown), len(shown)\n    alpha = strength*prior + clicks\n    beta = strength*(1-prior) + n-clicks\n    return alpha/(alpha+beta)\n\nprint(posterior(.25, [None]))  # .25: still unknown\nprint(posterior(.25, [0]))     # .20: exposed, no click\nprint(posterior(.25, [1,1]))   # .50: evidence overcomes prior',
      quiz('A new item was never exposed. What is its observed click rate?', 'Undefined; a prior can supply an estimate but not a measured rate.', 'Zero, because no clicks were logged.', 'The denominator is zero. No click count without exposure cannot establish a negative preference.', 'No opportunity to click is different from an exposed non-click.')
    ),
    'threshold-metrics': lesson(
      'Threshold metrics: count the decisions you actually made',
      'Place a cutoff among twelve labeled impressions and inspect each confusion cell.',
      'The ranking and labels stay fixed. Moving a cutoff changes selected records, and precision and recall follow from their actual counts.',
      'A business decision depends on costs and prevalence, not only a score model or an appealing F1.',
      'Define the decision, label window, exposure population, and error costs before selecting a threshold on validation data.',
      ['Precision is undefined when nothing is selected; recall is undefined with no positive labels.', 'Recall is nondecreasing as the threshold falls on a fixed dataset, but precision need not be monotone.', 'This balanced 12-record sample does not estimate production prevalence. Exposure and delayed-label effects remain outside the example.'],
      math('Confusion counts and decision cost', ['P=\\frac{TP}{TP+FP},\\quad R=\\frac{TP}{TP+FN}', 'F_1=\\frac{2TP}{2TP+FP+FN}', 'Cost=c_{FP}FP+c_{FN}FN'], 'All displayed rates are recomputed from the record IDs in the confusion cells.', [['TP', 'Clicked and selected by cutoff', 'At .59: IDs 1, 3, 4, 6'], ['FP', 'Not clicked but selected', 'At .59: IDs 2, 5'], ['cFN', 'Cost of one false negative', 'Switch from 1 to 5 without changing scores']]),
      'scores = [.94,.88,.81,.74,.68,.59,.52,.43,.34,.26,.17,.08]\nlabels = [1,0,1,1,0,1,0,1,0,0,1,0]\nt = .59\ntp = sum(p >= t and y == 1 for p,y in zip(scores,labels))\nfp = sum(p >= t and y == 0 for p,y in zip(scores,labels))\nfn = sum(p < t and y == 1 for p,y in zip(scores,labels))\nprecision = tp/(tp+fp) if tp+fp else None\nrecall = tp/(tp+fn) if tp+fn else None\nprint(tp, fp, fn, precision, recall)  # 4,2,2,2/3,2/3',
      quiz('Lowering the cutoff admits one true positive and no other record. What happens?', 'Recall increases; precision also increases unless already perfect.', 'Precision must decrease whenever a threshold is lowered.', 'On a fixed dataset, adding a true positive helps both ratios. Precision is not monotonic in threshold.', 'The usual recall-precision tradeoff is a tendency, not a theorem for every adjacent cutoff.')
    ),
    calibration: lesson(
      'Calibration: compare predicted event rates with actual outcomes',
      'Inspect the real labels inside reliability bins and transform logits without changing their order.',
      'A bin with mean predicted click probability .8 should contain approximately 80% clicks over enough comparable examples. That is not the same as classification accuracy.',
      'Good ranking can coexist with bad probability estimates, distorting expected-value decisions and threshold policies.',
      'Separate discrimination, calibration, and decision utility. Fit post-hoc maps on held-out calibration data and verify across segments and time.',
      ['ECE is empirical and depends on binning. Small samples can look calibrated by chance or hide within-bin errors.', 'Temperature scaling divides logits by a positive T and therefore preserves order, but may change operating-point metrics.', 'Brier score is a proper scoring rule sensitive to both calibration and discrimination; it is not a pure calibration metric.'],
      math('Binary-event calibration', ['\\bar p_b=\\frac{1}{n_b}\\sum_{i\\in b}p_i,\\quad \\bar y_b=\\frac{1}{n_b}\\sum_{i\\in b}y_i', 'ECE=\\sum_b\\frac{n_b}{n}|\\bar p_b-\\bar y_b|', 'Brier=\\frac{1}{n}\\sum_i(p_i-y_i)^2,\\quad p_i(T)=\\sigma(\\operatorname{logit}(p_i)/T)'], 'Empty bins receive zero weight and have no estimated rate. Final bin includes p=1.', [['y', 'Actual binary click outcome', '0 or 1, never a synthetic bin accuracy'], ['n_b/n', 'Observed bin frequency', 'Default upper bin contains 5 of 12 examples'], ['T', 'Positive logit temperature', '2 softens probabilities toward .5']]),
      'import math\n\ndef reliability(rows, bins=3):\n    groups = [[] for _ in range(bins)]\n    for p,y in rows:\n        groups[min(bins-1, int(p*bins))].append((p,y))\n    ece = 0.0\n    for group in groups:\n        if not group: continue\n        mean_p = sum(p for p,y in group)/len(group)\n        rate = sum(y for p,y in group)/len(group)\n        ece += len(group)/len(rows)*abs(mean_p-rate)\n    return ece\n# Transform p with sigmoid(logit(p)/T), then re-bin.',
      quiz('A positive temperature transform preserves ordering. What can still change?', 'Calibration, Brier score, and decisions at a fixed probability threshold.', 'Neither probabilities nor probability-based decisions can change.', 'Monotonicity preserves order, not probability values or the number exceeding .7.', 'The threshold is in probability space; temperature changes that space even when ranks stay fixed.')
    ),
    'ranking-metrics': lesson(
      'NDCG: keep judgments fixed and move the results',
      'Reorder four documents and inspect their exact discounted contributions and reciprocal rank.',
      'The same relevance gain is worth less at a lower position. The ideal denominator comes from all judged candidates, not from relabeling the current order.',
      'An ordering metric must reflect the display cutoff and relevance semantics while remaining separate from the training surrogate.',
      'State the query aggregation, judgment construction, gain function, cutoff, and handling of missing labels before comparing offline metrics.',
      ['Exponential gains make grade 3 worth 7, not 3. This convention is a utility choice, not a universal law.', 'The lesson displays one query: its RR becomes MRR only after averaging over queries.', 'Zero ideal gain is reported as NDCG zero here; evaluation libraries may instead exclude such queries. State the convention.'],
      math('A contribution at each rank', ['G(r)=2^r-1,\\quad D(j)=\\frac{1}{\\log_2(j+1)}', 'DCG@K=\\sum_{j=1}^K G(rel_j)D(j)', 'NDCG@K=\\frac{DCG@K}{IDCG@K}'], 'Ranks in the equations start at 1. The implementation uses index+2 for the logarithm.', [['G(3)', 'Gain for the exact answer', '7'], ['D(1), D(2)', 'Position discounts', '1 and .63093'], ['IDCG@3', 'Ideal grades [3,2,1]', '7 + 3/log2(3) + 1/log2(4) = 9.39279']]),
      'import math\n\ndef dcg(grades, k):\n    return sum((2**r-1)/math.log2(i+2)\n               for i,r in enumerate(grades[:k]))\n\ngrades = [0,1,2,3]  # current B,F,A,D order\nk = 3\nideal = dcg(sorted(grades, reverse=True), k)\nndcg = dcg(grades, k)/ideal if ideal else 0.0\nrr = next((1/(i+1) for i,r in enumerate(grades[:k])\n           if r >= 2), 0.0)\nprint(ndcg, rr)  # move D up; labels do not change',
      quiz('You move grade-3 D from rank 4 to rank 1 at K=3. What changes?', 'DCG and NDCG increase; IDCG stays fixed.', 'IDCG increases because D is now in the displayed list.', 'The ideal order already placed D first. Only the actual ranking changed.', 'IDCG comes from the complete judgment set and the cutoff, not the current displayed subset.')
    ),
  };

  const api = { dot, matvec, sigmoid, softplus, softmax, bce, normalize, gain, discount, dcg, rankingMetrics, swapDelta, pairwise, rankMovement, retrieve, funnel, factorStep, factorInitial, objective, confusion, temperature, calibration, posterior, coldRows, initial, reduce, render, content, catalog, queries, impressions, observations, exposureLog, rankItems, QUERY_W, ITEM_W };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window !== 'undefined') {
    window.AtelierLab.createModule({ id: 'ranking-labs', content, initial, render, reduce });
  }
})();
