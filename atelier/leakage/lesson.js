/* Lesson 11: a simulated churn dataset where leaky features look great until deployment. */
(function () {
  const A = window.Atelier;
  const { svg } = A;
  function mulberry32(a) { return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  const sig = (z) => 1 / (1 + Math.exp(-z));

  renderMathInElement(document.body, {
    delimiters: [{ left: '\\[', right: '\\]', display: true }, { left: '\\(', right: '\\)', display: false }],
    throwOnError: false,
  });

  /* ── 2. Timeline ────────────────────────────── */
  (function timeline() {
    const box = document.getElementById('timeline');
    A.responsive(box, (w) => {
      const W = Math.max(280, w), H = 110;
      box.innerHTML = '';
      const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': 'Feature window before prediction time, label window after' }, box);
      const x0 = 10, x1 = W - 10, xt = W * 0.58, y = 60;
      svg('line', { x1: x0, x2: x1, y1: y, y2: y, style: 'stroke: var(--ink-3); stroke-width: 1.5' }, s);
      svg('rect', { x: W * 0.2, y: y - 16, width: xt - W * 0.2, height: 32, rx: 6, style: 'fill: var(--accent-soft); stroke: var(--accent)' }, s);
      svg('rect', { x: xt, y: y - 16, width: W * 0.27, height: 32, rx: 6, style: 'fill: var(--good-soft); stroke: var(--good)' }, s);
      svg('text', { x: (W * 0.2 + xt) / 2, y: y + 5, 'text-anchor': 'middle', 'font-size': 12.5, 'font-weight': 600, style: 'fill: var(--accent)', text: 'features: last 30 days' }, s);
      svg('text', { x: xt + W * 0.135, y: y + 5, 'text-anchor': 'middle', 'font-size': 12.5, 'font-weight': 600, style: 'fill: var(--good)', text: 'label: churn in next 30 days' }, s);
      svg('line', { x1: xt, x2: xt, y1: 14, y2: y + 30, style: 'stroke: var(--ink); stroke-width: 2' }, s);
      svg('text', { x: xt, y: 11, 'text-anchor': 'middle', 'font-size': 12.5, 'font-weight': 700, style: 'fill: var(--ink)', text: 'prediction time t' }, s);
      svg('text', { x: x1, y: y + 44, 'text-anchor': 'end', class: 'tick', text: 'time →' }, s);
    });
  })();

  /* ── 3. Game: validate, then deploy ─────────── */
  (function game() {
    const r = mulberry32(77);
    const g = () => { const u = Math.max(r(), 1e-12), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const rows = [];
    const USERS = 1500, MONTHS = 12;
    for (let u = 0; u < USERS; u++) {
      const a = g(), join = 1 + Math.floor(r() * MONTHS);
      const urows = [];
      let toDate = 0;
      for (let m = join; m <= MONTHS; m++) {
        const sessions = Math.max(0, 8 - 2.2 * a + 1.3 * g());
        const tickets = Math.max(0, Math.round(1 + 0.8 * a + 0.8 * g()));
        const idle = Math.max(0, 5 + 2.5 * a + 2.2 * g());
        const y = r() < sig(-2.3 + 1.1 * a + 0.1 * (m - 6)) ? 1 : 0;
        const closed = y ? (r() < 0.9 ? 1 : 0) : (r() < 0.01 ? 1 : 0);
        toDate += sessions;
        urows.push({ u, m, sessions, tickets, idle, toDate, closed, y });
        if (y) break;
      }
      const lifetime = urows.reduce((s, x) => s + x.sessions, 0);
      urows.forEach((x) => { x.lifetime = lifetime; });
      rows.push(...urows);
    }
    const FEATS = [
      { k: 'sessions', name: 'sessions, last 30 days', src: 'events log, as of t' },
      { k: 'tickets', name: 'support tickets, last 30 days', src: 'support system, as of t' },
      { k: 'idle', name: 'days since last login', src: 'events log, as of t' },
      { k: 'lifetime', name: 'lifetime sessions', src: 'warehouse table, rebuilt nightly' },
      { k: 'closed', name: 'account closed flag', src: 'accounts table, current state' },
      { k: 'toDate', name: 'sessions to date', src: 'events log, as of t' },
    ];
    // At prediction time in month 12: the closed flag isn't known yet, and "lifetime" can only be computed up to now.
    const prodValue = (x, k) => (k === 'closed' ? 0 : k === 'lifetime' ? x.toDate : x[k]);
    const on = new Set(['sessions', 'tickets', 'lifetime', 'closed']);
    let split = 'random', last = null, attempts = [], solved = false;

    function auc(y, s) {
      const idx = s.map((v, i) => i).sort((a, b) => s[a] - s[b]);
      const rank = new Array(s.length);
      for (let i = 0; i < idx.length;) { let j = i; while (j + 1 < idx.length && s[idx[j + 1]] === s[idx[i]]) j++; const rr = (i + j) / 2 + 1; for (let k = i; k <= j; k++) rank[idx[k]] = rr; i = j + 1; }
      let np = 0, sr = 0; y.forEach((v, i) => { if (v) { np++; sr += rank[i]; } });
      const nn = y.length - np; return (sr - (np * (np + 1)) / 2) / (np * nn);
    }
    function train(X, y) {
      const d = X[0].length, mu = new Array(d).fill(0), sd = new Array(d).fill(0);
      X.forEach((x) => x.forEach((v, j) => { mu[j] += v / X.length; }));
      X.forEach((x) => x.forEach((v, j) => { sd[j] += (v - mu[j]) ** 2 / X.length; }));
      sd.forEach((v, j) => { sd[j] = Math.sqrt(v) || 1; });
      const Z = X.map((x) => x.map((v, j) => (v - mu[j]) / sd[j]));
      let w = new Array(d).fill(0), b = 0;
      for (let it = 0; it < 250; it++) {
        const gw = new Array(d).fill(0); let gb = 0;
        Z.forEach((z, i) => { const p = sig(b + z.reduce((s, v, j) => s + v * w[j], 0)); const e = p - y[i]; gb += e; z.forEach((v, j) => { gw[j] += e * v; }); });
        w = w.map((v, j) => v - 0.5 * (gw[j] / Z.length + 1e-3 * v)); b -= 0.5 * gb / Z.length;
      }
      return (x) => b + x.reduce((s, v, j) => s + ((v - mu[j]) / sd[j]) * w[j], 0);
    }
    const hist = rows.filter((x) => x.m <= 11), prod = rows.filter((x) => x.m === 12);
    function validate() {
      const ks = FEATS.filter((f) => on.has(f.k)).map((f) => f.k);
      if (!ks.length) { msg('Pick at least one feature.'); return; }
      let tr, va;
      if (split === 'random') { const rr = mulberry32(5); tr = []; va = []; hist.forEach((x) => (rr() < 0.25 ? va : tr).push(x)); }
      else { tr = hist.filter((x) => x.m <= 9); va = hist.filter((x) => x.m >= 10); }
      const f = train(tr.map((x) => ks.map((k) => x[k])), tr.map((x) => x.y));
      const vAuc = auc(va.map((x) => x.y), va.map((x) => f(ks.map((k) => x[k]))));
      // The deployed model is retrained on all history, like a real launch.
      const fAll = train(hist.map((x) => ks.map((k) => x[k])), hist.map((x) => x.y));
      last = { ks, split, vAuc, fAll, prod: null };
      attempts.push(last);
      msg('');
      render();
    }
    function deploy() {
      if (!last) { msg('Train and validate a model first.'); return; }
      if (last.prod !== null) { msg('That model is already deployed. Change something and validate again.'); return; }
      last.prod = auc(prod.map((x) => x.y), prod.map((x) => last.fAll(last.ks.map((k) => prodValue(x, k)))));
      msg('');
      render();
      if (!solved && last.prod >= 0.70 && Math.abs(last.prod - last.vAuc) <= 0.03) {
        solved = true;
        const deb = document.getElementById('lk-debrief');
        deb.style.display = 'block';
        deb.innerHTML = `<p><b>Solved: validation ${last.vAuc.toFixed(3)}, production ${last.prod.toFixed(3)}.</b> What the traps were:</p><ul>
          <li><b>Account closed flag</b> is target leakage. It's set <em>because</em> the user churned, so validation looks nearly perfect, and at prediction time it's always 0.</li>
          <li><b>Lifetime sessions</b> is temporal leakage. It's computed tonight over all history, so for old rows it includes sessions after the prediction time: survivors look engaged. In production it can only be computed up to now, and the model's learned relationship breaks. "Sessions to date" is the point-in-time version of the same idea.</li>
          <li><b>The random split</b> mixes months, so it can't reveal drift or future-dependent features. The time split mirrors deployment: train on the past, test on what comes next.</li></ul>`;
      }
    }
    function msg(t) { document.getElementById('lb-msg').textContent = t; }
    const chipBox = document.getElementById('feats');
    function chips() {
      chipBox.innerHTML = FEATS.map((f) => `<button class="${on.has(f.k) ? 'on' : ''}" data-k="${f.k}">${f.name}<small>${f.src}</small></button>`).join('');
      chipBox.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => { const k = b.dataset.k; if (on.has(k)) on.delete(k); else on.add(k); chips(); }));
    }
    function render() {
      const name = (k) => FEATS.find((f) => f.k === k).name;
      document.getElementById('lb').innerHTML = attempts.length ? '<tr><th>#</th><th>features</th><th>split</th><th class="num">validation AUC</th><th class="num">production AUC</th></tr>' +
        attempts.map((a, i) => {
          const gap = a.prod === null ? null : a.vAuc - a.prod;
          return `<tr><td>${i + 1}</td><td>${a.ks.map(name).join(', ')}</td><td>${a.split === 'random' ? 'random rows' : 'by time'}</td><td class="num">${a.vAuc.toFixed(3)}</td><td class="num">${a.prod === null ? '<span class="muted">not deployed</span>' : `<b style="color:${Math.abs(gap) <= 0.03 ? 'var(--good)' : 'var(--bad)'}">${a.prod.toFixed(3)}</b>${Math.abs(gap) > 0.03 ? ` <span class="small muted">(${gap > 0 ? '−' : '+'}${Math.abs(gap).toFixed(2)})</span>` : ''}`}</td></tr>`;
        }).join('') : '';
    }
    document.querySelectorAll('#split button').forEach((b) => b.addEventListener('click', () => { split = b.dataset.v; document.querySelectorAll('#split button').forEach((x) => x.classList.toggle('on', x === b)); }));
    document.getElementById('validate').addEventListener('click', validate);
    document.getElementById('deploy').addEventListener('click', deploy);
    chips();
  })();

  const ex = window.LEAK_EXERCISES;
  A.mountExercise(document.getElementById('ex-auc'), ex.auc);
  A.mountExercise(document.getElementById('ex-pit'), ex.pit);
})();
