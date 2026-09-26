/* Lesson 10: PCA on a 2-D cloud you can rotate an axis through, plus a 20-D scree plot. */
(function () {
  const A = window.Atelier;
  const { svg } = A;
  function mulberry32(a) { return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  const gaussFrom = (r) => () => { const u = Math.max(r(), 1e-12), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  function center(P) { const m = [0, 1].map((d) => P.reduce((s, p) => s + p[d], 0) / P.length); return P.map((p) => [p[0] - m[0], p[1] - m[1]]); }
  function cov(P) { const n = P.length; let a = 0, b = 0, c = 0; P.forEach(([x, y]) => { a += x * x; b += x * y; c += y * y; }); return [a / n, b / n, c / n]; }
  const bestAngle = ([a, b, c]) => 0.5 * Math.atan2(2 * b, a - c);
  const keptVar = ([a, b, c], th) => { const u = [Math.cos(th), Math.sin(th)]; return a * u[0] * u[0] + 2 * b * u[0] * u[1] + c * u[1] * u[1]; };

  renderMathInElement(document.body, {
    delimiters: [{ left: '\\[', right: '\\]', display: true }, { left: '\\(', right: '\\)', display: false }],
    throwOnError: false,
  });

  const base = (() => {
    const g = gaussFrom(mulberry32(31)), th = Math.PI / 6, c = Math.cos(th), s = Math.sin(th);
    return Array.from({ length: 30 }, () => { const x = Math.SQRT2 * g(), y = 0.71 * g(); return [c * x - s * y, s * x + c * y]; });
  })();

  /* ── 2. Game: find the axis ─────────────────── */
  (function game() {
    const LEVELS = [
      { name: 'Clean cloud', P: center(base), brief: 'Rotate the line until it keeps as much of the variance as possible.',
        debrief: (L) => `The best line runs along the long axis of the cloud, at about ${Math.round((L.best * 180) / Math.PI)}°. It keeps ${(100 * L.bestFrac).toFixed(1)}% of the variance: λ₁/(λ₁ + λ₂). Watch the red residuals as you rotate. They're shortest exactly when the blue spread is widest.` },
      { name: 'One outlier', P: center(base.concat([[0, 6.5]])), brief: 'Same cloud plus one unusual point. Find the best axis again.',
        debrief: (L) => `One point out of 31 swung the best axis from about ${Math.round((LEVELS[0].best * 180) / Math.PI)}° to about ${Math.round((((L.best * 180) / Math.PI) + 180) % 180)}°. PCA minimizes <em>squared</em> residuals, so a far point counts quadratically: it's exactly as outlier-sensitive as least squares. Inspect or clip outliers first, or use a robust variant.` },
    ];
    LEVELS.forEach((L) => { L.C = cov(L.P); L.best = bestAngle(L.C); L.tot = L.C[0] + L.C[2]; L.bestFrac = keptVar(L.C, L.best) / L.tot; L.R = Math.max(4.4, ...L.P.map((p) => Math.max(Math.abs(p[0]), Math.abs(p[1])) * 1.1)); });
    const done = [false, false];
    let li = 0, th = (100 * Math.PI) / 180, W = 600;
    const box = document.getElementById('pca-plane'), strip = document.getElementById('pca-strip'), out = document.getElementById('pca-out');
    const brief = document.getElementById('level-brief'), deb = document.getElementById('pca-debrief'), levelsBox = document.getElementById('levels');
    function render() {
      const L = LEVELS[li], S = Math.min(W, 520), R = L.R;
      const m = (p) => [S / 2 + (p[0] / R) * (S / 2), S / 2 - (p[1] / R) * (S / 2)];
      box.innerHTML = '';
      const s = svg('svg', { viewBox: `0 0 ${S} ${S}`, width: S, height: S, role: 'img', 'aria-label': 'Points, a projection line, projections and residuals', style: 'touch-action: none; overflow: hidden; border-radius: 8px' }, box);
      svg('rect', { x: 0, y: 0, width: S, height: S, style: 'fill: var(--surface-2)' }, s);
      const u = [Math.cos(th), Math.sin(th)];
      const [a0, b0] = m([-u[0] * R * 1.5, -u[1] * R * 1.5]), [a1, b1] = m([u[0] * R * 1.5, u[1] * R * 1.5]);
      L.P.forEach((p) => { const z = p[0] * u[0] + p[1] * u[1]; const q = m([z * u[0], z * u[1]]), pp = m(p); svg('line', { x1: pp[0], y1: pp[1], x2: q[0], y2: q[1], style: 'stroke: var(--bad); stroke-width: 1.2; stroke-dasharray: 3 2; opacity: 0.8' }, s); });
      svg('line', { x1: a0, y1: b0, x2: a1, y2: b1, style: 'stroke: var(--ink); stroke-width: 2' }, s);
      L.P.forEach((p) => { const pp = m(p); svg('circle', { cx: pp[0], cy: pp[1], r: 3.6, style: 'fill: var(--ink-3)' }, s); });
      L.P.forEach((p) => { const z = p[0] * u[0] + p[1] * u[1]; const q = m([z * u[0], z * u[1]]); svg('circle', { cx: q[0], cy: q[1], r: 3, style: 'fill: var(--accent)' }, s); });
      // strip of kept coordinates
      strip.innerHTML = '';
      const SH = 44, ss = svg('svg', { viewBox: `0 0 ${S} ${SH}`, width: S, height: SH, role: 'img', 'aria-label': 'Kept coordinates on one axis' }, strip);
      svg('line', { x1: 0, x2: S, y1: SH / 2, y2: SH / 2, class: 'axis' }, ss);
      L.P.forEach((p) => { const z = p[0] * u[0] + p[1] * u[1]; svg('circle', { cx: S / 2 + (z / R) * (S / 2), cy: SH / 2, r: 3, style: 'fill: var(--accent); opacity: 0.6' }, ss); });
      svg('text', { x: 4, y: 12, 'font-size': 11.5, style: 'fill: var(--ink-3)', text: 'kept coordinates z = x · u' }, ss);
      const kept = keptVar(L.C, th) / L.tot;
      const deg = ((((th * 180) / Math.PI) % 180) + 180) % 180;
      out.innerHTML = `<div class="readout"><b>${deg.toFixed(0)}°</b><span>angle</span></div><div class="readout"><b>${(100 * kept).toFixed(1)}%</b><span>variance kept</span></div><div class="readout"><b>${(100 * (1 - kept)).toFixed(1)}%</b><span>lost to residuals</span></div><div class="readout"><b>${(100 * L.bestFrac).toFixed(1)}%</b><span>best possible</span></div>`;
      if (!done[li] && kept >= L.bestFrac - 0.005) {
        done[li] = true; tabs(); showDebrief();
      }
    }
    function showDebrief() {
      deb.style.display = 'block';
      deb.innerHTML = `<p><b>Level ${li + 1} solved.</b> ${LEVELS[li].debrief(LEVELS[li])}</p>` + (li === 0 ? '<div class="btns"><button class="btn primary" data-next>Next level</button></div>' : '');
      const nb = deb.querySelector('[data-next]'); if (nb) nb.addEventListener('click', () => go(1));
    }
    function tabs() {
      levelsBox.innerHTML = LEVELS.map((L, i) => `<button class="btn${i === li ? ' primary' : ''}${done[i] ? ' done' : ''}" data-l="${i}">Level ${i + 1} · ${L.name}</button>`).join('');
      levelsBox.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => go(+b.dataset.l)));
    }
    function go(i) { li = i; th = (100 * Math.PI) / 180; brief.textContent = LEVELS[i].brief; deb.style.display = 'none'; if (done[i]) showDebrief(); tabs(); render(); }
    let dragging = false;
    const fromEvent = (e) => { const r = box.getBoundingClientRect(); const S = Math.min(W, 520); const x = e.clientX - r.left - S / 2, y = -(e.clientY - r.top - S / 2); if (Math.hypot(x, y) > 6) { th = Math.atan2(y, x); render(); } };
    box.addEventListener('pointerdown', (e) => { dragging = true; box.setPointerCapture(e.pointerId); fromEvent(e); });
    box.addEventListener('pointermove', (e) => { if (dragging) fromEvent(e); });
    box.addEventListener('pointerup', () => { dragging = false; });
    box.addEventListener('pointercancel', () => { dragging = false; });
    brief.textContent = LEVELS[0].brief;
    A.responsive(box, (w) => { W = Math.max(280, w); render(); });
    go(0);
  })();

  /* ── 5. Scree plot ──────────────────────────── */
  (function scree() {
    const D = 20, N = 500, K = 3, STR = [3, 2, 1.2];
    const box = document.getElementById('scree-chart'), sl = document.getElementById('noise'), so = document.getElementById('noise-out');
    const r0 = mulberry32(8), g0 = gaussFrom(r0);
    // Random orthonormal loadings (Gram–Schmidt on 3 random vectors)
    const Lm = [];
    for (let k = 0; k < K; k++) {
      let v = Array.from({ length: D }, g0);
      Lm.forEach((q) => { const d = v.reduce((s, x, i) => s + x * q[i], 0); v = v.map((x, i) => x - d * q[i]); });
      const n = Math.hypot(...v); Lm.push(v.map((x) => x / n));
    }
    const F = Array.from({ length: N }, () => STR.map((st) => st * g0()));
    const E = Array.from({ length: N }, () => Array.from({ length: D }, g0));
    function jacobiEig(M) {
      const n = M.length, a = M.map((r) => r.slice());
      for (let sweep = 0; sweep < 30; sweep++) {
        let off = 0;
        for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
        if (off < 1e-18) break;
        for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
          if (Math.abs(a[p][q]) < 1e-15) continue;
          const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
          const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (let k = 0; k < n; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
          for (let k = 0; k < n; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
        }
      }
      return a.map((r, i) => r[i]).sort((x, y) => y - x);
    }
    let W = 600;
    function render() {
      const sig = +sl.value; so.textContent = sig.toFixed(2);
      const X = F.map((f, n) => Array.from({ length: D }, (_, d) => Lm.reduce((s, l, k) => s + f[k] * l[d], 0) + sig * E[n][d]));
      const mu = Array.from({ length: D }, (_, d) => X.reduce((s, x) => s + x[d], 0) / N);
      const C = Array.from({ length: D }, (_, i) => Array.from({ length: D }, (_, j) => X.reduce((s, x) => s + (x[i] - mu[i]) * (x[j] - mu[j]), 0) / N));
      const ev = jacobiEig(C), tot = ev.reduce((a, b) => a + b, 0);
      box.innerHTML = '';
      const H = 210, m = { l: 34, r: 40, t: 12, b: 30 };
      const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': 'Eigenvalues and cumulative variance' }, box);
      const pw = W - m.l - m.r, ph = H - m.t - m.b, bw = pw / D;
      const ymax = Math.max(10, ev[0] * 1.08);
      const Y = (v) => m.t + ph - (v / ymax) * ph, Yc = (f) => m.t + ph - f * ph;
      [0, Math.round(ymax / 2), Math.round(ymax)].forEach((v) => { svg('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: 'axis' }, s); svg('text', { x: m.l - 5, y: Y(v) + 4, 'text-anchor': 'end', class: 'tick', text: v }, s); });
      [0, 0.5, 1].forEach((f) => svg('text', { x: W - m.r + 5, y: Yc(f) + 4, class: 'tick', text: `${Math.round(100 * f)}%` }, s));
      let cum = 0; const cpts = [];
      ev.forEach((e, i) => {
        svg('rect', { x: m.l + i * bw + 2, y: Y(e), width: bw - 4, height: Y(0) - Y(e), rx: 2, style: `fill: ${i < K ? 'var(--accent)' : 'var(--ink-3)'}; opacity: 0.85` }, s);
        cum += e; cpts.push([m.l + i * bw + bw / 2, Yc(cum / tot)]);
        if (i % 2 === 0 || i === D - 1) svg('text', { x: m.l + i * bw + bw / 2, y: H - m.b + 15, 'text-anchor': 'middle', class: 'tick', text: i + 1 }, s);
      });
      svg('path', { d: cpts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(''), fill: 'none', style: 'stroke: var(--w1); stroke-width: 2' }, s);
      svg('text', { x: m.l + pw / 2, y: H - 3, 'text-anchor': 'middle', class: 'tick', text: `component · first 3 keep ${(100 * (ev[0] + ev[1] + ev[2]) / tot).toFixed(0)}% of the variance` }, s);
    }
    sl.addEventListener('input', render);
    A.responsive(box, (w) => { W = Math.max(280, w); render(); });
  })();

  const ex = window.PCA_EXERCISES;
  A.mountExercise(document.getElementById('ex-pca'), ex.pca);
  A.mountExercise(document.getElementById('ex-lr'), ex.lr);
})();
