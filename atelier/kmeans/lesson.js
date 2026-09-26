/* Lesson 6: k-means on small 2-D datasets. The game's score is the k-means objective itself. */
(function () {
  const A = window.Atelier;
  const { svg } = A;
  const color = (i) => `var(--w${i})`;
  function mulberry32(a) { return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function blobs(seed, specs) {
    const r = mulberry32(seed);
    const g = () => { const u = Math.max(r(), 1e-12), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const pts = [];
    specs.forEach(([cx, cy, sd, n]) => { for (let i = 0; i < n; i++) pts.push([cx + sd * g(), cy + sd * g()]); });
    return pts;
  }
  const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
  const assign = (P, C) => P.map((p) => { let bi = 0, bd = Infinity; C.forEach((c, i) => { const d = d2(p, c); if (d < bd) { bd = d; bi = i; } }); return bi; });
  const inertia = (P, C) => P.reduce((s, p) => s + Math.min(...C.map((c) => d2(p, c))), 0);
  function update(P, C, lab) {
    const S = C.map(() => [0, 0, 0]);
    P.forEach((p, i) => { S[lab[i]][0] += p[0]; S[lab[i]][1] += p[1]; S[lab[i]][2]++; });
    return C.map((c, i) => (S[i][2] ? [S[i][0] / S[i][2], S[i][1] / S[i][2]] : c.slice()));
  }
  function lloyd(P, C0) {
    let C = C0.map((c) => c.slice());
    for (let it = 0; it < 200; it++) { const nc = update(P, C, assign(P, C)); if (nc.every((c, i) => d2(c, C[i]) < 1e-14)) break; C = nc; }
    return { C, J: inertia(P, C) };
  }
  const randInit = (P, k, r) => { const s = new Set(); while (s.size < k) s.add(Math.floor(r() * P.length)); return [...s].map((i) => P[i].slice()); };
  function ppInit(P, k, r) {
    const C = [P[Math.floor(r() * P.length)].slice()];
    while (C.length < k) {
      const D = P.map((p) => Math.min(...C.map((c) => d2(p, c))));
      let u = r() * D.reduce((a, b) => a + b, 0), i = 0;
      while (i < P.length - 1 && u > D[i]) { u -= D[i]; i++; }
      C.push(P[i].slice());
    }
    return C;
  }

  const DATA_A = blobs(11, [[-2.2, 1.4, 0.6, 20], [2.2, 1.4, 0.6, 20], [0, -1.9, 0.6, 20]]);
  const DATA_B = blobs(23, [[-3, 1.6, 0.5, 15], [-3, -1.6, 0.5, 15], [3, 1.6, 0.5, 15], [3, -1.6, 0.5, 15]]);
  function bestJ(P, k) { const r = mulberry32(5); let b = Infinity; for (let t = 0; t < 200; t++) b = Math.min(b, lloyd(P, ppInit(P, k, r)).J); return b; }
  const BEST = { A: bestJ(DATA_A, 3), B: bestJ(DATA_B, 4) };

  renderMathInElement(document.body, {
    delimiters: [{ left: '\\[', right: '\\]', display: true }, { left: '\\(', right: '\\)', display: false }],
    throwOnError: false,
  });

  /* Clip a convex polygon to the half-plane a·x <= b (Sutherland–Hodgman, one edge). */
  function clip(poly, a, b) {
    const out = [];
    for (let i = 0; i < poly.length; i++) {
      const P = poly[i], Q = poly[(i + 1) % poly.length];
      const fp = a[0] * P[0] + a[1] * P[1] - b, fq = a[0] * Q[0] + a[1] * Q[1] - b;
      if (fp <= 0) out.push(P);
      if ((fp < 0 && fq > 0) || (fp > 0 && fq < 0)) { const t = fp / (fp - fq); out.push([P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])]); }
    }
    return out;
  }

  /* ── 3. Game: place the centers ────────────── */
  (function game() {
    const X0 = -5, X1 = 5, Y0 = -3.6, Y1 = 3.6;
    const LEVELS = [
      { name: 'Three groups', P: DATA_A, best: BEST.A, C0: [[-0.6, 0.4], [0.4, 0.3], [0, -0.4]],
        brief: 'Drag the three centers until your inertia is within 3% of the best. Then press "Run Lloyd" and see if it agrees.',
        debrief: 'You were doing Lloyd\'s two moves by eye: pulling each center toward the middle of the points it owns, then watching points switch owner. On well-separated round blobs, almost any start converges to the same answer.' },
      { name: 'Lloyd\'s trap', P: DATA_B, best: BEST.B, C0: [[-3.2, 1.9], [-2.8, 1.3], [-3, -1.6], [3, 0]],
        brief: 'Four groups, and the centers start badly: two share the top-left group, one sits between the right-hand groups. Press "Run Lloyd" first. Then beat it by hand.',
        debrief: 'Lloyd stops where no <em>single</em> move helps. Pulling one of the two top-left centers toward the right would, for a while, make its points farther away: inertia goes up before it comes down, and Lloyd never goes uphill. You can jump there directly, which is what better initialization and restarts do for the algorithm.' },
    ];
    let li = 0, C = LEVELS[0].C0.map((c) => c.slice()), anim = 0, running = false, lloydStuck = null;
    const done = [false, false];
    const box = document.getElementById('plane');
    const read = document.getElementById('km-read');
    const brief = document.getElementById('level-brief');
    const deb = document.getElementById('km-debrief');
    const levelsBox = document.getElementById('levels');
    let W = 700, H = 500;
    const mx = (x) => ((x - X0) / (X1 - X0)) * W, my = (y) => H - ((y - Y0) / (Y1 - Y0)) * H;
    const ix = (px) => X0 + (px / W) * (X1 - X0), iy = (py) => Y0 + ((H - py) / H) * (Y1 - Y0);
    function render() {
      const L = LEVELS[li], P = L.P;
      box.innerHTML = '';
      const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': 'Points, centers and their regions' }, box);
      svg('rect', { x: 0, y: 0, width: W, height: H, rx: 8, style: 'fill: var(--surface-2)' }, s);
      C.forEach((c, i) => {
        let poly = [[X0, Y0], [X1, Y0], [X1, Y1], [X0, Y1]];
        C.forEach((o, j) => { if (j !== i) poly = clip(poly, [o[0] - c[0], o[1] - c[1]], (o[0] ** 2 + o[1] ** 2 - c[0] ** 2 - c[1] ** 2) / 2); });
        if (poly.length > 2) svg('polygon', { points: poly.map((p) => `${mx(p[0]).toFixed(1)},${my(p[1]).toFixed(1)}`).join(' '), style: `fill: ${color(i)}; opacity: 0.1; stroke: var(--surface); stroke-width: 2` }, s);
      });
      const lab = assign(P, C);
      P.forEach((p, i) => svg('circle', { cx: mx(p[0]), cy: my(p[1]), r: 4.2, style: `fill: ${color(lab[i])}; opacity: 0.9` }, s));
      C.forEach((c, i) => {
        const g = svg('g', { style: 'cursor: grab' }, s);
        svg('circle', { cx: mx(c[0]), cy: my(c[1]), r: 13, style: `fill: var(--surface); stroke: ${color(i)}; stroke-width: 3` }, g);
        svg('text', { x: mx(c[0]), y: my(c[1]) + 5.5, 'text-anchor': 'middle', 'font-size': 16, 'font-weight': 700, style: `fill: ${color(i)}`, text: '✕' }, g);
      });
      const J = inertia(P, C), gap = J / L.best - 1;
      read.innerHTML = `<div class="readout"><b>${J.toFixed(1)}</b><span>your inertia J</span></div><div class="readout"><b>${L.best.toFixed(1)}</b><span>best found (200 restarts)</span></div><div class="readout"><b style="color:${gap <= 0.03 ? 'var(--good)' : 'var(--ink)'}">${gap <= 0.0005 ? 'optimal' : '+' + (100 * gap).toFixed(1) + '%'}</b><span>above best</span></div>` +
        (lloydStuck ? `<div class="readout"><b>${lloydStuck.toFixed(1)}</b><span>where Lloyd stopped</span></div>` : '');
      if (gap <= 0.03 && !done[li] && !running) {
        done[li] = true;
        deb.style.display = 'block';
        deb.innerHTML = `<p><b>Level ${li + 1} solved.</b> ${L.debrief}</p>` + (li < LEVELS.length - 1 ? '<div class="btns"><button class="btn primary" data-next>Next level</button></div>' : '');
        const nb = deb.querySelector('[data-next]');
        if (nb) nb.addEventListener('click', () => go(li + 1));
        tabs();
      }
    }
    function tabs() {
      levelsBox.innerHTML = LEVELS.map((L, i) => `<button class="btn${i === li ? ' primary' : ''}${done[i] ? ' done' : ''}" data-l="${i}">Level ${i + 1} · ${L.name}</button>`).join('');
      levelsBox.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => go(+b.dataset.l)));
    }
    function go(i) {
      cancelAnimationFrame(anim); running = false;
      li = i; C = LEVELS[i].C0.map((c) => c.slice()); lloydStuck = null;
      brief.innerHTML = LEVELS[i].brief;
      deb.style.display = done[i] ? 'block' : 'none';
      if (done[i]) deb.innerHTML = `<p><b>Level ${i + 1} solved.</b> ${LEVELS[i].debrief}</p>`;
      tabs(); render();
    }
    function animateTo(target, then) {
      const from = C.map((c) => c.slice()), t0 = performance.now(), ms = 380;
      (function frame(now) {
        const u = Math.min(1, (now - t0) / ms), e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
        C = from.map((c, i) => [c[0] + (target[i][0] - c[0]) * e, c[1] + (target[i][1] - c[1]) * e]);
        render();
        if (u < 1) anim = requestAnimationFrame(frame); else if (then) then();
      })(t0);
    }
    const stepOnce = (then) => { const P = LEVELS[li].P; const nc = update(P, C, assign(P, C)); const moved = nc.some((c, i) => d2(c, C[i]) > 1e-12); animateTo(nc, () => then && then(moved)); };
    document.getElementById('km-step').addEventListener('click', () => { if (!running) stepOnce(); });
    document.getElementById('km-run').addEventListener('click', () => {
      if (running) return;
      running = true;
      const loop = () => stepOnce((moved) => {
        if (moved) { loop(); return; }
        running = false;
        const J = inertia(LEVELS[li].P, C);
        if (J > LEVELS[li].best * 1.03) {
          lloydStuck = J;
          deb.style.display = 'block';
          deb.innerHTML = `<p><b>Lloyd stopped at J = ${J.toFixed(1)}</b>, ${(J / LEVELS[li].best).toFixed(1)}× the best. Neither move helps: every point already goes to its nearest center, and every center already sits at its points' mean. That's a local minimum. Can you do better by hand?</p>`;
        }
        render();
      });
      loop();
    });
    document.getElementById('km-reset').addEventListener('click', () => go(li));
    let drag = -1;
    box.style.touchAction = 'none';
    const pt = (e) => { const r = box.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    box.addEventListener('pointerdown', (e) => {
      if (running) return;
      const [px, py] = pt(e);
      let bi = -1, bd = 30 * 30;
      C.forEach((c, i) => { const d = (mx(c[0]) - px) ** 2 + (my(c[1]) - py) ** 2; if (d < bd) { bd = d; bi = i; } });
      if (bi >= 0) { drag = bi; box.setPointerCapture(e.pointerId); }
    });
    box.addEventListener('pointermove', (e) => {
      if (drag < 0) return;
      const [px, py] = pt(e);
      C[drag] = [Math.min(X1, Math.max(X0, ix(px))), Math.min(Y1, Math.max(Y0, iy(py)))];
      render();
    });
    const end = () => { drag = -1; };
    box.addEventListener('pointerup', end); box.addEventListener('pointercancel', end);
    const rr = A.responsive(box, (w) => { W = Math.max(290, w); H = Math.round(W * 0.66); render(); });
    A.onTheme(rr);
    go(0);
  })();

  /* ── 4. Initialization comparison ───────────── */
  (function initBars() {
    const box = document.getElementById('init-bars');
    const r = mulberry32(99);
    const runs = { random: [], pp: [] };
    for (let t = 0; t < 300; t++) { runs.random.push(lloyd(DATA_B, randInit(DATA_B, 4, r)).J); runs.pp.push(lloyd(DATA_B, ppInit(DATA_B, 4, r)).J); }
    const best = Math.min(...runs.random, ...runs.pp);
    const ok = (J) => J.filter((x) => x < best * 1.001).length / J.length;
    const avg = (J) => J.reduce((a, b) => a + b, 0) / J.length;
    const row = (label, J) => `<span>${label}</span><div class="bar"><b style="width:${(100 * ok(J)).toFixed(1)}%"></b></div><span class="num">${(100 * ok(J)).toFixed(0)}%</span>`;
    box.innerHTML = '<span class="small muted">starting rule</span><span class="small muted">found the best</span><span></span>' + row('Uniform random', runs.random) + row('k-means++', runs.pp) +
      `<span class="small muted" style="grid-column:1/-1">Average final inertia: ${avg(runs.random).toFixed(1)} (random) vs ${avg(runs.pp).toFixed(1)} (k-means++); best ${best.toFixed(1)}.</span>`;
  })();

  /* ── 5. Scale: raw vs standardized ──────────── */
  (function scaleDemo() {
    const r = mulberry32(7);
    const g = () => { const u = Math.max(r(), 1e-12), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const P = [];
    [2, 8, 15].forEach((m) => { for (let i = 0; i < 30; i++) P.push([r() * 3600, Math.max(0, m + 1.0 * g())]); });
    const mean = [0, 1].map((d) => P.reduce((s, p) => s + p[d], 0) / P.length);
    const sd = [0, 1].map((d) => Math.sqrt(P.reduce((s, p) => s + (p[d] - mean[d]) ** 2, 0) / P.length));
    const Z = P.map((p) => [(p[0] - mean[0]) / sd[0], (p[1] - mean[1]) / sd[1]]);
    const fit = (Q) => { const rr = mulberry32(3); let best = null; for (let t = 0; t < 20; t++) { const f = lloyd(Q, ppInit(Q, 3, rr)); if (!best || f.J < best.J) best = f; } return assign(Q, best.C); };
    const labRaw = fit(P), labStd = fit(Z);
    // Order cluster colors by mean sessions so both panels use the same palette meaningfully.
    const order = (lab) => { const m = [0, 1, 2].map((c) => { const pts = P.filter((_, i) => lab[i] === c); return pts.reduce((s, p) => s + p[0] / 3600 + p[1], 0) / Math.max(1, pts.length); }); const idx = [0, 1, 2].sort((a, b) => m[a] - m[b]); return lab.map((c) => idx.indexOf(c)); };
    const plots = [['raw-plot', order(labRaw)], ['std-plot', order(labStd)]];
    plots.forEach(([id, lab]) => {
      const box = document.getElementById(id);
      const draw = (w) => {
        const W = Math.max(260, w), H = Math.round(W * 0.7), m = { l: 34, r: 10, t: 8, b: 34 };
        box.innerHTML = '';
        const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': 'Users by session length and sessions per week, colored by cluster' }, box);
        const pw = W - m.l - m.r, ph = H - m.t - m.b;
        const X = (x) => m.l + (x / 3600) * pw, Y = (y) => m.t + ph - (y / 20) * ph;
        [0, 5, 10, 15, 20].forEach((v) => { svg('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: 'axis' }, s); svg('text', { x: m.l - 5, y: Y(v) + 4, 'text-anchor': 'end', class: 'tick', text: v }, s); });
        [0, 1800, 3600].forEach((v) => svg('text', { x: X(v), y: H - m.b + 15, 'text-anchor': 'middle', class: 'tick', text: v }, s));
        svg('text', { x: m.l + pw / 2, y: H - 3, 'text-anchor': 'middle', class: 'tick', text: 'avg session length (s) · y: sessions per week' }, s);
        P.forEach((p, i) => svg('circle', { cx: X(p[0]), cy: Y(p[1]), r: 4, style: `fill: ${color(lab[i])}` }, s));
      };
      A.responsive(box, draw);
    });
  })();

  const ex = window.KM_EXERCISES;
  A.mountExercise(document.getElementById('ex-init'), ex.init);
  A.mountExercise(document.getElementById('ex-km'), ex.km);
})();
