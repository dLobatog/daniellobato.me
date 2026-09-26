/* Lesson 8: optimizers on a visible quadratic valley L(w) = ½ wᵀ H w. */
(function () {
  const A = window.Atelier;
  const { svg } = A;
  function mulberry32(a) { return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  const W0 = [-3, 1.5], TOL = 0.05;
  const hess = (lam, rot) => { const c = Math.cos(rot), s = Math.sin(rot); return [[lam[0] * c * c + lam[1] * s * s, (lam[0] - lam[1]) * c * s], [(lam[0] - lam[1]) * c * s, lam[0] * s * s + lam[1] * c * c]]; };
  const mul = (H, w) => [H[0][0] * w[0] + H[0][1] * w[1], H[1][0] * w[0] + H[1][1] * w[1]];
  const norm = (w) => Math.hypot(w[0], w[1]);

  /* Run an optimizer; returns the path and when it first got within TOL (or diverged). */
  function run({ lam, rot = 0, opt, N, eta, beta = 0.9, noise = 0, tau = null, seed = 0 }) {
    const H = hess(lam, rot), r = mulberry32(seed + 1);
    const gauss = () => { const u = Math.max(r(), 1e-12), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    let w = W0.slice(), v = [0, 0], m = [0, 0], s = [0, 0];
    const pts = [w.slice()];
    let reached = null, diverged = null;
    for (let t = 1; t <= N; t++) {
      const g0 = mul(H, w), g = noise ? [g0[0] + noise * gauss(), g0[1] + noise * gauss()] : g0;
      const e = tau ? eta / (1 + t / tau) : eta;
      if (opt === 'mom') { v = [beta * v[0] + g[0], beta * v[1] + g[1]]; w = [w[0] - e * v[0], w[1] - e * v[1]]; }
      else if (opt === 'adam') {
        m = [0.9 * m[0] + 0.1 * g[0], 0.9 * m[1] + 0.1 * g[1]]; s = [0.999 * s[0] + 0.001 * g[0] * g[0], 0.999 * s[1] + 0.001 * g[1] * g[1]];
        const bc1 = 1 - Math.pow(0.9, t), bc2 = 1 - Math.pow(0.999, t);
        w = [0, 1].map((i) => w[i] - e * (m[i] / bc1) / (Math.sqrt(s[i] / bc2) + 1e-8));
      } else w = [w[0] - e * g[0], w[1] - e * g[1]];
      pts.push(w.slice());
      if (!Number.isFinite(norm(w)) || norm(w) > 1e3) { diverged = t; break; }
      if (reached === null && norm(w) < TOL) reached = t;
    }
    return { pts, reached, diverged, final: norm(w) };
  }

  const X0 = -3.6, X1 = 3.6, Y0 = -2.2, Y1 = 2.2;
  function valley(box, W, lam, rot, paths) {
    const H = Math.round(W * (Y1 - Y0) / (X1 - X0));
    box.innerHTML = '';
    const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': 'Loss contours and optimizer path', style: 'overflow: hidden; border-radius: 8px' }, box);
    const mx = (x) => ((x - X0) / (X1 - X0)) * W, my = (y) => H - ((y - Y0) / (Y1 - Y0)) * H;
    svg('rect', { x: 0, y: 0, width: W, height: H, rx: 8, style: 'fill: var(--surface-2)' }, s);
    const g = svg('g', { transform: `rotate(${(-rot * 180) / Math.PI} ${mx(0)} ${my(0)})` }, s);
    [0.05, 0.3, 1, 2.5, 5, 10, 20, 40].forEach((c) => {
      svg('ellipse', { cx: mx(0), cy: my(0), rx: (Math.sqrt((2 * c) / lam[0]) / (X1 - X0)) * W, ry: (Math.sqrt((2 * c) / lam[1]) / (Y1 - Y0)) * H, style: 'fill: none; stroke: var(--ink-3); stroke-width: 1; opacity: 0.45' }, g);
    });
    svg('text', { x: mx(0) + 6, y: my(0) - 6, 'font-size': 12, style: 'fill: var(--ink-3)', text: 'min' }, s);
    svg('circle', { cx: mx(0), cy: my(0), r: 3.5, style: 'fill: var(--ink-3)' }, s);
    const clip = (v, lo, hi) => Math.max(lo - 0.3, Math.min(hi + 0.3, v));
    paths.forEach((p) => {
      const d = p.pts.map((w, i) => (i ? 'L' : 'M') + mx(clip(w[0], X0, X1)).toFixed(1) + ',' + my(clip(w[1], Y0, Y1)).toFixed(1)).join('');
      svg('path', { d, fill: 'none', style: `stroke: ${p.color}; stroke-width: ${p.thin ? 1 : 2}; opacity: ${p.thin ? 0.7 : 0.95}` }, s);
      if (!p.thin) p.pts.slice(0, 80).forEach((w) => { if (Math.abs(w[0]) < 4 && Math.abs(w[1]) < 2.6) svg('circle', { cx: mx(w[0]), cy: my(w[1]), r: 2.4, style: `fill: ${p.color}` }, s); });
    });
    svg('circle', { cx: mx(W0[0]), cy: my(W0[1]), r: 6, style: 'fill: var(--ink)' }, s);
  }

  renderMathInElement(document.body, {
    delimiters: [{ left: '\\[', right: '\\]', display: true }, { left: '\\(', right: '\\)', display: false }],
    throwOnError: false,
  });

  /* ── 1. One direction ───────────────────────── */
  (function one() {
    const box = document.getElementById('one-chart'), sl = document.getElementById('etal'), out = document.getElementById('etal-out'), ro = document.getElementById('one-out');
    let W = 600;
    function render() {
      const k = +sl.value, f = 1 - k;
      out.textContent = k.toFixed(2);
      box.innerHTML = '';
      const H = 180, m = { l: 34, r: 10, t: 10, b: 28 };
      const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': 'w over steps' }, box);
      const pw = W - m.l - m.r, ph = H - m.t - m.b;
      const X = (t) => m.l + (t / 20) * pw, Y = (v) => m.t + ph / 2 - (Math.max(-1.6, Math.min(1.6, v)) / 1.6) * (ph / 2);
      [-1, 0, 1].forEach((v) => { svg('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: 'axis', style: v === 0 ? 'stroke: var(--ink-3)' : '' }, s); svg('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end', class: 'tick', text: v }, s); });
      [0, 5, 10, 15, 20].forEach((t) => svg('text', { x: X(t), y: H - m.b + 16, 'text-anchor': 'middle', class: 'tick', text: 'step ' + t }, s));
      const pts = []; let w = 1; for (let t = 0; t <= 20; t++) { pts.push(w); w *= f; }
      const col = Math.abs(f) >= 1 ? 'var(--bad)' : 'var(--accent)';
      svg('path', { d: pts.map((v, t) => (t ? 'L' : 'M') + X(t).toFixed(1) + ',' + Y(v).toFixed(1)).join(''), fill: 'none', style: `stroke: ${col}; stroke-width: 2` }, s);
      pts.forEach((v, t) => svg('circle', { cx: X(t), cy: Y(v), r: 3.5, style: `fill: ${col}` }, s));
      const status = Math.abs(f) >= 1 ? 'diverges: every step overshoots more' : f < 0 ? 'converges, overshooting each step' : f === 0 ? 'lands on the minimum in one step' : 'converges smoothly';
      ro.innerHTML = `<div class="readout"><b>${f.toFixed(2)}</b><span>factor per step, 1 − ηλ</span></div><div class="readout"><b style="font-size:1.05rem;padding-top:5px;color:${Math.abs(f) >= 1 ? 'var(--bad)' : 'var(--ink)'}">${status}</b><span>behavior</span></div>`;
    }
    sl.addEventListener('input', render);
    A.responsive(box, (w) => { W = Math.max(280, w); render(); });
  })();

  /* ── 3. Game: descend the valley ────────────── */
  (function game() {
    const LEVELS = [
      { name: 'The stability limit', lam: [1, 20], opt: 'gd', N: 60, ctl: { eta: [0.005, 0.12, 0.02] },
        brief: 'Plain gradient descent, curvatures 1 and 20. Reach the minimum within <b>60 steps</b>. Your only knob is η.',
        ok: (r) => r.reached !== null && r.reached <= 60,
        debrief: 'The best η sits near 2/(λ<sub>min</sub> + λ<sub>max</sub>) ≈ 0.095. It\'s fast on the flat direction while the steep direction still converges, bouncing across the valley. Push past 0.1 and the steep direction explodes. Even at the best η the flat direction needs about 40 steps: the condition number at work.' },
      { name: 'Momentum', lam: [1, 20], opt: 'mom', N: 20, ctl: { eta: [0.005, 0.2, 0.05], beta: [0, 0.95, 0] },
        brief: 'Same valley, now with momentum \\(v \\leftarrow \\beta v + g,\\; w \\leftarrow w - \\eta v\\). Reach the minimum within <b>20 steps</b>, which gradient descent can\'t do.',
        ok: (r) => r.reached !== null && r.reached <= 20,
        debrief: 'Momentum lets the flat direction build up speed while oscillations across the steep direction cancel out. Tuned (around η ≈ 0.13, β ≈ 0.4) it needs about √κ-many steps instead of κ-many: here about 15 instead of 42. Too much β overshoots along the flat direction instead.' },
      { name: 'Noisy gradients', lam: [1, 4], opt: 'gd', N: 400, noise: 2, ctl: { eta: [0.003, 0.5, 0.03], sched: true },
        brief: 'Minibatch noise: every gradient has random error added. Get the <b>average distance to the minimum over the last 10 of 400 steps, averaged over 20 runs, below 0.11</b>.',
        ok: (r) => r.avg < 0.11,
        debrief: 'A constant rate hits a floor: large η keeps bouncing around the minimum by an amount that grows with η, and small η takes too long to get there. No constant η gets below about 0.15 here. A decaying schedule gets both, with big steps early and small ones late. That\'s why every serious training recipe ends with a decay.' },
    ];
    const done = [false, false, false];
    let li = 0, st = {};
    const box = document.getElementById('valley'), ctlBox = document.getElementById('controls'), out = document.getElementById('valley-out');
    const brief = document.getElementById('level-brief'), deb = document.getElementById('valley-debrief'), levelsBox = document.getElementById('levels');
    let W = 700;
    const logSlider = (id, [lo, hi], val) => { const v = Math.log(val / lo) / Math.log(hi / lo); return `<input id="${id}" type="range" min="0" max="1" step="0.001" value="${v.toFixed(3)}">`; };
    const fromLog = (x, [lo, hi]) => lo * Math.pow(hi / lo, x);
    function controls() {
      const L = LEVELS[li];
      let h = `<div class="ctl"><label for="c-eta">learning rate η${L.ctl.sched ? '₀' : ''}</label>${logSlider('c-eta', L.ctl.eta, st.eta)}<output id="c-eta-out"></output></div>`;
      if (L.ctl.beta) h += `<div class="ctl"><label for="c-beta">momentum β</label><input id="c-beta" type="range" min="0" max="0.95" step="0.01" value="${st.beta}"><output id="c-beta-out"></output></div>`;
      if (L.ctl.sched) h += `<div class="btns"><div class="seg" id="c-sched"><button data-s="0" class="${st.tau ? '' : 'on'}">Constant η</button><button data-s="1" class="${st.tau ? 'on' : ''}">Decay η₀ / (1 + t/10)</button></div></div>`;
      ctlBox.innerHTML = h;
      ctlBox.querySelector('#c-eta').addEventListener('input', (e) => { st.eta = fromLog(+e.target.value, L.ctl.eta); update(); });
      const b = ctlBox.querySelector('#c-beta'); if (b) b.addEventListener('input', (e) => { st.beta = +e.target.value; update(); });
      ctlBox.querySelectorAll('#c-sched button').forEach((x) => x.addEventListener('click', () => { st.tau = x.dataset.s === '1' ? 10 : null; ctlBox.querySelectorAll('#c-sched button').forEach((y) => y.classList.toggle('on', y === x)); update(); }));
    }
    function update() {
      const L = LEVELS[li];
      const eo = ctlBox.querySelector('#c-eta-out'); if (eo) eo.textContent = st.eta.toFixed(st.eta < 0.01 ? 4 : 3);
      const bo = ctlBox.querySelector('#c-beta-out'); if (bo) bo.textContent = st.beta.toFixed(2);
      const cfg = { lam: L.lam, opt: L.opt, N: L.N, eta: st.eta, beta: st.beta, noise: L.noise || 0, tau: st.tau };
      const r = run(cfg);
      if (L.noise) {
        let tot = 0;
        for (let sd = 0; sd < 20; sd++) { const rr = run({ ...cfg, seed: sd }); const last = rr.pts.slice(-10); tot += rr.diverged ? 99 : last.reduce((a, w) => a + norm(w), 0) / last.length; }
        r.avg = tot / 20;
      }
      valley(box, W, L.lam, 0, [{ pts: r.pts, color: 'var(--accent)', thin: !!L.noise }]);
      let status;
      if (r.diverged) status = `<b style="color:var(--bad)">diverged at step ${r.diverged}</b>`;
      else if (L.noise) status = `<b>${r.avg.toFixed(3)}</b>`;
      else status = r.reached ? `<b>${r.reached} steps</b>` : `<b>not within ${L.N}</b>`;
      out.innerHTML = `<div class="readout">${status}<span>${L.noise ? 'avg final distance (20 runs)' : 'to reach the minimum'}</span></div><div class="readout"><b>${L.noise ? '< 0.11' : '≤ ' + L.N}</b><span>goal</span></div>`;
      if (!done[li] && L.ok(r) && !r.diverged) {
        done[li] = true;
        showDebrief();
        tabs();
      }
    }
    function showDebrief() {
      deb.style.display = 'block';
      deb.innerHTML = `<p><b>Level ${li + 1} solved.</b> ${LEVELS[li].debrief}</p>` + (li < LEVELS.length - 1 ? '<div class="btns"><button class="btn primary" data-next>Next level</button></div>' : '');
      const nb = deb.querySelector('[data-next]'); if (nb) nb.addEventListener('click', () => go(li + 1));
    }
    function tabs() {
      levelsBox.innerHTML = LEVELS.map((L, i) => `<button class="btn${i === li ? ' primary' : ''}${done[i] ? ' done' : ''}" data-l="${i}">Level ${i + 1} · ${L.name}</button>`).join('');
      levelsBox.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => go(+b.dataset.l)));
    }
    function go(i) {
      li = i;
      const L = LEVELS[i];
      st = { eta: L.ctl.eta[2], beta: L.ctl.beta ? L.ctl.beta[2] : 0.9, tau: null };
      brief.innerHTML = L.brief;
      renderMathInElement(brief, { delimiters: [{ left: '\\(', right: '\\)', display: false }], throwOnError: false });
      deb.style.display = 'none';
      if (done[i]) showDebrief();
      tabs(); controls(); update();
    }
    li = 0; st = { eta: 0.02, beta: 0, tau: null };
    A.responsive(box, (w) => { W = Math.max(280, w); if (ctlBox.querySelector('#c-eta')) update(); });
    go(0);
  })();

  /* ── 4. Adam vs gradient descent, aligned vs rotated ── */
  (function adamCompare() {
    [['adam-a', 0], ['adam-b', Math.PI / 4]].forEach(([id, rot]) => {
      const box = document.getElementById(id), out = document.getElementById(id + '-out');
      const gd = run({ lam: [1, 20], rot, opt: 'gd', N: 100, eta: 0.095 });
      const ad = run({ lam: [1, 20], rot, opt: 'adam', N: 100, eta: 0.1 });
      const desc = (r) => (r.reached ? `reached in ${r.reached} steps` : `still ${r.final.toFixed(2)} away after 100`);
      out.innerHTML = `<span style="color:var(--accent)">GD</span> ${desc(gd)} · <span style="color:var(--w1)">Adam</span> ${desc(ad)}`;
      A.responsive(box, (w) => valley(box, Math.max(260, w), [1, 20], rot, [{ pts: gd.pts, color: 'var(--accent)' }, { pts: ad.pts, color: 'var(--w1)' }]));
    });
  })();

  A.mountExercise(document.getElementById('ex-opt'), window.OPT_EXERCISES.opt);
})();
