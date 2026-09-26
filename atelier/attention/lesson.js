/* Lesson 4: attention in two dimensions. One query, four keys and four values, shared by every figure. */
(function () {
  const A = window.Atelier;
  const { svg } = A;
  const TOK = [
    { name: 'trophy', k: [2.0, 0.5], v: [1.6, 1.2] },
    { name: 'suitcase', k: [0.7, 1.9], v: [-1.4, 1.0] },
    { name: "didn't fit", k: [-1.6, 0.6], v: [0.2, -1.3] },
    { name: 'the', k: [-0.6, -1.2], v: [-0.8, -0.6] },
  ];
  const SQ = Math.SQRT2, QMAX = 4, QK_R = 4.3, V_R = 2.1;
  const color = (i) => `var(--w${i})`;
  const halo = 'paint-order: stroke; stroke: var(--surface); stroke-width: 4px; stroke-linejoin: round;';
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
  function attend(q) {
    const s = TOK.map((t) => dot(q, t.k));
    const z = s.map((x) => x / SQ), m = Math.max(...z);
    const e = z.map((x) => Math.exp(x - m)), tot = e.reduce((a, b) => a + b, 0);
    const w = e.map((x) => x / tot);
    const o = [0, 1].map((d) => w.reduce((acc, wi, i) => acc + wi * TOK[i].v[d], 0));
    return { s, w, o };
  }
  function hull(pts) {
    const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lo = [], up = [];
    p.forEach((pt) => { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], pt) <= 0) lo.pop(); lo.push(pt); });
    p.slice().reverse().forEach((pt) => { while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], pt) <= 0) up.pop(); up.push(pt); });
    return lo.slice(0, -1).concat(up.slice(0, -1));
  }

  renderMathInElement(document.body, {
    delimiters: [{ left: '\\[', right: '\\]', display: true }, { left: '\\(', right: '\\)', display: false }],
    throwOnError: false,
  });

  /* A query/key plane + value plane + weight bars, driven by one draggable query. */
  function makeAttention({ qkBox, vBox, barsBox, q0, target, onChange }) {
    let q = q0.slice();
    let S = 320;
    const arrowHead = (s, x1, y1, x2, y2, col, wdt) => {
      const a = Math.atan2(y2 - y1, x2 - x1), h = 7 + wdt;
      svg('path', { d: `M${x2},${y2}L${x2 - h * Math.cos(a - 0.4)},${y2 - h * Math.sin(a - 0.4)}L${x2 - h * Math.cos(a + 0.4)},${y2 - h * Math.sin(a + 0.4)}Z`, style: `fill: ${col}` }, s);
    };
    function grid(s, R, map) {
      for (let v = -Math.floor(R); v <= Math.floor(R); v++) {
        const [x0, y0] = map([v, -R]), [x1, y1] = map([v, R]);
        svg('line', { x1: x0, y1: y0, x2: x1, y2: y1, style: `stroke: var(--rule); stroke-width: ${v === 0 ? 1.4 : 0.6}` }, s);
        const [a0, b0] = map([-R, v]), [a1, b1] = map([R, v]);
        svg('line', { x1: a0, y1: b0, x2: a1, y2: b1, style: `stroke: var(--rule); stroke-width: ${v === 0 ? 1.4 : 0.6}` }, s);
      }
    }
    function render() {
      const { w, o } = attend(q);
      // Query/key plane
      qkBox.innerHTML = '';
      const s1 = svg('svg', { viewBox: `0 0 ${S} ${S}`, width: S, height: S, role: 'img', 'aria-label': 'Query and key vectors' }, qkBox);
      const m1 = (p) => [S / 2 + (p[0] / QK_R) * (S / 2), S / 2 - (p[1] / QK_R) * (S / 2)];
      grid(s1, QK_R, m1);
      const [ox, oy] = m1([0, 0]);
      TOK.forEach((t, i) => {
        const [x, y] = m1(t.k);
        svg('line', { x1: ox, y1: oy, x2: x, y2: y, style: `stroke: ${color(i)}; stroke-width: 2.25` }, s1);
        arrowHead(s1, ox, oy, x, y, color(i), 1);
        const lx = x + (t.k[0] >= 0 ? 6 : -6), ly = y + (t.k[1] >= 0 ? -6 : 14);
        svg('text', { x: lx, y: ly, 'text-anchor': t.k[0] >= 0 ? 'start' : 'end', 'font-size': 12.5, 'font-weight': 600, style: halo + `fill: ${color(i)}`, text: t.name }, s1);
      });
      const [qx, qy] = m1(q);
      svg('line', { x1: ox, y1: oy, x2: qx, y2: qy, style: 'stroke: var(--ink); stroke-width: 3.5' }, s1);
      arrowHead(s1, ox, oy, qx, qy, 'var(--ink)', 2);
      svg('circle', { cx: qx, cy: qy, r: 12, style: 'fill: var(--ink); opacity: 0.18; cursor: grab' }, s1);
      svg('text', { x: qx + (q[0] >= 0 ? 12 : -12), y: qy + (q[1] >= 0 ? -12 : 20), 'text-anchor': q[0] >= 0 ? 'start' : 'end', 'font-size': 13, 'font-weight': 700, style: halo + 'fill: var(--ink)', text: 'query' }, s1);
      // Value plane
      vBox.innerHTML = '';
      const s2 = svg('svg', { viewBox: `0 0 ${S} ${S}`, width: S, height: S, role: 'img', 'aria-label': 'Value vectors and the attention output' }, vBox);
      const m2 = (p) => [S / 2 + (p[0] / V_R) * (S / 2), S / 2 - (p[1] / V_R) * (S / 2)];
      grid(s2, V_R, m2);
      const hp = hull(TOK.map((t) => t.v)).map(m2);
      svg('polygon', { points: hp.map((p) => p.join(',')).join(' '), style: 'fill: var(--surface-2); stroke: var(--ink-3); stroke-dasharray: 4 3; opacity: 0.8' }, s2);
      if (target) {
        const [tx, ty] = m2(target.p);
        svg('circle', { cx: tx, cy: ty, r: (target.r / V_R) * (S / 2), style: 'fill: none; stroke: var(--good); stroke-width: 2.5; stroke-dasharray: 5 3' }, s2);
        svg('circle', { cx: tx, cy: ty, r: 2.5, style: 'fill: var(--good)' }, s2);
      }
      const [px, py] = m2(o);
      TOK.forEach((t, i) => {
        const [x, y] = m2(t.v);
        svg('line', { x1: px, y1: py, x2: x, y2: y, style: `stroke: ${color(i)}; stroke-width: ${0.5 + 6 * w[i]}; opacity: ${0.25 + 0.6 * w[i]}` }, s2);
      });
      TOK.forEach((t, i) => {
        const [x, y] = m2(t.v);
        svg('circle', { cx: x, cy: y, r: 6, style: `fill: ${color(i)}` }, s2);
        svg('text', { x: x + (t.v[0] >= 0 ? 9 : -9), y: y + (t.v[1] >= 0 ? -8 : 16), 'text-anchor': t.v[0] >= 0 ? 'start' : 'end', 'font-size': 12.5, 'font-weight': 600, style: halo + `fill: ${color(i)}`, text: t.name }, s2);
      });
      svg('text', { x: px, y: py + 7, 'text-anchor': 'middle', 'font-size': 22, style: halo + 'fill: var(--ink)', text: '★' }, s2);
      // Bars
      const { s } = attend(q);
      barsBox.innerHTML = '<span class="hd">token</span><span class="hd">weight</span><span class="hd num">q·k</span><span class="hd num">w</span>' +
        TOK.map((t, i) => `<span style="color:${color(i)};font-weight:600">${t.name}</span><div class="bar"><b style="width:${(w[i] * 100).toFixed(1)}%;background:${color(i)}"></b></div><span class="num">${s[i].toFixed(2)}</span><span class="num">${w[i].toFixed(2)}</span>`).join('');
      if (onChange) onChange(q, o, w);
    }
    let dragging = false;
    const fromEvent = (e) => {
      const r = qkBox.getBoundingClientRect();
      const x = ((e.clientX - r.left - S / 2) / (S / 2)) * QK_R, y = -((e.clientY - r.top - S / 2) / (S / 2)) * QK_R;
      const n = Math.hypot(x, y);
      q = n > QMAX ? [(x / n) * QMAX, (y / n) * QMAX] : [x, y];
      render();
    };
    qkBox.style.touchAction = 'none';
    qkBox.addEventListener('pointerdown', (e) => { dragging = true; qkBox.setPointerCapture(e.pointerId); fromEvent(e); });
    qkBox.addEventListener('pointermove', (e) => { if (dragging) fromEvent(e); });
    qkBox.addEventListener('pointerup', () => { dragging = false; });
    qkBox.addEventListener('pointercancel', () => { dragging = false; });
    const rr = A.responsive(qkBox, (wd) => { S = Math.max(240, Math.min(380, wd)); render(); });
    A.onTheme(rr);
    return { setQ(v) { q = v.slice(); render(); }, setTarget(t) { target = t; render(); }, render };
  }

  /* ── 1–3. Explore ───────────────────────────── */
  makeAttention({ qkBox: document.getElementById('qk-plane'), vBox: document.getElementById('v-plane'), barsBox: document.getElementById('wbars'), q0: [1.3, 1.1] });

  /* ── 4. Game: be the query ──────────────────── */
  (function game() {
    const LEVELS = [
      { name: 'Retrieve trophy', target: TOK[0].v, brief: 'Put the output on <b style="color:var(--w0)">trophy</b>\'s value: attend to trophy and almost nothing else.',
        debrief: 'To retrieve one token you need two things: a query pointing near its key, and <em>length</em>. Length sharpens the softmax until the other tokens\' weights fade. Direction alone gives a blend.' },
      { name: 'Retrieve suitcase', target: TOK[1].v, brief: 'Now retrieve <b style="color:var(--w1)">suitcase</b>. Trophy\'s key is about as long and points not far away, so it competes.',
        debrief: 'Pointing straight at suitcase\'s key only just works, even at full length, because trophy\'s key scores well in that direction too. What matters is the <em>difference</em> of scores, \\(q\\cdot(k_{\\text{suitcase}} - k_{\\text{trophy}})\\), so the best query leans away from trophy: about 87° instead of suitcase\'s 70°. Heads can only separate tokens whose keys differ in some direction, which is one reason models use many heads.' },
      { name: 'Blend two', target: [(TOK[0].v[0] + TOK[1].v[0]) / 2, (TOK[0].v[1] + TOK[1].v[1]) / 2], brief: 'Take <b style="color:var(--w0)">trophy</b> and <b style="color:var(--w1)">suitcase</b> half-and-half, and nothing else.',
        debrief: 'Equal weights need equal scores, so the query must be perpendicular to \\(k_{\\text{trophy}} - k_{\\text{suitcase}}\\), and long enough to push the other two tokens out. That is the "soft" in soft attention: a head can read a mixture, which a hard lookup can\'t.' },
    ];
    const R = 0.25;
    const done = [false, false, false];
    let li = 0;
    const levelsBox = document.getElementById('levels');
    const brief = document.getElementById('level-brief');
    const deb = document.getElementById('g-debrief');
    const view = makeAttention({
      qkBox: document.getElementById('g-qk'), vBox: document.getElementById('g-v'), barsBox: document.getElementById('g-wbars'),
      q0: [0.9, 0.9], target: { p: LEVELS[0].target, r: R },
      onChange: (q, o) => {
        const L = LEVELS[li];
        if (!L) return;
        const d = Math.hypot(o[0] - L.target[0], o[1] - L.target[1]);
        brief.innerHTML = `${L.brief} <span class="muted">Distance to target: <b>${d.toFixed(2)}</b> (need &lt; ${R}). Query length ${Math.hypot(q[0], q[1]).toFixed(1)} of ${QMAX}.</span>`;
        if (d < R && !done[li]) {
          done[li] = true;
          deb.style.display = 'block';
          deb.innerHTML = `<p><b>Level ${li + 1} solved.</b> ${L.debrief}</p>` + (li < LEVELS.length - 1 ? '<div class="btns"><button class="btn primary" data-next>Next level</button></div>' : '<p>All three solved. You\'ve now done by hand what \\(W_Q\\) and \\(W_K\\) learn to do for every token and every head.</p>');
          renderMathInElement(deb, { delimiters: [{ left: '\\(', right: '\\)', display: false }], throwOnError: false });
          const nb = deb.querySelector('[data-next]');
          if (nb) nb.addEventListener('click', () => go(li + 1));
          tabs();
        }
      },
    });
    function tabs() {
      levelsBox.innerHTML = LEVELS.map((L, i) => `<button class="btn${i === li ? ' primary' : ''}${done[i] ? ' done' : ''}" data-l="${i}">Level ${i + 1} · ${L.name}</button>`).join('');
      levelsBox.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => go(+b.dataset.l)));
    }
    function go(i) {
      li = i;
      deb.style.display = 'none';
      if (done[i]) { deb.style.display = 'block'; deb.innerHTML = `<p><b>Level ${i + 1} solved.</b> ${LEVELS[i].debrief}</p>`; renderMathInElement(deb, { delimiters: [{ left: '\\(', right: '\\)', display: false }], throwOnError: false }); }
      tabs();
      view.setTarget({ p: LEVELS[i].target, r: R });
      view.setQ([0.9, 0.9]);
    }
    tabs();
  })();

  /* ── 5. Why √d ──────────────────────────────── */
  (function scaleChart() {
    const box = document.getElementById('scale-chart');
    const DS = [2, 4, 8, 16, 32, 64, 128, 256, 512];
    let seed = 7;
    const rng = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; };
    const gauss = () => { const u = Math.max(rng(), 1e-12), v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const maxw = (s) => { const m = Math.max(...s); const e = s.map((x) => Math.exp(x - m)); return 1 / e.reduce((a, b) => a + b, 0); };
    const rows = DS.map((d) => {
      let un = 0, sc = 0;
      for (let t = 0; t < 500; t++) {
        const q = Array.from({ length: d }, gauss);
        const s = Array.from({ length: 8 }, () => { let acc = 0; for (let i = 0; i < d; i++) acc += q[i] * gauss(); return acc; });
        un += maxw(s); sc += maxw(s.map((x) => x / Math.sqrt(d)));
      }
      return { d, un: un / 500, sc: sc / 500 };
    });
    let W = 600;
    function render() {
      box.innerHTML = '';
      const H = 210, m = { l: 40, r: 16, t: 14, b: 36 };
      const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': 'Largest softmax weight versus dimension, with and without scaling' }, box);
      const pw = W - m.l - m.r, ph = H - m.t - m.b;
      const X = (d) => m.l + (Math.log2(d) - 1) / 8 * pw, Y = (v) => m.t + ph - v * ph;
      [0, 0.25, 0.5, 0.75, 1].forEach((v) => { svg('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: 'axis' }, s); svg('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end', class: 'tick', text: v }, s); });
      DS.forEach((d) => svg('text', { x: X(d), y: H - m.b + 17, 'text-anchor': 'middle', class: 'tick', text: d }, s));
      svg('text', { x: m.l + pw / 2, y: H - 3, 'text-anchor': 'middle', class: 'tick', text: 'dimension d (log scale)' }, s);
      const line = (key, col, label, dy) => {
        svg('path', { d: rows.map((r, i) => (i ? 'L' : 'M') + X(r.d).toFixed(1) + ',' + Y(r[key]).toFixed(1)).join(''), fill: 'none', style: `stroke: ${col}; stroke-width: 2.5` }, s);
        rows.forEach((r) => svg('circle', { cx: X(r.d), cy: Y(r[key]), r: 3.5, style: `fill: ${col}` }, s));
        const last = rows[rows.length - 1];
        svg('text', { x: X(last.d) - 4, y: Y(last[key]) + dy, 'text-anchor': 'end', 'font-size': 12.5, 'font-weight': 600, style: halo + `fill: ${col}`, text: label }, s);
      };
      line('un', 'var(--bad)', 'softmax(q·k)', 18);
      line('sc', 'var(--accent)', 'softmax(q·k / √d)', -10);
    }
    A.responsive(box, (w) => { W = Math.max(280, w); render(); });
    A.onTheme(render);
  })();

  /* ── 6. Causal mask grid ────────────────────── */
  (function maskGrid() {
    const words = ['the', 'cat', 'sat', 'on', 'the'];
    const S = [[0.2, 0.9, 0.1, 0.4, 0.3], [0.5, 1.1, 1.4, 0.2, 0.6], [0.3, 1.8, 0.7, 1.2, 0.1], [0.4, 0.6, 1.5, 0.3, 1.1], [1.2, 0.8, 0.5, 1.4, 0.2]];
    const box = document.getElementById('mask-grid');
    let masked = true;
    function draw() {
      box.style.gridTemplateColumns = `3.2em repeat(${words.length}, minmax(0, 1fr))`;
      let h = '<div class="lab"></div>' + words.map((w) => `<div class="lab">${w}</div>`).join('');
      S.forEach((row, i) => {
        const z = row.map((v, j) => (masked && j > i ? -Infinity : v));
        const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), t = e.reduce((a, b) => a + b, 0);
        h += `<div class="lab" style="text-align:right;padding-right:6px">${words[i]}</div>` + e.map((v, j) => {
          const w = v / t;
          if (masked && j > i) return '<div style="background:var(--surface-2);color:var(--ink-3)">–</div>';
          return `<div style="background:color-mix(in srgb, var(--accent) ${Math.round(12 + w * 70)}%, transparent);color:var(--ink)">${w.toFixed(2)}</div>`;
        }).join('');
      });
      box.innerHTML = h;
    }
    document.querySelectorAll('#mask-toggle button').forEach((b) => b.addEventListener('click', () => {
      masked = b.dataset.m === '1';
      document.querySelectorAll('#mask-toggle button').forEach((x) => x.classList.toggle('on', x === b));
      draw();
    }));
    draw();
  })();

  /* ── 6. KV cache calculator ─────────────────── */
  (function kvCalc() {
    const F = [
      { k: 'layers', label: 'layers', v: 32 }, { k: 'kvh', label: 'KV heads', v: 32 }, { k: 'dh', label: 'head dimension', v: 128 },
      { k: 'n', label: 'context tokens', v: 8192 }, { k: 'b', label: 'batch size', v: 1 }, { k: 'bytes', label: 'bytes per value', v: 2 },
    ];
    const box = document.getElementById('kv-calc'), out = document.getElementById('kv-out');
    box.innerHTML = F.map((f) => `<div><label for="kv-${f.k}">${f.label}</label><input id="kv-${f.k}" type="number" min="1" value="${f.v}"></div>`).join('');
    function calc() {
      const g = (k) => Math.max(0, +document.getElementById('kv-' + k).value || 0);
      const perTok = 2 * g('layers') * g('kvh') * g('dh') * g('bytes');
      const total = perTok * g('n') * g('b');
      out.innerHTML = `<div class="readout"><b>${(total / 1e9).toFixed(2)} GB</b><span>total KV cache</span></div><div class="readout"><b>${(perTok / 1024).toFixed(0)} KiB</b><span>per token</span></div>`;
    }
    box.addEventListener('input', calc);
    calc();
  })();

  const ex = window.ATT_EXERCISES;
  A.mountExercise(document.getElementById('ex-attn'), ex.attn);
  A.mountExercise(document.getElementById('ex-decode'), ex.decode);
})();
