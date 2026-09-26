/* Lesson 3: one example through a 2-2-1 network. Forward values, parameters and gradients share one state. */
(function () {
  const A = window.Atelier;
  const { svg } = A;
  const X = [1.0, 0.6], Y = 0, ETA = 0.1;
  const P0 = { w: [[1.0, -0.5], [0.5, -1.0]], b: [0.1, 0.2], v: [1.5, 0.8], c: 0.1 };
  const clone = (P) => JSON.parse(JSON.stringify(P));
  const sig = (z) => 1 / (1 + Math.exp(-z));
  const f2 = (x) => (Math.abs(x) < 5e-4 ? '0' : x.toFixed(2));
  const f3 = (x) => (Math.abs(x) < 5e-5 ? '0' : x.toFixed(3));

  function forward(P, x = X) {
    const z = [0, 1].map((j) => x[0] * P.w[0][j] + x[1] * P.w[1][j] + P.b[j]);
    const h = z.map((v) => Math.max(0, v));
    const zo = h[0] * P.v[0] + h[1] * P.v[1] + P.c;
    const p = sig(zo);
    const L = -(Y * Math.log(p) + (1 - Y) * Math.log(1 - p));
    return { z, h, zo, p, L };
  }
  function backward(P, F, x = X) {
    const d = F.p - Y;
    const gv = F.h.map((h) => d * h);
    const dh = [0, 1].map((j) => d * P.v[j] * (F.z[j] > 0 ? 1 : 0));
    const gw = [0, 1].map((i) => dh.map((dj) => x[i] * dj));
    return { d, gv, gc: d, dh, gw, gb: dh.slice() };
  }
  const PARAMS = [
    { k: 'w00', label: 'w₁₁', desc: 'weight from x₁ to h₁', get: (P) => P.w[0][0], set: (P, v) => { P.w[0][0] = v; }, grad: (G) => G.gw[0][0] },
    { k: 'w10', label: 'w₂₁', desc: 'weight from x₂ to h₁', get: (P) => P.w[1][0], set: (P, v) => { P.w[1][0] = v; }, grad: (G) => G.gw[1][0] },
    { k: 'b0', label: 'b₁', desc: 'bias of h₁', get: (P) => P.b[0], set: (P, v) => { P.b[0] = v; }, grad: (G) => G.gb[0] },
    { k: 'w01', label: 'w₁₂', desc: 'weight from x₁ to h₂', get: (P) => P.w[0][1], set: (P, v) => { P.w[0][1] = v; }, grad: (G) => G.gw[0][1] },
    { k: 'w11', label: 'w₂₂', desc: 'weight from x₂ to h₂', get: (P) => P.w[1][1], set: (P, v) => { P.w[1][1] = v; }, grad: (G) => G.gw[1][1] },
    { k: 'b1', label: 'b₂', desc: 'bias of h₂', get: (P) => P.b[1], set: (P, v) => { P.b[1] = v; }, grad: (G) => G.gb[1] },
    { k: 'v0', label: 'v₁', desc: 'weight from h₁ to the output', get: (P) => P.v[0], set: (P, v) => { P.v[0] = v; }, grad: (G) => G.gv[0] },
    { k: 'v1', label: 'v₂', desc: 'weight from h₂ to the output', get: (P) => P.v[1], set: (P, v) => { P.v[1] = v; }, grad: (G) => G.gv[1] },
    { k: 'c', label: 'c', desc: 'bias of the output', get: (P) => P.c, set: (P, v) => { P.c = v; }, grad: (G) => G.gc },
  ];
  const byKey = Object.fromEntries(PARAMS.map((p) => [p.k, p]));

  renderMathInElement(document.body, {
    delimiters: [{ left: '\\[', right: '\\]', display: true }, { left: '\\(', right: '\\)', display: false }],
    throwOnError: false,
  });

  /* ── Network diagram ─────────────────────────── */
  function explain(key, P, stage) {
    const F = forward(P), G = backward(P, F);
    const sub = ['₁', '₂'];
    if (key === 'x0' || key === 'x1') {
      const i = +key[1];
      return `<b>x${sub[i]} = ${X[i]}</b>: ${i === 0 ? 'the user has clicked this creator before' : 'the item\'s normalized popularity'}. An input: data, not learned. It has no gradient we can use.`;
    }
    if (key === 'h0' || key === 'h1') {
      const j = +key[1];
      let s = `<b>h${sub[j]}</b>: z${sub[j]} = ${f2(P.w[0][j])}·${X[0]} + ${f2(P.w[1][j])}·${X[1]} + ${f2(P.b[j])} = <b>${f2(F.z[j])}</b>, then h${sub[j]} = max(0, ${f2(F.z[j])}) = <b>${f2(F.h[j])}</b>${F.z[j] <= 0 ? ' (the unit is off for this example)' : ''}.`;
      if (stage >= 3) s += `<br>Blame arriving: δ${sub[j]} = δ<sub>out</sub> × v${sub[j]} × ReLU′(z${sub[j]}) = ${f2(G.d)} × ${f2(P.v[j])} × ${F.z[j] > 0 ? 1 : 0} = <b style="color:var(--w1)">${f2(G.dh[j])}</b>.`;
      return s;
    }
    if (key === 'out') {
      let s = `<b>p</b>: z = ${f2(P.v[0])}·${f2(F.h[0])} + ${f2(P.v[1])}·${f2(F.h[1])} + ${f2(P.c)} = <b>${f2(F.zo)}</b>, then p = σ(${f2(F.zo)}) = <b>${f3(F.p)}</b>.`;
      if (stage >= 1) s += `<br>Blame at the logit: δ<sub>out</sub> = ∂L/∂z = p − y = <b style="color:var(--w1)">${f3(G.d)}</b>.`;
      return s;
    }
    if (key === 'loss') return `<b>Loss</b>: L = −ln(1 − p) = −ln(1 − ${f3(F.p)}) = <b>${f3(F.L)}</b>. The label is y = 0 (not clicked), so only the second term of cross-entropy is active.`;
    const p = byKey[key];
    let s = `<b>${p.label} = ${f2(p.get(P))}</b>: a parameter, the ${p.desc}. Training changes it.`;
    const shown = (key[0] === 'v' || key === 'c') ? stage >= 2 : stage >= 4;
    if (shown) {
      const g = p.grad(G);
      let why = '';
      if (key[0] === 'v') why = `δ<sub>out</sub> × h${sub[+key[1]]} = ${f2(G.d)} × ${f2(F.h[+key[1]])}`;
      else if (key === 'c') why = 'δ<sub>out</sub> × 1';
      else if (key[0] === 'w') why = `δ${sub[+key[2]]} × x${sub[+key[1]]} = ${f2(G.dh[+key[2]])} × ${X[+key[1]]}`;
      else why = `δ${sub[+key[1]]} × 1`;
      s += `<br>Gradient: ∂L/∂${p.label} = ${why} = <b style="color:var(--w1)">${f2(g)}</b>.`;
    }
    return s;
  }

  function drawNet(box, getState, info) {
    let W = 700;
    function render() {
      const { P, stage } = getState();
      const F = forward(P), G = backward(P, F);
      box.innerHTML = '';
      const small = W < 520;
      const H = small ? 270 : 300;
      const r = small ? 20 : 26;
      const fs = small ? 11.5 : 13;
      const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': 'Two-input, two-hidden-unit, one-output network with values' }, box);
      const pos = {
        x0: [W * (small ? 0.09 : 0.08), H * 0.28], x1: [W * (small ? 0.09 : 0.08), H * 0.74],
        h0: [W * (small ? 0.47 : 0.42), H * 0.28], h1: [W * (small ? 0.47 : 0.42), H * 0.74],
        out: [W * (small ? 0.86 : 0.74), H * 0.51], loss: [W * 0.925, H * 0.51],
      };
      const halo = 'paint-order: stroke; stroke: var(--surface); stroke-width: 4px; stroke-linejoin: round;';
      const edge = (a, b, key, val, grad, gradShown) => {
        const [x1, y1] = pos[a], [x2, y2] = pos[b];
        const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy);
        const sx = x1 + (dx / len) * r, sy = y1 + (dy / len) * r, ex = x2 - (dx / len) * r, ey = y2 - (dy / len) * r;
        svg('line', { x1: sx, y1: sy, x2: ex, y2: ey, style: 'stroke: var(--ink-3); stroke-width: 1.5' }, s);
        const t = 0.3, lx = sx + (ex - sx) * t, ly = sy + (ey - sy) * t;
        const g = svg('g', { 'data-k': key, style: 'cursor: pointer' }, s);
        svg('text', { x: lx, y: ly - 5, 'text-anchor': 'middle', 'font-size': fs, 'font-family': 'var(--mono)', style: halo + 'fill: var(--ink-2)', text: f2(val) }, g);
        if (gradShown) {
          const t2 = 0.68, gx = sx + (ex - sx) * t2, gy = sy + (ey - sy) * t2;
          const gg = svg('g', { 'data-k': key, style: 'cursor: pointer' }, s);
          svg('text', { x: gx, y: gy + 15, 'text-anchor': 'middle', 'font-size': fs, 'font-weight': 600, style: halo + 'fill: var(--w1)', text: `∇ ${f2(grad)}` }, gg);
        }
      };
      edge('x0', 'h0', 'w00', P.w[0][0], G.gw[0][0], stage >= 4);
      edge('x1', 'h0', 'w10', P.w[1][0], G.gw[1][0], stage >= 4);
      edge('x0', 'h1', 'w01', P.w[0][1], G.gw[0][1], stage >= 4);
      edge('x1', 'h1', 'w11', P.w[1][1], G.gw[1][1], stage >= 4);
      edge('h0', 'out', 'v0', P.v[0], G.gv[0], stage >= 2);
      edge('h1', 'out', 'v1', P.v[1], G.gv[1], stage >= 2);
      if (!small) {
        const [ox, oy] = pos.out, [lx, ly] = pos.loss;
        svg('line', { x1: ox + r, y1: oy, x2: lx - r * 0.8, y2: ly, style: 'stroke: var(--ink-3); stroke-width: 1.5' }, s);
      }
      const node = (key, label, val, sub, dead, delta) => {
        const [x, y] = pos[key];
        const g = svg('g', { 'data-k': key, style: 'cursor: pointer' }, s);
        svg('circle', { cx: x, cy: y, r, style: `fill: var(--accent-soft); stroke: var(--accent); stroke-width: 2; ${dead ? 'stroke-dasharray: 4 3; opacity: 0.6' : ''}` }, g);
        svg('text', { x, y: y + 4.5, 'text-anchor': 'middle', 'font-size': fs + 0.5, 'font-weight': 600, style: 'fill: var(--ink)', text: val }, g);
        svg('text', { x, y: y - r - 7, 'text-anchor': 'middle', 'font-size': fs, style: 'fill: var(--ink-2)', text: label }, g);
        if (sub) svg('text', { x, y: y + r + 15, 'text-anchor': 'middle', 'font-size': fs - 0.5, 'font-family': 'var(--mono)', style: 'fill: var(--ink-3)', text: sub }, g);
        if (delta !== null && delta !== undefined) svg('text', { x, y: y + r + (sub ? 31 : 16), 'text-anchor': 'middle', 'font-size': fs, 'font-weight': 600, style: halo + 'fill: var(--w1)', text: `δ ${f2(delta)}` }, g);
      };
      node('x0', 'x₁', f2(X[0]));
      node('x1', 'x₂', f2(X[1]));
      node('h0', 'h₁', f2(F.h[0]), `b ${f2(P.b[0])}`, F.z[0] <= 0, stage >= 3 ? G.dh[0] : null);
      node('h1', 'h₂', f2(F.h[1]), `b ${f2(P.b[1])}`, F.z[1] <= 0, stage >= 3 ? G.dh[1] : null);
      node('out', 'p', F.p.toFixed(2), `c ${f2(P.c)}`, false, stage >= 1 ? G.d : null);
      if (!small) node('loss', 'loss', F.L.toFixed(2));
      if (small) svg('text', { x: pos.out[0], y: H - 4, 'text-anchor': 'middle', 'font-size': fs, style: 'fill: var(--ink-2)', text: `loss ${F.L.toFixed(3)}` }, s);
    }
    box.addEventListener('click', (e) => {
      const g = e.target.closest('[data-k]');
      if (!g) return;
      const { P, stage } = getState();
      info.innerHTML = explain(g.dataset.k, P, stage);
    });
    const rerender = A.responsive(box, (w) => { W = Math.max(300, w); render(); });
    A.onTheme(rerender);
    return rerender;
  }

  const fwdState = { P: clone(P0), stage: 0 };
  drawNet(document.getElementById('net-fwd'), () => fwdState, document.getElementById('info-fwd'));

  /* ── Backward: the blame game ────────────────── */
  const bwd = { P: clone(P0), stage: 0 };
  const infoB = document.getElementById('info-bwd');
  const redraw = drawNet(document.getElementById('net-bwd'), () => bwd, infoB);
  const QUIZ = [
    { q: 'Start at the output logit z. How much blame arrives there, ∂L/∂z?', opts: ['p − y = 0.90', 'the loss itself, 2.31', 'the sigmoid slope p(1 − p) = 0.09', '−0.90'], right: 0,
      expl: '∂L/∂p = 1/(1 − p) = 10.0 and ∂p/∂z = p(1 − p) = 0.090. Their product is 0.90 = p − y: the same form as lesson 1\'s q − one-hot. It appears as δ under the output node.', key: 'out' },
    { q: 'An output weight\'s gradient is (blame at z) × (the activation it multiplies). What are ∂L/∂v₁ and ∂L/∂v₂?', opts: ['1.26 and 0', '1.35 and 0.72', '0.90 and 0.90', '1.50 and 0.80'], right: 0,
      expl: '∂z/∂v₁ = h₁ = 1.40, so 0.90 × 1.40 = 1.26. h₂ = 0, so v₂ gets no blame for this example: it didn\'t contribute to the mistake.', key: 'v0' },
    { q: 'Blame continues back into the hidden units: δⱼ = (blame at z) × vⱼ × (ReLU slope at zⱼ). How much reaches unit 2?', opts: ['0', '0.72 = 0.90 × 0.8', '0.90', '−0.72'], right: 0,
      expl: 'Unit 2 was off (z₂ = −0.9), so ReLU\'s slope there is 0 and the flow stops. Unit 1 was on (slope 1), so δ₁ = 0.90 × 1.5 = 1.35. The blame passes back <em>through</em> v₁\'s current value, not just to it.', key: 'h1' },
    { q: 'Last hop. The weights into unit 1 get δ₁ × their input. What are ∂L/∂w₁₁ (from x₁) and ∂L/∂w₂₁ (from x₂)?', opts: ['1.35 and 0.81', '1.35 and 1.35', '1.0 and 0.6', '0.90 and 0.54'], right: 0,
      expl: '1.35 × 1.0 and 1.35 × 0.6. The same δ₁ is reused for every weight into unit 1, and δ<sub>out</sub> was reused for everything behind it. That reuse is why backprop costs about one extra pass instead of one pass per parameter. Unit 2\'s weights and bias all get 0.', key: 'w10' },
  ];
  const game = document.getElementById('blame-game');
  function renderGame(chosen) {
    const i = bwd.stage;
    if (i < QUIZ.length) {
      const Q = QUIZ[i];
      game.innerHTML = `<div class="quiz"><div class="small muted">Step ${i + 1} of ${QUIZ.length}</div><div class="q">${Q.q}</div><div class="opts">${Q.opts.map((o, k) => `<button class="opt" data-k="${k}">${o}</button>`).join('')}</div></div>`;
      game.querySelectorAll('.opt').forEach((b) => b.addEventListener('click', () => {
        const k = +b.dataset.k;
        game.querySelectorAll('.opt').forEach((o, j) => { o.disabled = true; if (j === Q.right) o.classList.add('right'); });
        if (k !== Q.right) b.classList.add('wrong');
        bwd.stage = i + 1;
        redraw();
        infoB.innerHTML = explain(Q.key, bwd.P, bwd.stage);
        const ex = document.createElement('div');
        ex.className = 'expl';
        ex.innerHTML = Q.expl;
        const nx = document.createElement('div');
        nx.className = 'btns';
        nx.innerHTML = `<button class="btn primary">${i + 1 < QUIZ.length ? 'Next step' : 'Continue'}</button>`;
        nx.querySelector('button').addEventListener('click', () => renderGame());
        game.querySelector('.quiz').append(ex, nx);
      }));
    } else if (i === QUIZ.length) {
      const F = forward(bwd.P);
      game.innerHTML = `<div class="quiz"><div class="q">Every parameter now has a gradient. Update them all, parameter −= 0.1 × gradient, and rerun the forward pass.</div>
        <div class="btns"><button class="btn primary" data-apply>Apply the update</button></div>
        <p class="small muted" style="margin-top:8px">Loss now: ${F.L.toFixed(3)}</p></div>`;
      game.querySelector('[data-apply]').addEventListener('click', () => {
        const F0 = forward(bwd.P), G0 = backward(bwd.P, F0);
        const P = bwd.P;
        PARAMS.forEach((p) => p.set(P, p.get(P) - ETA * p.grad(G0)));
        bwd.stage = QUIZ.length + 1;
        const F1 = forward(P);
        redraw();
        game.innerHTML = `<div class="quiz"><div class="q">Loss ${F0.L.toFixed(3)} → ${F1.L.toFixed(3)}. The prediction dropped from ${F0.p.toFixed(2)} to ${F1.p.toFixed(2)}.</div>
          <p>Unit 1's weights moved, v₁ and c moved, and every parameter of unit 2 stayed exactly where it was, because it was off for this example. The diagram now shows the updated parameters and the new forward pass, with gradients recomputed at the new point.</p>
          <div class="btns"><button class="btn" data-reset>Start over</button></div></div>`;
        game.querySelector('[data-reset]').addEventListener('click', () => { bwd.P = clone(P0); bwd.stage = 0; redraw(); infoB.textContent = 'Answer the questions below the diagram. You can also click nodes and weights.'; renderGame(); });
      });
    }
  }
  renderGame();

  /* ── Gradient check ──────────────────────────── */
  (function gradCheck() {
    const wrap = document.getElementById('gc-params');
    const out = document.getElementById('gc-out');
    wrap.innerHTML = PARAMS.map((p) => `<button class="btn" data-k="${p.k}">${p.label}</button>`).join('');
    function show(k) {
      wrap.querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.k === k));
      const p = byKey[k], P = clone(P0), h = 1e-4;
      const G = backward(P, forward(P));
      const Pp = clone(P0); p.set(Pp, p.get(P) + h);
      const Pm = clone(P0); p.set(Pm, p.get(P) - h);
      const Lp = forward(Pp).L, Lm = forward(Pm).L;
      const fd = (Lp - Lm) / (2 * h), bp = p.grad(G);
      const rel = Math.abs(bp - fd) / Math.max(Math.abs(bp), Math.abs(fd), 1e-12);
      out.innerHTML = `<b>${p.label}</b> (${p.desc}) = ${f2(p.get(P))}<br>
        Nudge: L(${p.label} + 0.0001) = ${Lp.toFixed(7)}, L(${p.label} − 0.0001) = ${Lm.toFixed(7)}<br>
        Finite difference: (${Lp.toFixed(7)} − ${Lm.toFixed(7)}) / 0.0002 = <b>${fd.toFixed(5)}</b><br>
        Backprop: <b style="color:var(--w1)">${bp.toFixed(5)}</b> · ${Math.abs(bp) < 1e-12 && Math.abs(fd) < 1e-9 ? 'both exactly 0: this parameter has no effect on this example' : `relative error ${rel.toExponential(1)}`}`;
    }
    wrap.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => show(b.dataset.k)));
    show('w10');
  })();

  /* ── Saturation: BCE vs squared error ────────── */
  (function saturation() {
    const box = document.getElementById('sat-chart');
    const slider = document.getElementById('sat-z');
    const out = document.getElementById('sat-out');
    const ro = document.getElementById('sat-readouts');
    let z = 2.2, W = 600;
    const bce = (z) => sig(z);
    const mse = (z) => { const p = sig(z); return 2 * p * p * (1 - p); };
    function render() {
      box.innerHTML = '';
      const H = 210, m = { l: 40, r: 14, t: 14, b: 34 };
      const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': 'Blame at the output logit for cross-entropy and squared error' }, box);
      const pw = W - m.l - m.r, ph = H - m.t - m.b;
      const X = (v) => m.l + ((v + 6) / 14) * pw, Y = (v) => m.t + ph - v * ph;
      [0, 0.25, 0.5, 0.75, 1].forEach((v) => { svg('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: 'axis' }, s); svg('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end', class: 'tick', text: v }, s); });
      [-6, -4, -2, 0, 2, 4, 6, 8].forEach((v) => svg('text', { x: X(v), y: H - m.b + 17, 'text-anchor': 'middle', class: 'tick', text: v }, s));
      svg('text', { x: m.l + pw / 2, y: H - 3, 'text-anchor': 'middle', class: 'tick', text: 'output logit z (y = 0, so the right answer is p → 0)' }, s);
      const path = (f) => { let d = ''; for (let k = 0; k <= 200; k++) { const v = -6 + (14 * k) / 200; d += (k ? 'L' : 'M') + X(v).toFixed(1) + ',' + Y(f(v)).toFixed(1); } return d; };
      svg('path', { d: path(bce), fill: 'none', style: 'stroke: var(--accent); stroke-width: 2.5' }, s);
      svg('path', { d: path(mse), fill: 'none', style: 'stroke: var(--w1); stroke-width: 2.5' }, s);
      svg('text', { x: X(7.8), y: Y(bce(7.8)) + 16, 'text-anchor': 'end', 'font-size': 12.5, style: 'fill: var(--accent)', text: 'cross-entropy: p − y' }, s);
      svg('text', { x: X(7.8), y: Y(mse(7.8)) - 8, 'text-anchor': 'end', 'font-size': 12.5, style: 'fill: var(--w1)', text: 'squared error: 2(p − y)·p(1 − p)' }, s);
      svg('line', { x1: X(z), x2: X(z), y1: m.t, y2: m.t + ph, style: 'stroke: var(--ink-3); stroke-dasharray: 4 3' }, s);
      svg('circle', { cx: X(z), cy: Y(bce(z)), r: 5.5, style: 'fill: var(--accent)' }, s);
      svg('circle', { cx: X(z), cy: Y(mse(z)), r: 5.5, style: 'fill: var(--w1)' }, s);
    }
    function update() {
      slider.value = z; out.textContent = z.toFixed(2);
      const p = sig(z);
      ro.innerHTML = `<div class="readout"><b>${p.toFixed(3)}</b><span>prediction p</span></div>
        <div class="readout"><b style="color:var(--accent)">${bce(z).toFixed(3)}</b><span>blame, cross-entropy</span></div>
        <div class="readout"><b style="color:var(--w1)">${mse(z).toFixed(3)}</b><span>blame, squared error</span></div>`;
      render();
    }
    slider.addEventListener('input', () => { z = +slider.value; update(); });
    A.responsive(box, (w) => { W = Math.max(280, w); update(); });
  })();

  const ex = window.BP_EXERCISES;
  A.mountExercise(document.getElementById('ex-mlp'), ex.mlp);
  A.mountExercise(document.getElementById('ex-num'), ex.num);
})();
