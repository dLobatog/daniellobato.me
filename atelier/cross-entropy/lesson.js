/* Cross-entropy lesson: every figure is computed from the same data (20 occurrences) and one shared model q. */
(function () {
  const A = window.Atelier;
  const { svg, fmt } = A;

  const WORDS = ['mat', 'floor', 'sofa', 'moon'];
  const COUNTS = [10, 5, 4, 1];
  const N = 20;
  const P = COUNTS.map((c) => c / N);
  const H_BITS = -P.reduce((s, p) => s + p * Math.log2(p), 0);
  const Q_MIN = 0.005;
  const Q_MAX = 1 - 3 * Q_MIN;
  const color = (i) => `var(--w${i})`;
  const hex = (i) => A.cssVar(`--w${i}`);
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const tex = (el, s) => katex.render(s, el, { displayMode: true, throwOnError: false });

  const PRESETS = {
    start: [0.5, 0.25, 0.125, 0.125],
    uniform: [0.25, 0.25, 0.25, 0.25],
    A: [0.97, 0.01, 0.01, 0.01],
    B: P.slice(),
    C: [0.70, 0.18, 0.11, 0.01],
    D: [0.25, 0.25, 0.25, 0.25],
  };

  /* ── Shared model q ─────────────────────────── */
  let q = PRESETS.start.slice();
  const subs = [];
  const onQ = (f) => { subs.push(f); f(); };
  const setQ = (nq) => { q = nq.slice(); subs.forEach((f) => f()); };
  /* Move the model smoothly so you can watch which columns change. */
  let tweenId = 0;
  function tweenQ(target, ms = 650) {
    const from = q.slice(), id = ++tweenId, t0 = performance.now();
    (function frame(now) {
      if (id !== tweenId) return;
      const u = Math.min(1, (now - t0) / ms);
      const e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
      setQ(from.map((v, i) => v + (target[i] - v) * e));
      if (u < 1) requestAnimationFrame(frame);
    })(t0);
  }
  const surprise = (v) => -Math.log2(v);
  const crossEntropy = () => P.reduce((s, p, i) => s + p * surprise(q[i]), 0);

  /* Set one word's probability; the others share the rest in proportion, each kept >= Q_MIN. */
  function setWord(i, v) {
    v = clamp(v, Q_MIN, Q_MAX);
    const nq = q.slice();
    const others = [0, 1, 2, 3].filter((j) => j !== i);
    const target = 1 - v;
    const rest = others.reduce((s, j) => s + nq[j], 0);
    others.forEach((j) => { nq[j] = rest > 1e-12 ? (nq[j] * target) / rest : target / 3; });
    for (let k = 0; k < 4; k++) {
      const low = others.filter((j) => nq[j] <= Q_MIN);
      const free = others.filter((j) => nq[j] > Q_MIN);
      low.forEach((j) => { nq[j] = Q_MIN; });
      const need = target - low.length * Q_MIN;
      const s = free.reduce((a, j) => a + nq[j], 0);
      if (s > 0) free.forEach((j) => { nq[j] *= need / s; });
    }
    nq[i] = v;
    setQ(nq);
  }

  const wordLabel = (i) => `<span class="word"><i style="background:${color(i)}"></i>${WORDS[i]}</span>`;
  const halo = 'paint-order: stroke; stroke: var(--surface); stroke-width: 4px; stroke-linejoin: round;';

  /* ── 1. Surprise of one prediction ──────────── */
  function surpriseWidget() {
    const box = document.getElementById('surprise-chart');
    const slider = document.getElementById('surprise-slider');
    const out = document.getElementById('surprise-q');
    const ro = document.getElementById('surprise-readouts');
    const LOWEST = Math.pow(2, -10);
    let qv = 0.02;
    let W = 600;
    const H = 230;
    const m = { l: 46, r: 16, t: 16, b: 44 };
    const formatQ = (v) => (v >= 0.1 ? v.toFixed(2) : v >= 0.01 ? v.toFixed(3) : v.toFixed(4));

    function render() {
      box.innerHTML = '';
      const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': 'Surprise minus log2 q as a function of q' }, box);
      const pw = W - m.l - m.r, ph = H - m.t - m.b;
      const X = (x) => m.l + x * pw;
      const Y = (b) => m.t + ph - (b / 10) * ph;
      [0, 2, 4, 6, 8, 10].forEach((b) => {
        svg('line', { x1: m.l, x2: W - m.r, y1: Y(b), y2: Y(b), class: 'axis' }, s);
        svg('text', { x: m.l - 8, y: Y(b) + 4, 'text-anchor': 'end', class: 'tick', text: b }, s);
      });
      [0, 0.25, 0.5, 0.75, 1].forEach((x) => svg('text', { x: X(x), y: H - m.b + 18, 'text-anchor': 'middle', class: 'tick', text: x }, s));
      svg('text', { x: m.l + pw / 2, y: H - 6, 'text-anchor': 'middle', 'font-size': 13, text: 'q = probability given to the word that occurred' }, s);
      svg('text', { x: 12, y: m.t + ph / 2, 'text-anchor': 'middle', 'font-size': 13, transform: `rotate(-90 12 ${m.t + ph / 2})`, text: 'surprise (bits)' }, s);
      let d = '';
      for (let k = 0; k <= 240; k++) {
        const x = Math.pow(2, -10 + (10 * k) / 240);
        d += (k ? 'L' : 'M') + X(x).toFixed(1) + ',' + Y(-Math.log2(x)).toFixed(1);
      }
      svg('path', { d, fill: 'none', style: 'stroke: var(--accent); stroke-width: 2.5' }, s);
      const b = -Math.log2(qv);
      const cx = X(qv), cy = Y(b);
      svg('line', { x1: cx, x2: cx, y1: Y(0), y2: cy, style: 'stroke: var(--ink-3); stroke-dasharray: 4 3' }, s);
      svg('line', { x1: m.l, x2: cx, y1: cy, y2: cy, style: 'stroke: var(--ink-3); stroke-dasharray: 4 3' }, s);
      svg('circle', { cx, cy, r: 8, style: 'fill: var(--accent); stroke: var(--surface); stroke-width: 2.5; cursor: grab' }, s);
      const right = cx < W - 150;
      svg('text', { x: cx + (right ? 14 : -14), y: cy - 10, 'text-anchor': right ? 'start' : 'end', 'font-size': 14, 'font-weight': 600, style: halo + 'fill: var(--ink)', text: `${fmt(b)} bits` }, s);
    }

    function update() {
      const b = -Math.log2(qv);
      slider.value = Math.log2(qv);
      out.textContent = formatQ(qv);
      const k = Math.round(b);
      const coins = Math.abs(b - k) < 0.03
        ? (k === 0 ? 'no surprise at all' : `${k} fair coin${k > 1 ? 's' : ''} all landing heads`)
        : `between ${Math.floor(b)} and ${Math.ceil(b)} heads in a row`;
      ro.innerHTML = `
        <div class="readout"><b>1 in ${1 / qv >= 100 ? Math.round(1 / qv) : (1 / qv).toFixed(1)}</b><span>odds of that word</span></div>
        <div class="readout"><b>${fmt(b)} bits</b><span>surprise −log₂ q</span></div>
        <div class="readout"><b style="font-size:1.05rem;padding-top:4px">${coins}</b><span>as surprising as</span></div>`;
      render();
    }

    slider.addEventListener('input', () => { qv = Math.pow(2, +slider.value); update(); });
    let dragging = false;
    const fromEvent = (e) => {
      const r = box.getBoundingClientRect();
      const frac = (e.clientX - r.left - m.l) / (W - m.l - m.r);
      qv = clamp(frac, LOWEST, 1);
      update();
    };
    box.style.touchAction = 'pan-y';
    box.addEventListener('pointerdown', (e) => { dragging = true; box.setPointerCapture(e.pointerId); fromEvent(e); });
    box.addEventListener('pointermove', (e) => { if (dragging) fromEvent(e); });
    box.addEventListener('pointerup', () => { dragging = false; });
    box.addEventListener('pointercancel', () => { dragging = false; });
    A.responsive(box, (w) => { W = Math.max(280, w); update(); });
  }

  /* ── 2 & 4. One column per occurrence ───────── */
  function columnsChart(box, withFloor) {
    let W = 700;
    const readout = document.createElement('div');
    readout.className = 'readouts';
    readout.style.margin = '0 0 6px';
    box.before(readout);
    function render() {
      box.innerHTML = '';
      const small = W < 520;
      const H = small ? 250 : 290;
      const m = { l: 40, r: 52, t: 24, b: 50 };
      const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': 'Surprise of each of the 20 occurrences under the model' }, box);
      const defs = svg('defs', {}, s);
      const pat = svg('pattern', { id: `hatch-${box.id}`, width: 5, height: 5, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
      svg('rect', { width: 5, height: 5, style: 'fill: var(--surface); opacity: 0.55' }, pat);
      svg('line', { x1: 0, y1: 0, x2: 0, y2: 5, style: 'stroke: var(--ink); stroke-width: 2; opacity: 0.7' }, pat);

      const pw = W - m.l - m.r, ph = H - m.t - m.b;
      const YMAX = 8;
      const Y = (b) => m.t + ph - (Math.min(b, YMAX) / YMAX) * ph;
      [0, 2, 4, 6, 8].forEach((b) => {
        svg('line', { x1: m.l, x2: W - m.r, y1: Y(b), y2: Y(b), class: 'axis' }, s);
        svg('text', { x: m.l - 8, y: Y(b) + 4, 'text-anchor': 'end', class: 'tick', text: b }, s);
      });
      svg('text', { x: 4, y: m.t - 10, class: 'tick', text: 'surprise (bits)' }, s);

      const groupGap = small ? 8 : 16, inGap = small ? 1.5 : 3;
      const colW = (pw - 3 * groupGap - (N - 4) * inGap) / N;
      let x = m.l;
      WORDS.forEach((w, i) => {
        const gx0 = x;
        const sv = surprise(q[i]);
        const fv = -Math.log2(P[i]);
        for (let c = 0; c < COUNTS[i]; c++) {
          svg('rect', { x, y: Y(sv), width: colW, height: Y(0) - Y(sv), rx: 1.5, style: `fill: ${color(i)}; opacity: 0.9` }, s);
          if (withFloor && sv > fv) {
            svg('rect', { x, y: Y(sv), width: colW, height: Y(fv) - Y(sv), style: `fill: url(#hatch-${box.id})` }, s);
          } else if (withFloor && sv < fv) {
            svg('rect', { x: x + 0.75, y: Y(fv), width: Math.max(0, colW - 1.5), height: Y(sv) - Y(fv), style: 'fill: none; stroke: var(--ink-3); stroke-width: 1.5; stroke-dasharray: 3 2' }, s);
          }
          x += colW + (c < COUNTS[i] - 1 ? inGap : 0);
        }
        const gx1 = x;
        if (withFloor) {
          svg('line', { x1: gx0 - 3, x2: gx1 + 3, y1: Y(fv), y2: Y(fv), style: 'stroke: var(--ink); stroke-width: 2.5; stroke-dasharray: 6 3' }, s);
        }
        // The one-column group is too narrow for a centered label: start it at the column and let it run into the right margin.
        const narrow = i === WORDS.length - 1 && gx1 - gx0 < 44;
        const cx = narrow ? gx0 : (gx0 + gx1) / 2;
        const anchor = narrow ? 'start' : 'middle';
        svg('text', { x: cx, y: Y(0) + 19, 'text-anchor': anchor, 'font-size': 14, 'font-weight': 600, style: `fill: ${color(i)}`, text: w }, s);
        svg('text', { x: cx, y: Y(0) + 36, 'text-anchor': anchor, class: 'tick', text: `×${COUNTS[i]}` }, s);
        x += groupGap;
      });

      const ce = crossEntropy();
      svg('line', { x1: m.l, x2: m.l + pw, y1: Y(ce), y2: Y(ce), style: 'stroke: var(--ink); stroke-width: 1.75' }, s);
      let yCe = Y(ce) + 4, yH = Y(H_BITS) + 4;
      if (withFloor) {
        svg('line', { x1: m.l, x2: m.l + pw, y1: Y(H_BITS), y2: Y(H_BITS), style: 'stroke: var(--ink); stroke-width: 1.5; stroke-dasharray: 2 3' }, s);
        if (yH - yCe < 15) { const mid = (yH + yCe) / 2; yCe = mid - 7.5; yH = mid + 7.5; }
        svg('text', { x: m.l + pw + 6, y: yH, 'font-size': 12.5, style: 'fill: var(--ink-2)', text: 'H(p)' }, s);
      }
      svg('text', { x: m.l + pw + 6, y: yCe, 'font-size': 12.5, 'font-weight': 600, style: 'fill: var(--ink)', text: 'H(p,q)' }, s);
      readout.innerHTML = withFloor
        ? `<div class="readout"><b>${fmt(ce)}</b><span>H(p,q), bits</span></div><div class="readout"><b>= ${fmt(H_BITS)}</b><span>floor H(p)</span></div><div class="readout"><b>+ ${fmt(ce - H_BITS)}</b><span>gap KL(p‖q)</span></div>`
        : `<div class="readout"><b>${fmt(ce)} bits</b><span>average surprise H(p,q)</span></div>`;
    }
    const rerender = A.responsive(box, (w) => { W = Math.max(280, w); render(); });
    onQ(() => rerender());
    A.onTheme(() => rerender());
  }

  /* Table of per-word sliders bound to the shared q. */
  function qTable(table, kind) {
    const head = kind === 'ce'
      ? '<tr><th>word</th><th class="hide-sm">seen</th><th>model q</th><th class="num"></th><th class="num">−log₂ q</th></tr>'
      : '<tr><th>word</th><th>model q</th><th class="num"></th><th class="num hide-sm">p</th><th class="num">p·log₂(p/q)</th></tr>';
    table.innerHTML = head + WORDS.map((w, i) => `
      <tr>
        <td>${wordLabel(i)}</td>
        ${kind === 'ce' ? `<td class="hide-sm muted">×${COUNTS[i]}</td>` : ''}
        <td class="slider"><input type="range" min="${Q_MIN}" max="${Q_MAX}" step="0.005" aria-label="model probability for ${w}" data-i="${i}"></td>
        <td class="num" data-q></td>
        ${kind === 'kl' ? `<td class="num hide-sm">${P[i].toFixed(2)}</td>` : ''}
        <td class="num" data-v></td>
      </tr>`).join('');
    const rows = [...table.querySelectorAll('tr')].slice(1);
    rows.forEach((r, i) => r.querySelector('input').addEventListener('input', (e) => { tweenId++; setWord(i, +e.target.value); }));
    onQ(() => rows.forEach((r, i) => {
      const inp = r.querySelector('input');
      if (document.activeElement !== inp) inp.value = q[i];
      r.querySelector('[data-q]').textContent = q[i].toFixed(3);
      const v = kind === 'ce' ? surprise(q[i]) : P[i] * Math.log2(P[i] / q[i]);
      r.querySelector('[data-v]').textContent = (kind === 'kl' && v > 0.0049 ? '+' : '') + fmt(Math.abs(v) < 0.005 ? 0 : v);
    }));
  }

  function ceEquation() {
    const el = document.getElementById('ce-eq');
    const draw = () => {
      const c = (i, t) => `\\textcolor{${hex(i)}}{${t}}`;
      const sv = q.map(surprise);
      const narrow = el.clientWidth < 460;
      const join = (xs) => (narrow ? xs.slice(0, 2).join(' + ') + ' \\\\ &\\quad + ' + xs.slice(2).join(' + ') : xs.join(' + '));
      const l1 = join(WORDS.map((w, i) => `${COUNTS[i]}\\cdot ${c(i, fmt(sv[i]))}`));
      const l2 = join(WORDS.map((w, i) => `${P[i].toFixed(2)}\\cdot ${c(i, fmt(sv[i]))}`));
      const pre = narrow ? '\\tfrac{1}{20}\\,\\big(' : '\\tfrac{1}{20}\\big(';
      tex(el, `\\begin{aligned} \\text{average} &= ${pre}${l1}\\big) \\\\ &= ${l2} \\\\ &= ${fmt(crossEntropy())}\\ \\text{bits} \\end{aligned}`);
    };
    onQ(draw); A.onTheme(draw); A.responsive(el, draw);
  }

  function klEquation() {
    const el = document.getElementById('kl-eq');
    const draw = () => {
      const c = (i, t) => `\\textcolor{${hex(i)}}{${t}}`;
      const terms = P.map((p, i) => p * Math.log2(p / q[i]));
      const parts = terms.map((t, i) => {
        const v = Math.abs(t) < 0.005 ? 0 : t;
        const sign = v < 0 ? '-' : '+';
        return (i === 0 ? (v < 0 ? '-' : '') : ` ${sign} `) + c(i, fmt(Math.abs(v)));
      }).join('');
      const ce = crossEntropy();
      tex(el, `\\begin{aligned} \\mathrm{KL}(p\\|q) &= \\textstyle\\sum_x p(x)\\log_2\\frac{p(x)}{q(x)} \\\\ &= ${parts} \\\\ &= ${fmt(ce - H_BITS)}\\ \\text{bits} \\;=\\; \\underbrace{${fmt(ce)}}_{H(p,q)} - \\underbrace{${fmt(H_BITS)}}_{H(p)} \\end{aligned}`);
    };
    onQ(draw); A.onTheme(draw);
  }

  /* ── 5. Gradient descent on logits ──────────── */
  function gradientWidget() {
    const table = document.getElementById('gd-table');
    const box = document.getElementById('gd-chart');
    const ro = document.createElement('div');
    ro.className = 'readouts';
    box.after(ro);
    const H_NATS = H_BITS * Math.LN2;
    let z, hist, W = 600;
    const softmax = (v) => { const mx = Math.max(...v); const e = v.map((x) => Math.exp(x - mx)); const s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); };
    const loss = (qq) => -P.reduce((s, p, i) => s + p * Math.log(qq[i]), 0);
    function reset() { z = [0, 0, 0, 0]; hist = [loss(softmax(z))]; draw(); }
    function step(n) { for (let k = 0; k < n; k++) { const qq = softmax(z); z = z.map((v, i) => v - 1.0 * (qq[i] - P[i])); hist.push(loss(softmax(z))); } draw(); }

    table.innerHTML = '<tr><th>word</th><th class="num">logit z</th><th class="num">q</th><th class="num">p</th><th class="num">∂L/∂z = q − p</th></tr>' +
      WORDS.map((w, i) => `<tr><td>${wordLabel(i)}</td><td class="num" data-z></td><td class="num" data-q></td><td class="num">${P[i].toFixed(2)}</td><td class="num"><span data-g></span><span class="gradbar hide-sm"><b></b></span></td></tr>`).join('');
    const rows = [...table.querySelectorAll('tr')].slice(1);

    function chart() {
      box.innerHTML = '';
      const H = 170, m = { l: 46, r: 14, t: 14, b: 34 };
      const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': 'Loss per step' }, box);
      const pw = W - m.l - m.r, ph = H - m.t - m.b;
      const n = Math.max(20, hist.length - 1);
      const X = (k) => m.l + (k / n) * pw;
      const Y = (v) => m.t + ph - ((v - 1.1) / 0.3) * ph;
      [1.15, 1.25, 1.35].forEach((v) => {
        svg('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: 'axis' }, s);
        svg('text', { x: m.l - 8, y: Y(v) + 4, 'text-anchor': 'end', class: 'tick', text: v.toFixed(2) }, s);
      });
      for (let k = 0; k <= n; k += n > 40 ? 20 : 10) svg('text', { x: X(k), y: H - m.b + 16, 'text-anchor': 'middle', class: 'tick', text: k }, s);
      svg('text', { x: m.l + pw / 2, y: H - 3, 'text-anchor': 'middle', class: 'tick', text: 'step' }, s);
      svg('text', { x: 4, y: m.t - 2, class: 'tick', text: 'loss (nats)' }, s);
      svg('line', { x1: m.l, x2: W - m.r, y1: Y(H_NATS), y2: Y(H_NATS), style: 'stroke: var(--ink); stroke-dasharray: 2 3; stroke-width: 1.5' }, s);
      svg('text', { x: W - m.r - 4, y: Y(H_NATS) + 16, 'text-anchor': 'end', 'font-size': 13, style: halo + 'fill: var(--ink-2)', text: `floor H(p) = ${H_NATS.toFixed(3)}` }, s);
      const d = hist.map((v, k) => (k ? 'L' : 'M') + X(k).toFixed(1) + ',' + Y(v).toFixed(1)).join('');
      svg('path', { d, fill: 'none', style: 'stroke: var(--accent); stroke-width: 2' }, s);
      hist.forEach((v, k) => svg('circle', { cx: X(k), cy: Y(v), r: 3, style: 'fill: var(--accent)' }, s));
    }

    function draw() {
      const qq = softmax(z);
      rows.forEach((r, i) => {
        const g = qq[i] - P[i];
        r.querySelector('[data-z]').textContent = (z[i] >= 0 ? '+' : '') + z[i].toFixed(2);
        r.querySelector('[data-q]').textContent = qq[i].toFixed(3);
        r.querySelector('[data-g]').textContent = (g >= 0 ? '+' : '') + g.toFixed(3);
        const b = r.querySelector('.gradbar b');
        const w = Math.min(50, (Math.abs(g) / 0.3) * 50);
        b.style.width = w + '%';
        b.style.left = g >= 0 ? '50%' : 50 - w + '%';
      });
      const L = hist[hist.length - 1];
      ro.innerHTML = `
        <div class="readout"><b>${hist.length - 1}</b><span>steps</span></div>
        <div class="readout"><b>${L.toFixed(3)}</b><span>loss H(p,q), nats</span></div>
        <div class="readout"><b>${(L - H_NATS).toFixed(4)}</b><span>KL(p‖q), nats</span></div>`;
      chart();
    }

    document.getElementById('gd-step').addEventListener('click', () => step(1));
    document.getElementById('gd-step10').addEventListener('click', () => step(10));
    document.getElementById('gd-reset').addEventListener('click', reset);
    z = [0, 0, 0, 0]; hist = [loss(softmax(z))];
    A.responsive(box, (w) => { W = Math.max(280, w); draw(); });
    A.onTheme(() => chart());
  }

  /* ── 6. Forward vs reverse KL with a one-bump model ── */
  function directionWidget() {
    const box = document.getElementById('kl-dir-chart');
    const pbox = document.getElementById('kl-dir-penalty');
    const mu = document.getElementById('mu'), sigma = document.getElementById('sigma');
    const muOut = document.getElementById('mu-out'), sgOut = document.getElementById('sigma-out');
    const btns = [document.getElementById('fit-forward'), document.getElementById('fit-reverse')];
    const S_MIN = 4, S_MAX = 45;
    sigma.min = S_MIN;
    const st = { mu: 30, sg: 12 };
    const LN2PI = Math.log(2 * Math.PI);
    const logN = (x, m, s) => -0.5 * ((x - m) / s) ** 2 - Math.log(s) - 0.5 * LN2PI;
    const lse = (a, b) => { const mx = Math.max(a, b); return mx + Math.log(Math.exp(a - mx) + Math.exp(b - mx)); };
    const logP = (x) => lse(Math.log(0.5) + logN(x, 20, 6), Math.log(0.5) + logN(x, 80, 10));
    const DX = 0.5;
    const GRID = []; for (let x = -60; x <= 170; x += DX) GRID.push(x);
    const LP = GRID.map(logP);
    const fwd = (m, s) => { let t = 0; GRID.forEach((x, k) => { t += Math.exp(LP[k]) * (LP[k] - logN(x, m, s)); }); return t * DX; };
    const rev = (m, s) => { let t = 0; GRID.forEach((x, k) => { const lq = logN(x, m, s); t += Math.exp(lq) * (lq - LP[k]); }); return t * DX; };
    let W = 700;
    const XMIN = -10, XMAX = 120;

    function render() {
      box.innerHTML = ''; pbox.innerHTML = '';
      const m = { l: 14, r: 14, t: 10, b: 34 };
      const H = 200;
      const pw = W - m.l - m.r;
      const X = (x) => m.l + ((x - XMIN) / (XMAX - XMIN)) * pw;
      const xs = []; for (let x = XMIN; x <= XMAX; x += 0.5) xs.push(x);
      const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': 'Target p and model q' }, box);
      const ph = H - m.t - m.b, YM = 0.075;
      const Y = (v) => m.t + ph - (Math.min(v, YM) / YM) * ph;
      svg('line', { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), class: 'axis' }, s);
      [0, 20, 40, 60, 80, 100, 120].forEach((x) => svg('text', { x: X(x), y: H - m.b + 17, 'text-anchor': 'middle', class: 'tick', text: x }, s));
      svg('text', { x: m.l + pw / 2, y: H - 3, 'text-anchor': 'middle', class: 'tick', text: 'response length (tokens)' }, s);
      const area = (f) => 'M' + X(XMIN) + ',' + Y(0) + xs.map((x) => 'L' + X(x).toFixed(1) + ',' + Y(f(x)).toFixed(1)).join('') + 'L' + X(XMAX) + ',' + Y(0) + 'Z';
      svg('path', { d: area((x) => Math.exp(logP(x))), style: 'fill: var(--ink-3); opacity: 0.35' }, s);
      const qArea = area((x) => Math.exp(logN(x, st.mu, st.sg)));
      svg('path', { d: qArea, style: 'fill: var(--accent); opacity: 0.16' }, s);
      svg('path', { d: qArea, style: 'fill: none; stroke: var(--accent); stroke-width: 2.5' }, s);

      const F = fwd(st.mu, st.sg), R = rev(st.mu, st.sg);
      const panels = [
        { name: `forward KL(p‖q) = ${fmt(F)} nats`, sub: 'cost where p has mass that q lacks', f: (x) => Math.exp(logP(x)) * (logP(x) - logN(x, st.mu, st.sg)) },
        { name: `reverse KL(q‖p) = ${fmt(R)} nats`, sub: 'cost where q has mass that p lacks', f: (x) => { const lq = logN(x, st.mu, st.sg); return Math.exp(lq) * (lq - logP(x)); } },
      ];
      const PH = 96;
      const ps = svg('svg', { viewBox: `0 0 ${W} ${PH * 2 + 20}`, width: W, height: PH * 2 + 20, role: 'img', 'aria-label': 'Where each KL collects its cost' }, pbox);
      panels.forEach((pn, k) => {
        const top = k * (PH + 20) + 6;
        const vals = xs.map(pn.f);
        const mx = Math.max(0.004, ...vals.map(Math.abs));
        const y0 = top + PH * 0.72;
        const Yp = (v) => y0 - (v / mx) * (PH * 0.62);
        svg('line', { x1: m.l, x2: W - m.r, y1: y0, y2: y0, class: 'axis' }, ps);
        const pos = 'M' + X(XMIN) + ',' + y0 + xs.map((x, j) => 'L' + X(x).toFixed(1) + ',' + Yp(Math.max(0, vals[j])).toFixed(1)).join('') + 'L' + X(XMAX) + ',' + y0 + 'Z';
        const neg = 'M' + X(XMIN) + ',' + y0 + xs.map((x, j) => 'L' + X(x).toFixed(1) + ',' + Yp(Math.min(0, vals[j])).toFixed(1)).join('') + 'L' + X(XMAX) + ',' + y0 + 'Z';
        svg('path', { d: pos, style: 'fill: var(--bad); opacity: 0.55' }, ps);
        svg('path', { d: neg, style: 'fill: var(--good); opacity: 0.45' }, ps);
        svg('text', { x: m.l + 2, y: top + 12, 'font-size': 13.5, 'font-weight': 600, style: halo + 'fill: var(--ink)', text: pn.name }, ps);
        svg('text', { x: m.l + 2, y: top + 28, 'font-size': 12.5, style: halo + 'fill: var(--ink-3)', text: pn.sub }, ps);
      });
    }

    function sync() {
      mu.value = st.mu; sigma.value = st.sg;
      muOut.textContent = st.mu.toFixed(1); sgOut.textContent = st.sg.toFixed(1);
      render();
    }

    function minimize(f) {
      let m = st.mu, s = st.sg, best = f(m, s);
      let dm = 6, ds = 3;
      const path = [];
      for (let it = 0; it < 600 && (dm > 0.02 || ds > 0.02); it++) {
        let moved = false;
        for (const [a, b] of [[dm, 0], [-dm, 0], [0, ds], [0, -ds], [dm, ds], [dm, -ds], [-dm, ds], [-dm, -ds]]) {
          const m2 = clamp(m + a, 0, 100), s2 = clamp(s + b, S_MIN, S_MAX);
          const v = f(m2, s2);
          if (v < best - 1e-9) { best = v; m = m2; s = s2; moved = true; path.push([m, s]); break; }
        }
        if (!moved) { dm /= 2; ds /= 2; }
      }
      if (!path.length) return;
      btns.forEach((b) => { b.disabled = true; });
      const per = Math.max(1, Math.ceil(path.length / 50));
      let k = 0;
      (function frame() {
        k = Math.min(path.length, k + per);
        [st.mu, st.sg] = path[k - 1];
        sync();
        if (k < path.length) requestAnimationFrame(frame);
        else btns.forEach((b) => { b.disabled = false; });
      })();
    }

    mu.addEventListener('input', () => { st.mu = +mu.value; sync(); });
    sigma.addEventListener('input', () => { st.sg = +sigma.value; sync(); });
    btns[0].addEventListener('click', () => minimize(fwd));
    btns[1].addEventListener('click', () => minimize(rev));
    const rerender = A.responsive(box, (w) => { W = Math.max(280, w); sync(); });
    A.onTheme(() => rerender());
  }

  /* ── Wiring ─────────────────────────────────── */
  renderMathInElement(document.body, {
    delimiters: [{ left: '\\[', right: '\\]', display: true }, { left: '\\(', right: '\\)', display: false }],
    throwOnError: false,
  });
  document.getElementById('word-counts').innerHTML = WORDS.map((w, i) => `${wordLabel(i)} <span class="muted">×${COUNTS[i]}</span>`).map((h) => `<span>${h}</span>`).join('');

  surpriseWidget();
  qTable(document.getElementById('ce-table'), 'ce');
  qTable(document.getElementById('kl-table'), 'kl');
  columnsChart(document.getElementById('ce-chart'), false);
  columnsChart(document.getElementById('kl-chart'), true);
  ceEquation();
  klEquation();
  gradientWidget();
  directionWidget();

  document.querySelectorAll('.check[data-load]').forEach((c) => c.addEventListener('answered', (e) => {
    if (!e.detail.restoring) tweenQ(PRESETS[c.dataset.load]);
  }));
  document.querySelectorAll('[data-preset]').forEach((b) => b.addEventListener('click', () => {
    tweenQ(PRESETS[b.dataset.preset]);
    if (b.closest('.reveal-load')) document.getElementById('ce-chart').closest('.fig').scrollIntoView({ behavior: 'smooth', block: 'center' });
  }));

  const ex = window.CE_EXERCISES;
  A.mountExercise(document.getElementById('ex-ce'), ex.ce);
  A.mountExercise(document.getElementById('ex-kl'), ex.kl);
})();
