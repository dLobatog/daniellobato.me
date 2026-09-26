/* Lesson 2: every widget uses the same six-answer policy (softmax over logits) or groups sampled from it. */
(function () {
  const A = window.Atelier;
  const { svg, fmt } = A;
  const ANS = ['408', '410', '400', '390', '420', '380'];
  const R = [1.0, 0.8, 0.6, 0.5, 0.5, 0.4];
  const K = ANS.length;
  const softmax = (z) => { const m = Math.max(...z); const e = z.map((v) => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); };
  const expReward = (pi) => pi.reduce((s, p, i) => s + p * R[i], 0);
  const sample = (p, u) => { let c = 0; for (let i = 0; i < p.length; i++) { c += p[i]; if (u < c) return i; } return p.length - 1; };
  function mulberry32(a) { return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  const pstd = (xs) => { const m = mean(xs); return Math.sqrt(mean(xs.map((x) => (x - m) ** 2))); };
  const groupAdv = (rs) => { const m = mean(rs), s = pstd(rs); return rs.map((r) => (s < 1e-9 ? 0 : (r - m) / s)); };
  const halo = 'paint-order: stroke; stroke: var(--surface); stroke-width: 4px; stroke-linejoin: round;';

  renderMathInElement(document.body, {
    delimiters: [{ left: '\\[', right: '\\]', display: true }, { left: '\\(', right: '\\)', display: false }],
    throwOnError: false,
  });

  /* ── Answers table ─────────────────────────── */
  document.getElementById('answers-table').innerHTML =
    '<tr><th>answer</th>' + ANS.map((a) => `<td class="mono">${a}</td>`).join('') + '</tr>' +
    '<tr><th>reward r</th>' + R.map((r) => `<td class="mono">${r.toFixed(1)}</td>`).join('') + '</tr>';

  /* ── 1. One REINFORCE step by hand ─────────── */
  (function stepWidget() {
    const bars = document.getElementById('step-bars');
    const ro = document.getElementById('step-readouts');
    const note = document.getElementById('step-note');
    let z = new Array(K).fill(0), n = 0;
    bars.innerHTML = ANS.map((a, i) => `<span class="mono">${a}</span><div class="bar"><b class="${i === 0 ? 'best' : ''}"></b></div><span class="num"></span>`).join('');
    function draw(last) {
      const pi = softmax(z);
      bars.querySelectorAll('.bar b').forEach((b, i) => { b.style.width = (pi[i] * 100).toFixed(1) + '%'; });
      bars.querySelectorAll('.num').forEach((e, i) => { e.textContent = pi[i].toFixed(3); });
      ro.innerHTML = `<div class="readout"><b>${n}</b><span>updates</span></div><div class="readout"><b>${expReward(pi).toFixed(3)}</b><span>expected reward J</span></div>`;
      if (last) {
        const { y, r, piBefore } = last;
        note.innerHTML = `Sampled <b class="mono">${ANS[y]}</b>, reward ${r.toFixed(1)}. Its logit rose by ${r.toFixed(1)} × (1 − ${piBefore[y].toFixed(3)}) = ${(r * (1 - piBefore[y])).toFixed(3)}; every other logit fell by ${r.toFixed(1)} × its own probability.`;
      } else {
        note.textContent = 'Bars show π, the probability of each answer. Green is the best answer.';
      }
    }
    document.getElementById('step-go').addEventListener('click', () => {
      const pi = softmax(z);
      const y = sample(pi, Math.random());
      const r = R[y];
      z = z.map((v, i) => v + r * ((i === y ? 1 : 0) - pi[i]));
      n++;
      draw({ y, r, piBefore: pi });
    });
    document.getElementById('step-reset').addEventListener('click', () => { z = new Array(K).fill(0); n = 0; draw(); });
    draw();
  })();

  /* ── 2. Training curves: no baseline vs mean baseline vs GRPO ── */
  (function curves() {
    const box = document.getElementById('curves');
    const ESTS = [
      { name: 'No baseline', sub: 'A = r', f: (rs) => rs },
      { name: 'Group mean baseline', sub: 'A = r − mean(r)', f: (rs) => { const m = mean(rs); return rs.map((r) => r - m); } },
      { name: 'GRPO', sub: 'A = (r − mean) / std', f: groupAdv },
    ];
    const RUNS = 30, STEPS = 150, G = 4, LR = 1;
    let seed = 1;
    let data = null;
    function simulate() {
      data = ESTS.map((est) => {
        const runs = [];
        let ended = 0;
        for (let k = 0; k < RUNS; k++) {
          const rng = mulberry32(seed * 1000 + k);
          let z = new Array(K).fill(0);
          const J = [];
          for (let t = 0; t <= STEPS; t++) {
            const pi = softmax(z);
            J.push(expReward(pi));
            if (t === STEPS) { if (pi[0] > 0.9) ended++; break; }
            const ys = Array.from({ length: G }, () => sample(pi, rng()));
            const adv = est.f(ys.map((y) => R[y]));
            const g = new Array(K).fill(0);
            ys.forEach((y, j) => { for (let i = 0; i < K; i++) g[i] += adv[j] * ((i === y ? 1 : 0) - pi[i]); });
            z = z.map((v, i) => v + (LR * g[i]) / G);
          }
          runs.push(J);
        }
        return { runs, ended };
      });
    }
    function render() {
      box.innerHTML = '';
      ESTS.forEach((est, e) => {
        const cell = document.createElement('div');
        cell.innerHTML = `<div class="panel-title">${est.name}</div><div class="panel-sub mono">${est.sub}</div>`;
        const chart = document.createElement('div');
        chart.className = 'chart';
        cell.appendChild(chart);
        const foot = document.createElement('div');
        foot.className = 'small';
        foot.innerHTML = `Ended on 408: <b>${data[e].ended} / ${RUNS}</b>`;
        cell.appendChild(foot);
        box.appendChild(cell);
        const W = Math.max(220, chart.clientWidth || 260), H = 170, m = { l: 34, r: 8, t: 8, b: 24 };
        const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': `${est.name}: expected reward over training` }, chart);
        const pw = W - m.l - m.r, ph = H - m.t - m.b;
        const X = (t) => m.l + (t / STEPS) * pw, Y = (v) => m.t + ph - ((v - 0.6) / 0.4) * ph;
        [0.6, 0.7, 0.8, 0.9, 1.0].forEach((v) => {
          svg('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: 'axis' }, s);
          svg('text', { x: m.l - 5, y: Y(v) + 4, 'text-anchor': 'end', class: 'tick', text: v.toFixed(1) }, s);
        });
        [0, 50, 100, 150].forEach((t) => svg('text', { x: X(t), y: H - 6, 'text-anchor': 'middle', class: 'tick', text: t }, s));
        const line = (J) => J.map((v, t) => (t ? 'L' : 'M') + X(t).toFixed(1) + ',' + Y(Math.max(0.6, v)).toFixed(1)).join('');
        data[e].runs.forEach((J) => svg('path', { d: line(J), fill: 'none', style: 'stroke: var(--accent); stroke-width: 1; opacity: 0.28' }, s));
        const avg = data[e].runs[0].map((_, t) => mean(data[e].runs.map((J) => J[t])));
        svg('path', { d: line(avg), fill: 'none', style: 'stroke: var(--ink); stroke-width: 2.25' }, s);
      });
    }
    document.getElementById('curves-go').addEventListener('click', () => { seed++; simulate(); render(); });
    simulate();
    A.responsive(box, () => render());
    A.onTheme(render);
  })();

  /* ── 3 & 4. Game: predict the push ─────────── */
  function pushGame(root, rounds, eps) {
    let ri = 0, score = 0, total = 0, picks, checked;
    const reset = () => { picks = new Array(rounds[ri].rewards.length).fill(null); checked = false; };
    const truth = (rd) => {
      const adv = groupAdv(rd.rewards);
      return adv.map((a, i) => {
        let clipped = false;
        if (rd.ratios) { const rho = rd.ratios[i]; clipped = (a > 0 && rho > 1 + eps) || (a < 0 && rho < 1 - eps); }
        const dir = Math.abs(a) < 1e-9 || clipped ? 0 : Math.sign(a);
        return { a, clipped, dir };
      });
    };
    function render() {
      const rd = rounds[ri];
      const tr = truth(rd);
      const hasR = !!rd.ratios;
      const rows = rd.answers.map((ans, i) => {
        const t = tr[i];
        const seg = [-1, 0, 1].map((d) => {
          let cls = picks[i] === d ? 'on' : '';
          if (checked) cls = d === t.dir ? 'ok' : picks[i] === d ? 'bad' : '';
          return `<button data-i="${i}" data-d="${d}" class="${cls}" aria-label="${d > 0 ? 'more likely' : d < 0 ? 'less likely' : 'unchanged'}">${d > 0 ? '↑' : d < 0 ? '↓' : '0'}</button>`;
        }).join('');
        const w = Math.min(50, (Math.abs(t.a) / 2) * 50);
        const advCell = checked
          ? `<td class="num">${Math.abs(t.a) < 1e-9 ? '0' : (t.a > 0 ? '+' : '') + t.a.toFixed(2)} <span class="advbar hide-sm"><b style="width:${w}%;left:${t.a >= 0 ? 50 : 50 - w}%;background:${t.a >= 0 ? 'var(--good)' : 'var(--bad)'}"></b></span>${hasR && t.clipped ? ' <span class="tag">clipped</span>' : ''}</td>`
          : '<td class="num muted">?</td>';
        return `<tr><td class="ans">${ans}</td><td><span class="tag ${rd.rewards[i] >= 0.5 ? 'good' : 'bad'}">${rd.rewards[i]}</span></td>${hasR ? `<td class="num">${rd.ratios[i].toFixed(2)}</td>` : ''}<td><div class="seg">${seg}</div></td>${advCell}</tr>`;
      }).join('');
      root.innerHTML = `
        <div class="round-head"><b>Round ${ri + 1} of ${rounds.length}</b><span class="small muted">${total ? `score ${score} / ${total}` : ''}</span></div>
        <div class="round-note">${rd.note}</div>
        <div class="scroll-x"><table class="qtable push-table"><tr><th>completion</th><th>r</th>${hasR ? '<th class="num">ρ</th>' : ''}<th>your call</th><th class="num">Â</th></tr>${rows}</table></div>
        <div class="btns">${checked ? (ri < rounds.length - 1 ? '<button class="btn primary" data-next>Next round</button>' : '<button class="btn" data-again>Play again</button>') : '<button class="btn primary" data-check>Check</button>'}</div>
        <div class="small" data-err style="color:var(--bad);min-height:1.2em;margin-top:4px"></div>
        ${checked ? `<div class="debrief" style="margin-top:6px">${rd.explain}</div>` : ''}`;
      root.querySelectorAll('.seg button').forEach((b) => b.addEventListener('click', () => {
        if (checked) return;
        picks[+b.dataset.i] = +b.dataset.d;
        render();
      }));
      const c = root.querySelector('[data-check]');
      if (c) c.addEventListener('click', () => {
        if (picks.some((p) => p === null)) { root.querySelector('[data-err]').textContent = 'Make a call for every completion first.'; return; }
        checked = true;
        tr.forEach((t, i) => { total++; if (picks[i] === t.dir) score++; });
        render();
      });
      const nx = root.querySelector('[data-next]');
      if (nx) nx.addEventListener('click', () => { ri++; reset(); render(); });
      const ag = root.querySelector('[data-again]');
      if (ag) ag.addEventListener('click', () => { ri = 0; score = 0; total = 0; reset(); render(); });
    }
    reset();
    render();
  }

  pushGame(document.getElementById('game-grpo'), [
    { answers: ['408', '398', '418', '400'], rewards: [1, 0, 0, 0], note: 'One right answer out of four.',
      explain: 'Mean 0.25, std 0.43. The right answer gets Â = +1.73 and each wrong one −0.58. One success on a hard prompt is a strong signal, spread thinly as blame over the failures.' },
    { answers: ['408', '408', '398', '408'], rewards: [1, 1, 0, 1], note: 'Three right, one wrong.',
      explain: 'Mean 0.75. Now the right answers get only +0.58 and the wrong one −1.73. When most answers are already right, the lesson is mostly "don\'t do that".' },
    { answers: ['398', '418', '400', '388'], rewards: [0, 0, 0, 0], note: 'All four wrong.',
      explain: 'Every reward equals the mean, so every advantage is 0. The group can\'t tell which wrong answer was less wrong. No gradient, and four generations wasted.' },
    { answers: ['408', '408', '408', '408'], rewards: [1, 1, 1, 1], note: 'All four right.',
      explain: 'Also all zeros. A prompt that\'s too easy is as uninformative as one that\'s too hard. Useful prompts sit where the model sometimes succeeds.' },
    { answers: ['408, shown work', '408', '410', '"about 400"'], rewards: [0.9, 0.7, 0.8, 0.2], note: 'A reward model scores the same prompt with partial credit.',
      explain: 'Mean 0.65, std 0.27. The 0.7 answer is pushed <em>up</em> (Â = +0.19): advantages are relative to the group, not to a fixed bar. In a stronger group, the same 0.7 would be pushed down.' },
  ], 0.2);

  pushGame(document.getElementById('game-clip'), [
    { answers: ['408', '398', '418', '400'], rewards: [1, 0, 0, 0], ratios: [1.3, 1.0, 1.0, 1.0], note: 'Same group as round 1 above, one gradient step later. ε = 0.2.',
      explain: '408 has Â = +1.73, but ρ = 1.3 is past 1 + ε = 1.2, so it\'s clipped: no further push. The three wrong answers are inside the range and are pushed down as before.' },
    { answers: ['408', '398', '418', '400'], rewards: [1, 0, 0, 0], ratios: [1.1, 1.3, 0.7, 1.0], note: 'Different ratios this time. ε = 0.2.',
      explain: '408 is inside the range, so it\'s pushed up. 398 is bad and has become <em>more</em> likely (ρ = 1.3). For A &lt; 0 that is never clipped, so it\'s pushed down at full strength. 418 is bad and already 30% less likely (ρ = 0.7 &lt; 0.8): clipped, left alone. 400 is pushed down.' },
    { answers: ['408', '408', '398', '418'], rewards: [1, 1, 0, 0], ratios: [0.7, 1.25, 0.85, 1.0], note: 'Two right, two wrong. ε = 0.2.',
      explain: 'Advantages are ±1. The first 408 is good but became <em>less</em> likely (ρ = 0.7). For A &gt; 0 only the top is clipped, so it gets a full push back up. The second 408 is past 1.2: clipped. Both wrong answers are inside the range and pushed down. The rule: clipping stops you from going further in the direction the advantage wants, never from correcting a move in the wrong direction.' },
  ], 0.2);

  /* ── 4. Clipped objective plot ──────────────── */
  (function clipPlot() {
    const box = document.getElementById('clip-chart');
    const slider = document.getElementById('rho');
    const out = document.getElementById('rho-out');
    const ro = document.getElementById('clip-readouts');
    const EPS = 0.2, X0 = 0.3, X1 = 1.8;
    let sign = 1, rho = 1.3, W = 600;
    const clip = (r) => Math.min(1 + EPS, Math.max(1 - EPS, r));
    const L = (r) => Math.min(r * sign, clip(r) * sign);
    const grad = (r) => (r * sign <= clip(r) * sign ? sign : 0);
    function render() {
      box.innerHTML = '';
      const H = 220, m = { l: 40, r: 14, t: 14, b: 34 };
      const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': 'Clipped surrogate objective versus ratio' }, box);
      const pw = W - m.l - m.r, ph = H - m.t - m.b;
      const lo = sign > 0 ? 0.2 : -1.9, hi = sign > 0 ? 1.9 : -0.2;
      const X = (r) => m.l + ((r - X0) / (X1 - X0)) * pw, Y = (v) => m.t + ph - ((v - lo) / (hi - lo)) * ph;
      const flat = sign > 0 ? [1 + EPS, X1] : [X0, 1 - EPS];
      svg('rect', { x: X(flat[0]), y: m.t, width: X(flat[1]) - X(flat[0]), height: ph, style: 'fill: var(--surface-2)' }, s);
      svg('text', { x: (X(flat[0]) + X(flat[1])) / 2, y: m.t + 16, 'text-anchor': 'middle', 'font-size': 12.5, style: 'fill: var(--ink-3)', text: 'clipped: flat, no gradient' }, s);
      [0.5, 1, 1.5].forEach((r) => svg('text', { x: X(r), y: H - m.b + 17, 'text-anchor': 'middle', class: 'tick', text: r }, s));
      [0.8, 1.2].forEach((r) => {
        svg('line', { x1: X(r), x2: X(r), y1: m.t, y2: m.t + ph, style: 'stroke: var(--rule); stroke-dasharray: 3 3' }, s);
        svg('text', { x: X(r), y: H - m.b + 17, 'text-anchor': 'middle', class: 'tick', text: r }, s);
      });
      svg('text', { x: m.l + pw / 2, y: H - 3, 'text-anchor': 'middle', class: 'tick', text: 'ratio ρ = π_θ / π_old' }, s);
      for (let v = Math.ceil(lo * 2) / 2; v <= hi; v += 0.5) svg('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end', class: 'tick', text: v.toFixed(1) }, s);
      svg('path', { d: `M${X(X0)},${Y(X0 * sign)}L${X(X1)},${Y(X1 * sign)}`, style: 'stroke: var(--ink-3); stroke-dasharray: 5 4; fill: none' }, s);
      let d = '';
      for (let k = 0; k <= 150; k++) { const r = X0 + ((X1 - X0) * k) / 150; d += (k ? 'L' : 'M') + X(r).toFixed(1) + ',' + Y(L(r)).toFixed(1); }
      svg('path', { d, style: 'stroke: var(--accent); stroke-width: 2.75; fill: none' }, s);
      svg('text', { x: X(X1) - 4, y: Y(X1 * sign) + (sign > 0 ? -8 : 16), 'text-anchor': 'end', 'font-size': 12, style: halo + 'fill: var(--ink-3)', text: 'unclipped ρA' }, s);
      svg('circle', { cx: X(rho), cy: Y(L(rho)), r: 7, style: 'fill: var(--accent); stroke: var(--surface); stroke-width: 2.5' }, s);
    }
    function update() {
      slider.value = rho;
      out.textContent = rho.toFixed(2);
      const g = grad(rho);
      ro.innerHTML = `<div class="readout"><b>${(L(rho)).toFixed(2)}</b><span>objective L(ρ), A = ${sign > 0 ? '+1' : '−1'}</span></div>
        <div class="readout"><b>${g === 0 ? '0' : g > 0 ? '+1' : '−1'}</b><span>gradient dL/dρ</span></div>
        <div class="readout"><b style="font-size:1.05rem;padding-top:5px">${g === 0 ? 'clipped: no push' : g > 0 ? 'pushed up' : 'pushed down'}</b><span>what the step does</span></div>`;
      render();
    }
    slider.addEventListener('input', () => { rho = +slider.value; update(); });
    document.querySelectorAll('#clip-sign button').forEach((b) => b.addEventListener('click', () => {
      sign = +b.dataset.sign;
      document.querySelectorAll('#clip-sign button').forEach((x) => x.classList.toggle('on', x === b));
      update();
    }));
    let dragging = false;
    const fromEvent = (e) => { const r = box.getBoundingClientRect(); rho = Math.min(X1, Math.max(X0, X0 + ((e.clientX - r.left - 40) / (W - 54)) * (X1 - X0))); update(); };
    box.style.touchAction = 'pan-y';
    box.addEventListener('pointerdown', (e) => { dragging = true; box.setPointerCapture(e.pointerId); fromEvent(e); });
    box.addEventListener('pointermove', (e) => { if (dragging) fromEvent(e); });
    box.addEventListener('pointerup', () => { dragging = false; });
    A.responsive(box, (w) => { W = Math.max(280, w); update(); });
  })();

  /* ── 5. Length bias table ───────────────────── */
  (function lengthTable() {
    const table = document.getElementById('len-table');
    const LENS = [12, 30, 60, 150];
    const rewards = [1, 0, 1, 0];
    function draw() {
      const adv = groupAdv(rewards);
      const G = LENS.length, tot = LENS.reduce((a, b) => a + b, 0);
      const seq = adv.map((a, i) => (1000 * a) / (G * LENS[i]));
      const tok = adv.map((a) => (1000 * a) / tot);
      const mx = Math.max(...seq.map(Math.abs), ...tok.map(Math.abs), 1e-9);
      const bar = (v) => { const w = Math.min(50, (Math.abs(v) / mx) * 50); return `<span class="advbar"><b style="width:${w}%;left:${v >= 0 ? 50 : 50 - w}%;background:${v >= 0 ? 'var(--good)' : 'var(--bad)'}"></b></span>`; };
      table.innerHTML = '<tr><th>#</th><th>reward</th><th class="num">tokens</th><th class="num">Â</th><th class="num">GRPO</th><th class="num">DAPO</th></tr>' +
        LENS.map((len, i) => `<tr><td>${i + 1}</td><td><button class="btn" data-flip="${i}" style="padding:3px 10px"><span class="tag ${rewards[i] ? 'good' : 'bad'}">${rewards[i] ? 'right' : 'wrong'}</span></button></td>
          <td class="num">${len}</td><td class="num">${adv[i] >= 0 ? '+' : ''}${adv[i].toFixed(2)}</td>
          <td class="num">${seq[i].toFixed(1)}<span class="hide-sm"> ${bar(seq[i])}</span></td><td class="num">${tok[i].toFixed(1)}<span class="hide-sm"> ${bar(tok[i])}</span></td></tr>`).join('');
      table.querySelectorAll('[data-flip]').forEach((b) => b.addEventListener('click', () => { const i = +b.dataset.flip; rewards[i] = 1 - rewards[i]; draw(); }));
    }
    draw();
  })();

  /* ── 7. Exercises ───────────────────────────── */
  const ex = window.PG_EXERCISES;
  A.mountExercise(document.getElementById('ex-adv'), ex.adv);
  A.mountExercise(document.getElementById('ex-clip'), ex.clip);
  A.mountExercise(document.getElementById('ex-kl'), ex.kl);
})();
