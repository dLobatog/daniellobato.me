/* Kelly betting on the next word. Log-bankroll grows by log2(4 q(x)) per round, so the game's score is
   cross-entropy and the gap to a bettor who knows p is KL(p||q) per round. */
(function () {
  const A = window.Atelier;
  const { svg, fmt } = A;
  const root = document.getElementById('game-fig');
  const CHIPS = 20;
  const START = 100;
  const color = (i) => `var(--w${i})`;
  const log2 = Math.log2;
  const P1 = [0.5, 0.25, 0.2, 0.05];
  const P2 = [0.45, 0.3, 0.2, 0.05];
  const P3_AFTER = [0.1, 0.15, 0.25, 0.5];
  const SHIFT = 25;

  const LEVELS = [
    {
      name: 'Known odds', context: 'the cat sat on the', words: ['mat', 'floor', 'sofa', 'moon'], rounds: 30, showP: true,
      p: () => P1,
      brief: 'You are told the true frequencies. Can you keep up with the perfect bettor for 30 rounds?',
    },
    {
      name: 'Hidden odds', context: "I'd like a cup of", words: ['tea', 'coffee', 'water', 'juice'], rounds: 40, showP: false,
      p: () => P2,
      brief: 'New context, hidden frequencies. All you know is what you have seen. You can move chips between rounds.',
    },
    {
      name: 'The world changes', context: 'the cat sat on the', words: ['mat', 'floor', 'sofa', 'moon'], rounds: 50, showP: false,
      p: (t) => (t < SHIFT ? P1 : P3_AFTER),
      brief: 'Hidden frequencies again. At some point the text source changes, and you will not be told when.',
    },
  ];

  let li = 0;
  let st;

  const kl = (p, q) => p.reduce((s, pi, i) => (pi === 0 ? s : q[i] === 0 ? Infinity : s + pi * log2(pi / q[i])), 0);
  const sampleFrom = (p, u) => { let c = 0; for (let i = 0; i < p.length; i++) { c += p[i]; if (u < c) return i; } return p.length - 1; };
  function mulberry32(a) { return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

  function fresh() {
    return { bet: [5, 5, 5, 5], t: 0, y: 0, py: 0, hist: [{ y: 0, py: 0 }], seq: [], bankrupt: false, expGap: 0, last: null, msg: '', err: '', sim: null };
  }

  root.innerHTML = `
    <div class="game-tabs" role="tablist"></div>
    <p class="game-brief"></p>
    <div class="scroll-x"><table class="qtable game-table"></table></div>
    <div class="btns">
      <button class="btn primary" data-act="draw">Draw next word</button>
      <button class="btn" data-act="draw10">Draw 10</button>
      <button class="btn" data-act="restart">Restart level</button>
      <span class="game-left small muted" style="align-self:center"></span>
    </div>
    <div class="game-err small" style="color:var(--bad);min-height:1.3em;margin-top:6px"></div>
    <div class="game-msg" style="min-height:1.6em"></div>
    <div class="readouts game-stats"></div>
    <div class="chart game-plot" style="margin-top:10px"></div>
    <p class="fig-note">Height = log₂(bankroll / $100). Each round adds 2 − your surprise. Solid: you. Dashed: a bettor who knows the true frequencies.</p>
    <div class="game-debrief"></div>`;
  const $ = (sel) => root.querySelector(sel);

  function money(v) {
    if (v === 0) return '$0';
    if (v >= 1e6) return '$' + v.toExponential(2).replace('e+', '×10^');
    if (v < 0.01) return '$' + v.toExponential(1).replace('e-', '×10^-');
    return '$' + v.toFixed(2);
  }
  const used = () => st.bet.reduce((a, b) => a + b, 0);
  const over = () => st.bankrupt || st.t >= LEVELS[li].rounds;

  function counts(from) {
    const c = [0, 0, 0, 0];
    st.seq.slice(from).forEach((k) => c[k]++);
    return c;
  }

  function renderTabs() {
    const tabs = $('.game-tabs');
    tabs.innerHTML = LEVELS.map((L, i) => `<button class="btn${i === li ? ' primary' : ''}" role="tab" aria-selected="${i === li}" data-level="${i}">Level ${i + 1} · ${L.name}</button>`).join('');
    tabs.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => { li = +b.dataset.level; st = fresh(); render(); }));
  }

  function renderTable() {
    const L = LEVELS[li];
    const all = counts(0), recent = counts(Math.max(0, st.seq.length - 10));
    const extraHead = L.showP ? '<th class="num hide-sm">true p</th>' : li === 1 ? '<th class="num hide-sm">seen</th>' : '<th class="num hide-sm">seen</th><th class="num hide-sm">last 10</th>';
    $('.game-table').innerHTML = `<tr><th>next word</th><th>your chips</th><th class="num hide-sm">share</th><th></th>${extraHead}</tr>` +
      L.words.map((w, i) => {
        const chips = Array.from({ length: st.bet[i] }, () => `<i style="background:${color(i)}"></i>`).join('');
        const extra = L.showP ? `<td class="num hide-sm">${Math.round(L.p(0)[i] * 100)}%</td>` : li === 1 ? `<td class="num hide-sm">${all[i]}</td>` : `<td class="num hide-sm">${all[i]}</td><td class="num hide-sm">${recent[i]}</td>`;
        const note = L.showP ? `true ${Math.round(L.p(0)[i] * 100)}%` : li === 1 ? `seen ${all[i]}` : `seen ${all[i]} · last 10: ${recent[i]}`;
        const hit = st.last && st.last.k === i ? ' class="hit"' : '';
        return `<tr${hit}><td><span class="word"><i style="background:${color(i)}"></i>${w}</span><div class="show-sm small muted">${note}</div></td>
          <td><div class="chips">${chips || '<span class="muted small">none</span>'}</div><div class="show-sm small muted">${st.bet[i] * 5}%</div></td>
          <td class="num hide-sm">${st.bet[i] * 5}%</td>
          <td class="pm"><button class="btn" data-minus="${i}" aria-label="remove a chip from ${w}">−</button><button class="btn" data-plus="${i}" aria-label="add a chip to ${w}">+</button></td>${extra}</tr>`;
      }).join('');
    $('.game-table').querySelectorAll('[data-minus]').forEach((b) => b.addEventListener('click', () => {
      const i = +b.dataset.minus; st.err = '';
      if (st.bet[i] > 0) st.bet[i]--;
      render();
    }));
    $('.game-table').querySelectorAll('[data-plus]').forEach((b) => b.addEventListener('click', () => {
      const i = +b.dataset.plus; st.err = '';
      if (used() < CHIPS) st.bet[i]++;
      else st.err = 'All 20 chips are placed. Take one off another word first.';
      render();
    }));
  }

  function drawOnce() {
    const L = LEVELS[li];
    const p = L.p(st.t);
    const q = st.bet.map((b) => b / CHIPS);
    const k = sampleFrom(p, Math.random());
    st.expGap += kl(p, q);
    st.seq.push(k);
    st.py += 2 + log2(p[k]);
    if (q[k] === 0) { st.bankrupt = true; st.y = -Infinity; } else st.y += 2 + log2(q[k]);
    st.hist.push({ y: st.y, py: st.py });
    st.t++;
    st.last = { k, q: q[k] };
  }

  function act(n) {
    st.err = '';
    if (over()) { st.err = st.bankrupt ? 'You went bankrupt. Restart the level to try again.' : 'This level is finished. Restart it or try the next level.'; render(); return; }
    if (used() !== CHIPS) { st.err = `Place all 20 chips first (${CHIPS - used()} left).`; render(); return; }
    for (let i = 0; i < n && !over(); i++) drawOnce();
    render();
  }

  function renderMsg() {
    const L = LEVELS[li];
    $('.game-err').textContent = st.err;
    $('.game-left').textContent = used() === CHIPS ? '' : `${CHIPS - used()} chips left to place`;
    const m = $('.game-msg');
    if (!st.last) { m.innerHTML = `<span class="muted">Context: <em>${L.context} ___</em>. Place your chips and draw.</span>`; return; }
    const w = L.words[st.last.k];
    if (st.bankrupt) {
      m.innerHTML = `It was <b style="color:${color(st.last.k)}">${w}</b>. You had no chips on it: <b style="color:var(--bad)">bankrupt</b>. Surprise = −log₂ 0 = ∞.`;
    } else {
      m.innerHTML = `It was <b style="color:${color(st.last.k)}">${w}</b>. Bankroll ×${(4 * st.last.q).toFixed(2)}, surprise ${fmt(-log2(st.last.q))} bits.`;
    }
  }

  function renderStats() {
    const L = LEVELS[li];
    const you = st.bankrupt ? 0 : START * Math.pow(2, st.y);
    const perfect = START * Math.pow(2, st.py);
    const gap = st.bankrupt ? Infinity : st.py - st.y;
    $('.game-stats').innerHTML = `
      <div class="readout"><b>${st.t} / ${L.rounds}</b><span>round</span></div>
      <div class="readout"><b>${money(you)}</b><span>your bankroll</span></div>
      <div class="readout"><b>${money(perfect)}</b><span>perfect bettor</span></div>
      <div class="readout"><b>${st.t ? (gap >= 0 ? '' : '+') + fmt(Math.abs(gap)) : '–'}</b><span>${gap >= 0 ? 'bits behind' : 'bits ahead (luck)'}</span></div>`;
  }

  let W = 640;
  function renderPlot() {
    const box = $('.game-plot');
    box.innerHTML = '';
    const L = LEVELS[li];
    const H = 200, m = { l: 40, r: 12, t: 10, b: 30 };
    const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': 'Log2 bankroll per round, you versus the perfect bettor' }, box);
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const vals = st.hist.flatMap((h) => [h.y, h.py]).filter(Number.isFinite);
    let lo = Math.min(-3, ...vals), hi = Math.max(6, ...vals);
    lo = Math.floor(lo / 2) * 2; hi = Math.ceil(hi / 2) * 2;
    const X = (k) => m.l + (k / L.rounds) * pw;
    const Y = (v) => m.t + ph - ((v - lo) / (hi - lo)) * ph;
    const step = hi - lo > 24 ? 8 : hi - lo > 12 ? 4 : 2;
    for (let v = lo; v <= hi; v += step) {
      svg('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: 'axis', style: v === 0 ? 'stroke: var(--ink-3)' : '' }, s);
      svg('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end', class: 'tick', text: v }, s);
    }
    for (let k = 0; k <= L.rounds; k += 10) svg('text', { x: X(k), y: H - m.b + 16, 'text-anchor': 'middle', class: 'tick', text: k }, s);
    if (li === 2 && over()) {
      svg('line', { x1: X(SHIFT), x2: X(SHIFT), y1: m.t, y2: m.t + ph, style: 'stroke: var(--bad); stroke-dasharray: 4 3' }, s);
      svg('text', { x: X(SHIFT) + 4, y: m.t + 12, 'font-size': 12, style: 'fill: var(--bad)', text: 'source changed' }, s);
    }
    const path = (key) => {
      let d = '';
      for (let k = 0; k < st.hist.length; k++) {
        const v = st.hist[k][key];
        if (!Number.isFinite(v)) break;
        d += (k ? 'L' : 'M') + X(k).toFixed(1) + ',' + Y(v).toFixed(1);
      }
      return d;
    };
    svg('path', { d: path('py'), fill: 'none', style: 'stroke: var(--ink-3); stroke-width: 2; stroke-dasharray: 5 4' }, s);
    svg('path', { d: path('y'), fill: 'none', style: 'stroke: var(--accent); stroke-width: 2.5' }, s);
    if (st.bankrupt) {
      const k = st.hist.length - 2, v = st.hist[k].y;
      svg('text', { x: X(k + 1), y: Y(v) + 5, 'text-anchor': 'middle', 'font-size': 16, 'font-weight': 700, style: 'fill: var(--bad)', text: '✕' }, s);
    }
  }

  /* ── Debriefs ───────────────────────────────── */
  function debrief() {
    const d = $('.game-debrief');
    if (!over()) { d.innerHTML = ''; return; }
    const L = LEVELS[li];
    const actual = st.bankrupt ? Infinity : st.py - st.y;
    let h = '<div class="debrief">';
    if (st.bankrupt) {
      const w = L.words[st.last.k];
      h += `<p><b>Bankrupt on ${w}.</b> One outcome you gave probability 0 wipes out every gain before it, exactly as one \\(-\\log 0 = \\infty\\) term makes the average loss infinite. This is why a model must never assign exactly 0 to something that can happen.</p>`;
    } else {
      h += `<p><b>You finished ${fmt(Math.abs(actual))} bits ${actual >= 0 ? 'behind' : 'ahead of'} the perfect bettor</b> over ${st.t} rounds. Your bets predicted a gap of \\(\\sum_t \\mathrm{KL}(p\\|q_t) = ${fmt(st.expGap)}\\) bits. The difference between the two is luck: the draws you happened to get. The expected gap can't be negative. Beating the perfect bettor only happens by luck.</p>`;
    }
    if (li === 0) {
      h += '<p>The bet that never falls behind in expectation is 10 / 5 / 4 / 1: \\(q = p\\). Try it. The two lines move in lockstep, whatever the draws.</p>';
    }
    if (li === 1) {
      const c = counts(0), n = st.seq.length;
      h += `<p><b>The hidden frequencies were ${P2.map((v, i) => `${L.words[i]} ${Math.round(v * 100)}%`).join(', ')}.</b> You saw ${c.map((v, i) => `${L.words[i]} ${v}`).join(', ')} in ${n} draws.</p>
        <p>Which rule should you have used? Replay 2,000 games of 40 rounds with fixed rules and the same luck for each rule:</p>
        <div class="btns"><button class="btn" data-sim="2">Replay 2,000 games</button></div>`;
    }
    if (li === 2) {
      h += `<p><b>The source changed after round ${SHIFT}</b>: moon went from 5% to 50%, mat from 50% to 10%. The counts you'd collected became a description of a world that no longer existed.</p>
        <p>Compare a rule that trusts all history with one that only looks at the last 10 rounds:</p>
        <div class="btns"><button class="btn" data-sim="3">Replay 2,000 games</button></div>`;
    }
    h += '<div class="sim-out"></div></div>';
    d.innerHTML = h;
    renderMathInElement(d, { delimiters: [{ left: '\\(', right: '\\)', display: false }], throwOnError: false });
    d.querySelectorAll('[data-sim]').forEach((b) => b.addEventListener('click', () => { b.textContent = 'Replaying…'; setTimeout(() => { st.sim = simulate(+b.dataset.sim); showSim(); }, 20); }));
    if (st.sim) showSim();
  }

  /* Fixed betting rules replayed on identical draws (common random numbers), exact fractions instead of chips. */
  function simulate(level) {
    const GAMES = 2000;
    const L = LEVELS[level - 1];
    const rules = level === 2
      ? [
        { name: 'Bet the observed frequencies', f: (c, n) => (n ? c.map((x) => x / n) : [0.25, 0.25, 0.25, 0.25]) },
        { name: 'Observed counts + 1 each', f: (c, n) => c.map((x) => (x + 1) / (n + 4)) },
        { name: 'Always uniform', f: () => [0.25, 0.25, 0.25, 0.25] },
      ]
      : [
        { name: 'Counts + 1, all history', f: (seq) => { const c = [1, 1, 1, 1]; seq.forEach((k) => c[k]++); const n = seq.length + 4; return c.map((x) => x / n); } },
        { name: 'Counts + 1, last 10 rounds', f: (seq) => { const c = [1, 1, 1, 1]; seq.slice(-10).forEach((k) => c[k]++); const n = Math.min(10, seq.length) + 4; return c.map((x) => x / n); } },
      ];
    const out = rules.map((r) => ({ name: r.name, bankrupt: 0, gap: 0, gapA: 0, gapB: 0, survivors: 0 }));
    const rng = mulberry32(12345);
    for (let g = 0; g < GAMES; g++) {
      const draws = [];
      for (let t = 0; t < L.rounds; t++) { const p = L.p(t); draws.push({ k: sampleFrom(p, rng()), p }); }
      rules.forEach((r, ri) => {
        const c = [0, 0, 0, 0], seq = [];
        let gap = 0, gapA = 0, gapB = 0, dead = false;
        for (let t = 0; t < draws.length; t++) {
          const { k, p } = draws[t];
          const q = level === 2 ? r.f(c, t) : r.f(seq);
          if (q[k] === 0) { dead = true; break; }
          const gk = log2(p[k] / q[k]);
          gap += gk;
          if (t < SHIFT) gapA += gk; else gapB += gk;
          c[k]++; seq.push(k);
        }
        if (dead) out[ri].bankrupt++;
        else { out[ri].survivors++; out[ri].gap += gap / draws.length; out[ri].gapA += gapA / SHIFT; out[ri].gapB += gapB / (draws.length - SHIFT); }
      });
    }
    out.forEach((o) => { if (o.survivors) { o.gap /= o.survivors; o.gapA /= o.survivors; o.gapB /= o.survivors; } });
    return { level, out };
  }

  function showSim() {
    const box = $('.sim-out');
    if (!box || !st.sim || st.sim.level !== li + 1) return;
    const { level, out } = st.sim;
    const pct = (x) => `${((100 * x) / 2000).toFixed(1)}%`;
    const g = (o, key) => (o.survivors ? fmt(o[key], 3) : '–');
    root.querySelectorAll('[data-sim]').forEach((b) => { b.textContent = 'Replay again'; });
    if (level === 2) {
      box.innerHTML = `<div class="scroll-x"><table class="plain"><tr><th>Rule</th><th>Went bankrupt</th><th>Bits lost per round vs perfect (survivors)</th></tr>
        ${out.map((o) => `<tr><td>${o.name}</td><td>${pct(o.bankrupt)}</td><td>${g(o, 'gap')}</td></tr>`).join('')}</table></div>
        <p>Betting the raw observed frequencies is <strong>maximum likelihood</strong>, and it went bankrupt in every one of these games. After one draw of tea it puts everything on tea, and the first coffee ends the game. Adding one imaginary observation of every word (Laplace smoothing) never goes bankrupt and loses little per round. That is the betting version of why a model must keep some probability on outcomes it hasn't seen yet. Uniform survives too but ignores the evidence, so it loses KL(p‖uniform) = ${fmt(kl(P2, [0.25, 0.25, 0.25, 0.25]), 3)} bits every round.</p>`;
    } else {
      box.innerHTML = `<div class="scroll-x"><table class="plain"><tr><th>Rule</th><th>Bits lost per round, rounds 1–25</th><th>Bits lost per round, rounds 26–50</th></tr>
        ${out.map((o) => `<tr><td>${o.name}</td><td>${g(o, 'gapA')}</td><td>${g(o, 'gapB')}</td></tr>`).join('')}</table></div>
        <p>Before the change, the short window loses more: 10 draws are a noisy estimate (variance). After it, all-history keeps betting on a world that's gone (bias), and the window catches up within a few rounds. Neither rule wins everywhere. The window length is a bias–variance trade-off, and it depends on how fast the world moves. That's the same reason production models are retrained on recent data, and why a model's offline loss says nothing about next month.</p>`;
    }
  }

  function render() {
    renderTabs();
    $('.game-brief').innerHTML = `<b>Level ${li + 1}.</b> ${LEVELS[li].brief}`;
    renderTable();
    renderMsg();
    renderStats();
    renderPlot();
    debrief();
  }

  root.querySelector('[data-act="draw"]').addEventListener('click', () => act(1));
  root.querySelector('[data-act="draw10"]').addEventListener('click', () => act(10));
  root.querySelector('[data-act="restart"]').addEventListener('click', () => { st = fresh(); render(); });
  st = fresh();
  A.responsive($('.game-plot'), (w) => { W = Math.max(280, w); renderPlot(); });
  render();
})();
