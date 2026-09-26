/* Lesson 7: A/B tests. Everything is simulated in the browser from Bernoulli click models (normal approximation for speed). */
(function () {
  const A = window.Atelier;
  const { svg } = A;
  function mulberry32(a) { return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  let rng = mulberry32(Date.now() % 100000);
  const gauss = () => { const u = Math.max(rng(), 1e-12), v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const clicks = (n, p) => Math.max(0, Math.round(n * p + Math.sqrt(n * p * (1 - p)) * gauss()));
  // Two-sided normal tail via erfc (Abramowitz–Stegun 7.1.26 is enough for display)
  function erfc(x) { const z = Math.abs(x), t = 1 / (1 + 0.5 * z); const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))); return x >= 0 ? r : 2 - r; }
  function ztest(ca, na, cb, nb) { const p = (ca + cb) / (na + nb); const se = Math.sqrt(p * (1 - p) * (1 / na + 1 / nb)); const z = se > 0 ? (cb / nb - ca / na) / se : 0; return { z, p: erfc(Math.abs(z) / Math.SQRT2) }; }
  function invNorm(p) { // Acklam's approximation
    const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
    const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
    const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
    const pl = 0.02425;
    if (p < pl) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    if (p > 1 - pl) return -invNorm(1 - p);
    const q = p - 0.5, r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }

  renderMathInElement(document.body, {
    delimiters: [{ left: '\\[', right: '\\]', display: true }, { left: '\\(', right: '\\)', display: false }],
    throwOnError: false,
  });

  /* ── 1. A/A histogram ───────────────────────── */
  (function aa() {
    const box = document.getElementById('aa-chart'), out = document.getElementById('aa-out');
    let zs = [], W = 600;
    function run() {
      zs = [];
      for (let t = 0; t < 1000; t++) { const n = 10000; zs.push(ztest(clicks(n, 0.05), n, clicks(n, 0.05), n).z); }
      const sig = zs.filter((z) => Math.abs(z) > 1.96).length;
      out.innerHTML = `<div class="readout"><b>${sig}</b><span>of 1,000 had p &lt; 0.05</span></div><div class="readout"><b>${(sig / 10).toFixed(1)}%</b><span>false-positive rate</span></div>`;
      render();
    }
    function render() {
      box.innerHTML = '';
      const H = 190, m = { l: 30, r: 10, t: 10, b: 30 };
      const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': 'Histogram of A/A test z-scores' }, box);
      const pw = W - m.l - m.r, ph = H - m.t - m.b;
      const X = (z) => m.l + ((z + 4) / 8) * pw;
      const bins = new Array(32).fill(0);
      zs.forEach((z) => { const b = Math.floor((z + 4) / 0.25); if (b >= 0 && b < 32) bins[b]++; });
      const mx = Math.max(30, ...bins);
      svg('rect', { x: m.l, y: m.t, width: X(-1.96) - m.l, height: ph, style: 'fill: var(--bad-soft)' }, s);
      svg('rect', { x: X(1.96), y: m.t, width: W - m.r - X(1.96), height: ph, style: 'fill: var(--bad-soft)' }, s);
      bins.forEach((c, i) => { const z0 = -4 + i * 0.25; const h = (c / mx) * ph; svg('rect', { x: X(z0) + 1, y: m.t + ph - h, width: pw / 32 - 2, height: h, style: `fill: ${Math.abs(z0 + 0.125) > 1.96 ? 'var(--bad)' : 'var(--accent)'}; opacity: 0.85` }, s); });
      [-4, -2, -1.96, 0, 1.96, 2, 4].filter((v) => Math.abs(v) !== 2).forEach((v) => svg('text', { x: X(v), y: H - m.b + 16, 'text-anchor': 'middle', class: 'tick', text: v }, s));
      svg('line', { x1: m.l, x2: W - m.r, y1: m.t + ph, y2: m.t + ph, class: 'axis' }, s);
      if (!zs.length) svg('text', { x: W / 2, y: H / 2, 'text-anchor': 'middle', 'font-size': 13, style: 'fill: var(--ink-3)', text: 'Press "Run 1,000 A/A tests"' }, s);
    }
    document.getElementById('aa-run').addEventListener('click', run);
    A.responsive(box, (w) => { W = Math.max(280, w); render(); });
  })();

  /* ── 3. Sample size calculator ──────────────── */
  (function ss() {
    const F = [{ k: 'p', label: 'base rate (%)', v: 5 }, { k: 'mde', label: 'relative lift to detect (%)', v: 2 }, { k: 'alpha', label: 'α (two-sided)', v: 0.05 }, { k: 'pow', label: 'power', v: 0.8 }, { k: 'traffic', label: 'users per arm per day', v: 50000 }];
    const box = document.getElementById('ss-calc'), out = document.getElementById('ss-out');
    box.innerHTML = F.map((f) => `<div><label for="ss-${f.k}">${f.label}</label><input id="ss-${f.k}" type="number" step="any" value="${f.v}"></div>`).join('');
    function calc() {
      const g = (k) => +document.getElementById('ss-' + k).value;
      const p1 = g('p') / 100, p2 = p1 * (1 + g('mde') / 100), a = g('alpha'), pw = g('pow');
      if (!(p1 > 0 && p1 < 1 && p2 > 0 && p2 < 1 && a > 0 && a < 1 && pw > 0 && pw < 1 && p2 !== p1)) { out.innerHTML = '<span class="small" style="color:var(--bad)">Enter a base rate and lift that give rates between 0 and 100%, and α and power between 0 and 1.</span>'; return; }
      const zz = invNorm(1 - a / 2) + invNorm(pw);
      const n = Math.ceil((zz * zz * (p1 * (1 - p1) + p2 * (1 - p2))) / (p2 - p1) ** 2);
      const days = n / Math.max(1, g('traffic'));
      out.innerHTML = `<div class="readout"><b>${n.toLocaleString('en-US')}</b><span>users per arm</span></div><div class="readout"><b>${days < 1 ? '< 1' : days.toFixed(1)}</b><span>days at your traffic</span></div><div class="readout"><b>${(100 * p1).toFixed(2)}% → ${(100 * p2).toFixed(3)}%</b><span>absolute rates</span></div>`;
    }
    box.addEventListener('input', calc);
    calc();
  })();

  /* ── 4. Game: ship or not ───────────────────── */
  (function peekGame() {
    const root = document.getElementById('peek-game');
    const DAYS = 14, NPD = 2000, BASE = 0.05, LIFT = 0.10;
    let exps, ei, day, data, decided, tally;
    function newGame() {
      const kinds = [0, 0, 0, 0, 1, 1, 1, 1];
      for (let i = kinds.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [kinds[i], kinds[j]] = [kinds[j], kinds[i]]; }
      exps = kinds.map((real) => {
        const d = []; let ca = 0, cb = 0;
        for (let t = 1; t <= DAYS; t++) { ca += clicks(NPD, BASE); cb += clicks(NPD, BASE * (real ? 1 + LIFT : 1)); d.push({ n: t * NPD, ca, cb, ...ztest(ca, t * NPD, cb, t * NPD) }); }
        return { real, d };
      });
      ei = 0; day = 1; decided = null;
      tally = { falseShip: 0, rightShip: 0, missed: 0, rightKeep: 0, earlyNullShip: 0, days: 0 };
      render();
    }
    let W = 640;
    function chart(box, d, upto, done) {
      box.innerHTML = '';
      const H = 210, m = { l: 44, r: 12, t: 12, b: 30 };
      const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': 'Observed relative lift with 95% interval by day' }, box);
      const pw = W - m.l - m.r, ph = H - m.t - m.b;
      const lo = -30, hi = 40;
      const X = (t) => m.l + ((t - 0.5) / DAYS) * pw, Y = (v) => m.t + ph - ((v - lo) / (hi - lo)) * ph;
      [-20, -10, 0, 10, 20, 30, 40].forEach((v) => { svg('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: 'axis', style: v === 0 ? 'stroke: var(--ink-3)' : '' }, s); svg('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end', class: 'tick', text: (v > 0 ? '+' : '') + v + '%' }, s); });
      for (let t = 1; t <= DAYS; t += t < 2 ? 1 : 2) svg('text', { x: X(t), y: H - m.b + 16, 'text-anchor': 'middle', class: 'tick', text: 'day ' + t }, s);
      if (done) svg('line', { x1: m.l, x2: W - m.r, y1: Y(done.real ? 100 * LIFT : 0), y2: Y(done.real ? 100 * LIFT : 0), style: 'stroke: var(--good); stroke-width: 2; stroke-dasharray: 6 4' }, s);
      d.slice(0, upto).forEach((r, i) => {
        const pa = r.ca / r.n, pb = r.cb / r.n, lift = 100 * (pb / pa - 1);
        const se = 100 * Math.sqrt(pa * (1 - pa) / r.n + pb * (1 - pb) / r.n) / pa;
        const sig = r.p < 0.05;
        svg('line', { x1: X(i + 1), x2: X(i + 1), y1: Y(Math.min(hi, lift + 1.96 * se)), y2: Y(Math.max(lo, lift - 1.96 * se)), style: `stroke: ${sig ? 'var(--bad)' : 'var(--ink-3)'}; stroke-width: 2` }, s);
        svg('circle', { cx: X(i + 1), cy: Y(Math.max(lo, Math.min(hi, lift))), r: 4.5, style: `fill: ${sig ? 'var(--bad)' : 'var(--accent)'}` }, s);
      });
    }
    function render() {
      if (ei >= exps.length) return summary();
      const E = exps[ei], r = E.d[day - 1];
      const pa = r.ca / r.n, pb = r.cb / r.n;
      root.innerHTML = `<div class="round-head" style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px"><b>Experiment ${ei + 1} of ${exps.length} · day ${day} of ${DAYS}</b><span class="score"><span>shipped wins: ${tally.rightShip}</span><span>shipped duds: ${tally.falseShip}</span><span>missed wins: ${tally.missed}</span></span></div>
        <div class="chart" data-chart style="margin-top:6px"></div>
        <div class="readouts"><div class="readout"><b>${(100 * pa).toFixed(2)}% → ${(100 * pb).toFixed(2)}%</b><span>click rate A → B</span></div><div class="readout"><b>${(pb / pa - 1 >= 0 ? '+' : '') + (100 * (pb / pa - 1)).toFixed(1)}%</b><span>observed lift</span></div><div class="readout"><b style="color:${r.p < 0.05 ? 'var(--bad)' : 'var(--ink)'}">${r.p < 0.001 ? '< 0.001' : r.p.toFixed(3)}</b><span>p-value today</span></div></div>
        ${decided ? `<div class="debrief"><p>${decided}</p><div class="btns"><button class="btn primary" data-next>${ei < exps.length - 1 ? 'Next experiment' : 'See the results'}</button></div></div>`
          : `<div class="btns">${day < DAYS ? '<button class="btn" data-day>Next day</button>' : ''}<button class="btn primary" data-ship>Ship B</button><button class="btn" data-keep>Keep A</button></div>`}`;
      const box = root.querySelector('[data-chart]');
      chart(box, E.d, day, decided ? E : null);
      const on = (sel, f) => { const b = root.querySelector(sel); if (b) b.addEventListener('click', f); };
      on('[data-day]', () => { day++; render(); });
      on('[data-ship]', () => decide(true));
      on('[data-keep]', () => decide(false));
      on('[data-next]', () => { ei++; day = 1; decided = null; render(); });
    }
    function decide(ship) {
      const E = exps[ei], r = E.d[day - 1];
      tally.days += day;
      let msg;
      if (ship && E.real) { tally.rightShip++; msg = `<b>Correct: B really was +10% better.</b>`; }
      else if (ship && !E.real) { tally.falseShip++; if (day < DAYS) tally.earlyNullShip++; msg = `<b>B was identical to A.</b> You shipped a dud on day ${day}${r.p < 0.05 ? `, when p = ${r.p.toFixed(3)} had dipped below 0.05 by chance` : ''}.`; }
      else if (!ship && E.real) { tally.missed++; msg = `<b>B was really +10% better</b>, and you kept A.`; }
      else { tally.rightKeep++; msg = `<b>Correct: B was identical to A.</b>`; }
      const ever = E.d.findIndex((x) => x.p < 0.05);
      msg += ` The dashed green line shows the true effect. ${ever >= 0 ? `This test first showed p &lt; 0.05 on day ${ever + 1}.` : 'This test never reached p &lt; 0.05.'} On day 14 it ended at p = ${E.d[DAYS - 1].p.toFixed(3)}.`;
      decided = msg;
      render();
    }
    function summary() {
      root.innerHTML = `<div class="debrief" style="border-top:none;margin-top:0;padding-top:0"><p><b>You shipped ${tally.rightShip} of 4 real wins and ${tally.falseShip} of 4 duds</b>, deciding after ${(tally.days / exps.length).toFixed(1)} days on average.</p>
        <p>Now the rule most dashboards invite: check every day, ship the first time B looks significantly better. Replay it on 2,000 experiments where B is identical to A:</p>
        <div class="btns"><button class="btn primary" data-sim>Replay 2,000 null experiments</button><button class="btn" data-again>Play again</button></div><div data-simout></div></div>`;
      root.querySelector('[data-again]').addEventListener('click', newGame);
      root.querySelector('[data-sim]').addEventListener('click', (e) => {
        e.target.textContent = 'Replaying…';
        setTimeout(() => {
          let peekShip = 0, peekAny = 0, fixed = 0;
          for (let t = 0; t < 2000; t++) {
            let ca = 0, cb = 0, shipped = false, any = false;
            for (let dd = 1; dd <= DAYS; dd++) {
              ca += clicks(NPD, BASE); cb += clicks(NPD, BASE);
              const r = ztest(ca, dd * NPD, cb, dd * NPD);
              if (r.p < 0.05) { any = true; if (r.z > 0) shipped = true; }
              if (dd === DAYS && r.p < 0.05) fixed++;
            }
            if (shipped) peekShip++; if (any) peekAny++;
          }
          e.target.textContent = 'Replay again';
          root.querySelector('[data-simout]').innerHTML = `<div class="readouts"><div class="readout"><b>${(fixed / 20).toFixed(1)}%</b><span>significant at day 14 only (planned)</span></div><div class="readout"><b>${(peekAny / 20).toFixed(1)}%</b><span>significant on some day (either direction)</span></div><div class="readout"><b style="color:var(--bad)">${(peekShip / 20).toFixed(1)}%</b><span>would have shipped a dud</span></div></div>
            <p>With one planned look, the false-positive rate is the promised 5%. With 14 looks it's roughly four times that: each look is another chance for noise to cross the line, and you stop on the first crossing. Fixes: decide only at the planned horizon, or use a sequential method built for repeated looks (alpha spending, always-valid p-values).</p>`;
        }, 30);
      });
    }
    A.responsive(root, (w) => { W = Math.max(280, w - 2); if (exps && ei < exps.length) render(); });
    newGame();
  })();

  const ex = window.AB_EXERCISES;
  A.mountExercise(document.getElementById('ex-z'), ex.z);
  A.mountExercise(document.getElementById('ex-ss'), ex.ss);
})();
