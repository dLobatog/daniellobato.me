/* Lesson 5: conditioning as selecting the evidence group, drawn as 1,000 dots that move. */
(function () {
  const A = window.Atelier;
  const { svg } = A;
  const N = 1000, COLS = 40;
  const pct = (x, d = 1) => (100 * x).toFixed(d) + '%';
  const fmtInt = (x) => Math.round(x).toLocaleString('en-US');

  renderMathInElement(document.body, {
    delimiters: [{ left: '\\[', right: '\\]', display: true }, { left: '\\(', right: '\\)', display: false }],
    throwOnError: false,
  });

  // A fixed shuffle so false alarms land in scattered, stable places.
  const RANK = (() => {
    let s = 42; const r = () => { s = (s * 1664525 + 1013904223) % 4294967296; return s / 4294967296; };
    const idx = Array.from({ length: N }, (_, i) => i);
    for (let i = N - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    const rank = new Array(N); idx.forEach((v, k) => { rank[v] = k; });
    return rank;
  })();

  /* 1,000 dots: stage 0 population, stage 1 flagged ringed, stage 2 flagged gathered into their own group. */
  function dotGrid(box, countsBox) {
    let W = 700, stage = 0, params = { prior: 0.02, sens: 0.9, fpr: 0.05 };
    let dots = [], s = null, label = null, cur = null, anim = 0;
    function counts() {
      const nS = Math.round(N * params.prior), nL = N - nS;
      const nSF = Math.round(nS * params.sens), nLF = Math.round(nL * params.fpr);
      return { nS, nL, nSF, nLF };
    }
    function targets() {
      const { nS, nSF, nLF } = counts();
      const legitRanks = [];
      for (let i = nS; i < N; i++) legitRanks.push([RANK[i], i]);
      legitRanks.sort((a, b) => a[0] - b[0]);
      const flaggedLegit = new Set(legitRanks.slice(0, nLF).map((x) => x[1]));
      const sp = Math.min(13, (W - 8) / COLS), r = sp * 0.36;
      const gridH = Math.ceil(N / COLS) * sp;
      const flaggedOrder = [];
      for (let i = 0; i < N; i++) { const spam = i < nS; const flag = spam ? i < nSF : flaggedLegit.has(i); if (flag) flaggedOrder.push(i); }
      flaggedOrder.sort((a, b) => (a < nS ? 0 : 1) - (b < nS ? 0 : 1) || a - b);
      const gpos = new Map(flaggedOrder.map((i, k) => [i, k]));
      const groupTop = gridH + 34;
      const out = [];
      for (let i = 0; i < N; i++) {
        const spam = i < nS, flag = spam ? i < nSF : flaggedLegit.has(i);
        let x = 4 + (i % COLS) * sp + sp / 2, y = Math.floor(i / COLS) * sp + sp / 2;
        let op = spam ? 1 : 0.55;
        if (stage === 2) {
          if (flag) { const k = gpos.get(i); x = 4 + (k % COLS) * sp + sp / 2; y = groupTop + Math.floor(k / COLS) * sp + sp / 2; op = spam ? 1 : 0.75; }
          else op = 0.08;
        }
        out.push({ x, y, op, spam, ring: stage >= 1 && flag, r });
      }
      const H = stage === 2 ? groupTop + Math.max(1, Math.ceil(flaggedOrder.length / COLS)) * sp + 8 : gridH + 8;
      return { out, H, groupTop, nF: flaggedOrder.length };
    }
    function build() {
      box.innerHTML = '';
      s = svg('svg', { width: W, height: 10, viewBox: `0 0 ${W} 10`, role: 'img', 'aria-label': 'One thousand messages as dots' }, box);
      dots = Array.from({ length: N }, () => svg('circle', {}, s));
      label = svg('text', { x: 4, y: 0, 'font-size': 13, 'font-weight': 600, style: 'fill: var(--ink)' }, s);
      cur = null;
    }
    function paint(state, H, groupTop, nF) {
      s.setAttribute('height', H); s.setAttribute('viewBox', `0 0 ${W} ${H}`);
      state.forEach((d, i) => {
        const c = dots[i];
        c.setAttribute('cx', d.x.toFixed(1)); c.setAttribute('cy', d.y.toFixed(1)); c.setAttribute('r', d.r.toFixed(2));
        c.setAttribute('style', `fill: ${d.spam ? 'var(--w3)' : 'var(--ink-3)'}; opacity: ${d.op.toFixed(2)}; ${d.ring ? 'stroke: var(--ink); stroke-width: 1.4' : ''}`);
      });
      if (stage === 2) { label.setAttribute('y', groupTop - 10); label.textContent = `Flagged: ${nF} messages. This is the new "everything".`; }
      else label.textContent = '';
    }
    function update(animate = true) {
      const { out, H, groupTop, nF } = targets();
      const from = cur;
      cancelAnimationFrame(anim);
      if (!from || !animate) { cur = out; paint(out, H, groupTop, nF); renderCounts(); return; }
      const t0 = performance.now(), ms = 750;
      const Hmax = Math.max(H, parseFloat(s.getAttribute('height')) || H);
      (function frame(now) {
        const u = Math.min(1, (now - t0) / ms), e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
        const mid = out.map((d, i) => ({ ...d, x: from[i].x + (d.x - from[i].x) * e, y: from[i].y + (d.y - from[i].y) * e, op: from[i].op + (d.op - from[i].op) * e }));
        paint(mid, u < 1 ? Hmax : H, groupTop, nF);
        if (u < 1) anim = requestAnimationFrame(frame); else cur = out;
      })(t0);
      renderCounts();
    }
    function renderCounts() {
      if (!countsBox) return;
      const { nS, nL, nSF, nLF } = counts();
      let h = `<div class="readout"><b style="color:var(--w3)">${nS}</b><span>spam</span></div><div class="readout"><b>${nL}</b><span>legitimate</span></div>`;
      if (stage >= 1) h += `<div class="readout"><b style="color:var(--w3)">${nSF}</b><span>spam, flagged</span></div><div class="readout"><b>${nLF}</b><span>legitimate, flagged</span></div>`;
      if (stage >= 2) h += `<div class="readout"><b>${nSF} / ${nSF + nLF} = ${pct(nSF / Math.max(1, nSF + nLF))}</b><span>P(spam | flagged)</span></div>`;
      countsBox.innerHTML = h;
    }
    const rr = A.responsive(box, (w) => { W = Math.max(280, Math.min(560, w)); build(); update(false); });
    A.onTheme(() => update(false));
    return {
      setStage(k) { stage = k; if (s) update(); },
      setParams(p) { params = { ...params, ...p }; if (s) update(false); },
      counts, get stage() { return stage; }, rr,
    };
  }

  /* ── 1–2. Build the counts, then condition ──── */
  const g1 = dotGrid(document.getElementById('grid'), document.getElementById('grid-counts'));
  const stageBtns = document.getElementById('stage-btns');
  const STAGES = ['All 1,000', 'Apply the filter', 'Keep only flagged'];
  function drawStageBtns() {
    stageBtns.innerHTML = STAGES.map((t, i) => `<button class="btn${g1.stage === i ? ' primary' : ''}" data-s="${i}">${i + 1}. ${t}</button>`).join('');
    stageBtns.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => { g1.setStage(+b.dataset.s); drawStageBtns(); }));
  }
  drawStageBtns();
  document.querySelectorAll('.check[data-stage]').forEach((c) => c.addEventListener('answered', (e) => {
    const k = +c.dataset.stage;
    if (k > g1.stage) { g1.setStage(k); drawStageBtns(); }
  }));

  /* ── 3. Same filter, different inboxes ──────── */
  (function explorer() {
    const g2 = dotGrid(document.getElementById('grid2'), null);
    g2.setStage(2);
    const $ = (id) => document.getElementById(id);
    const priorOf = (v) => 0.001 * Math.pow(500, v);
    function upd() {
      const prior = priorOf(+$('prior').value), sens = +$('sens').value, fpr = +$('fpr').value;
      $('prior-out').textContent = pct(prior, prior < 0.01 ? 2 : 1);
      $('sens-out').textContent = pct(sens, 0);
      $('fpr-out').textContent = pct(fpr, 1);
      g2.setParams({ prior, sens, fpr });
      const post = (sens * prior) / (sens * prior + fpr * (1 - prior) || 1);
      $('grid2-out').innerHTML = `<div class="readout"><b>${pct(sens, 0)}</b><span>recall, P(flag | spam)</span></div><div class="readout"><b style="color:var(--w3)">${pct(post)}</b><span>precision, P(spam | flag)</span></div><div class="readout"><b>${pct(sens * prior + fpr * (1 - prior))}</b><span>of all mail gets flagged</span></div>`;
    }
    ['prior', 'sens', 'fpr'].forEach((id) => $(id).addEventListener('input', upd));
    upd();
  })();

  /* ── 4. Game: call the odds ─────────────────── */
  (function oddsGame() {
    const root = document.getElementById('odds-game');
    const A1 = { name: 'filter A', s: 0.9, f: 0.05 }, B = { name: 'filter B', s: 0.8, f: 0.1 };
    const CASES = [
      { text: 'Inbox with <b>2%</b> spam. Filter A (recall 90%, false-positive rate 5%) <b>flagged</b> the message.', prior: 0.02, ev: [[A1, true]], M: 100000 },
      { text: 'Same inbox, same filter. This time it did <b>not</b> flag the message.', prior: 0.02, ev: [[A1, false]], M: 100000 },
      { text: 'Same inbox. Filter A <b>and</b> an independent filter B (recall 80%, false-positive rate 10%) both flagged it.', prior: 0.02, ev: [[A1, true], [B, true]], M: 100000 },
      { text: 'A different inbox segment where <b>30%</b> of mail is spam. Filter A flagged the message.', prior: 0.3, ev: [[A1, true]], M: 100000 },
      { text: 'Card transactions: <b>0.1%</b> are fraud. A detector with recall 99% and false-positive rate 1% flags one.', prior: 0.001, ev: [[{ s: 0.99, f: 0.01 }, true]], M: 1000000, pos: 'fraud', neg: 'legitimate' },
    ];
    const truth = (c) => {
      let a = c.prior, b = 1 - c.prior;
      c.ev.forEach(([t, flag]) => { a *= flag ? t.s : 1 - t.s; b *= flag ? t.f : 1 - t.f; });
      return a / (a + b);
    };
    const klBits = (t, y) => t * Math.log2(t / y) + (1 - t) * Math.log2((1 - t) / (1 - y));
    let ci = 0, total = 0, neglect = 0, locked = false, guess = 50;
    function render() {
      const c = CASES[ci], t = truth(c);
      const P = c.pos || 'spam', Nn = c.neg || 'legitimate';
      if (!locked) {
        root.innerHTML = `<div class="round-head" style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px"><b>Case ${ci + 1} of ${CASES.length}</b><span class="small muted">${ci ? `bits lost so far: ${total.toFixed(3)}` : ''}</span></div>
          <p class="case">${c.text} Probability it's ${P}?</p>
          <div class="ctl"><label for="guess">your probability</label><input id="guess" type="range" min="0.5" max="99.5" step="0.5" value="${guess}"><output id="guess-out">${guess}%</output></div>
          <div class="btns"><button class="btn primary" data-lock>Lock in</button></div>`;
        const inp = root.querySelector('#guess');
        inp.addEventListener('input', () => { guess = +inp.value; root.querySelector('#guess-out').textContent = guess + '%'; });
        root.querySelector('[data-lock]').addEventListener('click', () => {
          locked = true;
          const y = guess / 100;
          const lost = klBits(t, y);
          total += lost;
          const firstFlag = c.ev.find(([, f]) => f);
          const naive = firstFlag ? firstFlag[0].s : 1 - c.ev[0][0].s;
          neglect += klBits(t, Math.min(0.995, Math.max(0.005, naive)));
          render();
        });
        return;
      }
      let a = c.M * c.prior, b = c.M * (1 - c.prior);
      const lines = [`Out of ${fmtInt(c.M)}: <b style="color:var(--w3)">${fmtInt(a)}</b> ${P}, ${fmtInt(b)} ${Nn}.`];
      c.ev.forEach(([tt, flag]) => {
        a *= flag ? tt.s : 1 - tt.s; b *= flag ? tt.f : 1 - tt.f;
        lines.push(`${flag ? 'Flagged' : 'Not flagged'}${tt.name ? ' by ' + tt.name : ''}: <b style="color:var(--w3)">${fmtInt(a)}</b> ${P}, ${fmtInt(b)} ${Nn}.`);
      });
      const y = guess / 100, lost = klBits(t, y);
      const wA = Math.max(2, 100 * t);
      root.innerHTML = `<div class="round-head" style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px"><b>Case ${ci + 1} of ${CASES.length}</b><span class="small muted">bits lost so far: ${total.toFixed(3)}</span></div>
        <p class="case">${c.text}</p>
        <div class="readouts"><div class="readout"><b>${guess}%</b><span>you said</span></div><div class="readout"><b style="color:var(--w3)">${pct(t)}</b><span>Bayes says</span></div><div class="readout"><b>${lost.toFixed(3)}</b><span>bits lost this case</span></div></div>
        <div class="comp"><div style="width:${wA}%;background:var(--w3)">${t > 0.08 ? P : ''}</div><div style="width:${100 - wA}%;background:var(--ink-3)">${1 - t > 0.12 ? Nn : ''}</div></div>
        <p class="small muted" style="margin-top:0">The group that matches the evidence, split by what the messages really are.</p>
        <p class="small">${lines.join('<br>')}<br>So P(${P} | evidence) = ${fmtInt(a)} / ${fmtInt(a + b)} = <b>${pct(t, 2)}</b>.</p>
        <div class="btns">${ci < CASES.length - 1 ? '<button class="btn primary" data-next>Next case</button>' : ''}</div>
        ${ci === CASES.length - 1 ? `<div class="debrief"><p><b>Total: ${total.toFixed(3)} bits lost.</b> Answering with the likelihood of the evidence every time (base-rate neglect: 90% when flagged, 10% when not) would have lost <b>${neglect.toFixed(3)}</b> bits. Cases 2 and 5 are where intuition fails most. A missing flag is strong evidence, and a flag on a rare event is weak evidence.</p><div class="btns"><button class="btn" data-again>Play again</button></div></div>` : ''}`;
      const nx = root.querySelector('[data-next]');
      if (nx) nx.addEventListener('click', () => { ci++; locked = false; guess = 50; render(); });
      const ag = root.querySelector('[data-again]');
      if (ag) ag.addEventListener('click', () => { ci = 0; total = 0; neglect = 0; locked = false; guess = 50; render(); });
    }
    render();
  })();

  const ex = window.BAYES_EXERCISES;
  A.mountExercise(document.getElementById('ex-post'), ex.post);
  A.mountExercise(document.getElementById('ex-recal'), ex.recal);
})();
