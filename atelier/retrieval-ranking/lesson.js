/* Lesson 9: a simulated two-stage recommender under a latency budget, plus ranking metrics. */
(function () {
  const A = window.Atelier;
  const { svg } = A;
  function mulberry32(a) { return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

  renderMathInElement(document.body, {
    delimiters: [{ left: '\\[', right: '\\]', display: true }, { left: '\\(', right: '\\)', display: false }],
    throwOnError: false,
  });

  /* ── 3. Budget the funnel ───────────────────── */
  (function funnel() {
    const N = 5000, R = 10, USERS = 120, BUDGET = 40, TARGET = 6;
    const RET = { 16: { mu: 1.6, ms: 4 }, 64: { mu: 2.2, ms: 10 }, 256: { mu: 2.7, ms: 22 } };
    const RANK = { small: { mu: 2.2, ms: 0.008 }, big: { mu: 3.0, ms: 0.04 } };
    const KS = Array.from({ length: 20 }, (_, i) => Math.round(10 * Math.pow(200, i / 19)));
    // Common random numbers: the same noise for every configuration, so differences come from the choices.
    const r = mulberry32(2024);
    const gauss = () => { const u = Math.max(r(), 1e-12), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const Z1 = Array.from({ length: USERS }, () => Float64Array.from({ length: N }, gauss));
    const Z2 = Array.from({ length: USERS }, () => Float64Array.from({ length: N }, gauss));
    const cache = new Map();
    function sim(dim, K, rk) {
      const key = `${dim}|${K}|${rk}`;
      if (cache.has(key)) return cache.get(key);
      const mu = RET[dim].mu, mr = RANK[rk].mu;
      let hits = 0, rec = 0;
      const idx = new Int32Array(N);
      for (let u = 0; u < USERS; u++) {
        const z = Z1[u], z2 = Z2[u];
        for (let i = 0; i < N; i++) idx[i] = i;
        const sc = new Float64Array(N);
        for (let i = 0; i < N; i++) sc[i] = (i < R ? mu : 0) + z[i];
        const cand = Array.from(idx).sort((a, b) => sc[b] - sc[a]).slice(0, K);
        rec += cand.filter((i) => i < R).length / R;
        const top = cand.map((i) => [i, (i < R ? mr : 0) + z2[i]]).sort((a, b) => b[1] - a[1]).slice(0, 10);
        hits += top.filter(([i]) => i < R).length;
      }
      const out = { recall: rec / USERS, hits: hits / USERS, lat: RET[dim].ms + K * RANK[rk].ms };
      cache.set(key, out);
      return out;
    }
    const st = { dim: 64, ki: 10, rk: 'small' };
    const box = document.getElementById('funnel'), out = document.getElementById('g-out'), deb = document.getElementById('g-debrief');
    let solved = false, W = 700;
    function draw(res) {
      box.innerHTML = '';
      const K = KS[st.ki];
      const H = 170, s = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': 'Funnel from corpus to candidates to top ten' }, box);
      const stages = [
        { label: 'corpus', n: N, rel: R },
        { label: `${K} candidates`, n: K, rel: res.recall * R },
        { label: 'top 10 shown', n: 10, rel: res.hits },
      ];
      const colW = W / 3, maxH = 110;
      stages.forEach((sg, i) => {
        const h = Math.max(10, (Math.log10(sg.n) / Math.log10(N)) * maxH);
        const x = i * colW + colW * 0.12, w = colW * 0.76, y = 20 + (maxH - h) / 2;
        svg('rect', { x, y, width: w, height: h, rx: 6, style: 'fill: var(--surface-2); stroke: var(--rule)' }, s);
        const rw = Math.min(w, Math.max(3, (sg.rel / R) * w));
        svg('rect', { x, y: y + h - 8, width: rw, height: 8, rx: 3, style: 'fill: var(--good)' }, s);
        svg('text', { x: x + w / 2, y: y + h / 2 + 2, 'text-anchor': 'middle', 'font-size': 13, 'font-weight': 600, style: 'fill: var(--ink)', text: sg.label }, s);
        svg('text', { x: x + w / 2, y: 20 + maxH + 24, 'text-anchor': 'middle', 'font-size': 12.5, style: 'fill: var(--good)', text: `${sg.rel.toFixed(1)} of 10 relevant` }, s);
        if (i < 2) svg('text', { x: (i + 1) * colW, y: 20 + maxH / 2 + 5, 'text-anchor': 'middle', 'font-size': 18, style: 'fill: var(--ink-3)', text: '→' }, s);
      });
    }
    function update() {
      const K = KS[st.ki];
      document.getElementById('g-k-out').textContent = K;
      const res = sim(st.dim, K, st.rk);
      const over = res.lat > BUDGET;
      draw(res);
      out.innerHTML = `<div class="readout"><b style="color:${over ? 'var(--bad)' : 'var(--ink)'}">${res.lat.toFixed(1)} ms</b><span>latency (budget ${BUDGET} ms)</span></div>
        <div class="readout"><b>${(100 * res.recall).toFixed(0)}%</b><span>retrieval recall@K</span></div>
        <div class="readout"><b style="color:${res.hits >= TARGET && !over ? 'var(--good)' : 'var(--ink)'}">${res.hits.toFixed(2)}</b><span>relevant in top 10 (goal ≥ ${TARGET})</span></div>`;
      if (!solved && !over && res.hits >= TARGET) {
        solved = true;
        deb.style.display = 'block';
        deb.innerHTML = `<p><b>Solved: ${res.hits.toFixed(2)} relevant in the top 10 at ${res.lat.toFixed(0)} ms.</b> Three things had to line up. First, the retriever's recall sets the ceiling, and the 16-d retriever can't reach the goal at any K. Second, the big ranker has to earn its cost: send it too many candidates and you blow the budget. Third, try the <em>small</em> ranker with a large K. It gets <em>worse</em> as K grows, because every extra candidate is another chance for an irrelevant item to outscore a relevant one. More candidates only help a ranker precise enough to handle them.</p>`;
      }
    }
    const seg = (id, key, parse) => document.querySelectorAll(`#${id} button`).forEach((b) => b.addEventListener('click', () => {
      st[key] = parse(b.dataset.v);
      document.querySelectorAll(`#${id} button`).forEach((x) => x.classList.toggle('on', x === b));
      update();
    }));
    seg('g-dim', 'dim', Number);
    seg('g-rank', 'rk', String);
    document.getElementById('g-k').addEventListener('input', (e) => { st.ki = +e.target.value; update(); });
    A.responsive(box, (w) => { W = Math.max(290, w); update(); });
  })();

  /* ── 5. Ranking metrics ─────────────────────── */
  (function metrics() {
    const TOTAL = 3;
    const rel = [true, false, false, true, false, false, false, false, false, false];
    const slots = document.getElementById('slots'), labels = document.getElementById('slotlabels'), out = document.getElementById('m-out');
    labels.innerHTML = rel.map((_, i) => `<span>${(1 / Math.log2(i + 2)).toFixed(2)}</span>`).join('');
    function draw() {
      slots.innerHTML = rel.map((r, i) => `<button class="${r ? 'rel' : ''}" data-i="${i}" aria-label="rank ${i + 1}: ${r ? 'relevant' : 'not relevant'}">${i + 1}</button>`).join('');
      slots.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
        const i = +b.dataset.i;
        const count = rel.filter(Boolean).length;
        if (!rel[i] && count >= TOTAL) return;
        rel[i] = !rel[i];
        draw();
      }));
      const dcg = rel.reduce((s, r, i) => s + (r ? 1 / Math.log2(i + 2) : 0), 0);
      let idcg = 0; for (let i = 0; i < TOTAL; i++) idcg += 1 / Math.log2(i + 2);
      const hits = rel.filter(Boolean).length, first = rel.indexOf(true);
      out.innerHTML = `<div class="readout"><b>${(dcg / idcg).toFixed(3)}</b><span>NDCG@10</span></div><div class="readout"><b>${first < 0 ? '0' : (1 / (first + 1)).toFixed(3)}</b><span>MRR</span></div><div class="readout"><b>${hits} / ${TOTAL}</b><span>recall@10</span></div><div class="readout"><b>${(hits / 10).toFixed(1)}</b><span>precision@10</span></div>`;
    }
    draw();
  })();

  const ex = window.RR_EXERCISES;
  A.mountExercise(document.getElementById('ex-ndcg'), ex.ndcg);
  A.mountExercise(document.getElementById('ex-inbatch'), ex.inbatch);
})();
