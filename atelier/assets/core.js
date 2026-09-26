/* Shared page behaviour: theme toggle, commit-then-reveal checks, follow-up reveals, SVG helper. */
(function () {
  const A = (window.Atelier = window.Atelier || {});
  const themeListeners = [];

  function store(k, v) {
    try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; }
  }

  A.onTheme = (f) => themeListeners.push(f);
  A.cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function applyTheme(t) {
    if (t) document.documentElement.setAttribute('data-theme', t);
    else document.documentElement.removeAttribute('data-theme');
    const btn = document.querySelector('.theme-btn');
    if (btn) btn.textContent = isDark() ? 'Light' : 'Dark';
    themeListeners.forEach((f) => f());
  }
  function isDark() {
    const t = document.documentElement.getAttribute('data-theme');
    if (t) return t === 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  }

  A.svg = function (tag, attrs, parent) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const k in attrs || {}) {
      if (k === 'text') el.textContent = attrs[k];
      else el.setAttribute(k, attrs[k]);
    }
    if (parent) parent.appendChild(el);
    return el;
  };

  /* Re-render a chart whenever its container changes width. */
  A.responsive = function (container, render) {
    let w = 0;
    const ro = new ResizeObserver(() => {
      const nw = Math.round(container.clientWidth);
      if (nw !== w) { w = nw; render(nw); }
    });
    ro.observe(container);
    // Render now if the container is already laid out; the observer handles later size changes.
    const now = Math.round(container.clientWidth);
    if (now > 0) { w = now; render(now); }
    return () => render(Math.round(container.clientWidth));
  };

  A.fmt = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : '∞');

  /* Checks: commit to an answer, then see the reasoning. A check with data-gate also hides the rest of its
     section until it is answered (or skipped), so the question comes before the explanation. */
  function initChecks() {
    const page = location.pathname;
    document.querySelectorAll('.check').forEach((box, idx) => {
      const key = `atelier:check:${page}:${idx}`;
      const opts = box.querySelectorAll('.opt');
      const locked = [];
      if (box.hasAttribute('data-gate')) {
        let el = box.nextElementSibling;
        while (el && !el.matches('.check[data-gate]')) { locked.push(el); el = el.nextElementSibling; }
      }
      const hint = document.createElement('div');
      hint.className = 'gate-hint';
      hint.innerHTML = (locked.length ? 'Answer to continue. ' : '') + '<button type="button" class="linkish">Skip: show me the answer</button>';
      box.appendChild(hint);
      locked.forEach((e) => e.classList.add('locked'));

      function finish(chosen, restoring) {
        if (box.classList.contains('done')) return;
        opts.forEach((o) => {
          o.disabled = true;
          if (o.hasAttribute('data-correct')) o.classList.add('right');
        });
        if (chosen && !chosen.hasAttribute('data-correct')) chosen.classList.add('wrong');
        box.classList.add('done');
        hint.remove();
        locked.forEach((e) => e.classList.remove('locked'));
        if (!restoring) store(key, chosen ? (chosen.hasAttribute('data-correct') ? 'right' : 'wrong') : 'skipped');
        box.dispatchEvent(new CustomEvent('answered', { detail: { correct: !!chosen && chosen.hasAttribute('data-correct'), restoring: !!restoring } }));
        window.dispatchEvent(new Event('resize'));
      }
      opts.forEach((b) => b.addEventListener('click', () => finish(b)));
      hint.querySelector('button').addEventListener('click', () => finish(null));
      if (store(key)) finish(null, true);
    });
    document.querySelectorAll('.followup').forEach((f) => {
      const b = f.querySelector('button');
      b.addEventListener('click', () => {
        f.classList.toggle('open');
        b.textContent = f.classList.contains('open') ? 'Hide key points' : 'Reveal key points';
      });
    });
    const reset = document.querySelector('[data-reset-checks]');
    if (reset) reset.addEventListener('click', () => {
      document.querySelectorAll('.check').forEach((_, idx) => { try { localStorage.removeItem(`atelier:check:${page}:${idx}`); } catch (e) { /* ignore */ } });
      location.reload();
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    const btn = document.querySelector('.theme-btn');
    const saved = store('atelier:theme');
    if (saved === 'light' || saved === 'dark') applyTheme(saved);
    else if (btn) btn.textContent = isDark() ? 'Light' : 'Dark';
    if (btn) btn.addEventListener('click', () => {
      const t = isDark() ? 'light' : 'dark';
      store('atelier:theme', t);
      applyTheme(t);
    });
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => applyTheme(document.documentElement.getAttribute('data-theme')));
    initChecks();
  });
})();
