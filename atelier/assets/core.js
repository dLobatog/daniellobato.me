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
    return () => render(Math.round(container.clientWidth));
  };

  A.fmt = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : '∞');

  function initChecks() {
    document.querySelectorAll('.check').forEach((box) => {
      const opts = box.querySelectorAll('.opt');
      opts.forEach((b) => b.addEventListener('click', () => {
        if (box.classList.contains('done')) return;
        opts.forEach((o) => {
          o.disabled = true;
          if (o.hasAttribute('data-correct')) o.classList.add('right');
        });
        if (!b.hasAttribute('data-correct')) b.classList.add('wrong');
        box.classList.add('done');
        box.dispatchEvent(new CustomEvent('answered', { detail: { correct: b.hasAttribute('data-correct') } }));
      }));
    });
    document.querySelectorAll('.followup').forEach((f) => {
      const b = f.querySelector('button');
      b.addEventListener('click', () => {
        f.classList.toggle('open');
        b.textContent = f.classList.contains('open') ? 'Hide key points' : 'Reveal key points';
      });
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
