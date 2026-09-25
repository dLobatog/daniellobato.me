(() => {
  const root = document.documentElement;
  let saved;
  try { saved = localStorage.getItem('atelier-reading-theme'); } catch (_) { /* Storage is optional. */ }
  root.dataset.readingTheme = saved === 'dark' ? 'dark' : 'light';
  document.addEventListener('DOMContentLoaded', () => {
    const nav = document.querySelector('.site-nav');
    if (!nav) return;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'reading-theme-toggle';
    const sync = () => {
      const dark = root.dataset.readingTheme === 'dark';
      button.textContent = dark ? 'Light reading' : 'Dark reading';
      button.setAttribute('aria-label', `Switch to ${dark ? 'light' : 'dark'} reading theme`);
    };
    button.addEventListener('click', () => {
      root.dataset.readingTheme = root.dataset.readingTheme === 'dark' ? 'light' : 'dark';
      try { localStorage.setItem('atelier-reading-theme', root.dataset.readingTheme); } catch (_) { /* Keep working without storage. */ }
      sync();
    });
    sync();
    nav.append(button);
  });
})();
