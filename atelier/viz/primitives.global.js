(function () {
  const SVG_NS = 'http://www.w3.org/2000/svg';

  const palette = {
    fg: '#e8e4de',
    dim: '#8a8680',
    accent: '#c9a96e',
    accent2: '#6ea5c9',
    accent3: '#c96e8a',
    line: 'rgba(255,255,255,0.10)',
    panel: 'rgba(255,255,255,0.04)',
    panelStrong: 'rgba(255,255,255,0.06)',
    grid: 'rgba(255,255,255,0.05)',
    bg: '#0d0d14',
  };

  function clamp(value, min = 0, max = 1) {
    return Math.min(max, Math.max(min, value));
  }

  function pct(value, digits = 0) {
    return `${(value * 100).toFixed(digits)}%`;
  }

  function num(value, digits = 2) {
    return Number(value).toFixed(digits);
  }

  function bits(value) {
    return `${Number(value).toFixed(2)} bits`;
  }

  function clearNode(node) {
    node.innerHTML = '';
  }

  function createSvg(root, viewBox = '0 0 740 360') {
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', viewBox);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    svg.classList.add('hybrid-svg');
    root.appendChild(svg);
    return svg;
  }

  function svgEl(name, attrs = {}, text = '') {
    const node = document.createElementNS(SVG_NS, name);
    Object.entries(attrs).forEach(([key, value]) => {
      if (value !== undefined && value !== null) node.setAttribute(key, String(value));
    });
    if (text) node.textContent = text;
    return node;
  }

  function append(parent, name, attrs = {}, text = '') {
    const node = svgEl(name, attrs, text);
    parent.appendChild(node);
    return node;
  }

  function group(parent, className) {
    const g = svgEl('g', className ? { class: className } : {});
    parent.appendChild(g);
    return g;
  }

  function html(tag, className, text = '') {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
  }

  function setTakeaway(context, text) {
    context.takeaway.textContent = text;
  }

  function setMetrics(context, rows) {
    context.metrics.innerHTML = rows
      .map(
        ([label, value]) => `
        <div class="metric-card">
          <span class="metric-label">${label}</span>
          <span class="metric-value">${value}</span>
        </div>
      `
      )
      .join('');
  }

  function buildHybridRoot(stageRoot, layout = 'standard') {
    clearNode(stageRoot);
    stageRoot.className = `viz-stage-root hybrid-stage hybrid-layout-${layout}`;
    return stageRoot;
  }

  function buildHeader(root, title, note) {
    const header = html('div', 'hv-header');
    if (title) header.appendChild(html('p', 'hv-title', title));
    if (note) header.appendChild(html('p', 'hv-note', note));
    root.appendChild(header);
    return header;
  }

  function buildGrid(root, className = '') {
    const node = html('div', `hv-grid ${className}`.trim());
    root.appendChild(node);
    return node;
  }

  function buildCard(root, title, className = '') {
    const card = html('div', `hv-card ${className}`.trim());
    if (title) card.appendChild(html('p', 'hv-card-label', title));
    root.appendChild(card);
    return card;
  }

  function buildBarRow(root, label, value, color, caption = '') {
    const row = html('div', 'hv-bar-row');
    row.title = `${label}: ${caption || pct(value)}`;
    const top = html('div', 'hv-bar-top');
    top.appendChild(html('span', 'hv-bar-label', label));
    top.appendChild(html('span', 'hv-bar-value', caption || pct(value)));
    const track = html('div', 'hv-bar-track');
    const fill = html('div', 'hv-bar-fill');
    fill.style.width = `${clamp(value) * 100}%`;
    fill.style.background = color;
    track.appendChild(fill);
    row.appendChild(top);
    row.appendChild(track);
    root.appendChild(row);
    return row;
  }

  function buildBadge(root, text, className = '') {
    const badge = html('div', `hv-badge ${className}`.trim(), text);
    root.appendChild(badge);
    return badge;
  }

  function ensureD3() {
    if (!window.d3) throw new Error('D3 not loaded');
    return window.d3;
  }

  window.AtelierVizPrimitives = {
    append,
    bits,
    buildBadge,
    buildBarRow,
    buildCard,
    buildGrid,
    buildHeader,
    buildHybridRoot,
    clamp,
    createSvg,
    ensureD3,
    group,
    html,
    num,
    pct,
    palette,
    setMetrics,
    setTakeaway,
    svgEl,
  };
})();
