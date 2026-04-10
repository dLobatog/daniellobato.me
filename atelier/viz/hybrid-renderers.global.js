(function () {
const { append, bits, buildBadge, buildBarRow, buildCard, buildGrid, buildHeader, buildHybridRoot, clamp, createSvg, ensureD3, group, html, num, pct, palette, setMetrics, setTakeaway, svgEl } = window.AtelierVizPrimitives;


function addGrid(svg, x, y, width, height, cols = 10, rows = 6) {
  const g = group(svg, 'hv-grid-layer');
  for (let i = 0; i <= cols; i += 1) {
    const px = x + (width * i) / cols;
    g.appendChild(svgEl('line', { x1: px, x2: px, y1: y, y2: y + height, stroke: palette.grid, 'stroke-width': 1 }));
  }
  for (let i = 0; i <= rows; i += 1) {
    const py = y + (height * i) / rows;
    g.appendChild(svgEl('line', { x1: x, x2: x + width, y1: py, y2: py, stroke: palette.grid, 'stroke-width': 1 }));
  }
}

function addText(parent, x, y, text, className = '', extra = {}) {
  const attrs = { x, y, class: className, ...extra };
  const node = svgEl('text', attrs, text);
  parent.appendChild(node);
  return node;
}

function addArrow(parent, fromX, fromY, toX, toY, color = palette.accent, curve = 0) {
  const path = curve
    ? `M ${fromX} ${fromY} C ${fromX + curve} ${fromY}, ${toX - curve} ${toY}, ${toX} ${toY}`
    : `M ${fromX} ${fromY} L ${toX} ${toY}`;
  const node = svgEl('path', {
    d: path,
    fill: 'none',
    stroke: color,
    'stroke-width': 2.2,
    'stroke-linecap': 'round',
    'stroke-linejoin': 'round',
    opacity: 0.9,
  });
  parent.appendChild(node);
  return node;
}

function buildInspector(root, initialDetail) {
  const inspector = html('div', 'hv-inspector');
  inspector.appendChild(html('p', 'hv-inline-label', 'Hover or click to inspect'));
  const title = html('p', 'hv-inspector-title', initialDetail.title);
  const body = html('p', 'hv-footnote', initialDetail.body);
  inspector.appendChild(title);
  inspector.appendChild(body);
  root.appendChild(inspector);
  return {
    set(detail) {
      title.textContent = detail.title;
      body.textContent = detail.body;
    },
  };
}

function addInteractiveNode(parent, { x, y, r = 22, label, color, caption = '', detail, ariaLabel }) {
  const target = group(parent, 'hv-node-target');
  target.setAttribute('tabindex', '0');
  target.setAttribute('role', 'button');
  target.setAttribute('aria-label', ariaLabel || `${detail.title}. ${detail.body}`);
  target.appendChild(svgEl('title', {}, ariaLabel || `${detail.title}: ${caption}`));
  target.appendChild(
    svgEl('circle', {
      cx: x,
      cy: y,
      r,
      fill: palette.panelStrong,
      stroke: color,
      'stroke-width': 1.4,
      class: 'hv-node-circle',
    })
  );
  addText(target, x - Math.max(8, label.length * 3.1), y + 6, label, 'hv-svg-symbol');
  if (caption) addText(target, x - Math.max(18, caption.length * 2.8), y + r + 18, caption, 'hv-svg-kicker');
  return { node: target, detail };
}

function addInteractiveBox(parent, { x, y, width, height, fill, stroke, label = '', caption = '', detail, ariaLabel }) {
  const target = group(parent, 'hv-node-target');
  target.setAttribute('tabindex', '0');
  target.setAttribute('role', 'button');
  target.setAttribute('aria-label', ariaLabel || `${detail.title}. ${detail.body}`);
  target.appendChild(svgEl('title', {}, ariaLabel || `${detail.title}: ${caption}`));
  target.appendChild(
    svgEl('rect', {
      x,
      y,
      width,
      height,
      rx: 16,
      fill,
      stroke,
      'stroke-width': 1.4,
      class: 'hv-region-box',
    })
  );
  if (label) addText(target, x + 12, y + 20, label, 'hv-svg-kicker');
  if (caption) addText(target, x + 12, y + 42, caption, 'hv-svg-caption');
  return { node: target, detail };
}

function addMeasureLine(parent, x1, y1, x2, y2, label, textX, textY, anchor = 'middle') {
  parent.appendChild(svgEl('line', { x1, y1, x2, y2, stroke: palette.dim, 'stroke-width': 1.1, opacity: 0.85 }));
  if (x1 === x2) {
    parent.appendChild(svgEl('line', { x1: x1 - 5, y1, x2: x1 + 5, y2: y1, stroke: palette.dim, 'stroke-width': 1.1, opacity: 0.85 }));
    parent.appendChild(svgEl('line', { x1: x2 - 5, y1: y2, x2: x2 + 5, y2, stroke: palette.dim, 'stroke-width': 1.1, opacity: 0.85 }));
  } else {
    parent.appendChild(svgEl('line', { x1, y1: y1 - 5, x2: x1, y2: y1 + 5, stroke: palette.dim, 'stroke-width': 1.1, opacity: 0.85 }));
    parent.appendChild(svgEl('line', { x1: x2, y1: y2 - 5, x2, y2: y2 + 5, stroke: palette.dim, 'stroke-width': 1.1, opacity: 0.85 }));
  }
  addText(parent, textX, textY, label, 'hv-svg-caption', { 'text-anchor': anchor });
}

function wireInspector(targets, inspector, initialIndex = 0) {
  if (!targets.length) return;
  const activate = (index) => {
    targets.forEach((target, targetIndex) => {
      target.node.classList.toggle('is-active', targetIndex === index);
    });
    inspector.set(targets[index].detail);
  };

  targets.forEach((target, index) => {
    target.node.addEventListener('mouseenter', () => activate(index));
    target.node.addEventListener('focus', () => activate(index));
    target.node.addEventListener('click', (event) => {
      event.stopPropagation();
      activate(index);
    });
    target.node.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        event.stopPropagation();
        activate(index);
      }
    });
  });

  activate(initialIndex);
}

function renderBayes(stageRoot, section, state, context) {
  const prior = state.prior;
  const sensitivity = state.sensitivity;
  const specificity = state.specificity;
  const falsePositiveRate = 1 - specificity;
  const tpMass = prior * sensitivity;
  const fpMass = (1 - prior) * falsePositiveRate;
  const fnMass = prior * (1 - sensitivity);
  const tnMass = (1 - prior) * specificity;
  const evidenceMass = tpMass + fpMass;
  const posterior = evidenceMass === 0 ? 0 : tpMass / evidenceMass;

  const root = buildHybridRoot(stageRoot, 'comparison');
  buildHeader(
    root,
    'Bayes is crop, then compare',
    'Start with the whole world. Highlight every way a positive test can happen. Then crop down to only those positives and ask how much of that cropped region still belongs to the hypothesis.'
  );
  const grid = buildGrid(root, 'hv-grid-bayes');

  const world = buildCard(grid, 'Whole possibility space');
  world.appendChild(
    html(
      'p',
      'hv-footnote',
      'Area is probability mass here. Wide columns mean common situations; tall highlighted slices mean a positive test is likely there.'
    )
  );
  const svg = createSvg(world, '0 0 440 316');
  const worldX = 64;
  const worldY = 58;
  const worldW = 308;
  const worldH = 210;
  const hW = Math.max(18, worldW * prior);
  const notHW = Math.max(18, worldW - hW);
  const tpH = Math.max(10, worldH * sensitivity);
  const fpH = Math.max(10, worldH * falsePositiveRate);

  addGrid(svg, worldX, worldY, worldW, worldH, 6, 6);
  svg.appendChild(svgEl('rect', { x: worldX, y: worldY, width: hW, height: worldH, rx: 18, fill: 'rgba(201,169,110,0.08)', stroke: 'rgba(201,169,110,0.22)' }));
  svg.appendChild(svgEl('rect', { x: worldX + hW, y: worldY, width: notHW, height: worldH, rx: 18, fill: 'rgba(110,165,201,0.04)', stroke: 'rgba(110,165,201,0.12)' }));

  const regionTargets = [
    addInteractiveBox(svg, {
      x: worldX,
      y: worldY,
      width: hW,
      height: tpH,
      fill: 'rgba(201,169,110,0.52)',
      stroke: palette.accent,
      label: '',
      caption: '',
      detail: {
        title: 'True-positive mass',
        body: `This gold area is P(H ∩ +) = P(H) × P(+|H) = ${pct(prior, 1)} × ${pct(sensitivity, 1)} = ${pct(tpMass, 1)} of the whole world.`,
      },
    }),
    addInteractiveBox(svg, {
      x: worldX + hW,
      y: worldY,
      width: notHW,
      height: fpH,
      fill: 'rgba(201,110,138,0.42)',
      stroke: palette.accent3,
      label: '',
      caption: '',
      detail: {
        title: 'False-positive mass',
        body: `This rose area is P(¬H ∩ +) = P(¬H) × P(+|¬H) = ${pct(1 - prior, 1)} × ${pct(falsePositiveRate, 1)} = ${pct(fpMass, 1)} of the whole world.`,
      },
    }),
    addInteractiveBox(svg, {
      x: worldX,
      y: worldY + tpH,
      width: hW,
      height: Math.max(14, worldH - tpH),
      fill: 'rgba(201,169,110,0.10)',
      stroke: 'rgba(201,169,110,0.20)',
      label: '',
      caption: '',
      detail: {
        title: 'False negatives',
        body: `This remaining gold area is P(H ∩ −) = ${pct(fnMass, 1)}. These are real hypothesis cases that the test missed.`,
      },
    }),
    addInteractiveBox(svg, {
      x: worldX + hW,
      y: worldY + fpH,
      width: notHW,
      height: Math.max(14, worldH - fpH),
      fill: 'rgba(110,165,201,0.05)',
      stroke: 'rgba(110,165,201,0.16)',
      label: '',
      caption: '',
      detail: {
        title: 'True-negative mass',
        body: `Most of the blue area is P(¬H ∩ −) = ${pct(tnMass, 1)}. These are non-hypothesis cases that correctly tested negative.`,
      },
    }),
  ];

  addText(svg, worldX + 14, worldY + worldH + 24, 'H', 'hv-svg-kicker');
  addText(svg, worldX + hW + 14, worldY + worldH + 24, '¬H', 'hv-svg-kicker');
  addMeasureLine(svg, worldX, worldY - 18, worldX + hW, worldY - 18, `P(H) = ${pct(prior, 1)}`, worldX + hW / 2, worldY - 24);
  addMeasureLine(svg, worldX + hW, worldY - 18, worldX + worldW, worldY - 18, `P(¬H) = ${pct(1 - prior, 1)}`, worldX + hW + notHW / 2, worldY - 24);
  addMeasureLine(svg, worldX - 18, worldY, worldX - 18, worldY + tpH, `P(+|H) = ${pct(sensitivity, 1)}`, worldX - 28, worldY + tpH / 2 + 4, 'end');
  addMeasureLine(svg, worldX + worldW + 18, worldY, worldX + worldW + 18, worldY + fpH, `P(+|¬H) = ${pct(falsePositiveRate, 1)}`, worldX + worldW + 28, worldY + fpH / 2 + 4, 'start');
  const worldLegend = html('div', 'hv-legend-row');
  const tpLegend = html('div', 'hv-legend-item');
  tpLegend.appendChild(html('span', 'hv-swatch hv-swatch-kl'));
  tpLegend.appendChild(html('span', 'hv-legend-text', `gold top area = P(H∩+) = ${pct(tpMass, 1)}`));
  const fpLegend = html('div', 'hv-legend-item');
  fpLegend.appendChild(html('span', 'hv-swatch hv-swatch-fp'));
  fpLegend.appendChild(html('span', 'hv-legend-text', `rose top area = P(¬H∩+) = ${pct(fpMass, 1)}`));
  worldLegend.appendChild(tpLegend);
  worldLegend.appendChild(fpLegend);
  world.appendChild(worldLegend);

  const crop = buildCard(grid, 'Keep only positive tests');
  crop.appendChild(html('p', 'hv-footnote', 'Conditioning on “positive” means throwing away every non-highlighted region. The positive band becomes the new universe.'));
  const stripMeta = html('div', 'hv-bayes-meta');
  stripMeta.appendChild(html('span', 'hv-inline-label', `positive-test pool P(+) = ${pct(evidenceMass, 1)} of the whole world`));
  crop.appendChild(stripMeta);

  const strip = html('div', 'hv-bayes-strip');
  const trueSegment = html('button', 'hv-bayes-segment hv-bayes-segment-true');
  trueSegment.type = 'button';
  trueSegment.style.flex = `${Math.max(tpMass, 1e-4)} 1 0`;
  trueSegment.innerHTML = `<span class="hv-bayes-segment-label">really H</span><span class="hv-bayes-segment-value">${pct(posterior, 1)}</span>`;
  const falseSegment = html('button', 'hv-bayes-segment hv-bayes-segment-false');
  falseSegment.type = 'button';
  falseSegment.style.flex = `${Math.max(fpMass, 1e-4)} 1 0`;
  falseSegment.innerHTML = `<span class="hv-bayes-segment-label">not H</span><span class="hv-bayes-segment-value">${pct(1 - posterior, 1)}</span>`;
  strip.appendChild(trueSegment);
  strip.appendChild(falseSegment);
  crop.appendChild(strip);

  const massGrid = html('div', 'hv-bayes-measures');
  massGrid.innerHTML = `
    <div class="hv-bayes-measure"><span class="hv-inline-label">true-positive mass</span><strong>${pct(tpMass, 1)}</strong></div>
    <div class="hv-bayes-measure"><span class="hv-inline-label">false-positive mass</span><strong>${pct(fpMass, 1)}</strong></div>
    <div class="hv-bayes-measure"><span class="hv-inline-label">posterior after +</span><strong>${pct(posterior, 1)}</strong></div>
  `;
  crop.appendChild(massGrid);

  const inspector = buildInspector(crop, {
    title: 'Posterior after a positive test',
    body: `After cropping to only positive tests, ${pct(posterior, 1)} of the remaining area belongs to H.`,
  });
  const inspectorTargets = [
    ...regionTargets,
    {
      node: trueSegment,
      detail: {
        title: 'Posterior share',
        body: `This gold share is the posterior P(H|+) = ${pct(posterior, 1)}. It asks: among all positive tests, what fraction actually came from H?`,
      },
    },
    {
      node: falseSegment,
      detail: {
        title: 'False-positive share inside positives',
        body: `This rose share is 1 − P(H|+) = ${pct(1 - posterior, 1)}. It shows how much of the positive-test pool still comes from ¬H.`,
      },
    },
  ];
  wireInspector(inspectorTargets, inspector, inspectorTargets.length - 2);
  crop.appendChild(html('p', 'hv-footnote', 'This is the Bayes crop: posterior = gold area inside the positive-test strip divided by the whole strip.'));

  setTakeaway(
    context,
    posterior > 0.8
      ? 'After cropping to positives, most of the remaining area is truly H, so the positive test is convincing.'
      : posterior > 0.4
        ? `The positive test helps, but ${pct(1 - posterior, 1)} of the positive-test strip still comes from false positives.`
        : 'Even after a positive test, most of the cropped strip still belongs to ¬H, so the posterior stays modest.'
  );
  setMetrics(context, [
    ['Posterior P(H|+)', pct(posterior, 1)],
    ['Positive-test pool P(+)', pct(evidenceMass, 1)],
    ['False-positive share', pct(1 - posterior, 1)],
  ]);
}

function renderEntropy(stageRoot, section, state, context) {
  const p = state.p;
  const q = state.q;
  const p0 = 1 - p;
  const q0 = 1 - q;
  const safe = (value) => Math.max(1e-6, Math.min(1 - 1e-6, value));
  const entropy = -(p * Math.log2(safe(p)) + p0 * Math.log2(safe(p0)));
  const crossEntropy = -(p * Math.log2(safe(q)) + p0 * Math.log2(safe(q0)));
  const kl = crossEntropy - entropy;
  const trueSurprise = [-Math.log2(safe(p)), -Math.log2(safe(p0))];
  const root = buildHybridRoot(stageRoot, 'comparison');
  buildHeader(root, 'Task surprise vs model mismatch', 'Entropy belongs to the world. KL is the avoidable penalty your model adds by believing the wrong distribution.');
  const grid = buildGrid(root, 'hv-grid-entropy');

  const outcomes = buildCard(grid, 'Outcome table');
  buildBadge(outcomes, p > 0.42 && p < 0.58 ? 'balanced world' : p < 0.14 || p > 0.86 ? 'almost predictable' : 'skewed world');
  [['likely outcome', p, q, trueSurprise[0], '#6ea5c9'], ['unlikely outcome', p0, q0, trueSurprise[1], '#c96e8a']].forEach(
    ([label, trueProb, modelProb, surprise, color]) => {
      const block = html('div', 'hv-outcome-block');
      block.appendChild(html('p', 'hv-inline-label', label));
      buildBarRow(block, 'true probability', trueProb, color, pct(trueProb));
      buildBarRow(block, 'model probability', modelProb, palette.accent, pct(modelProb));
      block.appendChild(html('p', 'hv-footnote', `If this outcome happens, it costs ${bits(surprise)} of surprise.`));
      outcomes.appendChild(block);
    }
  );
  outcomes.appendChild(
    html(
      'p',
      'hv-footnote',
      'Balanced worlds stay hard to guess in advance. Skewed worlds feel easier because one outcome dominates.'
    )
  );

  const average = buildCard(grid, 'Loss decomposition');
  average.appendChild(html('p', 'hv-footnote', 'Blue is unavoidable task surprise. Gold is the extra mismatch your model adds on top.'));
  const svg = createSvg(average, '0 0 440 248');
  addGrid(svg, 22, 18, 396, 184, 8, 5);
  addText(svg, 34, 36, 'same vertical scale', 'hv-svg-kicker');
  const baseY = 198;
  const maxH = 122;
  const axisMax = Math.max(1.1, crossEntropy * 1.12, entropy * 1.2);
  const scale = maxH / axisMax;
  const taskH = entropy * scale;
  const extraH = Math.max(kl, 0) * scale;
  const totalH = crossEntropy * scale;
  const leftX = 96;
  const rightX = 252;
  const barWidth = 78;
  svg.appendChild(svgEl('rect', { x: leftX, y: baseY - taskH, width: barWidth, height: taskH, rx: 20, fill: palette.accent2, opacity: 0.9 }));
  svg.appendChild(svgEl('rect', { x: rightX, y: baseY - taskH, width: barWidth, height: taskH, rx: 20, fill: palette.accent2, opacity: 0.9 }));
  svg.appendChild(svgEl('rect', { x: rightX, y: baseY - totalH, width: barWidth, height: extraH, rx: 20, fill: palette.accent, opacity: 0.95 }));
  addText(svg, leftX + 18, 62, 'H(p)', 'hv-svg-kicker');
  addText(svg, leftX + 2, 86, num(entropy, 3), 'hv-svg-number');
  addText(svg, rightX + 6, 62, 'H(p, q)', 'hv-svg-kicker');
  addText(svg, rightX - 8, 86, num(crossEntropy, 3), 'hv-svg-number');
  addText(svg, leftX - 6, 224, 'task entropy', 'hv-svg-caption');
  addText(svg, rightX - 4, 224, 'total loss', 'hv-svg-caption');
  svg.appendChild(svgEl('line', { x1: 58, x2: 394, y1: baseY, y2: baseY, stroke: palette.line }));

  const legend = html('div', 'hv-legend-row');
  const taskLegend = html('div', 'hv-legend-item');
  taskLegend.appendChild(html('span', 'hv-swatch hv-swatch-task'));
  taskLegend.appendChild(html('span', 'hv-legend-text', `task entropy H(p) = ${num(entropy, 3)}`));
  const mismatchLegend = html('div', 'hv-legend-item');
  mismatchLegend.appendChild(html('span', 'hv-swatch hv-swatch-kl'));
  mismatchLegend.appendChild(html('span', 'hv-legend-text', `extra KL = ${num(kl, 3)}`));
  legend.appendChild(taskLegend);
  legend.appendChild(mismatchLegend);
  average.appendChild(legend);
  average.appendChild(html('p', 'hv-footnote', 'The right bar stacks the blue task baseline with the gold mismatch penalty.'));

  setTakeaway(
    context,
    Math.abs(kl) < 0.04
      ? 'Your model is paying almost no avoidable penalty, so most of the loss belongs to the task itself.'
      : kl > 0.35
        ? 'Most of the loss is not intrinsic uncertainty. It is the model paying for a mismatched belief.'
        : 'Some surprise belongs to the task; the rest is the cost of your model betting on the wrong shape.'
  );
  setMetrics(context, [
    ['Entropy', num(entropy, 3)],
    ['Cross-entropy', num(crossEntropy, 3)],
    ['KL gap', num(kl, 3)],
  ]);
}

function renderNeuron(stageRoot, section, state, context) {
  const c1 = state.x1 * state.w1;
  const c2 = state.x2 * state.w2;
  const z = c1 + c2 + state.bias;
  const gate = Math.max(0, z);

  const root = buildHybridRoot(stageRoot, 'pipeline');
  buildHeader(root, 'Weighted evidence -> score -> gate', 'A neuron is not mystical. It adds signed evidence, shifts the threshold with a bias, then decides whether any signal gets through.');
  const grid = buildGrid(root, 'hv-grid-neuron');

  const left = buildCard(grid, 'Contribution ledger');
  buildBarRow(left, 'input x1 contribution', clamp((c1 + 1.5) / 3, 0, 1), palette.accent2, num(c1));
  buildBarRow(left, 'input x2 contribution', clamp((c2 + 1.5) / 3, 0, 1), palette.accent3, num(c2));
  buildBarRow(left, 'bias offset', clamp((state.bias + 1.5) / 3, 0, 1), palette.accent, num(state.bias));
  buildBadge(left, z > 0.35 ? 'gate opens' : z < 0 ? 'gate stays closed' : 'near threshold');
  left.appendChild(
    html(
      'p',
      'hv-footnote',
      'Positive contributions push the neuron toward firing. Negative contributions pull it back. The bias shifts the whole score before the gate sees it.'
    )
  );

  const right = buildCard(grid, 'Signal flow');
  const svg = createSvg(right, '0 0 470 272');
  addGrid(svg, 18, 18, 434, 236, 9, 6);
  const items = [
    { x: 34, y: 76, label: 'x1 · w1', value: num(c1), color: palette.accent2 },
    { x: 34, y: 150, label: 'x2 · w2', value: num(c2), color: palette.accent3 },
    { x: 34, y: 224, label: 'bias', value: num(state.bias), color: palette.accent },
  ];
  items.forEach((item) => {
    svg.appendChild(svgEl('rect', { x: item.x, y: item.y - 26, width: 98, height: 44, rx: 16, fill: item.color, opacity: 0.16, stroke: item.color }));
    addText(svg, item.x + 16, item.y - 2, item.label, 'hv-svg-label');
    addText(svg, item.x + 72, item.y + 14, item.value, 'hv-svg-kicker');
    addArrow(svg, item.x + 100, item.y - 6, 220, 136, item.color, 28);
  });
  svg.appendChild(svgEl('circle', { cx: 248, cy: 136, r: 38, fill: palette.panelStrong, stroke: palette.fg, 'stroke-width': 1.2 }));
  addText(svg, 236, 130, 'Σ', 'hv-svg-symbol');
  addText(svg, 210, 192, `z = ${num(z)}`, 'hv-svg-number');
  addArrow(svg, 286, 136, 358, 136, palette.fg, 0);
  svg.appendChild(svgEl('rect', { x: 356, y: 92, width: 80, height: 88, rx: 20, fill: palette.panelStrong, stroke: palette.accent, 'stroke-width': 1.2 }));
  addText(svg, 379, 122, 'gate', 'hv-svg-label');
  addText(svg, 376, 148, gate > 0 ? 'passes' : 'stops', 'hv-svg-kicker');
  addText(svg, 374, 170, `a = ${num(gate)}`, 'hv-svg-kicker');

  setTakeaway(
    context,
    z > 0.4
      ? 'The combined evidence clears the gate, so the neuron passes a positive signal onward.'
      : z < 0
        ? 'The evidence does not clear the gate, so the neuron stays quiet.'
        : 'The neuron is near threshold, where small changes in evidence can flip the output.'
  );
  setMetrics(context, [
    ['Pre-activation z', num(z)],
    ['Gate output', num(gate)],
    ['Bias', num(state.bias)],
  ]);
}

function renderActivationBasics(stageRoot, section, state, context) {
  const d3 = ensureD3();
  const z = state.z;
  const slope = state.slope;
  const sig = 1 / (1 + Math.exp(-(z * slope)));
  const tanh = Math.tanh(z * slope);
  const relu = Math.max(0, z * slope);
  const grads = [
    slope * sig * (1 - sig),
    slope * (1 - tanh * tanh),
    z * slope > 0 ? slope : 0,
  ];

  const root = buildHybridRoot(stageRoot, 'regime');
  buildHeader(root, 'Same input, different regimes', 'The curve matters because it decides both the output and how much gradient is still alive at that point.');
  const grid = buildGrid(root, 'hv-grid-activations');

  const defs = [
    { label: 'Sigmoid', color: palette.accent2, fn: (x) => 1 / (1 + Math.exp(-(x * slope))), min: -0.05, max: 1.05, out: sig, grad: grads[0] },
    { label: 'Tanh', color: palette.accent3, fn: (x) => Math.tanh(x * slope), min: -1.05, max: 1.05, out: tanh, grad: grads[1] },
    { label: 'ReLU', color: palette.accent, fn: (x) => Math.max(0, x * slope), min: -0.2, max: 3.2, out: relu, grad: grads[2] },
  ];

  defs.forEach((def) => {
    const card = buildCard(grid, def.label);
    const svg = createSvg(card, '0 0 220 220');
    const x = d3.scaleLinear().domain([-3, 3]).range([24, 196]);
    const y = d3.scaleLinear().domain([def.min, def.max]).range([170, 34]);
    addGrid(svg, 24, 34, 172, 136, 4, 4);
    svg.appendChild(svgEl('rect', { x: x(z) - 10, y: 34, width: 20, height: 136, rx: 10, fill: 'rgba(201, 169, 110, 0.08)' }));
    svg.appendChild(
      svgEl('path', {
        d: d3
          .line()
          .x((d) => x(d))
          .y((d) => y(def.fn(d)))(d3.range(-3, 3.05, 0.1)),
        fill: 'none',
        stroke: def.color,
        'stroke-width': 3,
        'stroke-linecap': 'round',
      })
    );
    svg.appendChild(svgEl('line', { x1: x(0), x2: x(0), y1: 34, y2: 170, stroke: palette.line }));
    svg.appendChild(svgEl('circle', { cx: x(z), cy: y(def.out), r: 5.5, fill: def.color }));
    addText(svg, 26, 196, `output ${num(def.out)}`, 'hv-svg-kicker');
    addText(svg, 118, 196, `slope ${num(def.grad)}`, 'hv-svg-kicker');
    buildBadge(card, def.grad < 0.12 ? 'saturated / weak gradient' : def.label === 'ReLU' && def.out === 0 ? 'off / dead side' : 'responsive center');
    card.appendChild(
      html(
        'p',
        'hv-footnote',
        def.grad < 0.12
          ? 'The activation is still producing an output, but the gradient is nearly gone.'
          : def.label === 'ReLU' && def.out === 0
            ? 'ReLU has shut the unit off on the negative side, so no gradient passes there.'
            : 'This is the productive regime where changes in z still move the output meaningfully.'
      )
    );
  });

  setTakeaway(
    context,
    Math.abs(z * slope) < 0.4
      ? 'Near zero, sigmoid and tanh are most responsive, while ReLU is right at its hinge.'
      : z * slope > 1.8
        ? 'Large positive inputs saturate sigmoid and tanh, but ReLU keeps growing with a strong slope.'
        : 'Activation choice changes both the output shape and how much gradient is still alive at that point.'
  );
  setMetrics(context, [
    ['Sigmoid slope', num(grads[0])],
    ['Tanh slope', num(grads[1])],
    ['ReLU slope', num(grads[2])],
  ]);
}

function renderOutputFunctions(stageRoot, section, state, context) {
  const binaryProb = 1 / (1 + Math.exp(-state.binary));
  const logits = [state.classA, state.classB, 0];
  const exps = logits.map((v) => Math.exp(v));
  const total = exps.reduce((sum, value) => sum + value, 0);
  const probs = exps.map((value) => value / total);

  const root = buildHybridRoot(stageRoot, 'comparison');
  buildHeader(root, 'One yes/no probability vs one shared class distribution', 'Sigmoid handles a single binary decision. Softmax turns several logits into one contest for probability mass.');
  const grid = buildGrid(root, 'hv-grid-outputs');

  const left = buildCard(grid, 'Binary output');
  left.appendChild(html('p', 'hv-big-number', pct(binaryProb)));
  buildBarRow(left, 'positive class probability', binaryProb, palette.accent2);
  left.appendChild(html('p', 'hv-footnote', `One logit, one probability. Confidence rises smoothly as the logit moves positive.`));

  const right = buildCard(grid, 'Multiclass competition');
  [['Class A', probs[0], palette.accent], ['Class B', probs[1], palette.accent2], ['Class C', probs[2], palette.accent3]].forEach(
    ([label, value, color]) => buildBarRow(right, label, value, color)
  );
  right.appendChild(html('p', 'hv-footnote', 'Raising one class does not live in isolation: it steals probability mass from the others.'));

  setTakeaway(
    context,
    Math.abs(logits[0] - logits[1]) < 0.2
      ? 'Softmax is showing a real contest: when two class logits are close, the distribution stays split.'
      : 'Sigmoid handles one yes/no probability, while softmax coordinates a whole field of competing classes.'
  );
  setMetrics(context, [
    ['Binary prob', pct(binaryProb)],
    ['Top class', `Class ${['A', 'B', 'C'][probs.indexOf(Math.max(...probs))]}`],
    ['Top prob', pct(Math.max(...probs))],
  ]);
}

function renderForwardPass(stageRoot, section, state, context) {
  const h1 = clamp(state.x1 * 0.9 + state.x2 * 0.25, 0, 1);
  const h2 = clamp(state.x1 * 0.28 + state.x2 * 0.82, 0, 1);
  const score = 0.7 * h1 + 0.9 * h2 - 0.5;
  const prediction = 1 / (1 + Math.exp(-score * 3));
  const dominant = h1 >= h2 ? 'hidden feature 1' : 'hidden feature 2';

  const root = buildHybridRoot(stageRoot, 'comparison');
  buildHeader(root, 'Inputs become hidden features, then a prediction', 'A forward pass is just feature construction. Hidden units mix the raw inputs in different ways, and the output layer reads those hidden features.');
  const grid = buildGrid(root, 'hv-grid-forward');

  const left = buildCard(grid, 'Feature construction');
  buildBarRow(left, 'input x1', state.x1, palette.accent2, pct(state.x1));
  buildBarRow(left, 'input x2', state.x2, palette.accent3, pct(state.x2));
  buildBarRow(left, 'hidden feature 1', h1, palette.accent, pct(h1));
  buildBarRow(left, 'hidden feature 2', h2, palette.accent2, pct(h2));
  buildBadge(left, `${dominant} is carrying more of the signal`);
  left.appendChild(
    html(
      'p',
      'hv-footnote',
      'Both hidden units see the same raw inputs, but with different weights. That is why they respond to different patterns before the output layer combines them.'
    )
  );

  const right = buildCard(grid, 'Inspect the network');
  right.appendChild(html('p', 'hv-footnote', 'Hover or click a node to see what role it plays in the forward pass.'));
  const svg = createSvg(right, '0 0 400 246');
  addGrid(svg, 24, 20, 352, 182, 6, 4);
  const inputNodes = [
    {
      x: 74, y: 82, label: 'x1', color: palette.accent2, value: state.x1, caption: pct(state.x1),
      detail: {
        title: 'Input x1',
        body: `This is a raw feature from the data. Right now it contributes ${pct(state.x1)} before the network has transformed anything.`,
      },
    },
    {
      x: 74, y: 162, label: 'x2', color: palette.accent3, value: state.x2, caption: pct(state.x2),
      detail: {
        title: 'Input x2',
        body: `This is the second raw feature. Neural nets start by mixing signals like this into more useful hidden features.`,
      },
    },
  ];
  const hiddenNodes = [
    {
      x: 202, y: 82, label: 'h1', color: palette.accent, value: h1, caption: pct(h1),
      detail: {
        title: 'Hidden feature 1',
        body: `This hidden unit mostly listens to x1, so it rises to ${pct(h1)} when that first signal is strong.`,
      },
    },
    {
      x: 202, y: 162, label: 'h2', color: palette.accent2, value: h2, caption: pct(h2),
      detail: {
        title: 'Hidden feature 2',
        body: `This hidden unit weighs x2 more heavily, so it reacts to a different pattern than h1.`,
      },
    },
  ];
  const outputNode = {
    x: 326, y: 122, label: 'ŷ', color: palette.fg, value: prediction, caption: pct(prediction),
    detail: {
      title: 'Prediction',
      body: `The output node combines both hidden features into one score, then turns that score into a ${pct(prediction)} probability.`,
    },
  };

  addArrow(svg, 90, 78, 166, 78, `rgba(110,165,201,${0.35 + state.x1 * 0.55})`, 0);
  addArrow(svg, 90, 78, 166, 154, `rgba(110,165,201,${0.18 + state.x1 * 0.35})`, 18);
  addArrow(svg, 90, 154, 166, 78, `rgba(201,110,138,${0.18 + state.x2 * 0.25})`, 18);
  addArrow(svg, 90, 154, 166, 154, `rgba(201,110,138,${0.35 + state.x2 * 0.55})`, 0);
  addArrow(svg, 210, 78, 278, 110, `rgba(201,169,110,${0.35 + h1 * 0.5})`, 20);
  addArrow(svg, 210, 154, 278, 122, `rgba(110,165,201,${0.35 + h2 * 0.5})`, 20);
  addText(svg, 34, 32, 'INPUTS', 'hv-svg-kicker');
  addText(svg, 152, 32, 'HIDDEN FEATURES', 'hv-svg-kicker');
  addText(svg, 286, 32, 'PREDICTION', 'hv-svg-kicker');
  const inspector = buildInspector(right, outputNode.detail);
  const targets = [
    ...inputNodes.map((node) => addInteractiveNode(svg, node)),
    ...hiddenNodes.map((node) => addInteractiveNode(svg, node)),
    addInteractiveNode(svg, { ...outputNode, r: 28 }),
  ];
  wireInspector(targets, inspector, targets.length - 1);
  right.appendChild(
    html(
      'p',
      'hv-footnote',
      'The output layer never sees the raw inputs directly. It only reads the hidden features that the earlier layer has already built.'
    )
  );

  setTakeaway(
    context,
    prediction > 0.72
      ? `The forward pass has assembled a strong positive case, mostly through ${dominant}.`
      : prediction < 0.4
        ? 'The hidden features are not building enough evidence yet, so the output stays cautious.'
        : 'The forward pass is combining moderate hidden signals, so the output lands in the uncertain middle.'
  );
  setMetrics(context, [
    ['Hidden feature 1', pct(h1)],
    ['Hidden feature 2', pct(h2)],
    ['Prediction', pct(prediction)],
  ]);
}

function renderChainRule(stageRoot, section, state, context) {
  const factors = [Math.abs(state.error), state.slope, state.input];
  const gradient = factors.reduce((acc, value) => acc * value, 1);
  const root = buildHybridRoot(stageRoot, 'pipeline');
  buildHeader(root, 'Backprop works because gradients are products of local sensitivities', 'Each local derivative acts like a gain knob. One tiny factor can choke the whole signal.');
  const grid = buildGrid(root, 'hv-grid-chain');
  const left = buildCard(grid, 'Local factors');
  [['output error', factors[0], palette.accent3], ['activation slope', factors[1], palette.accent2], ['input path', factors[2], palette.accent]].forEach(
    ([label, value, color]) => buildBarRow(left, label, value, color)
  );
  const right = buildCard(grid, 'Product path');
  const svg = createSvg(right, '0 0 360 220');
  addGrid(svg, 18, 18, 324, 184, 6, 4);
  const xs = [58, 152, 246, 314];
  ['w', 'z', 'ŷ', 'L'].forEach((label, index) => {
    svg.appendChild(svgEl('circle', { cx: xs[index], cy: 112, r: 22, fill: palette.panelStrong, stroke: index === 3 ? palette.accent3 : palette.fg }));
    addText(svg, xs[index] - 8, 118, label, 'hv-svg-symbol');
  });
  addArrow(svg, xs[0] + 22, 112, xs[1] - 22, 112, palette.accent, 0);
  addArrow(svg, xs[1] + 22, 112, xs[2] - 22, 112, palette.accent2, 0);
  addArrow(svg, xs[2] + 22, 112, xs[3] - 22, 112, palette.accent3, 0);
  addText(svg, 34, 174, `${num(factors[2])} × ${num(factors[1])} × ${num(factors[0])} = ${num(gradient)}`, 'hv-svg-number');

  setTakeaway(
    context,
    gradient < 0.08
      ? 'A small local factor is suppressing the whole gradient, which is exactly the vanishing-gradient intuition.'
      : 'The chain rule makes blame compositional: each stage contributes a local sensitivity to the final update signal.'
  );
  setMetrics(context, [
    ['Error term', num(factors[0])],
    ['Activation slope', num(factors[1])],
    ['Final gradient', num(gradient)],
  ]);
}

function renderBackprop(stageRoot, section, state, context) {
  const delta = state.error;
  const dw1 = state.x1 * delta;
  const dw2 = state.x2 * delta;
  const db = delta;
  const maxGrad = Math.max(Math.abs(dw1), Math.abs(dw2), Math.abs(db), 1e-6);
  const dominant = Math.abs(dw1) >= Math.abs(dw2) ? 'w1' : 'w2';

  const root = buildHybridRoot(stageRoot, 'comparison');
  buildHeader(root, 'One local delta becomes several parameter gradients', 'Backprop does not invent a different error for each weight. It sends one shared delta backward, and each parameter scales it by the activity that passed through it.');
  const grid = buildGrid(root, 'hv-grid-backprop');
  const left = buildCard(grid, 'Inspect the blame flow');
  left.appendChild(html('p', 'hv-footnote', 'Hover or click a node to see why one gradient becomes larger than another.'));
  const svg = createSvg(left, '0 0 420 272');
  addGrid(svg, 24, 20, 372, 204, 7, 5);
  addText(svg, 44, 40, 'UPSTREAM DELTA', 'hv-svg-kicker');
  addText(svg, 38, 160, 'FORWARD ACTIVITY', 'hv-svg-kicker');
  addText(svg, 286, 40, 'PARAMETER GRADIENTS', 'hv-svg-kicker');
  addArrow(svg, 124, 80, 272, 80, `rgba(201,169,110,${0.35 + Math.abs(delta) * 0.5})`, 0);
  addArrow(svg, 124, 80, 272, 150, `rgba(110,165,201,${0.2 + Math.abs(dw1) * 0.6})`, 28);
  addArrow(svg, 124, 80, 272, 216, `rgba(201,110,138,${0.2 + Math.abs(dw2) * 0.6})`, 28);
  addArrow(svg, 122, 170, 272, 150, `rgba(110,165,201,${0.2 + state.x1 * 0.6})`, 28);
  addArrow(svg, 122, 226, 272, 216, `rgba(201,110,138,${0.2 + state.x2 * 0.6})`, 28);
  const targets = [
    addInteractiveNode(svg, {
      x: 96,
      y: 80,
      label: 'δ',
      color: palette.accent,
      caption: num(delta),
      detail: {
        title: 'Shared local delta',
        body: delta >= 0
          ? `This neuron pushed the loss upward by ${num(delta)}. Every incoming parameter will see this same signed correction signal.`
          : `The local delta is ${num(delta)}, so pushing this neuron upward would actually help the loss. The sign flips the update direction.`,
      },
    }),
    addInteractiveNode(svg, {
      x: 96,
      y: 170,
      label: 'x1',
      color: palette.accent2,
      caption: pct(state.x1),
      detail: {
        title: 'Input activity x1',
        body: `This path carried ${pct(state.x1)} of activity during the forward pass. Bigger activity means the matching weight feels more blame now.`,
      },
    }),
    addInteractiveNode(svg, {
      x: 96,
      y: 226,
      label: 'x2',
      color: palette.accent3,
      caption: pct(state.x2),
      detail: {
        title: 'Input activity x2',
        body: `This input only contributes ${pct(state.x2)}. If that is smaller than x1, its matching weight gradient also comes back smaller.`,
      },
    }),
    addInteractiveNode(svg, {
      x: 308,
      y: 80,
      label: 'db',
      color: palette.accent,
      caption: num(db, 3),
      detail: {
        title: 'Bias gradient',
        body: `The bias does not multiply an input, so its gradient is just the local delta itself: ${num(db, 3)}.`,
      },
    }),
    addInteractiveNode(svg, {
      x: 308,
      y: 150,
      label: 'dw1',
      color: palette.accent2,
      caption: num(dw1, 3),
      detail: {
        title: 'Weight gradient for w1',
        body: `This gradient is x1 × δ = ${pct(state.x1)} × ${num(delta, 3)} = ${num(dw1, 3)}. More forward activity on x1 means a bigger update here.`,
      },
    }),
    addInteractiveNode(svg, {
      x: 308,
      y: 216,
      label: 'dw2',
      color: palette.accent3,
      caption: num(dw2, 3),
      detail: {
        title: 'Weight gradient for w2',
        body: `This gradient is x2 × δ = ${pct(state.x2)} × ${num(delta, 3)} = ${num(dw2, 3)}. This path only gets big blame when that input was active.`,
      },
    }),
  ];
  const inspector = buildInspector(left, targets[0].detail);
  wireInspector(targets, inspector, 0);
  left.appendChild(
    html(
      'p',
      'hv-footnote',
      'Every incoming weight sees the same delta. Their gradients only split because the forward pass sent different amounts of activity down each path.'
    )
  );
  const right = buildCard(grid, 'Who gets more blame?');
  buildBarRow(right, '∂L/∂w1 = x1·δ', Math.abs(dw1) / maxGrad, palette.accent2, num(dw1, 3));
  buildBarRow(right, '∂L/∂w2 = x2·δ', Math.abs(dw2) / maxGrad, palette.accent3, num(dw2, 3));
  buildBarRow(right, '∂L/∂b = δ', Math.abs(db) / maxGrad, palette.accent, num(db, 3));
  buildBadge(right, `${dominant} gets the largest update signal`);
  right.appendChild(
    html(
      'p',
      'hv-footnote',
      'Bias does not multiply an input, so its gradient is just the local delta itself. A weight gets a larger gradient when more signal flowed through that path.'
    )
  );

  setTakeaway(
    context,
    Math.abs(dw1) > Math.abs(dw2)
      ? `w1 gets slightly more blame here because ${num(dw1, 3)} is larger than ${num(dw2, 3)}.`
      : `w2 gets more blame here because ${num(dw2, 3)} is larger than ${num(dw1, 3)}.`
  );
  setMetrics(context, [
    ['Local δ', num(delta, 3)],
    ['Grad w1', num(dw1, 3)],
    ['Grad w2', num(dw2, 3)],
  ]);
}

function renderGradientDescent(stageRoot, section, state, context) {
  const d3 = ensureD3();
  const lr = state.lr;
  const curvature = state.curvature;
  const start = state.position;
  const fn = (x) => 0.22 + curvature * (x * x) * 0.13;
  const points = [];
  let x = start;
  for (let i = 0; i < 10; i += 1) {
    points.push({ x, y: fn(x) });
    x -= lr * (2 * curvature * x);
  }
  const end = points.at(-1).x;
  const root = buildHybridRoot(stageRoot, 'plot');
  buildHeader(root, 'Training is a path across the loss surface', 'A healthy learning rate settles into the valley. A bad one crawls, bounces, or explodes.');
  const svg = createSvg(root, '0 0 740 340');
  const sx = d3.scaleLinear().domain([-3.5, 3.5]).range([64, 680]);
  const sy = d3.scaleLinear().domain([0.15, 2.2]).range([274, 44]);
  addGrid(svg, 64, 44, 616, 230, 10, 6);
  svg.appendChild(
    svgEl('path', {
      d: d3.line().x((d) => sx(d)).y((d) => sy(fn(d)))(d3.range(-3.5, 3.55, 0.05)),
      fill: 'none',
      stroke: palette.accent,
      'stroke-width': 3,
    })
  );
  points.forEach((point, index) => {
    if (index < points.length - 1) {
      addArrow(svg, sx(point.x), sy(point.y), sx(points[index + 1].x), sy(points[index + 1].y), palette.accent2, 18);
    }
    svg.appendChild(svgEl('circle', { cx: sx(point.x), cy: sy(point.y), r: 4.5, fill: palette.accent2 }));
  });
  setTakeaway(
    context,
    Math.abs(end) < 0.35
      ? 'These steps are healthy: they make real progress without losing control.'
      : lr > 0.6
        ? 'The update is too aggressive, so the path keeps overshooting the valley instead of settling into it.'
        : 'The update is safe, but it is moving too cautiously to cover ground quickly.'
  );
  setMetrics(context, [
    ['Start', num(start)],
    ['End', num(end)],
    ['Step size', num(lr)],
  ]);
}

function renderOptimizers(stageRoot, section, state, context) {
  const d3 = ensureD3();
  const noise = state.noise;
  const lr = state.lr;
  const steps = 11;
  const makeSeries = (kind) => {
    let x = 2.8;
    let velocity = 0;
    let second = 0;
    const series = [];
    for (let i = 0; i < steps; i += 1) {
      const grad = x + Math.sin(i * 1.2) * noise * 0.9;
      series.push({ x: i, y: x });
      if (kind === 'sgd') x -= lr * grad;
      if (kind === 'momentum') {
        velocity = 0.78 * velocity + 0.22 * grad;
        x -= lr * velocity;
      }
      if (kind === 'adam') {
        velocity = 0.85 * velocity + 0.15 * grad;
        second = 0.9 * second + 0.1 * grad * grad;
        x -= (lr * velocity) / (Math.sqrt(second) + 0.2);
      }
    }
    return series;
  };
  const series = [
    { key: 'sgd', label: 'SGD', color: palette.accent3 },
    { key: 'momentum', label: 'Momentum', color: palette.accent },
    { key: 'adam', label: 'Adam', color: palette.accent2 },
  ].map((item) => ({ ...item, points: makeSeries(item.key) }));

  const root = buildHybridRoot(stageRoot, 'plot');
  buildHeader(root, 'Same gradients, different optimizer personalities', 'SGD reacts immediately, momentum smooths direction, and Adam also rescales by recent gradient magnitude.');
  const svg = createSvg(root, '0 0 740 340');
  const sx = d3.scaleLinear().domain([0, steps - 1]).range([70, 682]);
  const sy = d3.scaleLinear().domain([-0.8, 3.2]).range([276, 44]);
  addGrid(svg, 70, 44, 612, 232, 10, 6);
  series.forEach((item) => {
    svg.appendChild(
      svgEl('path', {
        d: d3.line().x((d) => sx(d.x)).y((d) => sy(d.y))(item.points),
        fill: 'none',
        stroke: item.color,
        'stroke-width': 3,
      })
    );
    const last = item.points.at(-1);
    svg.appendChild(svgEl('circle', { cx: sx(last.x), cy: sy(last.y), r: 5, fill: item.color }));
  });
  const legend = buildGrid(root, 'hv-grid-optimizer-legend');
  series.forEach((item) => buildBarRow(legend, item.label, clamp(1 - Math.abs(item.points.at(-1).y) / 3), item.color, `finish ${num(item.points.at(-1).y)}`));

  setTakeaway(
    context,
    noise > 0.55
      ? 'With noisy minibatches, momentum and Adam hold a steadier direction than plain SGD.'
      : 'When the gradient is cleaner, the main difference is how aggressively each optimizer smooths or rescales the path.'
  );
  setMetrics(context, [
    ['Noise', pct(noise)],
    ['Base LR', num(lr)],
    ['Best finisher', series.slice().sort((a, b) => Math.abs(a.points.at(-1).y) - Math.abs(b.points.at(-1).y))[0].label],
  ]);
}

function renderLrSchedule(stageRoot, section, state, context) {
  const d3 = ensureD3();
  const initial = state.initial;
  const decay = state.decay;
  const epochs = d3.range(0, 41);
  const constant = epochs.map((t) => ({ x: t, y: initial * (1 - decay * 0.12) }));
  const step = epochs.map((t) => ({ x: t, y: initial * Math.pow(1 - decay * 0.55, Math.floor(t / 12)) }));
  const cosine = epochs.map((t) => ({ x: t, y: 0.12 + (initial - 0.12) * 0.5 * (1 + Math.cos((Math.PI * t) / 40)) * (1 - decay * 0.18) }));
  const series = [
    { label: 'Constant', points: constant, color: palette.accent3 },
    { label: 'Step decay', points: step, color: palette.accent },
    { label: 'Cosine', points: cosine, color: palette.accent2 },
  ];

  const root = buildHybridRoot(stageRoot, 'plot');
  buildHeader(root, 'Schedules cool optimization over time', 'The real question is how much update energy you keep late in training, not which line looks prettiest.');
  const svg = createSvg(root, '0 0 740 340');
  const sx = d3.scaleLinear().domain([0, 40]).range([70, 682]);
  const sy = d3.scaleLinear().domain([0, Math.max(initial + 0.08, 1.05)]).range([280, 44]);
  addGrid(svg, 70, 44, 612, 236, 10, 6);
  series.forEach((item) => {
    svg.appendChild(
      svgEl('path', {
        d: d3.line().x((d) => sx(d.x)).y((d) => sy(d.y))(item.points),
        fill: 'none',
        stroke: item.color,
        'stroke-width': 3,
      })
    );
  });
  const legend = buildGrid(root, 'hv-grid-optimizer-legend');
  series.forEach((item) => buildBarRow(legend, item.label, clamp(item.points.at(-1).y / initial, 0, 1), item.color, `late LR ${num(item.points.at(-1).y)}`));

  setTakeaway(
    context,
    decay > 0.72
      ? 'Aggressive decay gives late-stage stability, but it can leave the optimizer with too little energy to keep improving.'
      : 'A schedule buys you two phases: larger exploratory steps early, smaller refining steps later.'
  );
  setMetrics(context, [
    ['Initial LR', num(initial)],
    ['Decay strength', pct(decay)],
    ['Cosine late LR', num(cosine.at(-1).y)],
  ]);
}

function renderTransformerBlock(stageRoot, section, state, context) {
  const contextNeed = state.contextNeed;
  const rewrite = state.rewrite;
  const residual = clamp(0.34 + (1 - Math.abs(contextNeed - rewrite)) * 0.44);
  const root = buildHybridRoot(stageRoot, 'wide');
  buildHeader(root, 'A transformer block has two jobs', 'Attention mixes context across tokens. The MLP rewrites each token after that context arrives. Residual paths keep the previous state alive.');
  const svg = createSvg(root, '0 0 740 320');
  addGrid(svg, 20, 20, 700, 280, 10, 6);
  const tokenXs = [72, 116, 160, 204];
  svg.appendChild(svgEl('rect', { x: 42, y: 84, width: 176, height: 96, rx: 22, fill: 'rgba(110, 165, 201, 0.05)' }));
  svg.appendChild(svgEl('rect', { x: 248, y: 84, width: 184, height: 96, rx: 22, fill: 'rgba(201, 169, 110, 0.05)' }));
  svg.appendChild(svgEl('rect', { x: 460, y: 84, width: 212, height: 96, rx: 22, fill: 'rgba(201, 110, 138, 0.05)' }));
  tokenXs.forEach((x, index) => {
    svg.appendChild(svgEl('rect', { x, y: 122, width: 26, height: 40, rx: 10, fill: palette.panelStrong, stroke: index === 2 ? palette.accent : palette.accent2 }));
    addText(svg, x + 6, 147, `t${index + 1}`, 'hv-svg-kicker');
  });
  svg.appendChild(svgEl('rect', { x: 254, y: 94, width: 170, height: 74, rx: 18, fill: palette.panelStrong, stroke: palette.accent }));
  addText(svg, 280, 120, 'self-attention', 'hv-svg-label');
  addText(svg, 280, 145, 'mixes context across tokens', 'hv-svg-kicker');
  svg.appendChild(svgEl('rect', { x: 470, y: 94, width: 170, height: 74, rx: 18, fill: palette.panelStrong, stroke: palette.accent2 }));
  addText(svg, 508, 120, 'MLP / feed-forward', 'hv-svg-label');
  addText(svg, 500, 145, 'rewrites each token locally', 'hv-svg-kicker');
  [0.2, 0.34, 0.46, 0.62].forEach((offset, index) => addArrow(svg, tokenXs[index] + 26, 142, 254, 126, `rgba(201,169,110,${offset + contextNeed * 0.25})`, 28));
  addArrow(svg, 424, 130, 470, 130, palette.fg, 0);
  svg.appendChild(svgEl('rect', { x: 316, y: 196, width: 132, height: 28, rx: 14, fill: palette.panel, stroke: palette.line }));
  addText(svg, 334, 214, 'residual + norm', 'hv-svg-kicker');
  svg.appendChild(svgEl('rect', { x: 534, y: 196, width: 132, height: 28, rx: 14, fill: palette.panel, stroke: palette.line }));
  addText(svg, 552, 214, 'residual + norm', 'hv-svg-kicker');
  addText(svg, 74, 108, 'token rail', 'hv-svg-kicker');
  addText(svg, 288, 108, 'context mixing', 'hv-svg-kicker');
  addText(svg, 522, 108, 'per-token rewrite', 'hv-svg-kicker');

  const bars = buildGrid(root, 'hv-grid-transformer');
  buildBarRow(bars, 'context mixing', contextNeed, palette.accent);
  buildBarRow(bars, 'feature rewrite', rewrite, palette.accent2);
  buildBarRow(bars, 'residual carry', residual, palette.accent3);

  setTakeaway(
    context,
    contextNeed > rewrite + 0.2
      ? 'This block is mostly earning its keep by pulling in outside context before a lighter rewrite.'
      : rewrite > contextNeed + 0.2
        ? 'This block is mostly transforming token state after only modest context mixing.'
        : 'A healthy transformer block does both jobs: gather context, then refine the token representation.'
  );
  setMetrics(context, [
    ['Context mixing', pct(contextNeed)],
    ['Feature rewrite', pct(rewrite)],
    ['Residual carry', pct(residual)],
  ]);
}

function renderAttention(stageRoot, section, state, context) {
  const match = state.match;
  const sharpness = state.sharpness;
  const rawScores = [0.26 + match * 0.14, 0.44 + match * 0.24, 0.78 + match * 0.52, 0.31 + (1 - match) * 0.18];
  const tempered = rawScores.map((score) => Math.exp(score * sharpness * 2.2));
  const total = tempered.reduce((sum, value) => sum + value, 0);
  const weights = tempered.map((value) => value / total);

  const root = buildHybridRoot(stageRoot, 'comparison');
  buildHeader(root, 'Attention = score -> normalize -> mix', 'The query scores candidate tokens, softmax turns those scores into weights, then the values get blended into one context vector.');
  const grid = buildGrid(root, 'hv-grid-attention');

  const left = buildCard(grid, 'Three-step machine');
  const svg = createSvg(left, '0 0 440 248');
  addGrid(svg, 18, 18, 404, 212, 6, 4);
  svg.appendChild(svgEl('rect', { x: 28, y: 84, width: 82, height: 76, rx: 18, fill: palette.panelStrong, stroke: palette.line }));
  addText(svg, 48, 118, 'query', 'hv-svg-label');
  addText(svg, 52, 142, 'token', 'hv-svg-kicker');
  const labels = ['the', 'model', 'found', 'evidence'];
  svg.appendChild(svgEl('rect', { x: 146, y: 46, width: 110, height: 146, rx: 18, fill: 'rgba(255,255,255,0.025)', stroke: palette.line }));
  svg.appendChild(svgEl('rect', { x: 268, y: 46, width: 108, height: 146, rx: 18, fill: 'rgba(255,255,255,0.025)', stroke: palette.line }));
  labels.forEach((label, index) => {
    const y = 58 + index * 38;
    svg.appendChild(svgEl('rect', { x: 160, y, width: 86, height: 24, rx: 12, fill: palette.panelStrong, stroke: [palette.accent2, palette.accent2, palette.accent3, palette.accent][index] }));
    addText(svg, 188, y + 16, label, 'hv-svg-kicker');
    addArrow(svg, 110, 122, 160, y + 12, `rgba(201,169,110,${0.22 + weights[index] * 0.6})`, 16);
  });
  addText(svg, 170, 38, 'score', 'hv-svg-kicker');
  addText(svg, 292, 38, 'softmax', 'hv-svg-kicker');
  addText(svg, 324, 218, 'mixed context', 'hv-svg-kicker');
  weights.forEach((weight, index) => {
    const y = 58 + index * 38;
    svg.appendChild(svgEl('rect', { x: 278, y: y + 5, width: 84, height: 12, rx: 6, fill: palette.panel }));
    svg.appendChild(svgEl('rect', { x: 278, y: y + 5, width: 84 * weight, height: 12, rx: 6, fill: [palette.accent2, palette.accent2, palette.accent3, palette.accent][index] }));
  });
  svg.appendChild(svgEl('rect', { x: 278, y: 198, width: 110, height: 20, rx: 10, fill: palette.panel }));
  let offset = 278;
  weights.forEach((weight, index) => {
    const w = 110 * weight;
    svg.appendChild(svgEl('rect', { x: offset, y: 198, width: w, height: 20, rx: 10, fill: [palette.accent2, palette.accent2, palette.accent3, palette.accent][index], opacity: 0.9 }));
    offset += w;
  });

  const right = buildCard(grid, 'Normalized weights');
  labels.forEach((label, index) => buildBarRow(right, label, weights[index], [palette.accent2, palette.accent2, palette.accent3, palette.accent][index], pct(weights[index])));

  setTakeaway(
    context,
    sharpness > 1.15
      ? 'The model is concentrating hard on one token, which is powerful when the dependency is clear but brittle when the signal is noisy.'
      : sharpness < 0.55
        ? 'The weights stay broad, which keeps options open but dilutes the strongest dependency.'
        : 'Attention behaves like a soft lookup: it does not choose one source, it builds a weighted blend of relevant context.'
  );
  setMetrics(context, [
    ['Top weight', pct(Math.max(...weights))],
    ['Query match', pct(match)],
    ['Sharpness', num(sharpness)],
  ]);
}

function renderPositional(stageRoot, section, state, context) {
  const distance = state.distance;
  const pressure = state.context;
  const orderSignal = clamp(0.92 - distance * 0.28);
  const relativeHelp = clamp(0.42 + distance * 0.5 - pressure * 0.12);
  const longContextStrain = clamp(pressure * (0.55 + distance * 0.45));

  const root = buildHybridRoot(stageRoot, 'comparison');
  buildHeader(root, 'Order has to be injected from outside attention', 'Without a position signal, the model sees the same tokens but not their arrangement. Position is what turns a bag of tokens into a sequence.');
  const grid = buildGrid(root, 'hv-grid-positional');

  const left = buildCard(grid, 'Same tokens, different meaning');
  const svg = createSvg(left, '0 0 440 250');
  addGrid(svg, 18, 18, 404, 214, 7, 5);
  const top = ['dog', 'bites', 'man'];
  const bottom = ['man', 'bites', 'dog'];
  [top, bottom].forEach((row, rowIndex) => {
    const y = rowIndex === 0 ? 84 : 170;
    row.forEach((token, index) => {
      const x = 54 + index * 110;
      svg.appendChild(svgEl('rect', { x, y, width: 74, height: 30, rx: 15, fill: palette.panelStrong, stroke: index === 1 ? palette.accent : palette.accent2 }));
      addText(svg, x + 18, y + 20, token, 'hv-svg-kicker');
      svg.appendChild(svgEl('circle', { cx: x + 14, cy: y - 16, r: 11, fill: 'rgba(201,169,110,0.10)', stroke: palette.accent }));
      addText(svg, x + 10, y - 12, `${index + 1}`, 'hv-svg-kicker');
    });
  });
  addText(svg, 48, 66, 'order 1', 'hv-svg-kicker');
  addText(svg, 48, 152, 'order 2', 'hv-svg-kicker');
  addText(svg, 284, 56, 'same tokens, different arrangement', 'hv-svg-caption');
  addText(svg, 266, 214, orderSignal > 0.78 ? 'position keeps these meanings separate' : 'weak position signal makes order easier to confuse', 'hv-svg-caption');

  const right = buildCard(grid, 'What the position signal buys you');
  buildBarRow(right, 'order separability', orderSignal, palette.accent2, pct(orderSignal));
  buildBarRow(right, 'relative-distance help', relativeHelp, palette.accent, pct(relativeHelp));
  buildBarRow(right, 'long-context strain', longContextStrain, palette.accent3, pct(longContextStrain));
  right.appendChild(html('p', 'hv-footnote', 'Absolute position says where a token is. Relative position says how far tokens are from each other. Long contexts stress both choices.'));

  setTakeaway(
    context,
    longContextStrain > 0.7
      ? 'When the context stretches long, the real question is whether the positional scheme still preserves useful relative order.'
      : distance < 0.3
        ? 'For local relations, even simple position signals can separate the sequence cleanly.'
        : 'Position encoding matters because attention alone cannot tell “same words, different order” apart.'
  );
  setMetrics(context, [
    ['Order signal', pct(orderSignal)],
    ['Relative help', pct(relativeHelp)],
    ['Long-context strain', pct(longContextStrain)],
  ]);
}

function renderKvCache(stageRoot, section, state, context) {
  const tokens = Math.round(4 + state.tokens * 20);
  const layers = Math.round(8 + state.layers * 32);
  const noCacheWork = tokens * (tokens + 1) / 2;
  const cacheWork = tokens;
  const speedup = noCacheWork / cacheWork;
  const memoryPressure = clamp((tokens / 24) * 0.56 + (layers / 40) * 0.44);

  const root = buildHybridRoot(stageRoot, 'comparison');
  buildHeader(root, 'KV cache trades memory for decode speed', 'Old tokens do not change during autoregressive generation. The cache stores their keys and values so each new token only adds one more slice.');
  const grid = buildGrid(root, 'hv-grid-kv');

  const left = buildCard(grid, 'Without cache vs with cache');
  buildBarRow(left, 'recomputed attention work', clamp(noCacheWork / 200, 0, 1), palette.accent3, `${Math.round(noCacheWork)} units`);
  buildBarRow(left, 'incremental work with cache', clamp(cacheWork / 200, 0, 1), palette.accent2, `${cacheWork} units`);
  buildBarRow(left, 'cache memory pressure', memoryPressure, palette.accent, pct(memoryPressure));
  left.appendChild(html('p', 'hv-footnote', 'The longer the generation gets, the more wasteful full recomputation becomes. The tradeoff is that you now have to keep the cache resident in memory.'));

  const right = buildCard(grid, 'Decode timeline');
  const svg = createSvg(right, '0 0 440 250');
  addGrid(svg, 18, 18, 404, 214, 8, 5);
  for (let i = 0; i < 6; i += 1) {
    const x = 46 + i * 50;
    svg.appendChild(svgEl('rect', { x, y: 102, width: 34, height: 44, rx: 10, fill: palette.panelStrong, stroke: palette.accent2 }));
    addText(svg, x + 9, 128, `t${i + 1}`, 'hv-svg-kicker');
    if (i < 5) addArrow(svg, x + 34, 124, x + 50, 124, palette.line, 0);
  }
  svg.appendChild(svgEl('rect', { x: 350, y: 96, width: 44, height: 56, rx: 14, fill: 'rgba(201,169,110,0.10)', stroke: palette.accent }));
  addText(svg, 360, 128, 'new', 'hv-svg-kicker');
  addText(svg, 44, 78, 'cached past K/V', 'hv-svg-kicker');
  addText(svg, 314, 78, 'one fresh slice', 'hv-svg-kicker');
  addText(svg, 44, 194, `speedup grows with sequence length: ~${num(speedup, 1)}x here`, 'hv-svg-caption');

  setTakeaway(
    context,
    memoryPressure > 0.72
      ? 'The cache is doing its job for speed, but the memory bill is starting to dominate the serving tradeoff.'
      : speedup > 5
        ? 'The generation is long enough that cache reuse is now buying a meaningful speedup.'
        : 'On short replies the cache still helps, but its biggest win appears once the sequence starts getting long.'
  );
  setMetrics(context, [
    ['Sequence length', `${tokens} tokens`],
    ['Approx speedup', `${num(speedup, 1)}x`],
    ['Memory pressure', pct(memoryPressure)],
  ]);
}

function renderMdp(stageRoot, section, state, context) {
  const delay = state.rewardDelay;
  const stochasticity = state.stochasticity;
  const creditDifficulty = clamp(delay * 0.66 + stochasticity * 0.34);
  const markovClarity = clamp(1 - stochasticity * 0.42);

  const root = buildHybridRoot(stageRoot, 'wide');
  buildHeader(root, 'An RL environment is a loop, not a labeled dataset', 'The agent acts, changes the future state it will see, and only later finds out whether that action was actually good.');
  const grid = buildGrid(root, 'hv-grid-mdp');

  const left = buildCard(grid, 'Environment loop');
  const svg = createSvg(left, '0 0 430 250');
  addGrid(svg, 18, 18, 394, 214, 7, 5);
  const steps = [
    { x: 48, label: 'state', color: palette.accent2 },
    { x: 156, label: 'action', color: palette.accent },
    { x: 268, label: 'reward', color: palette.accent3 },
    { x: 356, label: 'next state', color: palette.fg },
  ];
  steps.forEach((step, index) => {
    svg.appendChild(svgEl('rect', { x: step.x, y: 110, width: 56, height: 34, rx: 14, fill: palette.panelStrong, stroke: step.color }));
    addText(svg, step.x + 10, 131, step.label, 'hv-svg-kicker');
    if (index < steps.length - 1) addArrow(svg, step.x + 56, 127, steps[index + 1].x, 127, palette.accent, 14);
  });
  addText(svg, 42, 72, 'action changes the future data distribution', 'hv-svg-caption');
  svg.appendChild(svgEl('path', {
    d: `M 296 92 C 336 ${92 - delay * 54}, 370 ${92 - delay * 54}, 384 110`,
    fill: 'none',
    stroke: `rgba(201,110,138,${0.35 + delay * 0.5})`,
    'stroke-width': 3,
    'stroke-linecap': 'round',
  }));
  addText(svg, 286, 56, 'delayed reward path', 'hv-svg-kicker');

  const right = buildCard(grid, 'Why RL feels harder');
  buildBarRow(right, 'credit-assignment difficulty', creditDifficulty, palette.accent3, pct(creditDifficulty));
  buildBarRow(right, 'environment uncertainty', stochasticity, palette.accent, pct(stochasticity));
  buildBarRow(right, 'state clarity', markovClarity, palette.accent2, pct(markovClarity));
  right.appendChild(html('p', 'hv-footnote', 'Delayed rewards make blame assignment harder. Stochastic environments make the loop noisier even when the policy is reasonable.'));

  setTakeaway(
    context,
    creditDifficulty > 0.68
      ? 'The hard part is not choosing an action. It is figuring out which earlier action deserves credit once the reward arrives much later.'
      : 'The MDP picture matters because RL is about acting inside a loop where today’s choice shapes tomorrow’s state.'
  );
  setMetrics(context, [
    ['Reward delay', pct(delay)],
    ['Stochasticity', pct(stochasticity)],
    ['Credit difficulty', pct(creditDifficulty)],
  ]);
}

function renderDiffusion(stageRoot, section, state, context) {
  const noise = state.noise;
  const steps = state.steps;
  const cleanup = clamp((steps - 3) / 17);
  const structure = clamp(0.18 + cleanup * 0.56 + (1 - noise) * 0.26);

  const root = buildHybridRoot(stageRoot, 'wide');
  buildHeader(root, 'Generation by progressive denoising', 'Diffusion does not create the sample in one jump. It starts from noise and repeatedly removes uncertainty until structure emerges.');
  const grid = buildGrid(root, 'hv-grid-diffusion');

  const left = buildCard(grid, 'Denoising path');
  const svg = createSvg(left, '0 0 460 250');
  addGrid(svg, 18, 18, 424, 214, 8, 4);
  const phases = [
    { x: 42, label: 'noise', radius: 17, blur: 0.1 },
    { x: 152, label: 'coarse form', radius: 15, blur: 0.4 },
    { x: 262, label: 'mid structure', radius: 13, blur: 0.7 },
    { x: 372, label: 'clean sample', radius: 11, blur: 1.0 },
  ];
  phases.forEach((phase, index) => {
    const y = 126;
    svg.appendChild(svgEl('circle', { cx: phase.x, cy: y, r: 46, fill: `rgba(255,255,255,${0.03 + phase.blur * 0.03})`, stroke: palette.line }));
    for (let i = 0; i < 6; i += 1) {
      const dx = Math.cos((i / 6) * Math.PI * 2) * (18 + (1 - phase.blur) * 10 * noise);
      const dy = Math.sin((i / 6) * Math.PI * 2) * (16 + (1 - phase.blur) * 8 * noise);
      svg.appendChild(svgEl('circle', { cx: phase.x + dx, cy: y + dy, r: phase.radius - i * 1.5, fill: [palette.accent2, palette.accent, palette.accent3][i % 3], opacity: 0.12 + phase.blur * 0.1 }));
    }
    addText(svg, phase.x - 24, 206, phase.label, 'hv-svg-kicker');
    if (index < phases.length - 1) addArrow(svg, phase.x + 54, y, phases[index + 1].x - 54, y, palette.accent, 18);
  });
  addText(svg, 110, 48, `${steps} reverse steps of cleanup`, 'hv-svg-caption');

  const right = buildCard(grid, 'Generation tradeoff');
  buildBarRow(right, 'starting noise', noise, palette.accent3, pct(noise));
  buildBarRow(right, 'denoising progress', cleanup, palette.accent2, `${steps} steps`);
  buildBarRow(right, 'visible structure', structure, palette.accent, pct(structure));
  right.appendChild(html('p', 'hv-footnote', 'More steps usually give cleaner structure, but the process is slower. Higher starting noise means the model has more uncertainty to undo.'));

  setTakeaway(
    context,
    steps < 6
      ? 'With very few reverse steps, the sample still carries a lot of unresolved noise.'
      : structure > 0.72
        ? 'Enough denoising steps have turned the noisy latent into a clearly structured sample.'
        : 'Diffusion feels intuitive when you read it as repeated cleanup rather than one-shot creation.'
  );
  setMetrics(context, [
    ['Noise', pct(noise)],
    ['Steps', `${steps}`],
    ['Structure', pct(structure)],
  ]);
}

function renderMatrixFactorization(stageRoot, section, state, context) {
  const alignment = state.alignment;
  const sparsity = state.sparsity;
  const affinity = clamp(alignment * (1 - sparsity * 0.28));
  const coldStartRisk = clamp(sparsity * 0.72 + (1 - alignment) * 0.18);

  const root = buildHybridRoot(stageRoot, 'comparison');
  buildHeader(root, 'Users and items share one preference space', 'Matrix factorization replaces a sparse interaction table with geometry: user and item vectors that align more strongly when the predicted preference is high.');
  const grid = buildGrid(root, 'hv-grid-mf');

  const left = buildCard(grid, 'Shared latent space');
  const svg = createSvg(left, '0 0 430 250');
  addGrid(svg, 30, 24, 360, 190, 6, 6);
  svg.appendChild(svgEl('line', { x1: 56, x2: 382, y1: 118, y2: 118, stroke: palette.line }));
  svg.appendChild(svgEl('line', { x1: 206, x2: 206, y1: 34, y2: 214, stroke: palette.line }));
  const user = { x: 206 + alignment * 104, y: 118 - alignment * 54 };
  const item = { x: 206 + alignment * 98, y: 118 - alignment * 68 + sparsity * 38 };
  svg.appendChild(svgEl('circle', { cx: user.x, cy: user.y, r: 10, fill: palette.accent2 }));
  svg.appendChild(svgEl('rect', { x: item.x - 10, y: item.y - 10, width: 20, height: 20, rx: 6, fill: palette.accent, opacity: 0.92 }));
  addText(svg, user.x + 14, user.y + 4, 'user', 'hv-svg-kicker');
  addText(svg, item.x + 14, item.y + 4, 'item', 'hv-svg-kicker');
  addArrow(svg, user.x, user.y, item.x, item.y, palette.fg, 22);
  addText(svg, 44, 46, 'learned taste axes', 'hv-svg-kicker');
  addText(svg, 42, 228, `closer alignment -> larger dot product`, 'hv-svg-caption');

  const right = buildCard(grid, 'What the factorization is buying');
  buildBarRow(right, 'predicted affinity', affinity, palette.accent2, pct(affinity));
  buildBarRow(right, 'interaction sparsity', sparsity, palette.accent3, pct(sparsity));
  buildBarRow(right, 'cold-start risk', coldStartRisk, palette.accent, pct(coldStartRisk));
  right.appendChild(html('p', 'hv-footnote', 'The geometry can generalize across similar users and items, but sparse histories still make placement in the space much harder.'));

  setTakeaway(
    context,
    sparsity > 0.7
      ? 'The shared latent space is useful, but sparse behavior makes it much harder to place users and items confidently.'
      : affinity > 0.76
        ? 'Strong user-item alignment is exactly what the dot product is trying to capture.'
        : 'Matrix factorization works when one shared embedding space is enough to express taste compatibility.'
  );
  setMetrics(context, [
    ['Affinity', pct(affinity)],
    ['Sparsity', pct(sparsity)],
    ['Cold-start risk', pct(coldStartRisk)],
  ]);
}

function renderTwoTower(stageRoot, section, state, context) {
  const latency = state.latency;
  const interaction = state.interaction;
  const precomputeGain = clamp(0.38 + latency * 0.56);
  const rerankNeed = clamp(0.26 + interaction * 0.62 - latency * 0.18);

  const root = buildHybridRoot(stageRoot, 'comparison');
  buildHeader(root, 'Two-tower retrieval wins by making the item side precomputable', 'The query tower runs online. The item tower runs offline. That split is what makes large-scale nearest-neighbor retrieval fast enough to serve.');
  const grid = buildGrid(root, 'hv-grid-two-tower');

  const left = buildCard(grid, 'Serving architecture');
  const svg = createSvg(left, '0 0 440 250');
  addGrid(svg, 18, 18, 404, 214, 7, 5);
  svg.appendChild(svgEl('rect', { x: 36, y: 100, width: 94, height: 42, rx: 18, fill: palette.panelStrong, stroke: palette.accent2 }));
  addText(svg, 58, 124, 'query tower', 'hv-svg-kicker');
  svg.appendChild(svgEl('rect', { x: 180, y: 74, width: 104, height: 42, rx: 18, fill: palette.panelStrong, stroke: palette.accent }));
  addText(svg, 200, 98, 'item tower', 'hv-svg-kicker');
  svg.appendChild(svgEl('rect', { x: 180, y: 144, width: 104, height: 42, rx: 18, fill: 'rgba(201,169,110,0.08)', stroke: palette.accent }));
  addText(svg, 198, 168, 'ANN index', 'hv-svg-kicker');
  svg.appendChild(svgEl('rect', { x: 326, y: 100, width: 74, height: 42, rx: 18, fill: palette.panelStrong, stroke: palette.accent3 }));
  addText(svg, 340, 124, 'retrieve', 'hv-svg-kicker');
  addArrow(svg, 130, 120, 180, 120, palette.accent2, 14);
  addArrow(svg, 284, 164, 326, 122, palette.accent, 20);
  addText(svg, 184, 56, 'offline, precomputed', 'hv-svg-caption');
  addText(svg, 34, 196, 'online query encoding', 'hv-svg-caption');

  const right = buildCard(grid, 'Tradeoff surface');
  buildBarRow(right, 'precompute gain', precomputeGain, palette.accent2, pct(precomputeGain));
  buildBarRow(right, 'cross-feature richness', interaction, palette.accent3, pct(interaction));
  buildBarRow(right, 'need for reranker', rerankNeed, palette.accent, pct(rerankNeed));
  right.appendChild(html('p', 'hv-footnote', 'Two-tower gets you speed and scale. What you give up is some pair-specific interaction detail, which is why heavier rankers often come later.'));

  setTakeaway(
    context,
    rerankNeed > 0.72
      ? 'The serving win is real, but you now need a stronger second-stage ranker to recover the interaction detail you left out.'
      : latency > 0.75
        ? 'This is where two-tower shines: the system is strongly latency-constrained, so precomputing the item side is a huge advantage.'
        : 'Two-tower is best understood as a serving compromise: cheap retrieval now, richer interaction later.'
  );
  setMetrics(context, [
    ['Precompute gain', pct(precomputeGain)],
    ['Interaction richness', pct(interaction)],
    ['Rerank need', pct(rerankNeed)],
  ]);
}

function renderRetrievalFunnel(stageRoot, section, state, context) {
  const candidates = state.candidates;
  const recall = state.recall;
  const precision = state.precision;
  const finalPool = Math.max(8, Math.round(candidates * 0.04));
  const recoveryChance = clamp(recall * precision);

  const root = buildHybridRoot(stageRoot, 'wide');
  buildHeader(root, 'Retrieval protects recall. Ranking spends compute on precision.', 'If the relevant item never enters the candidate set, no later stage can rescue it.');
  const grid = buildGrid(root, 'hv-grid-retrieval');

  const left = buildCard(grid, 'Pipeline funnel');
  const svg = createSvg(left, '0 0 460 250');
  addGrid(svg, 24, 18, 412, 214, 6, 5);
  const widths = [260, 160, 78];
  const labels = [`retrieved ${candidates}`, `reranked ${Math.round(candidates * 0.18)}`, `final ${finalPool}`];
  widths.forEach((w, index) => {
    const y = 44 + index * 60;
    svg.appendChild(svgEl('rect', { x: 90, y, width: w, height: 34, rx: 17, fill: [palette.accent2, palette.accent, palette.accent3][index], opacity: 0.82 }));
    addText(svg, 102, y + 22, labels[index], 'hv-svg-kicker');
  });
  addArrow(svg, 350, 61, 380, 121, palette.fg, 18);
  addArrow(svg, 290, 121, 320, 181, palette.fg, 18);
  addText(svg, 90, 222, recall < 0.6 ? 'weak retrieval leaves the ranker boxed in' : 'healthy retrieval keeps good options alive for ranking', 'hv-svg-caption');

  const right = buildCard(grid, 'Stage quality');
  buildBarRow(right, 'retrieval recall', recall, palette.accent2, pct(recall));
  buildBarRow(right, 'ranking precision', precision, palette.accent3, pct(precision));
  buildBarRow(right, 'chance the right item survives', recoveryChance, palette.accent, pct(recoveryChance));
  right.appendChild(html('p', 'hv-footnote', 'Early stages are cheap but blunt. Later stages are smarter but only get to see what survived the funnel.'));

  setTakeaway(
    context,
    recall < 0.58
      ? 'The ranker is not the bottleneck here. The relevant item is often disappearing too early in retrieval.'
      : precision < 0.58
        ? 'Recall is healthy enough, so the remaining win is in spending more intelligence on ranking the shortlist.'
        : 'The clean mental model is a funnel: retrieval buys coverage, ranking buys order.'
  );
  setMetrics(context, [
    ['Candidates', `${candidates}`],
    ['Recall', pct(recall)],
    ['Precision', pct(precision)],
  ]);
}

function renderTreeSplit(stageRoot, section, state, context) {
  const separation = state.separation;
  const noise = state.noise;
  const gain = clamp(separation * (1 - noise * 0.72));
  const childPurity = clamp(0.28 + gain * 0.68);

  const root = buildHybridRoot(stageRoot, 'comparison');
  buildHeader(root, 'A split is good when the children are simpler than the parent', 'Trees do not fit one smooth surface. They keep carving the data into local regions that are easier to predict.');
  const grid = buildGrid(root, 'hv-grid-tree');

  const left = buildCard(grid, 'Before and after the split');
  const svg = createSvg(left, '0 0 430 250');
  addGrid(svg, 18, 18, 394, 214, 6, 5);
  svg.appendChild(svgEl('rect', { x: 48, y: 78, width: 106, height: 90, rx: 24, fill: palette.panelStrong, stroke: palette.line }));
  [palette.accent2, palette.accent3, palette.accent, palette.accent3, palette.accent2, palette.accent].forEach((color, idx) => {
    const cx = 72 + (idx % 3) * 28;
    const cy = 104 + Math.floor(idx / 3) * 30;
    svg.appendChild(svgEl('circle', { cx, cy, r: 8, fill: color, opacity: 0.85 }));
  });
  addArrow(svg, 170, 122, 228, 122, palette.fg, 16);
  svg.appendChild(svgEl('rect', { x: 238, y: 60, width: 74, height: 64, rx: 20, fill: 'rgba(110,165,201,0.08)', stroke: palette.accent2 }));
  svg.appendChild(svgEl('rect', { x: 238, y: 134, width: 74, height: 64, rx: 20, fill: 'rgba(201,110,138,0.08)', stroke: palette.accent3 }));
  addText(svg, 252, 92, 'left child', 'hv-svg-kicker');
  addText(svg, 250, 166, 'right child', 'hv-svg-kicker');
  addText(svg, 52, 58, 'mixed parent', 'hv-svg-kicker');
  addText(svg, 238, 222, 'split succeeds when the children are more pure', 'hv-svg-caption');

  const right = buildCard(grid, 'Impurity picture');
  buildBarRow(right, 'feature separation', separation, palette.accent2, pct(separation));
  buildBarRow(right, 'label noise', noise, palette.accent3, pct(noise));
  buildBarRow(right, 'impurity reduction', gain, palette.accent, pct(gain));
  right.appendChild(html('p', 'hv-footnote', 'Strong separation helps only if label noise is not too high. Trees are always asking whether one more split makes the child nodes easier to predict.'));

  setTakeaway(
    context,
    gain > 0.62
      ? 'This is a high-value split because the children look much simpler than the mixed parent.'
      : noise > 0.6
        ? 'The feature may separate the data, but label noise is blurring away most of the gain.'
        : 'A useful split is one that makes the children more homogeneous than the parent.'
  );
  setMetrics(context, [
    ['Separation', pct(separation)],
    ['Noise', pct(noise)],
    ['Gain', pct(gain)],
  ]);
}

function renderBoosting(stageRoot, section, state, context) {
  const d3 = ensureD3();
  const rounds = Math.round(2 + state.rounds * 10);
  const rate = state.rate;
  const residuals = [];
  let residual = 1;
  for (let i = 0; i < rounds; i += 1) {
    residuals.push({ x: i + 1, y: residual });
    residual *= (1 - rate * 0.55);
  }
  const finalResidual = residuals.at(-1).y;

  const root = buildHybridRoot(stageRoot, 'wide');
  buildHeader(root, 'Boosting is stagewise residual correction', 'Each new tree is not replacing the ensemble. It is trying to shave down what the current ensemble still gets wrong.');
  const svg = createSvg(root, '0 0 740 340');
  const sx = d3.scaleLinear().domain([1, rounds]).range([80, 670]);
  const sy = d3.scaleLinear().domain([0, 1.05]).range([272, 44]);
  addGrid(svg, 80, 44, 590, 228, 8, 6);
  svg.appendChild(
    svgEl('path', {
      d: d3.line().x((d) => sx(d.x)).y((d) => sy(d.y))(residuals),
      fill: 'none',
      stroke: palette.accent2,
      'stroke-width': 3,
      'stroke-linecap': 'round',
    })
  );
  residuals.forEach((point) => {
    svg.appendChild(svgEl('circle', { cx: sx(point.x), cy: sy(point.y), r: 5, fill: palette.accent }));
  });
  addText(svg, 82, 28, 'residual error after each tree', 'hv-svg-kicker');
  addText(svg, 462, 80, `final residual ${num(finalResidual, 2)}`, 'hv-svg-caption');

  const legend = buildGrid(root, 'hv-grid-optimizer-legend');
  buildBarRow(legend, 'boosting rounds', clamp(rounds / 12), palette.accent2, `${rounds}`);
  buildBarRow(legend, 'learning rate', rate, palette.accent, pct(rate));
  buildBarRow(legend, 'residual left', finalResidual, palette.accent3, num(finalResidual, 2));

  setTakeaway(
    context,
    rate > 0.7
      ? 'Aggressive boosting shrinks the residual quickly, but it also makes each stage more capable of overcorrecting.'
      : finalResidual < 0.3
        ? 'Many small corrections have now driven most of the remaining residual down.'
        : 'Boosting feels intuitive once you stop imagining one big tree and instead watch the residual get chipped away round by round.'
  );
  setMetrics(context, [
    ['Rounds', `${rounds}`],
    ['Rate', pct(rate)],
    ['Residual left', num(finalResidual, 2)],
  ]);
}

// Delegate to the React+Vite bundle (atelier/react-viz/dist/atelier-viz.iife.js).
// Phase 1 viz live in React — entropy, optimizers, attention.
function renderReactViz(vizKey) {
  return function render(stageRoot, section, state, context) {
    if (!window.AtelierReactViz) {
      stageRoot.innerHTML =
        '<div style="padding:16px;color:#c96e8a;font-family:monospace;font-size:12px">' +
        '[atelier] React viz bundle not loaded. Ensure ./react-viz/dist/atelier-viz.iife.js is included before studio.js.' +
        '</div>';
      return;
    }
    window.AtelierReactViz.mount(vizKey, { stageRoot, section, state, context });
  };
}

const hybridRenderers = {
  distribution: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('distribution') },
  loss: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('loss') },
  expectation: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('expectation') },
  bayes: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('bayes') },
  entropy: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('entropy') },
  vectors: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('vectors') },
  'dot-products': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('dot-products') },
  'matrix-multiply': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('matrix-multiply') },
  eigen: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('eigen') },
  svd: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('svd') },
  'mle-map': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('mle-map') },
  'bias-variance': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('bias-variance') },
  regularization: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('regularization') },
  neuron: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('neuron') },
  'activation-basics': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('activation-basics') },
  'output-functions': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('output-functions') },
  'forward-pass': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('forward-pass') },
  'chain-rule': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('chain-rule') },
  backprop: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('backprop') },
  'gradient-descent': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('gradient-descent') },
  'learning-rate': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('learning-rate') },
  optimizers: { rendererKind: 'plot', viewerLayout: 'wide', render: renderReactViz('optimizers') },
  'lr-schedule': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('lr-schedule') },
  initialization: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('initialization') },
  normalization: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('normalization') },
  residuals: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('residuals') },
  'gradient-flow': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('gradient-flow') },
  tokenization: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('tokenization') },
  embeddings: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('embeddings') },
  positional: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('positional') },
  'transformer-block': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('transformer-block') },
  attention: { rendererKind: 'diagram', viewerLayout: 'wide', render: renderReactViz('attention') },
  'kv-cache': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('kv-cache') },
  mdp: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('mdp') },
  diffusion: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('diffusion') },
  'matrix-factorization': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('matrix-factorization') },
  'two-tower': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('two-tower') },
  'retrieval-funnel': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('retrieval-funnel') },
  'tree-split': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('tree-split') },
  boosting: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('boosting') },
  'threshold-metrics': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('threshold-metrics') },
  'ranking-metrics': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('ranking-metrics') },
  calibration: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('calibration') },
  'online-offline': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('online-offline') },
  'serving-skew': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('serving-skew') },
  'data-drift': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('data-drift') },
  'value-functions': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('value-functions') },
  'td-learning': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('td-learning') },
  'q-learning': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('q-learning') },
  dqn: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('dqn') },
  bandit: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('bandit') },
  'flash-attention': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('flash-attention') },
  'pretrain-finetune': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('pretrain-finetune') },
  peft: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('peft') },
  lora: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('lora') },
  quantization: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('quantization') },
  distillation: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('distillation') },
  rag: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('rag') },
  'cold-start': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('cold-start') },
  'feature-leakage': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('feature-leakage') },
  'feature-shift': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('feature-shift') },
  'serving-tradeoffs': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('serving-tradeoffs') },
  guidance: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('guidance') },
  dpo: { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('dpo') },
  'reward-hacking': { rendererKind: 'hybrid', viewerLayout: 'wide', render: renderReactViz('reward-hacking') },
};

function isHybridViz(vizId) {
  return Boolean(hybridRenderers[vizId]);
}

window.AtelierHybrid = { hybridRenderers, isHybridViz };
})();
