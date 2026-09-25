/* Native function lessons. Each concept shares its state across mounted views. */
(() => {
  'use strict';

  function sigmoid(z) {
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    const e = Math.exp(z);
    return e / (1 + e);
  }

  function softmax(logits) {
    if (!Array.isArray(logits) || !logits.length || !logits.every(Number.isFinite)) {
      throw new TypeError('Softmax requires a nonempty array of finite logits.');
    }
    const max = Math.max(...logits);
    const weights = logits.map(z => Math.exp(z - max));
    const total = weights.reduce((sum, x) => sum + x, 0);
    return weights.map(x => x / total);
  }

  function activation(kind, z) {
    if (!Number.isFinite(z)) throw new TypeError('Activation input must be finite.');
    if (kind === 'sigmoid') {
      const value = sigmoid(z);
      return { value, derivative: value * (1 - value), backwardMultiplier: value * (1 - value) };
    }
    if (kind === 'tanh') {
      const value = Math.tanh(z);
      return { value, derivative: 1 - value * value, backwardMultiplier: 1 - value * value };
    }
    if (kind === 'relu') {
      return { value: Math.max(0, z), derivative: z === 0 ? null : z > 0 ? 1 : 0, backwardMultiplier: z > 0 ? 1 : 0 };
    }
    if (kind === 'linear') return { value: z, derivative: 1, backwardMultiplier: 1 };
    throw new RangeError(`Unknown activation: ${kind}`);
  }

  function outputProbabilities(logits) {
    const exclusive = softmax(logits);
    return { independent: logits.map(sigmoid), exclusive };
  }

  const limit = x => Math.max(-6, Math.min(6, Math.round(x * 4) / 4));
  const number = (x, digits = 3) => (Math.abs(x) < 0.5 * 10 ** -digits ? 0 : x).toFixed(digits);
  const slopeNumber = x => x !== 0 && Math.abs(x) < 0.001 ? x.toExponential(2) : number(x);
  const signed = x => `${x > 0 ? '+' : ''}${number(x, 2)}`;
  const percent = x => `${number(100 * x, 1)}%`;
  const names = ['cat', 'dog', 'bird'];
  const functions = [
    { kind: 'sigmoid', name: 'Sigmoid', range: [-0.15, 1.15], top: '1', bottom: '0' },
    { kind: 'tanh', name: 'tanh', range: [-1.3, 1.3], top: '+1', bottom: '-1' },
    { kind: 'relu', name: 'ReLU', range: [-0.8, 6.8], top: '+6', bottom: '0' },
  ];
  const scenarios = {
    both: { name: 'Cat and dog', logits: [2, 2, -1] },
    dog: { name: 'More dog evidence', logits: [2, 4, -1] },
    tied: { name: 'No preference', logits: [0, 0, 0] },
  };
  const instances = new Map();
  const bindings = new WeakMap();
  const button = (action, value, label, selected) => `<button type="button" class="fn-button" data-fn-action="${action}" data-fn-value="${value}"${selected === undefined ? '' : ` aria-pressed="${selected}"`}>${label}</button>`;

  function regime(kind, z, enabled) {
    if (!enabled) return 'Straight through';
    if (kind === 'relu') return z === 0 ? 'At the hinge' : z < 0 ? 'Off: zero gradient' : 'Linear: gradient passes';
    return activation(kind, z).derivative < 0.03 ? 'Saturated: small gradient' : 'Responsive region';
  }

  function plotMarkup(spec, s) {
    const kind = s.enabled ? spec.kind : 'linear';
    const result = activation(kind, s.z);
    const previous = activation(s.before.enabled ? spec.kind : 'linear', s.before.z);
    const [low, high] = s.enabled ? spec.range : [-6.8, 6.8];
    const x = z => 16 + (z + 6) / 12 * 288;
    const y = value => 168 - (value - low) / (high - low) * 156;
    const points = Array.from({ length: 121 }, (_, i) => {
      const z = -6 + i / 10;
      return `${i ? 'L' : 'M'}${number(x(z), 2)},${number(y(activation(kind, z).value), 2)}`;
    }).join(' ');
    const slope = result.derivative;
    const tangent = slope === null ? '' : `<path class="fn-tangent" d="M${x(s.z - 0.75)},${y(result.value - slope * 0.75)} L${x(s.z + 0.75)},${y(result.value + slope * 0.75)}"/>`;
    const derivative = slope === null ? 'Undefined' : slopeNumber(slope);
    return `<article class="fn-function">
      <header class="fn-function-heading"><h4>${spec.name}</h4><span class="fn-regime">${regime(spec.kind, s.z, s.enabled)}</span></header>
      <div class="fn-plot" data-fn-plot="${spec.kind}" tabindex="0" role="slider" aria-label="${spec.name} input z. Left and right arrows change the shared input; Home and End jump to its limits."
        aria-valuemin="-6" aria-valuemax="6" aria-valuenow="${s.z}" aria-valuetext="z ${s.z}, output ${number(result.value)}, derivative ${derivative}">
        <div class="fn-plot-frame"><span class="fn-y-tick" style="top:${y(Number(s.enabled ? spec.top : 6)) / 180 * 100}%" aria-hidden="true">${s.enabled ? spec.top : '+6'}</span>
        <span class="fn-y-tick" style="top:${y(Number(s.enabled ? spec.bottom : -6)) / 180 * 100}%" aria-hidden="true">${s.enabled ? spec.bottom : '-6'}</span>
        <svg viewBox="0 0 320 180" aria-hidden="true" focusable="false">
          <path class="fn-grid" d="M16,${y(0)} H304 M160,12 V168"/>
          <path class="fn-guide" d="M${x(s.z)},12 V168"/>
          <path class="fn-curve" d="${points}"/>
          ${tangent}
          <circle class="fn-point" cx="${x(s.z)}" cy="${y(result.value)}" r="5"/>
        </svg></div>
        <div class="fn-x-axis" aria-hidden="true"><span>-6</span><span>0</span><span>+6</span></div>
      </div>
      <dl class="fn-values"><div><dt>Output f(z)</dt><dd>${number(result.value)}</dd></div><div><dt>Local derivative</dt><dd class="fn-slope">${derivative}</dd></div></dl>
      <p class="fn-previous">Before: output ${number(previous.value)}, derivative ${previous.derivative === null ? 'undefined' : slopeNumber(previous.derivative)}.</p>
      <p class="fn-gradient">${slope === null ? 'At exactly 0, ReLU has no derivative. This demo uses 0 for backprop.' : `Incoming gradient 1 &rarr; outgoing ${slopeNumber(result.backwardMultiplier)}.`}</p>
    </article>`;
  }

  function activationMarkup(s) {
    return `<div class="lesson fn-lesson fn-activation">
      <header class="fn-heading"><p class="fn-kicker">Same input. Different response.</p><h3>What does an activation change?</h3><p>A neuron makes a score <strong>z</strong>. Watch both its output and the gradient that can pass back through it.</p></header>
      <div class="fn-toolbar">
        <div class="fn-choice-group" role="group" aria-label="Apply or bypass the activation">${button('enabled', 'true', 'Apply activations', s.enabled)}${button('enabled', 'false', 'Bypass: f(z) = z', !s.enabled)}</div>
        <div class="fn-input-control"><span>Shared input <strong>z = ${signed(s.z)}</strong></span><div class="fn-stepper">${button('z-step', '-0.25', '<span aria-hidden="true">&minus;</span><span class="fn-sr-only">Decrease input by 0.25</span>')}${button('z-step', '0.25', '<span aria-hidden="true">+</span><span class="fn-sr-only">Increase input by 0.25</span>')}</div></div>
      </div>
      <div class="fn-choice-group fn-presets" role="group" aria-label="Explore an input regime">${[[-4, 'Negative: -4'], [0, 'At zero: 0'], [1, 'Positive: +1'], [4, 'Far positive: +4']].map(([z, label]) => button('z', z, label, s.z === z)).join('')}</div>
      <p class="fn-instruction">Click a plot to move z. Focus it and use arrow keys. <span class="fn-line-key"></span> Orange tangent = local slope.</p>
      <div class="fn-functions">${functions.map(spec => plotMarkup(spec, s)).join('')}</div>
      <p class="fn-plot-readout" data-fn-hover>Every plot shares the x scale; y scales differ. Compare the numerical derivatives, not the drawn angles.</p>
      <p class="fn-takeaway" data-fn-status role="status">${!s.enabled ? '<strong>No bend, no new shape.</strong> Stacking only weighted sums and biases still gives one affine map. Nonlinear activations break that limitation.' : s.z === 0 ? '<strong>A bend changes what the network can learn.</strong> At zero, sigmoid and tanh pass some gradient; ReLU has a sharp corner, so its derivative is undefined there.' : Math.abs(s.z) >= 4 ? `<strong>Output can stay large while learning slows.</strong> Sigmoid and tanh are nearly flat here: little gradient gets through. ReLU ${s.z < 0 ? 'blocks this negative input and its gradient' : 'passes a positive input with slope 1'}.` : '<strong>Two jobs, one function:</strong> nonlinear shape adds expressive power; local slope scales the gradient during learning.'}</p>
    </div>`;
  }

  function deltaLabel(current, previous) {
    const change = (current - previous) * 100;
    return Math.abs(change) < 0.00001 ? 'Unchanged' : `${change > 0 ? '+' : ''}${number(change, 1)} percentage points`;
  }

  function outputPanel(s, type, probabilities, before) {
    const independent = type === 'independent';
    const total = probabilities.reduce((a, b) => a + b, 0);
    return `<section class="fn-task fn-${type}">
      <header><p class="fn-kicker">${independent ? 'Image tags / separate yes-no decisions' : 'Next token / one shared distribution'}</p><h4>${independent ? 'Can it contain cat AND dog?' : 'Choose one: cat, dog, or bird?'}</h4></header>
      <p class="fn-task-note">${independent ? 'One sigmoid per tag. More than one tag can be true.' : 'Softmax over this toy 3-token vocabulary. Exactly one token is next.'}</p>
      <div class="fn-outcomes" role="group" aria-label="${independent ? 'Image tag' : 'Next token'} probabilities; select a score to edit">
        ${names.map((name, i) => `<button type="button" class="fn-outcome" data-fn-action="select" data-fn-value="${i}" data-fn-focus="${type}-${i}" aria-pressed="${s.selected === i}" aria-label="Edit ${name} score. ${percent(probabilities[i])}. ${deltaLabel(probabilities[i], before[i])}."><span class="fn-outcome-name">${name}</span><strong>${percent(probabilities[i])}</strong><span class="fn-change">${deltaLabel(probabilities[i], before[i])}</span></button>`).join('')}
      </div>
      ${independent ? `<div class="fn-mass" aria-hidden="true"></div><div class="fn-mass-label"><strong>Total: ${percent(total)}</strong><span>Not constrained to 100%. These are separate questions.</span></div>` : `<div class="fn-mass" aria-hidden="true">${probabilities.map((p, i) => `<span class="fn-mass-${i}" style="width:${p * 100}%"></span>`).join('')}</div><div class="fn-mass-label"><strong>Total: 100%</strong><span>Fixed probability budget: one grows, others shrink.</span></div>`}
    </section>`;
  }

  function mobileComparison(s, current, before) {
    const shortDelta = (p, old) => deltaLabel(p, old).replace('percentage points', 'pp');
    return `<section class="fn-mobile-comparison" aria-label="Compare independent image tags with one next-token choice">
      <div class="fn-comparison-head"><span>Label</span><span><strong>Sigmoid</strong>Each tag: yes/no</span><span><strong>Softmax</strong>One next token</span></div>
      <div role="group" aria-label="Select a label to edit its score">${names.map((name, i) => `<button type="button" class="fn-comparison-row" data-fn-action="select" data-fn-value="${i}" data-fn-focus="mobile-${i}" aria-pressed="${s.selected === i}" aria-label="Edit ${name}. Sigmoid ${percent(current.independent[i])}, ${deltaLabel(current.independent[i], before.independent[i])}. Softmax ${percent(current.exclusive[i])}, ${deltaLabel(current.exclusive[i], before.exclusive[i])}."><span class="fn-outcome-name">${name}</span><span><strong>${percent(current.independent[i])}</strong><span class="fn-change">${shortDelta(current.independent[i], before.independent[i])}</span></span><span><strong>${percent(current.exclusive[i])}</strong><span class="fn-change">${shortDelta(current.exclusive[i], before.exclusive[i])}</span></span></button>`).join('')}</div>
      <p class="fn-comparison-foot">Sigmoid totals need not equal 100%. Softmax always shares 100%. <span>pp = percentage points.</span></p>
    </section>`;
  }

  function outputMarkup(s) {
    const current = outputProbabilities(s.logits);
    const before = outputProbabilities(s.before);
    const selected = s.selected;
    const other = selected === 0 ? 1 : 0;
    const changed = s.logits.some((z, i) => z !== s.before[i]);
    const changedIndices = s.logits.map((z, i) => z !== s.before[i] ? i : -1).filter(i => i >= 0);
    const changedIndex = changedIndices.length === 1 ? changedIndices[0] : selected;
    const affected = changedIndex === 0 ? 1 : 0;
    let observation = '<strong>Try increasing the dog score.</strong> The cat sigmoid will stay put, but its softmax share will shrink. Same scores, different task assumptions.';
    if (changed && changedIndices.length === 1) {
      observation = `<strong>${names[changedIndex]} score: ${signed(s.before[changedIndex])} &rarr; ${signed(s.logits[changedIndex])}.</strong> ${names[affected]} sigmoid stays ${percent(current.independent[affected])}; its softmax moves from ${percent(before.exclusive[affected])} to ${percent(current.exclusive[affected])}. Only softmax makes the unchanged score compete.`;
    } else if (changed) {
      observation = '<strong>Read the change under each outcome.</strong> Sigmoids answer separate yes/no questions. Softmax redistributes one shared 100% across the choices.';
    }
    return `<div class="lesson fn-lesson fn-outputs">
      <header class="fn-heading"><p class="fn-kicker">Choose the output for the task</p><h3>Can two answers both be right?</h3><p>Same example scores, two different tasks.</p></header>
      <div class="fn-choice-group fn-presets" role="group" aria-label="Output function scenarios">${Object.entries(scenarios).map(([key, scenario]) => button('scenario', key, scenario.name, scenario.logits.every((z, i) => z === s.logits[i]))).join('')}</div>
      <div class="fn-score-editor">
        <div class="fn-score-rail" role="group" aria-label="Choose a raw score to edit">${names.map((name, i) => button('select', i, `${name} <strong>${signed(s.logits[i])}</strong>`, selected === i)).join('')}</div>
        <div class="fn-input-control"><span>Raw ${names[selected]} score</span><span class="fn-stepper">${button('score-step', '-0.5', '<span aria-hidden="true">&minus;</span><span class="fn-sr-only">Decrease selected score by 0.5</span>')}<input class="fn-number" type="number" inputmode="decimal" min="-6" max="6" step="0.5" value="${s.logits[selected]}" data-fn-number aria-label="${names[selected]} raw score">${button('score-step', '0.5', '<span aria-hidden="true">+</span><span class="fn-sr-only">Increase selected score by 0.5</span>')}</span></div>
      </div>
      <div class="fn-task-comparison">${outputPanel(s, 'independent', current.independent, before.independent)}${outputPanel(s, 'exclusive', current.exclusive, before.exclusive)}</div>
      ${mobileComparison(s, current, before)}
      <p class="fn-takeaway" data-fn-status role="status">${observation}</p>
      <details class="fn-calculation"><summary>Follow the numbers + binary equivalence</summary><div><p>For <strong>${names[selected]}</strong>, sigmoid uses only its own score: <code>1 / (1 + exp(-(${number(s.logits[selected], 1)}))) = ${number(current.independent[selected])}</code>.</p><p>Softmax divides its exponentiated score by the total: <code>exp(${number(s.logits[selected], 1)}) / (${s.logits.map(z => `exp(${number(z, 1)})`).join(' + ')}) = ${number(current.exclusive[selected])}</code>.</p><p>The ${names[other]} score is ${signed(s.logits[other])}. Changing it affects the softmax denominator, but never enters ${names[selected]}'s sigmoid.</p><p><strong>Binary equivalence:</strong> sigmoid(z) is exactly the second probability of softmax([0, z]). At z = ${number(s.logits[selected], 1)}, both give <strong>${percent(current.independent[selected])}</strong>. The distinction above is <em>separate sigmoid decisions</em> versus <em>normalized multiclass competition</em>.</p><p>This is an illustrative 3-label example, not measured model output.</p></div></details>
    </div>`;
  }

  function initialState(kind) {
    return kind === 'activation-basics' ? { z: 1, enabled: true, before: { z: 1, enabled: true } } : { logits: [2, 2, -1], before: [2, 2, -1], selected: 1 };
  }

  function update(instance, action, value) {
    const s = instance.state;
    if (instance.kind === 'activation-basics') {
      if (!['z', 'z-step', 'enabled'].includes(action)) return;
      const z = action === 'enabled' ? s.z : limit(action === 'z-step' ? s.z + Number(value) : Number(value));
      const enabled = action === 'enabled' ? value === 'true' : s.enabled;
      if (!Number.isFinite(z) || (z === s.z && enabled === s.enabled)) return;
      s.before = { z: s.z, enabled: s.enabled };
      s.z = z;
      s.enabled = enabled;
    } else if (action === 'select') {
      if (![0, 1, 2].includes(Number(value))) return;
      s.selected = Number(value);
    } else if (action === 'scenario') {
      if (!scenarios[value]) return;
      s.before = [...s.logits];
      s.logits = [...scenarios[value].logits];
    } else if (action === 'score' || action === 'score-step') {
      const next = limit(action === 'score-step' ? s.logits[s.selected] + Number(value) : Number(value));
      if (!Number.isFinite(next)) return;
      if (next === s.logits[s.selected]) { paint(instance); return; }
      s.before = [...s.logits];
      s.logits[s.selected] = next;
    } else return;
    paint(instance);
  }

  function rememberFocus(root) {
    const active = root.ownerDocument.activeElement;
    if (!root.contains(active)) return null;
    if (active.dataset.fnFocus) return el => el.dataset.fnFocus === active.dataset.fnFocus;
    if (active.hasAttribute('data-fn-plot')) return el => el.dataset.fnPlot === active.dataset.fnPlot;
    if (active.hasAttribute('data-fn-number')) return el => el.hasAttribute('data-fn-number');
    if (active.matches('.fn-calculation summary')) return el => el.matches('.fn-calculation summary');
    return el => el.dataset.fnAction === active.dataset.fnAction && el.dataset.fnValue === active.dataset.fnValue;
  }

  function paint(instance) {
    const markup = instance.kind === 'activation-basics' ? activationMarkup(instance.state) : outputMarkup(instance.state);
    for (const root of instance.roots) {
      const binding = bindings.get(root);
      if (binding.wasConnected && !root.isConnected) {
        instance.roots.delete(root);
        continue;
      }
      binding.wasConnected ||= root.isConnected;
      const match = rememberFocus(root);
      const open = root.querySelector('.fn-calculation')?.open;
      root.innerHTML = markup;
      if (open) root.querySelector('.fn-calculation').open = true;
      if (match) [...root.querySelectorAll('[data-fn-action], [data-fn-plot], [data-fn-number], .fn-calculation summary')].find(match)?.focus({ preventScroll: true });
    }
  }

  function plotZ(plot, clientX) {
    const rect = plot.querySelector('svg').getBoundingClientRect();
    return limit(((clientX - rect.left) / rect.width * 320 - 16) / 288 * 12 - 6);
  }

  function render(kind, root) {
    if (!has(kind)) throw new RangeError(`Unknown lesson: ${kind}`);
    if (!root || typeof root.addEventListener !== 'function') throw new TypeError('A DOM root is required.');
    if (!instances.has(kind)) instances.set(kind, { kind, state: initialState(kind), roots: new Set() });
    const instance = instances.get(kind);
    let binding = bindings.get(root);
    if (!binding) {
      binding = { instance, wasConnected: root.isConnected };
      bindings.set(root, binding);
      root.addEventListener('click', event => {
        const target = event.target.closest('[data-fn-action], [data-fn-plot]');
        if (!target || !root.contains(target)) return;
        event.stopPropagation();
        if (target.hasAttribute('data-fn-plot')) {
          target.focus({ preventScroll: true });
          update(binding.instance, 'z', plotZ(target, event.clientX));
        } else update(binding.instance, target.dataset.fnAction, target.dataset.fnValue);
      });
      root.addEventListener('change', event => {
        if (!event.target.hasAttribute('data-fn-number')) return;
        event.stopPropagation();
        if (event.target.value.trim() === '' || !Number.isFinite(event.target.valueAsNumber)) { paint(binding.instance); return; }
        update(binding.instance, 'score', event.target.valueAsNumber);
      });
      root.addEventListener('keydown', event => {
        const plot = event.target.closest('[data-fn-plot]');
        if (!plot) return;
        const keys = { ArrowLeft: -0.25, ArrowDown: -0.25, ArrowRight: 0.25, ArrowUp: 0.25, PageDown: -1, PageUp: 1 };
        if (event.key in keys) {
          event.preventDefault(); event.stopPropagation();
          update(binding.instance, 'z-step', keys[event.key]);
        } else if (event.key === 'Home' || event.key === 'End') {
          event.preventDefault(); event.stopPropagation();
          update(binding.instance, 'z', event.key === 'Home' ? -6 : 6);
        }
      });
      root.addEventListener('pointermove', event => {
        const plot = event.target.closest('[data-fn-plot]');
        if (!plot || event.pointerType === 'touch') return;
        const z = plotZ(plot, event.clientX);
        const result = activation(binding.instance.state.enabled ? plot.dataset.fnPlot : 'linear', z);
        root.querySelector('[data-fn-hover]').textContent = `Preview at z = ${signed(z)}: ${plot.dataset.fnPlot} output ${number(result.value)}, derivative ${result.derivative === null ? 'undefined' : slopeNumber(result.derivative)}. Click to compare all three here.`;
      });
      root.addEventListener('pointerleave', () => {
        const note = root.querySelector('[data-fn-hover]');
        if (note) note.textContent = 'Every plot shares the x scale; y scales differ. Compare the numerical derivatives, not the drawn angles.';
      });
    } else if (binding.instance !== instance) {
      binding.instance.roots.delete(root);
      binding.instance = instance;
    }
    binding.wasConnected = root.isConnected;
    instance.roots.add(root);
    paint(instance);
  }

  const content = {
    'activation-basics': {
      title: 'Activations change the shape, and the gradient',
      summary: 'A weighted sum makes a score. An activation bends or gates it, so stacked layers can do more than one linear model.',
      what: 'An <strong>activation</strong> is a function applied to a neuron\'s score. Sigmoid squeezes it into 0 to 1; tanh into -1 to 1; ReLU keeps positive scores and zeros negative ones. The curve\'s local slope is also the multiplier on the gradient flowing backward.',
      why: 'Stacking only weighted sums and biases still produces one affine map. Nonlinear activations make more complex boundaries possible. But a nearly flat activation can also make a neuron slow to learn.',
      interview: '<strong>Two jobs:</strong> activations add nonlinearity in the forward pass and scale gradients by their local derivative in the backward pass. A large output does not imply a large gradient.',
      details: [
        'Saturation means the curve is nearly flat. Sigmoid and tanh have small derivatives for large-magnitude inputs; repeated multiplication by small derivatives can make earlier gradients vanish.',
        'ReLU has derivative 0 for negative inputs and 1 for positive inputs. At exactly zero it is not differentiable; this lesson uses the common backprop convention of 0. A negative input once is not proof of a permanently dead neuron.',
        'The orange tangent and numerical derivative describe local sensitivity, not the full network gradient. The lesson fixes the incoming gradient at 1 to isolate this multiplier.',
        'Do not confuse a hidden activation with the combined output loss. For a sigmoid output with binary cross-entropy, the loss gradient with respect to its logit simplifies to p - y; the sigmoid derivative alone does not determine that gradient.',
      ],
      math: {
        title: 'Output forward, derivative backward',
        formula: [String.raw`\sigma(z)=\frac{1}{1+e^{-z}},\qquad \sigma'(z)=\sigma(z)(1-\sigma(z))`, String.raw`\tanh'(z)=1-\tanh^2(z)`, String.raw`\operatorname{ReLU}(z)=\max(0,z),\quad \operatorname{ReLU}'(z)=\begin{cases}0&z<0\\1&z>0\\\text{undefined}&z=0\end{cases}`, String.raw`\frac{\partial L}{\partial z}=\frac{\partial L}{\partial a}\,f'(z),\qquad a=f(z)`],
        note: 'At z = 0, sigmoid outputs 0.5 but its derivative is only 0.25. An incoming gradient of 2 becomes 2 x 0.25 = 0.5.',
        annotations: [
          ['z', 'Score before the activation.', 'A neuron combines image features into a score of 4.'],
          ['a=f(z)', 'Activated output sent forward.', 'Sigmoid turns score 4 into about 0.982.'],
          ["f'(z)", 'Local change in output per tiny change in score.', 'At score 4, sigmoid has slope about 0.018: a small score change barely changes the output.'],
          [String.raw`\partial L/\partial a`, 'Gradient arriving from the rest of the network.', 'If the incoming gradient is 2, sigmoid at zero passes back 0.5.'],
          ['L', 'The loss being minimized.', 'A classification loss measures how wrong the prediction was.'],
        ],
      },
      code: { title: 'Outputs and local gradients', lang: 'python', snippet: 'import numpy as np\n\nz = np.array([-4.0, 0.0, 4.0])\nsigmoid = np.exp(-np.logaddexp(0.0, -z))\ntanh = np.tanh(z)\nrelu = np.maximum(z, 0.0)\n\nsigmoid_grad = sigmoid * (1.0 - sigmoid)\ntanh_grad = 1.0 - tanh ** 2\n# ReLU is not differentiable at 0. Use 0 by convention.\nrelu_grad = (z > 0.0).astype(float)\nincoming_grad = np.ones_like(z)\ngrad_to_score = incoming_grad * sigmoid_grad' },
      quiz: { prompt: 'A sigmoid outputs about 0.982 at z = 4. What happens to an incoming gradient of 1?', options: [
        { text: 'It becomes about 0.982, because that is the output.', correct: false, explanation: 'Backprop multiplies by the derivative, not the output. These are different quantities.' },
        { text: 'It becomes about 0.018, because the curve is nearly flat.', correct: true, explanation: 'The derivative is 0.982 x (1 - 0.982), about 0.018. A large output can still pass very little gradient.' },
        { text: 'It stays 1, as it would through a positive ReLU.', correct: false, explanation: 'Positive ReLU has slope 1. Sigmoid at 4 is saturated and has a much smaller slope.' },
      ] }, controls: [], presets: [],
    },
    'output-functions': {
      title: 'Can several labels be true, or must one choice win?',
      summary: 'An image can contain both a cat and a dog. A next-token prediction chooses one token. Those tasks need different probability constraints.',
      what: 'A <strong>logit</strong> is a raw model score. Independent sigmoids turn each score into its own yes/no probability. Softmax turns a group of scores into <strong>one distribution that sums to 1</strong>. Raising one softmax score takes probability away from its competitors.',
      why: 'Image tagging and document tagging can have multiple correct labels. Next-token prediction and single-label classification choose one of mutually exclusive options. The output should encode the task, not an arbitrary preference for a formula.',
      interview: '<strong>Binary equivalence:</strong> sigmoid(z) equals the second probability of softmax([0, z]). The important distinction is multiple independent sigmoid decisions versus one normalized multiclass decision.',
      details: [
        'Independent outputs mean the sigmoid normalization does not couple the probabilities. They do not claim that real-world labels are statistically independent; the network can still share features.',
        'For binary softmax with logits [a, b], the second probability is sigmoid(b - a). Subtracting the same constant from every logit changes no softmax probability.',
        'Training commonly pairs independent logits with binary cross-entropy and mutually exclusive logits with categorical cross-entropy. Stable loss implementations accept raw logits; do not apply sigmoid or softmax twice.',
        'In recommendations, a pointwise click model can use sigmoid for each user-item pair. A listwise softmax instead models relative preference within a candidate set; those shares are not independent click-through probabilities.',
        'The 3-token example is deliberately small. A real language model normalizes over its full vocabulary. Output probabilities are model estimates, not automatically calibrated real-world frequencies.',
      ],
      math: {
        title: 'Separate probabilities or a shared denominator',
        formula: [String.raw`p_i=\sigma(z_i)=\frac{1}{1+e^{-z_i}}`, String.raw`p_i=\operatorname{softmax}(\mathbf z)_i=\frac{e^{z_i}}{\sum_j e^{z_j}}`, String.raw`\operatorname{softmax}([0,z])=[1-\sigma(z),\sigma(z)]`],
        note: 'For scores [2, 2, -1], separate sigmoids give about [0.881, 0.881, 0.269]. Softmax gives about [0.488, 0.488, 0.024]. Only the second list must sum to 1.',
        annotations: [
          ['z_i', 'Raw score for candidate i, before conversion to probability.', 'In the toy image, the cat and dog scores are both 2.'],
          ['p_i', 'Model probability for a label or one exclusive choice.', 'An image can have cat probability 0.881 and dog probability 0.881 at the same time.'],
          [String.raw`e^{z_i}`, 'Positive weight obtained by exponentiating a score.', 'Score 2 contributes exp(2), about 7.39, to the softmax weights.'],
          [String.raw`\sum_j e^{z_j}`, 'Total weight of all competing candidates.', 'For [2, 2, -1], the total is about 15.15; cat receives 7.39 / 15.15, about 0.488.'],
          ['[0,z]', 'Two binary scores expressed relative to a zero baseline.', 'With z = 2, binary softmax gives [0.119, 0.881]; its second value is sigmoid(2).'],
        ],
      },
      code: { title: 'Identical scores, different tasks', lang: 'python', snippet: 'import numpy as np\n\nscores = np.array([2.0, 2.0, -1.0])  # cat, dog, bird\n\n# Multiple image tags can be true at once.\ntag_probs = np.exp(-np.logaddexp(0.0, -scores))\n\n# One next token: subtract max for numerical stability.\nweights = np.exp(scores - scores.max())\nnext_token_probs = weights / weights.sum()\n\n# Binary equivalence: sigmoid(z) == softmax([0, z])[1].\nz = 2.0\npair = np.array([0.0, z])\npair_weights = np.exp(pair - pair.max())\nbinary_softmax = pair_weights / pair_weights.sum()\nassert np.isclose(binary_softmax[1],\n                  np.exp(-np.logaddexp(0.0, -z)))' },
      quiz: { prompt: 'Only the dog score rises; the cat score stays fixed. What happens to the cat probability?', options: [
        { text: 'It falls for both independent sigmoid and softmax.', correct: false, explanation: 'An independent cat sigmoid only uses the cat score, so changing the dog score cannot change it.' },
        { text: 'It is unchanged for both.', correct: false, explanation: 'Softmax uses a shared denominator. The larger dog weight takes part of the fixed probability budget.' },
        { text: 'It stays fixed with independent sigmoid, but falls with softmax.', correct: true, explanation: 'Exactly. Separate yes/no decisions do not compete for normalization; mutually exclusive choices do. Binary sigmoid is still mathematically equivalent to a two-class softmax.' },
      ] }, controls: [], presets: [],
    },
  };
  function has(kind) { return Object.prototype.hasOwnProperty.call(content, kind); }
  const api = { has, render, content, sigmoid, softmax, activation, outputProbabilities };
  if (typeof window !== 'undefined') {
    window.AtelierNeuralFunctions = api;
    (window.AtelierLessonModules ||= []).push(api);
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})();
