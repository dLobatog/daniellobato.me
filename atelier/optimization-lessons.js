/* Native, deterministic optimization lessons. No framework or animation loop. */
(() => {
  'use strict';

  const START = Object.freeze([-3, 1.2]);
  const BUDGET = 24;
  const METHODS = ['sgd', 'momentum', 'adam'];
  const SCHEDULES = ['constant', 'step', 'cosine'];
  const NAMES = { sgd: 'SGD', momentum: 'Momentum', adam: 'Adam', constant: 'Constant', step: 'Step decay', cosine: 'Cosine' };
  const KINDS = ['gradient-descent', 'optimizers', 'lr-schedule'];
  const PRESETS = {
    small: { label: 'Too small', lr: 0.02 },
    useful: { label: 'Useful step', lr: 0.12 },
    hot: { label: 'Overshoot', lr: 0.26 },
  };

  function validPoint(point) {
    if (!Array.isArray(point) || point.length !== 2 || !point.every(Number.isFinite)) {
      throw new TypeError('A point must contain two finite weights.');
    }
  }
  function loss(point) {
    validPoint(point);
    return (point[0] ** 2 + 9 * point[1] ** 2) / 2;
  }
  function gradient(point) {
    validPoint(point);
    return [point[0], 9 * point[1]];
  }
  function initialState(point = START) {
    validPoint(point);
    if (!Number.isFinite(loss(point)) || loss(point) > 1e10 || point.some(value => Math.abs(value) > 1e6)) {
      throw new RangeError('The starting point is outside the safe numerical range.');
    }
    return { step: 0, point: [...point], loss: loss(point), velocity: [0, 0], m: [0, 0], v: [0, 0] };
  }

  function optimizerStep(previous, options = {}) {
    const { method = 'sgd', lr = 0.12, beta1 = 0.9, beta2 = 0.999, momentum = 0.9, epsilon = 1e-8 } = options;
    if (!METHODS.includes(method)) throw new RangeError('Unknown optimizer.');
    if (!Number.isFinite(lr) || lr < 0) throw new RangeError('Learning rate must be finite and nonnegative.');
    if (![beta1, beta2, momentum].every(b => Number.isFinite(b) && b >= 0 && b < 1) || !Number.isFinite(epsilon) || epsilon <= 0) {
      throw new RangeError('Moment coefficients must be in [0, 1); epsilon must be positive.');
    }
    const g = gradient(previous.point);
    const step = previous.step + 1;
    const velocity = g.map((value, i) => momentum * previous.velocity[i] + value);
    const m = g.map((value, i) => beta1 * previous.m[i] + (1 - beta1) * value);
    const v = g.map((value, i) => beta2 * previous.v[i] + (1 - beta2) * value ** 2);
    const mHat = m.map(value => value / (1 - beta1 ** step));
    const vHat = v.map(value => value / (1 - beta2 ** step));
    const direction = method === 'sgd' ? g : method === 'momentum' ? velocity : mHat.map((value, i) => value / (Math.sqrt(vHat[i]) + epsilon));
    const delta = direction.map(value => -lr * value);
    const point = previous.point.map((value, i) => value + delta[i]);
    // Stop before a non-finite value or a meaningless huge coordinate reaches the UI.
    const nextLoss = point.every(Number.isFinite) ? loss(point) : Infinity;
    if (!Number.isFinite(nextLoss) || nextLoss > 1e10 || point.some(value => Math.abs(value) > 1e6)) {
      return { stopped: true, reason: 'The loss exceeded the safety limit. Reset or choose a smaller learning rate.' };
    }
    return { step, point, loss: nextLoss, velocity, m, v, mHat, vHat, gradient: g, direction, delta, lr, stopped: false };
  }

  // index is zero-based: index 12 is the thirteenth update.
  function scheduleRate(schedule, index, budget = BUDGET, peak = 0.18) {
    if (!SCHEDULES.includes(schedule)) throw new RangeError('Unknown schedule.');
    if (!Number.isInteger(budget) || budget < 2 || !Number.isInteger(index) || index < 0 || index >= budget || !Number.isFinite(peak) || peak < 0) {
      throw new RangeError('Schedule requires a valid update index, budget, and peak rate.');
    }
    if (schedule === 'constant') return peak;
    if (schedule === 'step') return index < Math.floor(budget / 2) ? peak : peak * 0.1;
    return peak * (1 + Math.cos(Math.PI * index / (budget - 1))) / 2;
  }

  function trajectory(options = {}) {
    const { method = 'sgd', lr = 0.12, schedule, budget = BUDGET, start = START } = options;
    if (!Number.isInteger(budget) || budget < 1 || budget > 10000) throw new RangeError('Invalid update budget.');
    const points = [initialState(start)];
    for (let index = 0; index < budget; index++) {
      const next = optimizerStep(points[points.length - 1], { ...options, method, lr: schedule ? scheduleRate(schedule, index, budget, lr) : lr });
      if (next.stopped) return { points, stopped: true, reason: next.reason };
      points.push(next);
    }
    return { points, stopped: false, reason: '' };
  }

  // Liang-Barsky clipping keeps even a diverging real path inside finite SVG bounds.
  function clipSegment(a, b, bounds) {
    validPoint(a);
    validPoint(b);
    const [xmin, xmax, ymin, ymax] = bounds;
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const p = [-dx, dx, -dy, dy];
    const q = [a[0] - xmin, xmax - a[0], a[1] - ymin, ymax - a[1]];
    let enter = 0, leave = 1;
    for (let i = 0; i < 4; i++) {
      if (p[i] === 0) { if (q[i] < 0) return null; }
      else {
        const r = q[i] / p[i];
        if (p[i] < 0) enter = Math.max(enter, r);
        else leave = Math.min(leave, r);
        if (enter > leave) return null;
      }
    }
    return [[a[0] + enter * dx, a[1] + enter * dy], [a[0] + leave * dx, a[1] + leave * dy]];
  }

  const content = {
    'gradient-descent': {
      title: 'Gradient descent: the slope tells you where, not how far',
      summary: 'Move two weights down a loss valley. A sensible step lowers the loss; a bigger step can cross the valley and land higher.',
      what: 'The <strong>gradient</strong> measures how the loss changes when each weight moves a little. It points uphill. Gradient descent subtracts it, scaled by a <strong>learning rate</strong>. The slope is local: a large step need not improve the loss.',
      why: 'Training adjusts many weights using this same rule. Our two-weight model makes the direction and the step size visible.',
      interview: 'The negative gradient is a local downhill direction. The learning rate still has to be small enough: <strong>a downhill direction is not a guarantee of a downhill destination.</strong>',
      details: [
        'This is a deterministic quadratic, L(w) = (w1 squared + 9 times w2 squared) / 2. The vertical direction has nine times the curvature. Ellipses connect weights with equal loss; the minimum is (0, 0).',
        'For this quadratic, constant-rate gradient descent converges from every starting point when 0 < learning rate < 2/9. The overshoot preset uses 0.26, outside that range. This numerical threshold is specific to this loss.',
        'We use the exact gradient. In minibatch SGD, the gradient is instead estimated from part of the training data.',
      ],
      math: {
        title: 'One update, with actual numbers',
        formula: ['L(w)=\\tfrac12(w_1^2+9w_2^2),\\quad \\nabla L(w)=(w_1,9w_2)', 'w_{t+1}=w_t-\\eta\\nabla L(w_t)', '(-3,1.2)-0.12(-3,10.8)=(-2.64,-0.096)'],
        note: 'The first useful step lowers loss from 10.980 to 3.526. Negative w2 means it crossed the center, but the loss still fell.',
        annotations: [
          ['w_t', 'The two weights before update t + 1.', 'Start at w = (-3, 1.2); the best pair for this toy model is (0, 0).'],
          ['L(w)', 'The error assigned to those weights.', 'At (-3, 1.2), loss is (9 + 12.96) / 2 = 10.98.'],
          ['\\nabla L(w_t)', 'The slope with respect to each weight.', 'The initial gradient is (-3, 10.8): increasing w1 a little helps, while increasing w2 hurts.'],
          ['\\eta', 'The learning rate, which scales the move.', 'With 0.12, weight 1 moves +0.36 and weight 2 moves -1.296.'],
        ],
      },
      code: { title: 'Trace the exact path', lang: 'python', snippet: 'import numpy as np\n\nw = np.array([-3.0, 1.2])\nlr = 0.12  # try 0.02 or 0.26\n\ndef loss(w):\n    return 0.5 * (w[0] ** 2 + 9 * w[1] ** 2)\n\nfor step in range(24):\n    before = loss(w)\n    grad = np.array([w[0], 9 * w[1]])\n    w = w - lr * grad\n    print(step + 1, w, before, loss(w))' },
      quiz: { prompt: 'The gradient is correct, but the next loss is higher. What can explain it?', options: [
        { text: 'The learning rate carried the weights too far along a local downhill direction.', correct: true, explanation: 'Yes. In the overshoot preset, the vertical error grows because the learning rate exceeds 2/9.' },
        { text: 'A correct gradient guarantees every finite step lowers loss.', correct: false, explanation: 'Only sufficiently small steps have that local guarantee away from a stationary point.' },
        { text: 'Crossing zero always makes the loss worse.', correct: false, explanation: 'The useful preset crosses w2 = 0 but lands much closer to the minimum.' },
      ] }, controls: [], presets: [],
    },
    optimizers: {
      title: 'Same slopes. Different memories. Different paths.',
      summary: 'Compare SGD, momentum, and Adam on the same valley, from the same starting point, with the same number of updates.',
      what: '<strong>SGD</strong> uses today\'s gradient. <strong>Momentum</strong> also carries a velocity from previous gradients. <strong>Adam</strong> divides a running gradient average by a running size estimate, separately for each weight.',
      why: 'An optimizer changes how a gradient becomes a move. Inspect a step to see that transformation, rather than treating the method as a speed setting.',
      interview: '<strong>Adam is not universally better.</strong> Its normalization changes the meaning of the learning rate. Compare tuned methods at equal compute budgets, not just at an identical numerical rate.',
      details: [
        'These paths use exact gradients without sampling noise. The label SGD means the usual SGD update applied to the full toy loss; this is not a simulation of noisy minibatches.',
        'Momentum here uses velocity = 0.9 times old velocity + gradient, without a (1 - beta) factor. Both conventions exist, but their learning rates are not numerically interchangeable.',
        'Adam uses beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, zero initial moments, and bias correction at every step. There is no weight decay.',
        'The shared learning rate makes the recurrences comparable, not the tuning optimal. Momentum can carry the weights past the minimum; Adam can be slower than SGD on this smooth problem.',
      ],
      math: {
        title: 'What each method remembers',
        formula: ['g_t=\\nabla L(w_{t-1}),\\qquad w_t=w_{t-1}-\\eta g_t\\quad\\text{(SGD)}', 'u_t=0.9u_{t-1}+g_t,\\qquad w_t=w_{t-1}-\\eta u_t', 'm_t=\\beta_1m_{t-1}+(1-\\beta_1)g_t,\\quad v_t=\\beta_2v_{t-1}+(1-\\beta_2)g_t^2', '\\hat m_t=\\frac{m_t}{1-\\beta_1^t},\\quad\\hat v_t=\\frac{v_t}{1-\\beta_2^t},\\quad w_t=w_{t-1}-\\eta\\frac{\\hat m_t}{\\sqrt{\\hat v_t}+\\epsilon}'],
        note: 'All operations on vectors are coordinatewise. Updates start at t = 1; all memories start at zero. Momentum uses an unnormalized accumulated velocity.',
        annotations: [
          ['g_t', 'Current slope, before taking this update.', 'At the shared start it is (-3, 10.8), so the vertical slope is much steeper.'],
          ['u_t', 'Momentum velocity: old direction plus the new gradient.', 'For the first weight, u1 = -3 and u2 = 0.9(-3) - 2.64 = -5.34 at learning rate 0.12.'],
          ['m_t,\\ v_t', 'Running averages of gradients and squared gradients.', 'Adam\'s first first-weight moments are -0.3 and 0.009. The squared moment measures size, not sign.'],
          ['\\hat m_t,\\ \\hat v_t', 'Moments corrected for starting at zero.', 'At t = 1, correction restores -3 and 9 for the first weight. Its normalized direction is approximately -1.'],
          ['\\beta_1,\\beta_2,\\epsilon', 'Memory rates and a small denominator safeguard.', 'We use 0.9, 0.999, and 0.00000001. Epsilon prevents division by zero.'],
          ['\\eta', 'Scale applied to the method\'s direction.', 'At 0.12, SGD initially moves (+0.36, -1.296); Adam moves approximately (+0.12, -0.12).'],
        ],
      },
      code: { title: 'Three real update rules', lang: 'python', snippet: 'import numpy as np\n\ndef run(method, lr=0.12, steps=24):\n    w = np.array([-3.0, 1.2])\n    velocity = np.zeros(2)\n    m = np.zeros(2)\n    v = np.zeros(2)\n    for t in range(1, steps + 1):\n        g = np.array([w[0], 9 * w[1]])\n        if method == "sgd":\n            direction = g\n        elif method == "momentum":\n            velocity = 0.9 * velocity + g\n            direction = velocity\n        elif method == "adam":\n            m = 0.9 * m + 0.1 * g\n            v = 0.999 * v + 0.001 * g * g\n            m_hat = m / (1 - 0.9 ** t)\n            v_hat = v / (1 - 0.999 ** t)\n            direction = m_hat / (np.sqrt(v_hat) + 1e-8)\n        else:\n            raise ValueError(method)\n        w = w - lr * direction\n    return w' },
      quiz: { prompt: 'Adam has a higher loss after 24 steps here. What can we conclude?', options: [
        { text: 'SGD is always a better optimizer.', correct: false, explanation: 'This is one noiseless quadratic, one start, and a shared learning rate, not a benchmark of tuned optimizers.' },
        { text: 'At these settings and this budget, Adam\'s path was less effective.', correct: true, explanation: 'Exactly. The recurrence changes the path; the best learning rate and optimizer depend on the problem.' },
        { text: 'Bias correction must have been omitted.', correct: false, explanation: 'Bias correction is included. It corrects zero-initialization bias; it does not guarantee faster convergence.' },
      ] }, controls: [], presets: [],
    },
    'lr-schedule': {
      title: 'A schedule changes how far you trust the next slope',
      summary: 'Keep SGD fixed. Change only its learning-rate schedule, then inspect the update where the step budget shrinks.',
      what: 'A <strong>learning-rate schedule</strong> changes the multiplier on the gradient over time. Step decay cuts it at a chosen update. Cosine decay reduces it smoothly. Neither changes the gradient formula.',
      why: 'Smaller late steps can reduce motion caused by noisy gradients. They can also slow useful progress when there is no noise, as this example shows.',
      interview: '<strong>Decay is a tradeoff, not a guarantee.</strong> It reduces later movement; choosing when to decay depends on the training budget and the behavior of the loss.',
      details: [
        'All three paths use full-gradient SGD, start at (-3, 1.2), and get 24 updates. Constant uses 0.18 throughout. Step decay uses 0.18 for updates 1-12, then 0.018. Cosine goes from 0.18 on update 1 to zero on update 24.',
        'On this noiseless quadratic, a stable constant rate already converges. Reducing it early can leave more error after the same update budget. This is deliberately not a claim that decay always helps.',
        'Warmup gradually raises the early learning rate. It can help large models avoid unstable initial updates while representations and optimizer statistics develop. This small stable problem does not need it, so it is not an extra control here.',
      ],
      math: {
        title: 'Same gradient rule, different multiplier',
        formula: ['w_t=w_{t-1}-\\eta_t\\nabla L(w_{t-1})', '\\eta_t^{\\mathrm{step}}=\\begin{cases}0.18&t\\leq12\\\\0.018&t>12\\end{cases}', '\\eta_t^{\\mathrm{cosine}}=\\frac{0.18}{2}\\left(1+\\cos\\left(\\pi\\frac{t-1}{23}\\right)\\right),\\quad t=1,\\ldots,24'],
        note: 'The displayed update numbers are one-based. Cosine includes both endpoints: 0.18 at update 1 and exactly 0 at update 24.',
        annotations: [
          ['t', 'Which update is being applied.', 'Update 13 is the first update after the step schedule cuts the rate.'],
          ['\\eta_t', 'The learning rate used for that particular update.', 'Step decay changes 0.18 to 0.018. At the same weights, that makes the move ten times smaller.'],
          ['\\nabla L(w_{t-1})', 'The slope at this path\'s current weights.', 'Constant and step decay have identical gradients before update 13 because their first 12 updates match.'],
          ['23', 'The number of gaps between 24 scheduled rates.', 'Using (t - 1)/23 makes the cosine end at zero on update 24, not one update later.'],
        ],
      },
      code: { title: 'Change only the schedule', lang: 'python', snippet: 'import math\nimport numpy as np\n\ndef run(schedule, steps=24):\n    w = np.array([-3.0, 1.2])\n    for i in range(steps):  # zero-based index\n        if schedule == "constant":\n            lr = 0.18\n        elif schedule == "step":\n            lr = 0.18 if i < steps // 2 else 0.018\n        elif schedule == "cosine":\n            lr = 0.09 * (1 + math.cos(math.pi * i / (steps - 1)))\n        else:\n            raise ValueError(schedule)\n        grad = np.array([w[0], 9 * w[1]])\n        w = w - lr * grad\n    return w' },
      quiz: { prompt: 'At update 13, step decay and constant SGD have identical weights. Why is one move smaller?', options: [
        { text: 'Step decay changed the loss function.', correct: false, explanation: 'The same loss and gradient are used. Only the multiplier changes.' },
        { text: 'The step schedule multiplies the same gradient by a ten-times-smaller rate.', correct: true, explanation: 'Yes: 0.018 instead of 0.18. Later the gradients can differ because the paths have separated.' },
        { text: 'Smaller steps must reach the minimum in fewer updates.', correct: false, explanation: 'Not here. A smaller stable step can preserve more error within a fixed budget.' },
      ] }, controls: [], presets: [],
    },
  };

  const instances = new Map();
  const bindings = new WeakMap();
  const escape = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
  function number(value) {
    if (!Number.isFinite(value)) return 'not available';
    if (value === 0) return '0';
    if (Math.abs(value) < 0.001 || Math.abs(value) >= 10000) return value.toExponential(2);
    return value.toFixed(3).replace(/\.?0+$/, '');
  }
  const pair = values => `(${values.map(number).join(', ')})`;
  function button(action, value, text, options = {}) {
    return `<button type="button" class="opt-button ${options.className || ''}" data-opt-action="${action}" data-opt-value="${value}" data-opt-key="${action}:${value}"${options.pressed === undefined ? '' : ` aria-pressed="${options.pressed}"`}${options.disabled ? ' disabled' : ''}>${text}</button>`;
  }
  function model(instance) {
    const { kind, state } = instance;
    const ids = kind === 'gradient-descent' ? ['sgd'] : kind === 'optimizers' ? METHODS : SCHEDULES;
    const tracks = Object.fromEntries(ids.map(id => [id, trajectory({ method: kind === 'lr-schedule' ? 'sgd' : id, lr: kind === 'lr-schedule' ? 0.18 : PRESETS[state.preset].lr, schedule: kind === 'lr-schedule' ? id : undefined })]));
    const active = kind === 'gradient-descent' ? 'sgd' : state.active;
    const track = tracks[active];
    const selected = Math.min(state.selected, track.points.length - 1);
    const end = Math.min(state.completed, track.points.length - 1);
    const after = track.points[selected || 1] || track.points[0];
    const before = track.points[Math.max(0, selected - 1)];
    return { ids, tracks, active, track, selected, end, before, after, current: track.points[selected] };
  }

  function plotGeometry(compact) {
    const width = compact ? 400 : 640;
    const height = compact ? 300 : 424;
    const boundY = 2.5;
    const scale = (width - 64) / 8;
    return { width, height, bounds: [-4, 4, -boundY, boundY], map: point => [width / 2 + point[0] * scale, height / 2 - point[1] * scale], scale };
  }
  function inside(point, bounds) {
    return point[0] >= bounds[0] && point[0] <= bounds[1] && point[1] >= bounds[2] && point[1] <= bounds[3];
  }
  function segmentPath(a, b, geometry) {
    const clipped = clipSegment(a, b, geometry.bounds);
    if (!clipped) return '';
    const [p, q] = clipped.map(geometry.map);
    return `M${p.map(number).join(',')}L${q.map(number).join(',')}`;
  }
  function pointsPath(points, geometry) {
    return points.slice(1).map((point, i) => segmentPath(points[i].point, point.point, geometry)).join(' ');
  }
  function mapMarkup(instance, view, compact) {
    const { width, height, map, scale, bounds } = plotGeometry(compact);
    const geometry = plotGeometry(compact);
    const contourLevels = [0.5, 2, 5, 10, 18];
    const contours = contourLevels.map(level => {
      const points = Array.from({ length: 129 }, (_, i) => ({ point: [Math.sqrt(2 * level) * Math.cos(i * Math.PI / 64), Math.sqrt(2 * level / 9) * Math.sin(i * Math.PI / 64)] }));
      return `<path class="opt-contour" d="${pointsPath(points, geometry)}"/>`;
    }).join('');
    const path = view.ids.map(id => `<path class="opt-path opt-tone-${view.ids.indexOf(id)} ${id === view.active ? 'opt-path-active' : ''}" d="${pointsPath(view.tracks[id].points.slice(0, instance.state.completed + 1), geometry)}"/>`).join('');
    const highlighted = segmentPath(view.before.point, view.after.point, geometry);
    const [cx, cy] = map([0, 0]);
    const plotted = view.track.points.slice(0, view.end + 1).filter(point => inside(point.point, bounds));
    const selectedPoint = view.track.points[view.selected];
    // Nearby steps remain reachable with Previous/Next and arrow keys, without a pile of overlapping hit targets.
    const chosen = inside(selectedPoint.point, bounds) ? [selectedPoint] : [];
    for (const point of [...plotted].reverse()) {
      if (chosen.every(other => Math.hypot(...map(point.point).map((value, i) => value - map(other.point)[i])) >= (compact ? 38 : 32))) chosen.push(point);
    }
    const nodes = chosen.map(point => {
      const [x, y] = map(point.point);
      return `<button type="button" class="opt-point opt-tone-${view.ids.indexOf(view.active)}${point.step === view.selected ? ' opt-point-selected' : ''}" style="left:${x / width * 100}%;top:${y / height * 100}%" data-opt-action="inspect" data-opt-value="${point.step}" data-opt-key="point:${point.step}" tabindex="${point.step === view.selected ? 0 : -1}" aria-pressed="${point.step === view.selected}" aria-label="${NAMES[view.active]}, step ${point.step}: weights ${pair(point.point)}, loss ${number(point.loss)}. Use arrow keys to inspect other steps." title="Step ${point.step}: w = ${pair(point.point)}, loss ${number(point.loss)}"><span></span></button>`;
    }).join('');
    const endpoint = map(view.after.point);
    const preview = view.selected === 0 && inside(view.after.point, bounds) ? `<circle class="opt-preview-dot" cx="${endpoint[0]}" cy="${endpoint[1]}" r="6"/>` : '';
    return `<figure class="opt-figure">
      <figcaption class="opt-figure-caption"><strong>A top-down view of loss</strong><span>Each ellipse joins equal loss. The cross is the minimum.</span></figcaption>
      <div class="opt-map" style="aspect-ratio:${width}/${height}" role="group" aria-label="Optimization paths. Tap a point to inspect a step, or use the Previous and Next update buttons.">
        <svg viewBox="0 0 ${width} ${height}" aria-hidden="true" focusable="false">
          <path class="opt-axis" d="M32,${cy}H${width - 32}M${cx},${cy - bounds[3] * scale}V${cy + bounds[3] * scale}"/>
          ${contours}${path}
          <path class="opt-selected-segment${view.selected === 0 ? ' opt-preview' : ''}" d="${highlighted}"/>
          ${preview}<path class="opt-minimum" d="M${cx - 5},${cy}h10M${cx},${cy - 5}v10"/>
        </svg>${nodes}
        <span class="opt-y-label">w<sub>2</sub>: +${bounds[3]}</span>
        <span class="opt-y-label opt-y-bottom">-${bounds[3]}</span>
      </div>
      <div class="opt-x-label"><span>-4</span><span>Weight w<sub>1</sub></span><span>+4</span></div>
      <p class="opt-plot-note">${!inside(selectedPoint.point, bounds) ? `<button type="button" class="opt-button opt-outside-step" data-opt-action="inspect" data-opt-value="${view.selected}" data-opt-key="point:${view.selected}" aria-label="Step ${view.selected}, outside the plot. Weights ${pair(selectedPoint.point)}, loss ${number(selectedPoint.loss)}. Use arrow keys to inspect other steps.">Step ${view.selected}: outside this view</button> Actual weights and loss remain in the inspector. The plot scale has not changed.` : view.selected === 0 ? 'The dashed segment previews the first update. Press Next update to apply it.' : `The highlighted segment is update ${view.selected}. Tap a point to inspect it; arrow keys reach every completed step.`}</p>
    </figure>`;
  }

  function updateMarkup(instance, view) {
    const { before, after, selected, active } = view;
    if (!after.gradient) return '<div class="opt-inspector"><p>No safe update is available. Choose a smaller learning rate.</p></div>';
    const isAdam = active === 'adam', isMomentum = active === 'momentum';
    const label = isAdam ? 'Normalize remembered slopes' : isMomentum ? 'Add remembered velocity' : 'Use the current slope';
    const rows = [
      ['Weights before', pair(before.point)],
      ['Gradient here', pair(after.gradient)],
    ];
    if (isMomentum) {
      rows.push(['Previous velocity', pair(before.velocity)], ['0.9 velocity + gradient', pair(after.velocity)]);
    }
    if (isAdam) {
      rows.push(['Corrected mean, m-hat', pair(after.mHat)], ['Corrected size, sqrt(v-hat)', pair(after.vHat.map(Math.sqrt))], ['Normalized direction', pair(after.direction)]);
    }
    rows.push(['Learning rate', number(after.lr)], ['Move = -rate x direction', pair(after.delta)], ['Weights after', pair(after.point)]);
    const comparison = after.loss > before.loss + 1e-12 ? 'rose' : after.loss < before.loss - 1e-12 ? 'fell' : 'stayed the same';
    return `<aside class="opt-inspector" aria-label="Exact update inspector">
      <div class="opt-inspector-title"><span>${selected === 0 ? 'Preview' : `Update ${selected}`}</span><h4>${label}</h4></div>
      <dl class="opt-calculation">${rows.map(([name, value], i) => `<div${i === rows.length - 2 ? ' class="opt-move-row"' : ''}><dt>${name}</dt><dd>${value}</dd></div>`).join('')}</dl>
      <div class="opt-loss-change"><span>Loss before <strong>${number(before.loss)}</strong></span><span aria-hidden="true">&rarr;</span><span>Loss after <strong>${number(after.loss)}</strong></span></div>
      <p class="opt-result">${selected === 0 ? `This proposed update ${comparison === 'rose' ? 'raises' : comparison === 'fell' ? 'lowers' : 'does not change'} the loss.` : `The loss ${comparison}.`} ${after.loss > before.loss + 1e-12 ? 'A correct gradient can still lead to a worse destination.' : after.lr === 0 ? 'The rate is zero, not the gradient. Training has stopped moving.' : ''}</p>
      <p class="opt-equation">${isAdam ? 'w next = w - rate x m-hat / (sqrt(v-hat) + epsilon)' : isMomentum ? 'velocity = 0.9 x velocity + gradient; w next = w - rate x velocity' : 'w next = w - rate x gradient'}</p>
    </aside>`;
  }
  function comparisonMarkup(instance, view) {
    return `<div class="opt-comparison" role="group" aria-label="Choose a path to inspect">${view.ids.map((id, i) => {
      const path = view.tracks[id];
      const point = path.points[Math.min(instance.state.selected, path.points.length - 1)];
      return button('active', id, `<span class="opt-method-name"><i class="opt-line-sample opt-tone-${i}" aria-hidden="true"></i>${NAMES[id]}</span><span>Loss ${number(point.loss)}</span>`, { pressed: view.active === id, className: 'opt-method' });
    }).join('')}</div>`;
  }
  function scheduleMarkup(instance, view) {
    const width = 640, height = 120, left = 16, right = 624;
    const index = Math.max(0, view.selected - 1);
    const x = i => left + (right - left) * i / (BUDGET - 1);
    const y = rate => height - 12 - rate / 0.18 * (height - 30);
    const selectedX = x(index);
    const lines = SCHEDULES.map((id, color) => {
      const points = Array.from({ length: BUDGET }, (_, i) => {
        const corner = id === 'step' && i > 0 ? `${x(i)},${y(scheduleRate(id, i - 1))} ` : '';
        return `${corner}${x(i)},${y(scheduleRate(id, i))}`;
      }).join(' ');
      return `<polyline class="opt-path opt-tone-${color} ${id === view.active ? 'opt-path-active' : ''}" points="${points}"/>`;
    }).join('');
    return `<div class="opt-schedule">
      <div class="opt-schedule-heading"><strong>The rate used on each update</strong><span>Update ${index + 1}: <b>${number(scheduleRate(view.active, index))}</b></span></div>
      <div class="opt-schedule-chart"><span class="opt-schedule-high">0.18</span><span class="opt-schedule-low">0</span>
        <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-hidden="true"><path class="opt-axis" d="M16,108H624"/>${lines}<path class="opt-schedule-cursor" d="M${selectedX},10V112"/></svg>
        <div class="opt-schedule-hitpoints" role="group" aria-label="Inspect a scheduled update">${Array.from({ length: BUDGET }, (_, i) => `<button type="button" data-opt-action="scheduled" data-opt-value="${i + 1}" data-opt-key="scheduled:${i + 1}" tabindex="${i === index ? 0 : -1}" aria-label="Inspect update ${i + 1}: ${NAMES[view.active]} learning rate ${number(scheduleRate(view.active, i))}. Arrow keys move between updates." title="Update ${i + 1}: rate ${number(scheduleRate(view.active, i))}"></button>`).join('')}</div>
      </div><div class="opt-schedule-axis"><span>Update 1</span><span>Update 12</span><span>Update 24</span></div>
      <p class="opt-small">Tap the schedule to jump to an update. Its computed path and weights appear above.</p>
    </div>`;
  }
  function takeaway(instance, view) {
    if (instance.kind === 'gradient-descent') {
      if (instance.state.preset === 'hot') return 'The vertical error is multiplied by -1.34 each update: it flips side and grows. This is divergence, not exploration.';
      if (instance.state.preset === 'small') return 'A smaller learning rate makes safer but slower progress here. The horizontal error keeps 98% of its size after every update.';
      return 'Crossing the center is not automatically bad. At this rate the vertical error shrinks to 8% of its size, while the horizontal error keeps 88%.';
    }
    if (instance.kind === 'lr-schedule') {
      if (view.selected === 13 && view.active === 'step') return 'At update 13, step decay starts from exactly the same weights as constant SGD but makes a ten-times-smaller move. That is the schedule, not a new gradient rule.';
      return 'Here the gradients have no noise, and the constant rate is stable. Decay reduces motion but can leave more error after 24 updates. Less movement is not automatically better learning.';
    }
    if (view.active === 'adam') return 'Adam rescales each coordinate using its gradient history. On the first update, very different slopes produce nearly equal-sized moves. That is useful in some problems, not a guarantee of winning this one.';
    if (view.active === 'momentum') return 'Momentum carries velocity forward. It can travel faster in a consistent direction, but that same memory can carry it past the minimum.';
    return 'SGD follows the current slope with no memory. This smooth, noiseless problem can suit it well; minibatch noise and independently tuned rates could change the comparison.';
  }
  function markup(instance, compact) {
    const view = model(instance), { kind, state } = instance;
    const gd = kind === 'gradient-descent', schedules = kind === 'lr-schedule';
    const headings = { 'gradient-descent': 'A downhill direction can still overshoot.', optimizers: 'Change the update rule, not the problem.', 'lr-schedule': 'The same slope. A different-sized move.' };
    const presets = !schedules ? `<div class="opt-presets" role="group" aria-label="Learning-rate stories">${Object.entries(PRESETS).map(([key, preset]) => button('preset', key, `${gd ? preset.label : key === 'small' ? 'Small rate' : key === 'useful' ? 'Stable SGD rate' : 'Unstable SGD rate'} <span class="opt-rate">${number(preset.lr)}</span>`, { pressed: state.preset === key })).join('')}</div>` : '';
    return `<header class="opt-heading"><h3>${headings[kind]}</h3><p>Two weights. One loss: <span class="opt-inline-math">L = (w<sub>1</sub><sup>2</sup> + 9w<sub>2</sub><sup>2</sup>) / 2</span>. Start at <span class="opt-inline-math">(-3, 1.2)</span>; aim for <span class="opt-inline-math">(0, 0)</span>.</p></header>
      ${presets}${gd ? '' : comparisonMarkup(instance, view)}
      <nav class="opt-navigation" aria-label="Step through the optimization">
        <div class="opt-step-buttons">${button('previous', '', 'Previous', { disabled: view.selected === 0 })}${button('next', '', 'Next update', { className: 'opt-primary', disabled: view.selected >= view.track.points.length - 1 })}</div>
        <span class="opt-step-count">Step <strong>${view.selected}</strong> of ${BUDGET}${view.selected === 0 ? ' (start)' : ''}${gd ? ` &middot; Loss <strong>${number(view.current.loss)}</strong>` : ''}</span>
        <div class="opt-secondary-buttons">${button('finish', '', 'Show all 24', { disabled: state.completed === BUDGET && view.selected === BUDGET })}${button('reset', '', 'Reset', { disabled: state.completed === 0 })}</div>
      </nav>
      <div class="opt-workspace">${mapMarkup(instance, view, compact)}${updateMarkup(instance, view)}</div>
      ${schedules ? scheduleMarkup(instance, view) : ''}
      ${view.track.stopped ? `<p class="opt-warning" role="note">${escape(view.track.reason)} Last safe update: ${view.track.points.length - 1}.</p>` : ''}
      <p class="opt-takeaway">${takeaway(instance, view)}</p>
      ${gd ? '' : `<p class="opt-small">${schedules ? 'Same SGD rule and 24-update budget. Only the learning-rate schedule changes.' : 'Same start, learning rate, and update budget. Exact gradients, no sampling noise; rates are not separately tuned.'}</p>`}`;
  }

  function getInstance(kind) {
    if (!instances.has(kind)) {
      const completed = kind === 'optimizers' ? 6 : kind === 'lr-schedule' ? 13 : 0;
      instances.set(kind, { kind, state: { preset: 'useful', active: kind === 'lr-schedule' ? 'step' : 'sgd', completed, selected: completed }, roots: new Set() });
    }
    return instances.get(kind);
  }
  function paintRoot(root, announce = false) {
    const binding = bindings.get(root);
    if (!binding) return;
    const activeElement = root.ownerDocument.activeElement;
    const focusKey = root.contains(activeElement) ? activeElement.dataset.optKey : null;
    binding.content.innerHTML = markup(binding.instance, binding.compact);
    if (focusKey) {
      const focus = [...root.querySelectorAll('[data-opt-key]')].find(element => element.dataset.optKey === focusKey && !element.disabled);
      (focus || root.querySelector('[data-opt-action="previous"]:not(:disabled)') || root.querySelector('[data-opt-action="next"]'))?.focus({ preventScroll: true });
    }
    if (announce) {
      const view = model(binding.instance);
      binding.status.textContent = `${NAMES[view.active]}, step ${view.selected}. Weights ${pair(view.current.point)}. Loss ${number(view.current.loss)}.`;
    }
  }
  function paint(instance) {
    instance.roots.forEach(root => {
      const binding = bindings.get(root);
      if (!root.isConnected && binding?.seenConnected) { unmount(root); return; }
      if (root.isConnected && binding) binding.seenConnected = true;
      paintRoot(root, true);
    });
  }
  function act(instance, action, value) {
    const s = instance.state;
    const limit = model(instance).track.points.length - 1;
    if (action === 'preset' && PRESETS[value]) {
      s.preset = value;
      s.completed = instance.kind === 'optimizers' ? 6 : 0;
      s.selected = s.completed;
    } else if (action === 'active' && (instance.kind === 'optimizers' ? METHODS : SCHEDULES).includes(value)) s.active = value;
    else if (action === 'reset') { s.completed = 0; s.selected = 0; }
    else if (action === 'finish') { s.completed = BUDGET; s.selected = Math.min(BUDGET, limit); }
    else if (action === 'previous') s.selected = Math.max(0, s.selected - 1);
    else if (action === 'next') { s.selected = Math.min(limit, s.selected + 1); s.completed = Math.max(s.completed, s.selected); }
    else if (action === 'inspect' || action === 'scheduled') {
      const index = Number(value);
      if (!Number.isInteger(index)) return;
      s.selected = Math.max(0, Math.min(index, action === 'scheduled' ? limit : Math.min(s.completed, limit)));
      if (action === 'scheduled') s.completed = Math.max(s.completed, s.selected);
    } else return;
    paint(instance);
  }
  function unmount(root) {
    const binding = bindings.get(root);
    if (!binding) return;
    binding.observer?.disconnect();
    root.removeEventListener('click', binding.click);
    root.removeEventListener('keydown', binding.keydown);
    binding.instance.roots.delete(root);
    bindings.delete(root);
  }
  function render(kind, root) {
    if (!KINDS.includes(kind)) return null;
    if (!root || !root.ownerDocument) throw new TypeError('A DOM root is required.');
    let binding = bindings.get(root);
    if (binding && (binding.instance.kind !== kind || !root.contains(binding.content))) { unmount(root); binding = null; }
    if (!binding) {
      const instance = getInstance(kind);
      root.innerHTML = '<div class="lesson opt-lesson"><div class="opt-content"></div><p class="opt-sr" role="status" aria-live="polite" aria-atomic="true"></p></div>';
      binding = { instance, content: root.querySelector('.opt-content'), status: root.querySelector('.opt-sr'), compact: root.clientWidth < 560, seenConnected: root.isConnected };
      binding.click = event => {
        const button = event.target.closest?.('[data-opt-action]');
        if (!button || !root.contains(button) || button.disabled) return;
        event.stopPropagation();
        act(instance, button.dataset.optAction, button.dataset.optValue);
      };
      binding.keydown = event => {
        const target = event.target.closest?.('[data-opt-action="inspect"], [data-opt-action="scheduled"]');
        if (!target || !root.contains(target)) return;
        const key = event.key;
        if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(key)) return;
        event.preventDefault();
        event.stopPropagation();
        const schedule = target.dataset.optAction === 'scheduled';
        const index = Number(target.dataset.optValue);
        const first = schedule ? 1 : 0;
        const last = schedule ? BUDGET : instance.state.completed;
        const next = key === 'Home' ? first : key === 'End' ? last : index + (key === 'ArrowRight' || key === 'ArrowUp' ? 1 : -1);
        const value = Math.max(first, Math.min(last, next));
        act(instance, schedule ? 'scheduled' : 'inspect', value);
        root.querySelector(`[data-opt-key="${schedule ? 'scheduled' : 'point'}:${value}"]`)?.focus({ preventScroll: true });
      };
      bindings.set(root, binding);
      instance.roots.add(root);
      root.addEventListener('click', binding.click);
      root.addEventListener('keydown', binding.keydown);
      const Resize = root.ownerDocument.defaultView?.ResizeObserver;
      if (Resize) {
        binding.observer = new Resize(entries => {
          const width = entries[0]?.contentRect.width;
          if (!width) return;
          const compact = width < 560;
          if (compact !== binding.compact) { binding.compact = compact; paintRoot(root); }
        });
        binding.observer.observe(root);
      }
    }
    paintRoot(root);
    // A parent can call api.unmount(root), or invoke the returned cleanup during lazy unmount.
    const dispose = () => unmount(root);
    dispose.destroy = dispose;
    dispose.unmount = dispose;
    return dispose;
  }

  const api = { has: kind => KINDS.includes(kind), render, unmount, content, loss, gradient, initialState, optimizerStep, trajectory, scheduleRate, clipSegment, START, BUDGET };
  if (typeof window !== 'undefined') {
    window.AtelierOptimization = api;
    (window.AtelierLessonModules ||= []).push(api);
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})();
