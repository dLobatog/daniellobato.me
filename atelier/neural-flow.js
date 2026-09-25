/* One delivery, four views. The math is independent of the DOM and shared by every view. */
(() => {
  'use strict';

  const KINDS = ['neuron', 'forward-pass', 'chain-rule', 'backprop'];
  const PARAMETERS = ['distance', 'stops', 'bias'];
  const EXAMPLE = Object.freeze({
    inputs: Object.freeze({ distance: 2, stops: 1 }),
    parameters: Object.freeze({ distance: 3, stops: 4, bias: 5 }),
    target: 19,
    learningRate: 0.1,
  });

  function finite(value, name) {
    if (!Number.isFinite(value)) throw new TypeError(`${name} must be a finite number`);
    return value;
  }

  function forward(inputs = EXAMPLE.inputs, parameters = EXAMPLE.parameters) {
    const contributions = {
      distance: finite(inputs.distance, 'Distance') * finite(parameters.distance, 'Distance weight'),
      stops: finite(inputs.stops, 'Stops') * finite(parameters.stops, 'Stop weight'),
      bias: finite(parameters.bias, 'Bias'),
    };
    return { contributions, prediction: contributions.distance + contributions.stops + contributions.bias };
  }

  function loss(prediction, target) {
    const error = finite(prediction, 'Prediction') - finite(target, 'Target');
    return 0.5 * error * error;
  }

  function derive(inputs = EXAMPLE.inputs, parameters = EXAMPLE.parameters, target = EXAMPLE.target) {
    const result = forward(inputs, parameters);
    const error = result.prediction - finite(target, 'Target');
    const local = { distance: inputs.distance, stops: inputs.stops, bias: 1 };
    return {
      ...result, inputs: { ...inputs }, parameters: { ...parameters }, target, error,
      loss: loss(result.prediction, target),
      local,
      gradients: Object.fromEntries(PARAMETERS.map(key => [key, error * local[key]])),
    };
  }

  function chain(parameter, inputs = EXAMPLE.inputs, parameters = EXAMPLE.parameters, target = EXAMPLE.target) {
    if (!PARAMETERS.includes(parameter)) throw new RangeError('Unknown parameter');
    const result = derive(inputs, parameters, target);
    return { predictionToLoss: result.error, parameterToPrediction: result.local[parameter], gradient: result.gradients[parameter] };
  }

  function trainStep(inputs = EXAMPLE.inputs, parameters = EXAMPLE.parameters, target = EXAMPLE.target, learningRate = EXAMPLE.learningRate) {
    finite(learningRate, 'Learning rate');
    if (learningRate < 0) throw new RangeError('Learning rate must not be negative');
    const before = derive(inputs, parameters, target);
    const updates = Object.fromEntries(PARAMETERS.map(key => [key, -learningRate * before.gradients[key]]));
    const next = Object.fromEntries(PARAMETERS.map(key => [key, parameters[key] + updates[key]]));
    return { before, after: derive(inputs, next, target), updates, learningRate };
  }

  function finiteDifference(parameter, inputs = EXAMPLE.inputs, parameters = EXAMPLE.parameters, target = EXAMPLE.target, epsilon = 1e-5) {
    if (!PARAMETERS.includes(parameter)) throw new RangeError('Unknown parameter');
    finite(epsilon, 'Epsilon');
    if (epsilon <= 0) throw new RangeError('Epsilon must be positive');
    const plus = { ...parameters, [parameter]: parameters[parameter] + epsilon };
    const minus = { ...parameters, [parameter]: parameters[parameter] - epsilon };
    return (loss(forward(inputs, plus).prediction, target) - loss(forward(inputs, minus).prediction, target)) / (2 * epsilon);
  }

  const content = {
    neuron: {
      kicker: 'One delivery, one neuron',
      question: 'How does a neuron predict a delivery time?',
      title: 'A neuron turns inputs into one prediction',
      summary: 'Multiply each input by a learned weight, then add a bias. Here, the result is a delivery time.',
      what: 'The <strong>inputs</strong> describe this delivery: distance and extra stops. The <strong>weights</strong> are learned rules, such as minutes per kilometer. The <strong>bias</strong> is a base time added to every prediction.',
      why: 'You can point to every contribution instead of treating a neuron as a mysterious black box. An extra kilometer changes the prediction by exactly the distance weight.',
      interview: 'A neuron computes a weighted sum plus bias, usually followed by an activation. This regression output uses the identity activation: it keeps the sum unchanged.',
      details: [
        'These illustrative weights are 3 minutes per kilometer, 4 minutes per extra stop, and a 5-minute base time. They are parameters, not measured facts about this particular trip.',
        'A hidden neuron often applies a nonlinear activation after the sum. This example deliberately uses a linear output so that the arithmetic remains visible.',
        'A learned weight describes how the prediction responds to an input. It does not, by itself, prove a causal relationship in the real world.',
      ],
      math: {
        title: 'Add the three contributions',
        formula: ['\\hat{y}=w_d x_d+w_s x_s+b', '\\hat{y}=3\\cdot2+4\\cdot1+5=15\\text{ min}'],
        note: 'Each product has units of minutes. For this linear output, there is no additional gate after the sum.',
        annotations: [
          ['x_d,\\ x_s', 'Observed inputs: distance and extra stops.', 'This delivery is 2 km long and has 1 extra stop.'],
          ['w_d,\\ w_s', 'Learned amounts of time per unit of each input.', '3 min/km adds 6 minutes for 2 km; 4 min/stop adds 4 minutes for 1 stop.'],
          ['b', 'A learned base time, added once.', '5 minutes are added even when distance and extra stops are both zero.'],
          ['\\hat{y}', 'The predicted delivery time.', 'The three contributions add to 6 + 4 + 5 = 15 minutes.'],
        ],
      },
      code: {
        title: 'One linear neuron', lang: 'python',
        snippet: 'distance_km, extra_stops = 2.0, 1.0\nminutes_per_km, minutes_per_stop = 3.0, 4.0\nbase_minutes = 5.0\n\nprediction = (\n    distance_km * minutes_per_km\n    + extra_stops * minutes_per_stop\n    + base_minutes\n)  # 15 minutes',
      },
      quiz: {
        prompt: 'Keep the same weights and stops. What happens when the trip becomes 1 km longer?',
        options: [
          { text: 'The prediction increases by 1 minute.', correct: false, explanation: 'One is the input change in kilometers. Multiply it by the learned rate of 3 min/km.' },
          { text: 'The prediction increases by 3 minutes.', correct: true, explanation: 'The extra contribution is 1 km times 3 min/km. The prediction goes from 15 to 18 minutes.' },
          { text: 'The bias increases from 5 to 6 minutes.', correct: false, explanation: 'Changing an input does not train the model. The weights and bias remain unchanged.' },
        ],
      }, controls: [], presets: [],
    },
    'forward-pass': {
      kicker: 'Use the rules; do not change them',
      question: 'How do inputs become a prediction?',
      title: 'The forward pass makes a prediction, then measures its error',
      summary: 'The same delivery produces a 15-minute prediction. The observed time was 19 minutes. A loss turns that mismatch into a training objective.',
      what: 'A <strong>forward pass</strong> evaluates the model with its current parameters. When a target is available, a loss compares the prediction with that target. Neither calculation changes the parameters.',
      why: 'This separates using the model from training it. At inference time, the delivery time is predicted before the actual arrival time is known.',
      interview: 'The forward pass computes intermediate values and a prediction. During training, a loss compares that prediction with a target; backward differentiation and an optimizer update come afterward.',
      details: [
        'For this single example, the signed error is prediction minus observed time: 15 - 19 = -4 minutes. The negative sign means the model predicts an arrival that is too early.',
        'We use half the squared error. Squaring penalizes either direction of error; the factor of one half makes its derivative equal to the signed error.',
        'A larger network repeats this process across layers. Its intermediate activations are needed later to compute gradients.',
      ],
      math: {
        title: 'Prediction, error, loss',
        formula: ['\\hat{y}=3\\cdot2+4\\cdot1+5=15', 'e=\\hat{y}-y=15-19=-4', 'L=\\tfrac12(\\hat{y}-y)^2=8'],
        note: 'Prediction and error are measured in minutes. This squared-error loss is measured in squared minutes, not accuracy or a percentage.',
        annotations: [
          ['\\hat{y}', 'The value computed by the model.', 'Its current rules predict 15 minutes for this trip.'],
          ['y', 'The observed target, available in a training example.', 'The delivery actually took 19 minutes.'],
          ['e', 'Signed prediction error.', '15 - 19 = -4 means the prediction was 4 minutes too early.'],
          ['L', 'A nonnegative penalty for the mismatch.', 'Half of (-4) squared is 8. A perfect prediction would have loss 0.'],
        ],
      },
      code: {
        title: 'Evaluate without changing any weights', lang: 'python',
        snippet: 'x_distance, x_stops = 2.0, 1.0\nw_distance, w_stops, bias = 3.0, 4.0, 5.0\ntarget = 19.0\n\nprediction = w_distance * x_distance + w_stops * x_stops + bias\nerror = prediction - target  # -4 minutes\nloss = 0.5 * error**2        # 8 squared minutes\n# No parameter has changed.',
      },
      quiz: {
        prompt: 'After computing the prediction and loss, which parameters have changed?',
        options: [
          { text: 'None. We have only evaluated the current model.', correct: true, explanation: 'Prediction and loss are forward calculations. Computing gradients and updating parameters are separate operations.' },
          { text: 'All weights have moved to reduce the error.', correct: false, explanation: 'A high loss does not change weights automatically. An optimizer must apply an update.' },
          { text: 'Only the bias has changed.', correct: false, explanation: 'The bias is still 5 minutes. It is a parameter, and no update has been applied.' },
        ],
      }, controls: [], presets: [],
    },
    'chain-rule': {
      kicker: 'Follow one path backward',
      question: 'How does a small weight change reach the loss?',
      title: 'The chain rule multiplies local sensitivities',
      summary: 'The distance weight changes the prediction. The prediction changes the loss. Multiplying those two sensitivities connects the weight to the loss.',
      what: 'A <strong>derivative</strong> is the local rate at which one quantity changes with another. Along a chain of computations, multiply these local rates. They come from the actual operations, not from independent knobs.',
      why: 'This makes a gradient testable: nudge one weight slightly, recompute the loss, and compare the observed change with the gradient\'s prediction.',
      interview: 'The chain rule multiplies local derivatives along a computation path. When a parameter influences the loss through several paths, their contributions add.',
      details: [
        'For the distance weight, the local derivative of w_d times x_d is x_d = 2, holding the input fixed. The stop contribution and bias do not depend on w_d.',
        'For half the squared error, the derivative with respect to the prediction is prediction minus target: -4. Their product is -8.',
        'This is the chain rule for composed functions. It is not the product rule: that rule adds terms when both factors in a product depend on the variable being differentiated.',
        'A gradient gives a local, first-order approximation. A finite change of 0.01 in the distance weight changes the loss by -0.0798, close to the predicted -0.08, but not exactly equal.',
      ],
      math: {
        title: 'Two local rates, one gradient',
        formula: ['\\frac{\\partial L}{\\partial w_d}=\\frac{\\partial L}{\\partial\\hat{y}}\\frac{\\partial\\hat{y}}{\\partial w_d}', '=(\\hat{y}-y)x_d=(-4)\\cdot2=-8', '\\Delta L\\approx\\frac{\\partial L}{\\partial w_d}\\Delta w_d'],
        note: 'These derivatives are evaluated at the current parameters. The last line is an approximation for a small change, not a promise for a large step.',
        annotations: [
          ['\\partial L/\\partial\\hat{y}', 'How sensitive the loss is to the prediction.', 'At 15 minutes versus a 19-minute target, this local rate is -4. Raising the prediction slightly lowers loss.'],
          ['\\partial\\hat{y}/\\partial w_d', 'How sensitive the prediction is to the distance weight.', 'The trip is 2 km, so increasing that weight by 0.01 min/km raises the prediction by 0.02 minutes.'],
          ['\\partial L/\\partial w_d', 'How sensitive the loss is to the distance weight.', 'Multiply -4 by 2 to get -8. A 0.01 weight increase should lower loss by about 0.08.'],
          ['\\Delta', 'A finite change rather than a derivative.', 'Recomputing after the 0.01 nudge gives loss 7.9202 instead of 8: an actual change of -0.0798.'],
        ],
      },
      code: {
        title: 'Check the chain with a small nudge', lang: 'python',
        snippet: 'x_distance, x_stops = 2.0, 1.0\nw_distance, w_stops, bias = 3.0, 4.0, 5.0\ntarget = 19.0\n\ndef loss_at(weight):\n    prediction = weight * x_distance + w_stops * x_stops + bias\n    return 0.5 * (prediction - target)**2\n\nprediction = w_distance * x_distance + w_stops * x_stops + bias\ngradient = (prediction - target) * x_distance  # -8\neps = 1e-5\nnumerical = (loss_at(w_distance + eps)\n             - loss_at(w_distance - eps)) / (2 * eps)\nassert abs(gradient - numerical) < 1e-7',
      },
      quiz: {
        prompt: 'Why is the distance-weight gradient -8 rather than just -4?',
        options: [
          { text: 'There are two inputs, so every gradient is doubled.', correct: false, explanation: 'The multiplier is this input\'s value, not the number of inputs. The stop-weight gradient is -4 times 1, or -4.' },
          { text: 'The learning rate doubles the error.', correct: false, explanation: 'The learning rate is not part of the gradient. It is used later by the optimizer.' },
          { text: 'The loss sensitivity -4 is multiplied by the distance input 2.', correct: true, explanation: 'Changing the distance weight changes the prediction at a rate of 2. The chain rule multiplies that rate by the loss sensitivity, -4.' },
        ],
      }, controls: [], presets: [],
    },
    backprop: {
      kicker: 'Differentiate first; update second',
      question: 'Which rules should change, and by how much?',
      title: 'Backprop finds gradients. The optimizer changes the weights.',
      summary: 'Reuse the forward calculation to find a sensitivity for each parameter. Then take one small step opposite those gradients and evaluate the model again.',
      what: '<strong>Backpropagation</strong> applies the chain rule backward through the computation. It calculates gradients without changing parameters. <strong>Gradient descent</strong> uses those gradients and a learning rate to make an update.',
      why: 'Separating these jobs explains why different optimizers can use the same gradients. It also shows why an update can fail when its learning rate is too large.',
      interview: 'Forward computes activations and loss; backward computes parameter gradients; the optimizer updates parameters. For an affine neuron, each weight gradient is the upstream derivative times its input.',
      details: [
        'The shared upstream derivative is -4. Multiply it by distance 2, stops 1, and the bias derivative 1 to obtain gradients -8, -4, and -4.',
        'All three gradients are evaluated at the same old parameters. With learning rate 0.1, update them together: (3, 4, 5) becomes (3.8, 4.4, 5.4).',
        'The new prediction is 17.4 minutes and the loss is 1.28 instead of 8. That is progress on this one example, not evidence of generalization to unseen deliveries.',
        'Training normally uses many examples, often averaging their gradients over a batch. Repeating linear neurons without nonlinear activations still produces a linear model.',
      ],
      math: {
        title: 'Gradients, then one simultaneous update',
        formula: ['g_d=(\\hat{y}-y)x_d=-8,\\quad g_s=(\\hat{y}-y)x_s=-4,\\quad g_b=\\hat{y}-y=-4', '\\theta_{\\mathrm{new}}=\\theta-\\eta\\nabla_{\\theta}L', '(3,4,5)-0.1(-8,-4,-4)=(3.8,4.4,5.4)', 'L_{\\mathrm{new}}=\\tfrac12(17.4-19)^2=1.28'],
        note: 'Backprop produces the gradient vector. The update rule is a separate choice. This small learning rate lowers the loss for our example; arbitrary learning rates need not.',
        annotations: [
          ['g_d,\\ g_s,\\ g_b', 'The loss derivatives for the two weights and the bias.', 'For this trip they are -8, -4, and -4. Negative gradients favor small increases in those parameters.'],
          ['\\theta', 'The collection of trainable parameters.', 'The original distance weight, stop weight, and bias are (3, 4, 5).'],
          ['\\eta', 'The learning rate, which scales the update.', 'With 0.1, the distance-weight update is -0.1 times -8, or +0.8.'],
          ['\\theta_{\\mathrm{new}}', 'The parameters after applying the update.', 'Using (3.8, 4.4, 5.4), the same inputs now predict 17.4 minutes.'],
          ['L_{\\mathrm{new}}', 'The loss from a fresh forward calculation.', 'Half of (17.4 - 19) squared is 1.28, down from 8.'],
        ],
      },
      code: {
        title: 'One training step, three distinct jobs', lang: 'python',
        snippet: 'x_distance, x_stops, target = 2.0, 1.0, 19.0\nw_distance, w_stops, bias = 3.0, 4.0, 5.0\nlearning_rate = 0.1\n\n# Forward: predict, then measure mismatch.\nprediction = w_distance * x_distance + w_stops * x_stops + bias\nerror = prediction - target\nloss_before = 0.5 * error**2  # 8\n\n# Backward: calculate all gradients at the OLD parameters.\ng_distance = error * x_distance  # -8\ng_stops = error * x_stops        # -4\ng_bias = error                  # -4\n\n# Optimizer: apply the updates.\nw_distance -= learning_rate * g_distance\nw_stops -= learning_rate * g_stops\nbias -= learning_rate * g_bias\n\nprediction = w_distance * x_distance + w_stops * x_stops + bias\nloss_after = 0.5 * (prediction - target)**2  # 1.28',
      },
      quiz: {
        prompt: 'Backprop has computed gradients, but no optimizer step has run. What is the prediction for the same trip?',
        options: [
          { text: '17.4 minutes, because gradients already changed the model.', correct: false, explanation: '17.4 is the prediction after the update. Merely computing a gradient does not change the weight it describes.' },
          { text: '15 minutes, because the parameters are still unchanged.', correct: true, explanation: 'Backprop produces sensitivities. The optimizer must apply an update before the same inputs can produce a new prediction.' },
          { text: '19 minutes, because backprop corrects the prediction exactly.', correct: false, explanation: 'Backprop does not set predictions to targets. Even the small update in this lesson leaves some error.' },
        ],
      }, controls: [], presets: [],
    },
  };

  const STEPS = {
    neuron: ['Observe the inputs', 'Apply the learned rates', 'Add a base time'],
    'forward-pass': ['Read the inputs', 'Compute contributions', 'Make the prediction', 'Measure the mismatch'],
    'chain-rule': ['Loss responds to prediction', 'Prediction responds to weight', 'Multiply the local rates', 'Check with a small nudge'],
    backprop: ['Forward: measure error', 'Backward: find gradients', 'Optimizer: choose a step', 'Forward again: compare'],
  };

  const names = { distance: 'Distance weight', stops: 'Stop weight', bias: 'Base time (bias)' };
  const units = { distance: 'min/km', stops: 'min/stop', bias: 'min' };
  const format = (value, digits = 4) => {
    const rounded = Number(value.toFixed(digits));
    return String(Object.is(rounded, -0) ? 0 : rounded).replace('-', '\u2212');
  };
  const signed = value => `${value > 0 ? '+' : ''}${format(value)}`;
  const arrow = () => '<span class="nf-arrow" aria-hidden="true"><svg viewBox="0 0 40 24" focusable="false"><path d="M2 12H36M27 3L36 12L27 21" /></svg></span>';

  function createState(kind) {
    if (!KINDS.includes(kind)) throw new RangeError('Unknown lesson');
    return { step: 0, selected: kind === 'chain-rule' ? 'local-loss' : 'distance', parameter: 'distance', extraKm: false };
  }

  function goToStep(kind, state, step) {
    const highlights = {
      neuron: ['distance', 'distance', 'bias'],
      'forward-pass': ['distance', 'distance', 'prediction', 'loss'],
      'chain-rule': ['local-loss', 'local-input', 'gradient', 'gradient'],
    };
    return { ...state, step, selected: highlights[kind]?.[step] || state.selected };
  }

  function reduceState(kind, state, action, value) {
    if (!KINDS.includes(kind)) return state;
    if (action === 'next') return goToStep(kind, state, (state.step + 1) % STEPS[kind].length);
    if (action === 'previous') return goToStep(kind, state, Math.max(0, state.step - 1));
    if (action === 'reset') return createState(kind);
    if (action === 'extra-km' && kind === 'neuron') return { ...state, extraKm: !state.extraKm, selected: 'distance' };
    if (action === 'parameter' && kind === 'chain-rule' && PARAMETERS.includes(value)) return { ...state, parameter: value };
    if (action === 'inspect' && [...PARAMETERS, 'prediction', 'loss', 'local-loss', 'local-input', 'gradient'].includes(value)) return { ...state, selected: value };
    return state;
  }

  const inspectAttrs = (value, selected) => `data-nf-action="inspect" data-nf-value="${value}" data-nf-focus="inspect-${value}" aria-pressed="${value === selected}"`;

  function contributionMarkup(key, result, state, phase) {
    const isBias = key === 'bias';
    const input = key === 'distance' ? `${format(result.inputs.distance)} km` : `${format(result.inputs.stops)} extra stop`;
    const label = key === 'distance' ? 'Distance' : key === 'stops' ? 'Stops' : 'Base time';
    return `<button type="button" class="nf-contribution ${state.selected === key ? 'nf-selected' : ''}" ${inspectAttrs(key, state.selected)}>
      <span class="nf-contribution-top"><span class="nf-label">${label}</span><strong class="${phase === 'input' && !isBias ? 'nf-emphasis' : ''}">${isBias ? `${format(result.parameters.bias)} min` : input}</strong></span>
      <span class="nf-contribution-bottom"><span class="nf-rule ${phase === 'weights' && !isBias || phase === 'bias' && isBias ? 'nf-emphasis' : ''}">${isBias ? 'Added to every delivery' : `&times; ${format(result.parameters[key])} ${units[key]}`}</span><span class="nf-contribution-value">${isBias ? '+' : '='} ${format(result.contributions[key])} min</span></span>
    </button>`;
  }

  function predictionMarkup(result, state, phase = 'prediction') {
    return `<button type="button" class="nf-prediction ${phase === 'prediction' ? 'nf-active-node' : ''}" ${inspectAttrs('prediction', state.selected)}>
      <span class="nf-label">Prediction</span><strong class="nf-big">${format(result.prediction)} <span>min</span></strong>
      <span class="nf-equation">${format(result.contributions.distance)} + ${format(result.contributions.stops)} + ${format(result.contributions.bias)}</span><span class="nf-small">Sum the contributions</span>
    </button>`;
  }

  function affineMarkup(result, state, phase) {
    return `<div class="nf-affine-flow"><div class="nf-contributions" role="group" aria-label="Inspect each contribution">${PARAMETERS.map(key => contributionMarkup(key, result, state, phase)).join('')}</div>${arrow()}${predictionMarkup(result, state, phase)}</div>`;
  }

  function contributionNote(result, selected) {
    if (selected === 'distance') return `<strong>Distance:</strong> ${format(result.inputs.distance)} km is the input. The learned rate is ${format(result.parameters.distance)} min/km, so this trip contributes <strong>${format(result.contributions.distance)} minutes</strong>.`;
    if (selected === 'stops') return `<strong>Stops:</strong> ${format(result.inputs.stops)} extra stop &times; ${format(result.parameters.stops)} min/stop contributes <strong>${format(result.contributions.stops)} minutes</strong>. The rate is a parameter; the stop count is an input.`;
    if (selected === 'bias') return `<strong>Bias:</strong> the base time adds <strong>${format(result.parameters.bias)} minutes</strong> regardless of distance or stops. It is learned, just like the weights.`;
    if (selected === 'loss') return `<strong>Loss:</strong> (${format(result.prediction)} &minus; ${format(result.target)}) squared, divided by 2, is <strong>${format(result.loss)}</strong>. It measures mismatch, not a probability.`;
    return `<strong>Prediction:</strong> ${format(result.contributions.distance)} + ${format(result.contributions.stops)} + ${format(result.contributions.bias)} = <strong>${format(result.prediction)} minutes</strong>. This output keeps the sum as-is, with no nonlinear activation.`;
  }

  function neuronMarkup(state) {
    const result = derive({ ...EXAMPLE.inputs, distance: state.extraKm ? 3 : 2 });
    const descriptions = [
      `A ${format(result.inputs.distance)} km delivery has one extra stop. <strong>Inputs are facts about this trip</strong>; the learned rules are shown alongside them.`,
      'The distance weight is a rate: <strong>3 minutes per kilometer</strong>. Try an extra kilometer; only the distance contribution changes.',
      'Add the <strong>5-minute base time</strong> once. This linear output simply keeps the total as its prediction.',
    ];
    return { body: `${affineMarkup(result, state, ['input', 'weights', 'bias'][state.step])}
      <div class="nf-experiment"><button type="button" class="nf-button nf-experiment-button" data-nf-action="extra-km" data-nf-focus="extra-km" aria-pressed="${state.extraKm}">${state.extraKm ? 'Return to 2 km' : 'Make it 1 km farther'}</button><span>${state.extraKm ? 'Only the input changed. Prediction: 15 to 18 min.' : 'Keep the weights and the stop count fixed.'}</span></div>`,
      description: descriptions[state.step], note: contributionNote(result, state.selected) };
  }

  function lossMarkup(result, state) {
    return `<div class="nf-mismatch"><div><span class="nf-label">Observed delivery</span><strong>${format(result.target)} min</strong></div>${arrow()}
      <button type="button" class="nf-loss" ${inspectAttrs('loss', state.selected)}><span class="nf-label">Loss &middot; half the squared error</span><strong>${format(result.loss)} <span>min<sup>2</sup></span></strong><span class="nf-equation">&frac12; &times; (${format(result.prediction)} &minus; ${format(result.target)})<sup>2</sup></span></button></div>`;
  }

  function forwardMarkup(state) {
    const result = derive();
    const descriptions = [
      'Start with the same 2 km delivery and one extra stop. <strong>The parameters stay fixed throughout this forward pass.</strong>',
      'Apply each learned rule: distance contributes <strong>6 minutes</strong>, the stop contributes <strong>4</strong>, and the base time adds <strong>5</strong>.',
      'The model predicts <strong>15 minutes</strong>. This is all you can compute at inference time, before the actual delivery time is known.',
      'The delivery took <strong>19 minutes</strong>, so the prediction was 4 minutes too early. Half the squared error is <strong>8</strong>. No weights have changed.',
    ];
    return { body: `${affineMarkup(result, state, ['input', 'weights', 'prediction', 'prediction'][state.step])}${state.step === 3 ? lossMarkup(result, state) : ''}`,
      description: descriptions[state.step], note: contributionNote(result, state.selected) };
  }

  function parameterChoices(state) {
    return `<div class="nf-choices" role="group" aria-label="Follow one parameter">${PARAMETERS.map(key => `<button type="button" class="nf-button" data-nf-action="parameter" data-nf-value="${key}" data-nf-focus="parameter-${key}" aria-pressed="${state.parameter === key}">${names[key]}</button>`).join('')}</div>`;
  }

  function chainMarkup(state) {
    const result = derive(), parameter = state.parameter;
    const rates = chain(parameter);
    const nudged = derive(EXAMPLE.inputs, { ...EXAMPLE.parameters, [parameter]: EXAMPLE.parameters[parameter] + 0.01 });
    const descriptions = [
      `The prediction is 15 but the target is 19. Locally, raising the prediction by a tiny amount changes loss at a rate of <strong>${format(rates.predictionToLoss)}</strong>.`,
      `For the ${names[parameter].toLowerCase()}, the prediction changes at a rate of <strong>${format(rates.parameterToPrediction)}</strong>. ${parameter === 'bias' ? 'Bias is added directly, so its local derivative is 1.' : `That rate is the ${parameter === 'distance' ? 'distance' : 'stop'} input, held fixed.`}`,
      `Multiply the two local rates: <strong>${format(rates.predictionToLoss)} &times; ${format(rates.parameterToPrediction)} = ${format(rates.gradient)}</strong>. A small increase in this parameter should lower loss.`,
      `Nudge only the ${names[parameter].toLowerCase()} by <strong>+0.01 ${units[parameter]}</strong>. Recomputing the loss checks what the gradient predicted.`,
    ];
    let note = `These rates come from the computation: <strong>input ${format(rates.parameterToPrediction)}</strong> and <strong>prediction error ${format(rates.predictionToLoss)}</strong>. They are not adjustable multipliers.`;
    if (state.selected === 'local-loss') note = '<strong>Loss sensitivity:</strong> the derivative of half the squared error is prediction minus target: 15 &minus; 19 = &minus;4. The sign favors increasing this too-small prediction.';
    if (state.selected === 'local-input') note = parameter === 'bias' ? '<strong>Bias sensitivity:</strong> add 0.01 minute to the bias and the prediction increases by exactly 0.01 minute. The local rate is 1.' : `<strong>Weight sensitivity:</strong> the input ${format(rates.parameterToPrediction)} is held fixed. Changing this weight by 0.01 changes the prediction by ${format(rates.parameterToPrediction * 0.01)} minutes.`;
    if (state.selected === 'gradient') note = `<strong>Whole-path gradient:</strong> multiply, rather than add, the local rates. A +0.01 parameter change predicts a loss change of about <strong>${format(rates.gradient * 0.01)}</strong>.`;
    return { description: descriptions[state.step], note,
      body: `${parameterChoices(state)}
      <div class="nf-chain-path" aria-label="The computation goes from parameter to prediction to loss">
        <div class="nf-path-node"><span class="nf-label">${names[parameter]}</span><strong>${format(EXAMPLE.parameters[parameter])} <span>${units[parameter]}</span></strong></div>${arrow()}
        <div class="nf-path-node"><span class="nf-label">Prediction</span><strong>${format(result.prediction)} <span>min</span></strong></div>${arrow()}
        <div class="nf-path-node"><span class="nf-label">Loss</span><strong>${format(result.loss)} <span>min<sup>2</sup></span></strong></div>
      </div>
      <div class="nf-local-rates" role="group" aria-label="Inspect the local derivatives">
        <button type="button" class="nf-rate ${state.step === 1 ? 'nf-active-node' : ''}" ${inspectAttrs('local-input', state.selected)}><span class="nf-label">Parameter &rarr; prediction</span><strong>&times; ${format(rates.parameterToPrediction)}</strong><span>${parameter === 'bias' ? 'Bias is added directly' : `${parameter === 'distance' ? 'Distance' : 'Stop'} input, held fixed`}</span></button>
        <button type="button" class="nf-rate ${state.step === 0 ? 'nf-active-node' : ''}" ${inspectAttrs('local-loss', state.selected)}><span class="nf-label">Prediction &rarr; loss</span><strong>&times; ${format(rates.predictionToLoss)}</strong><span>Prediction &minus; target</span></button>
      </div>
      ${state.step >= 2 ? `<button type="button" class="nf-total-gradient" ${inspectAttrs('gradient', state.selected)}><span class="nf-label">Whole-path gradient</span><strong>${format(rates.predictionToLoss)} &times; ${format(rates.parameterToPrediction)} = ${format(rates.gradient)}</strong><span>How loss responds to this parameter</span></button>` : ''}
      ${state.step === 3 ? `<div class="nf-probe"><div><span class="nf-label">Gradient predicts</span><strong>${format(rates.gradient * 0.01)}</strong><span>Approximate loss change</span></div><div><span class="nf-label">Recalculation gives</span><strong>${format(nudged.loss - result.loss)}</strong><span>Loss: ${format(result.loss)} &rarr; ${format(nudged.loss)}</span></div><p>A derivative is a local approximation. The small difference comes from the curve of the squared loss.</p></div>` : ''}`,
    };
  }

  function gradientRows(state, training) {
    const { before, after, updates } = training;
    const showGradient = state.step >= 1;
    const showUpdate = state.step >= 2;
    return `<div class="nf-parameter-list" role="group" aria-label="Inspect gradients and parameter updates">${PARAMETERS.map(key => `<button type="button" class="nf-parameter-row" ${inspectAttrs(key, state.selected)}>
      <span class="nf-parameter-name"><span class="nf-label">${names[key]}</span><strong>${format(before.parameters[key])}${state.step === 3 ? ` &rarr; ${format(after.parameters[key])}` : ''} <span>${units[key]}</span></strong></span>
      ${showGradient ? `<span class="nf-parameter-gradient"><span class="nf-label">Gradient</span><strong>${format(before.error)} &times; ${format(before.local[key])} = ${format(before.gradients[key])}</strong></span>` : '<span class="nf-small">Current parameter; not changed</span>'}
      ${showUpdate ? `<span class="nf-parameter-update"><span class="nf-label">${state.step === 3 ? 'Update applied' : 'Proposed update'}</span><strong>${signed(updates[key])}</strong></span>` : ''}
    </button>`).join('')}</div>`;
  }

  function backpropMarkup(state) {
    const training = trainStep(), { before, after } = training;
    const descriptions = [
      'The same trip is predicted at <strong>15 minutes</strong>, but took 19. The signed error is &minus;4 and the loss is 8. Start with the current parameters below.',
      'Backprop reuses the error: multiply <strong>&minus;4</strong> by each input, or by 1 for the bias. <strong>The parameters still have not changed.</strong>',
      'Choose learning rate <strong>0.1</strong>. Subtract 0.1 times each gradient to propose an update. Apply all three together, using gradients from the old parameters.',
      'After one update, the same trip is predicted at <strong>17.4 minutes</strong>. The loss falls from <strong>8 to 1.28</strong>. Backprop found a direction; the optimizer took the step.',
    ];
    const parameter = PARAMETERS.includes(state.selected) ? state.selected : 'distance';
    const note = state.step === 0 ? contributionNote(before, parameter) : state.step === 1
      ? `<strong>${names[parameter]}:</strong> upstream sensitivity ${format(before.error)} &times; local derivative ${format(before.local[parameter])} = gradient <strong>${format(before.gradients[parameter])}</strong>. This is a sensitivity, not yet a weight change.`
      : `<strong>${names[parameter]}:</strong> ${format(before.parameters[parameter])} &minus; 0.1 &times; (${format(before.gradients[parameter])}) = <strong>${format(after.parameters[parameter])} ${units[parameter]}</strong>. ${state.step === 3 ? 'The update has been applied.' : 'This is the proposed value; the old value remains until you apply the update.'}`;
    return { description: descriptions[state.step], note,
      body: `${state.step === 3 ? `<div class="nf-comparison"><div><span class="nf-label">Before the update</span><strong class="nf-big">${format(before.prediction)} <span>min</span></strong><span>${format(-before.error)} min too early</span><span>Loss <strong>${format(before.loss)}</strong></span></div><div class="nf-after"><span class="nf-label">After the update</span><strong class="nf-big">${format(after.prediction)} <span>min</span></strong><span>${format(-after.error)} min too early</span><span>Loss <strong>${format(after.loss)}</strong></span></div><p>Same inputs, same 19-minute target. Only the learned parameters changed.</p></div>` : `<div class="nf-error-summary"><div><span class="nf-label">Prediction</span><strong>15 min</strong></div><div><span class="nf-label">Observed</span><strong>19 min</strong></div><div><span class="nf-label">${state.step === 0 ? 'Loss' : 'Upstream derivative'}</span><strong>${state.step === 0 ? '8' : '&minus;4'}</strong></div></div>`}
      ${gradientRows(state, training)}
      ${state.step === 2 ? '<div class="nf-update-rule"><span class="nf-label">Gradient descent &middot; learning rate 0.1</span><strong>New value = old value &minus; 0.1 &times; gradient</strong><span>No update applied yet. Use the button below to take the step.</span></div>' : ''}`,
    };
  }

  const renderers = { neuron: neuronMarkup, 'forward-pass': forwardMarkup, 'chain-rule': chainMarkup, backprop: backpropMarkup };

  function markup(kind, state) {
    const view = renderers[kind](state), steps = STEPS[kind];
    const nextLabel = state.step === steps.length - 1 ? 'Start again' : kind === 'backprop' && state.step === 2 ? 'Apply one update' : 'Next';
    return `<section class="lesson nf-lesson" data-nf-kind="${kind}" aria-label="${content[kind].question}">
      <header class="nf-header"><p class="nf-kicker">${content[kind].kicker}</p><h3>${content[kind].question}</h3></header>
      <nav class="nf-navigation" aria-label="${content[kind].title} steps">
        <button type="button" class="nf-button" data-nf-action="previous" data-nf-focus="previous" ${state.step === 0 ? 'disabled' : ''}>Previous</button>
        <span class="nf-step-count" aria-label="Step ${state.step + 1} of ${steps.length}">${state.step + 1} / ${steps.length}</span>
        <button type="button" class="nf-button nf-next" data-nf-action="next" data-nf-focus="next">${nextLabel}</button>
      </nav>
      <div class="nf-walkthrough"><span class="nf-step-label">${state.step + 1}. ${steps[state.step]}</span><p class="nf-description">${view.description}</p></div>
      <div class="nf-stage">${view.body}</div>
      <div class="nf-note" role="status" aria-live="polite" aria-atomic="true"><span class="nf-sr-only">Step ${state.step + 1} of ${steps.length}: ${steps[state.step]}. </span>${view.note}</div>
    </section>`;
  }

  const instances = new Map();
  const bindings = new WeakMap();

  function paint(instance, forceRoot) {
    const html = markup(instance.kind, instance.state);
    for (const root of instance.roots) {
      // A cleared lazy root is not an invitation to remount. Explicit render() revives it.
      if (root !== forceRoot && (!root.isConnected || root.firstElementChild?.dataset.nfKind !== instance.kind)) {
        instance.roots.delete(root);
        continue;
      }
      const active = root.ownerDocument.activeElement;
      const focusKey = root.contains(active) ? active.dataset?.nfFocus : null;
      const oldNote = root.querySelector('.nf-note');
      root.innerHTML = html;
      if (oldNote) {
        const newNote = root.querySelector('.nf-note');
        const noteHTML = newNote.innerHTML;
        newNote.replaceWith(oldNote);
        if (oldNote.innerHTML !== noteHTML) oldNote.innerHTML = noteHTML;
      }
      if (focusKey) {
        const controls = [...root.querySelectorAll('[data-nf-focus]')];
        const replacement = controls.find(element => element.dataset.nfFocus === focusKey && !element.disabled)
          || controls.find(element => element.dataset.nfFocus === 'next');
        replacement?.focus({ preventScroll: true });
      }
    }
  }

  function render(kind, root) {
    if (!KINDS.includes(kind) || !root || typeof root.addEventListener !== 'function' || !root.ownerDocument) return false;
    if (!instances.has(kind)) instances.set(kind, { kind, state: createState(kind), roots: new Set() });
    const instance = instances.get(kind);
    const existing = bindings.get(root);
    if (existing && existing.instance !== instance) existing.instance.roots.delete(root);
    if (!existing) {
      root.addEventListener('click', event => {
        const button = event.target.closest?.('[data-nf-action]');
        if (!button || !root.contains(button) || button.disabled) return;
        event.stopPropagation();
        const bound = bindings.get(root).instance;
        bound.state = reduceState(bound.kind, bound.state, button.dataset.nfAction, button.dataset.nfValue);
        paint(bound, root);
      });
      root.addEventListener('keydown', event => {
        if (['Enter', ' '].includes(event.key) && event.target.closest?.('[data-nf-action]')) {
          // Keep native button activation, but do not let a parent viewer open on that key.
          event.stopPropagation();
        }
      });
    }
    bindings.set(root, { instance });
    instance.roots.add(root);
    paint(instance, root);
    return true;
  }

  const api = { has: kind => KINDS.includes(kind), render, content, EXAMPLE, forward, loss, derive, chain, trainStep, finiteDifference, createState, reduceState };
  if (typeof window !== 'undefined') {
    window.AtelierNeuralFlow = api;
    (window.AtelierLessonModules ||= []).push(api);
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})();
