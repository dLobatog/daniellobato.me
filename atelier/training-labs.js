/* Exact, small training mechanisms. No DOM access is needed to test the models. */
(() => {
  'use strict';
  const sum = xs => xs.reduce((a, b) => a + b, 0);
  const mean = xs => sum(xs) / xs.length;
  const dot = (a, b) => sum(a.map((v, i) => v * b[i]));
  const mv = (a, x) => a.map(row => dot(row, x));
  const transpose = a => a[0].map((_, j) => a.map(row => row[j]));
  const mm = (a, b) => a.map(row => transpose(b).map(col => dot(row, col)));
  const norm = x => Math.sqrt(dot(x, x));
  const add = (a, b) => a.map((v, i) => v + b[i]);
  const eye = n => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +(i === j)));
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const softmax = (z, temperature = 1) => {
    const m = Math.max(...z);
    const e = z.map(v => Math.exp((v - m) / temperature));
    return e.map(v => v / sum(e));
  };
  function quadraticTrace(lr = .22, steps = 8) {
    let x = [2, 1];
    const rows = [];
    for (let i = 0; i <= steps; i++) {
      const grad = [x[0], 8 * x[1]];
      rows.push({ x, grad, loss: (x[0] ** 2 + 8 * x[1] ** 2) / 2 });
      x = x.map((v, j) => v - lr * grad[j]);
    }
    return rows;
  }
  function normalGenerator(seed) {
    let s = seed >>> 0;
    const uniform = () => { s = (1664525 * s + 1013904223) >>> 0; return (s + 1) / 4294967297; };
    return () => Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
  }
  function initializationTrace(mode = 'he', depth = 6, width = 8) {
    const std = { small: .1, xavier: Math.sqrt(1 / width), he: Math.sqrt(2 / width), large: Math.sqrt(4 / width) }[mode];
    const random = normalGenerator(4);
    let x = Array.from({ length: width }, (_, i) => i % 2 ? -1 : 1);
    return Array.from({ length: depth }, () => {
      const weights = Array.from({ length: width }, () => Array.from({ length: width }, () => random() * std));
      const z = mv(weights, x), output = z.map(v => Math.max(0, v));
      const row = { input: x, weights, z, output, std, rms: Math.sqrt(mean(output.map(v => v * v))), dead: output.filter(v => v === 0).length };
      x = output;
      return row;
    });
  }
  function gradientTrace(mode = 'tanh', gain = 1.3, depth = 6) {
    const w = [[gain, .15], [-.1, gain]];
    let x = [2, -2];
    const rows = [];
    for (let i = 0; i < depth; i++) {
      const z = mv(w, x);
      const output = z.map(v => mode === 'tanh' ? Math.tanh(v) : mode === 'relu' ? Math.max(0, v) : v);
      const derivative = z.map((v, j) => mode === 'tanh' ? 1 - output[j] ** 2 : mode === 'relu' ? +(v > 0) : 1);
      const jacobian = w.map((row, j) => row.map(v => v * derivative[j]));
      rows.push({ input: x, z, output, derivative, jacobian, weights: w });
      x = output;
    }
    let gradient = [1, 1]; // L = sum of final activations.
    for (let i = depth - 1; i >= 0; i--) {
      rows[i].outGradient = gradient;
      gradient = mv(transpose(rows[i].jacobian), gradient);
      rows[i].inGradient = gradient;
    }
    return rows;
  }
  function normalizeMatrix(x, mode = 'layer', eps = 1e-5) {
    return x.map((row, i) => row.map((v, j) => {
      const group = mode === 'batch' ? x.map(r => r[j]) : row;
      const mu = mode === 'rms' ? 0 : mean(group);
      const variance = mean(group.map(n => (n - mu) ** 2));
      return { value: (v - mu) / Math.sqrt(variance + eps), group, mean: mu, variance, denominator: Math.sqrt(variance + eps) };
    }));
  }
  function residualTrace(mode = 'small', depth = 6) {
    const branch = mode === 'cancel' ? [[-1, 0], [0, -1]] : mode === 'positive' ? [[.4, .1], [0, .4]] : [[-.1, .05], [0, -.1]];
    const residual = branch.map((r, i) => r.map((v, j) => v + +(i === j)));
    let plainX = [1, -1], residualX = [1, -1], plainJ = eye(2), residualJ = eye(2);
    const rows = [{ plainX, residualX, plainJ, residualJ }];
    for (let i = 0; i < depth; i++) {
      plainX = mv(branch, plainX); residualX = mv(residual, residualX);
      plainJ = mm(branch, plainJ); residualJ = mm(residual, residualJ);
      rows.push({ plainX, residualX, plainJ, residualJ });
    }
    return { branch, residual, rows };
  }
  const attentionKeys = [[1, 0], [0, 1], [2, 1], [-1, 2]];
  const attentionValues = [[1, 0], [0, 2], [3, 1], [-1, 2]];
  function onlineAttention(scores, values, tileSize = 2) {
    let m = -Infinity, l = 0, numerator = values[0].map(() => 0);
    const trace = [];
    for (let start = 0; start < scores.length; start += tileSize) {
      const tile = scores.slice(start, start + tileSize);
      const nextM = Math.max(m, ...tile), alpha = Math.exp(m - nextM);
      const weights = tile.map(v => Math.exp(v - nextM));
      const oldL = l, oldNumerator = numerator;
      l = alpha * l + sum(weights);
      numerator = numerator.map((v, j) => alpha * v + sum(weights.map((p, k) => p * values[start + k][j])));
      trace.push({ start, scores: tile, oldM: m, m: nextM, alpha, oldL, l, weights, oldNumerator, numerator, output: numerator.map(v => v / l) });
      m = nextM;
    }
    const probabilities = softmax(scores);
    const dense = values[0].map((_, j) => sum(probabilities.map((p, i) => p * values[i][j])));
    return { trace, output: numerator.map(v => v / l), dense, probabilities };
  }
  function attentionExample(query = 0) {
    const q = query ? [0, 2] : [1, 1];
    const scores = attentionKeys.map(k => dot(q, k) / Math.sqrt(2));
    return { q, scores, ...onlineAttention(scores, attentionValues) };
  }
  const tokens = ['<user>', 'Rank', 'these', 'items', '<assistant>', 'A', 'then', 'B', '<eos>'];
  const targetProbabilities = [.8, .25, .5, .4, .9, .6, .7, .2, .8];
  function tokenLoss(mode = 'sft') {
    const rows = tokens.map((token, i) => ({ token, p: targetProbabilities[i], mask: +(mode === 'pretrain' || i >= 5), nll: -Math.log(targetProbabilities[i]) }));
    const count = sum(rows.map(r => r.mask));
    return { rows, count, loss: sum(rows.map(r => r.nll * r.mask)) / count };
  }
  const baseW = [[1, .2, 0, -.1], [0, 1, .3, 0], [.1, 0, 1, .2], [0, -.2, 0, 1]];
  const headW = [[.4, -.2, .1, .3], [-.1, .3, .2, -.2]];
  function peftStep(mode = 'adapter', apply = false) {
    const x = [1, .5, -1, 2], target = [1, 0], A = [[1, -1, .5, .25]], B = [[0], [0], [0], [0]];
    const h = mv(baseW, x), output = mv(headW, h), error = output.map((v, i) => v - target[i]);
    const dh = mv(transpose(headW), error), ax = mv(A, x)[0];
    const matrices = { W: baseW, H: headW, b: [[0, 0, 0, 0]], c: [[0, 0]], A, B };
    const gradients = { W: dh.map(v => x.map(t => v * t)), H: error.map(v => h.map(t => v * t)), b: [dh], c: [error], A: [x.map(() => 0)], B: dh.map(v => [v * ax]) };
    const trainable = key => mode === 'full' ? !['A', 'B'].includes(key) : mode === 'head' ? ['H', 'c'].includes(key) : mode === 'bias' ? ['b', 'c'].includes(key) : ['A', 'B'].includes(key);
    const active = mode === 'adapter' ? Object.keys(matrices) : ['W', 'H', 'b', 'c'];
    const updated = Object.fromEntries(Object.entries(matrices).map(([key, matrix]) => [key, matrix.map((row, i) => row.map((v, j) => v - (apply && trainable(key) ? .1 * gradients[key][i][j] : 0)))]));
    const nextH = add(add(mv(updated.W, x), updated.b[0]), mode === 'adapter' ? mv(updated.B, mv(updated.A, x)) : [0, 0, 0, 0]);
    const nextOutput = add(mv(updated.H, nextH), updated.c[0]);
    return { matrices, gradients, trainable, active, updated, output, nextOutput, loss: dot(error, error) / 2, nextLoss: sum(nextOutput.map((v, i) => (v - target[i]) ** 2)) / 2, count: sum(active.filter(trainable).map(k => matrices[k].flat().length)), dh, ax };
  }
  const loraW = [[1, 0, .5], [0, 1, -.5], [.5, .5, 1], [-1, 0, 1]];
  function loraExample(rank = 1, zero = false) {
    const A = [[1, -1, .5], [0, 1, 1]].slice(0, rank);
    const B = [[.2, -.1], [.4, .2], [-.2, .3], [.1, -.2]].map(row => row.slice(0, rank).map(v => zero ? 0 : v));
    const delta = mm(B, A), merged = loraW.map((row, i) => add(row, delta[i]));
    const x = [1, 2, -1], ax = mv(A, x), base = mv(loraW, x), correction = mv(B, ax);
    return { W: loraW, A, B, delta, merged, x, ax, base, correction, output: add(base, correction), mergedOutput: mv(merged, x), parameters: rank * 7 };
  }
  const quantWeights = [-.9, -.55, -.2, .08, .3, .65, 1, 5];
  function quantize(values = quantWeights, bits = 4, limit = 5) {
    const qmax = 2 ** (bits - 1) - 1, scale = limit / qmax;
    const rows = values.map(x => {
      const clipped = clamp(x, -limit, limit);
      const q = clamp(Math.round(clipped / scale), -qmax, qmax), reconstructed = q * scale;
      return { x, clipped, q, reconstructed, error: reconstructed - x };
    });
    return { qmax, scale, rows, mse: mean(rows.map(r => r.error ** 2)), bulkMse: mean(rows.slice(0, -1).map(r => r.error ** 2)), packedBytes: Math.ceil(values.length * bits / 8), scaleBytes: 4 };
  }
  const teacherLogits = [3, 2, -1];
  function distillation(logits = [0, 0, 0], temperature = 2, objective = 'soft') {
    const teacher = softmax(teacherLogits, temperature), student = softmax(logits, temperature);
    const target = objective === 'hard' ? [1, 0, 0] : teacher;
    const terms = target.map((p, i) => p ? p * Math.log(p / student[i]) : 0);
    return { teacher, student, target, terms, kl: sum(terms), loss: sum(terms) * temperature ** 2, gradient: student.map((p, i) => temperature * (p - target[i])) };
  }
  function studentTrace(temperature = 2, objective = 'soft', steps = 0) {
    let logits = [0, 0, 0];
    for (let i = 0; i < steps; i++) {
      const d = distillation(logits, temperature, objective);
      logits = logits.map((v, j) => v - .5 * d.gradient[j]);
    }
    return { logits, ...distillation(logits, temperature, objective) };
  }
  const requests = [{ name: 'A', arrival: 0, prompt: 8, tokens: 4 }, { name: 'B', arrival: 4, prompt: 4, tokens: 2 }, { name: 'C', arrival: 9, prompt: 6, tokens: 3 }];
  function servingTimeline(policy = 'batch', longContext = false) {
    const jobs = requests.map(r => ({ ...r, prompt: r.prompt * (longContext ? 4 : 1) }));
    const batches = policy === 'batch' ? [jobs] : jobs.map(r => [r]);
    let clock = 0, peakKVTokens = 0;
    const results = [], events = [];
    for (const batch of batches) {
      const start = Math.max(clock, policy === 'batch' ? 10 : batch[0].arrival);
      const prefill = 2 + .5 * sum(batch.map(r => r.prompt));
      clock = start + prefill;
      const records = batch.map(r => ({ ...r, start, prefillEnd: clock, emitted: [] }));
      events.push({ type: 'prefill', start, end: clock, names: batch.map(r => r.name).join(',') });
      for (let step = 0; step < Math.max(...batch.map(r => r.tokens)); step++) {
        const active = records.filter(r => r.tokens > step), duration = 2 + .5 * active.length, begin = clock;
        clock += duration;
        active.forEach(r => r.emitted.push(clock));
        // The newly sampled token has no KV entry until the next forward pass.
        const cachedTokens = sum(active.map(r => r.prompt + step));
        peakKVTokens = Math.max(peakKVTokens, cachedTokens);
        events.push({ type: 'decode', start: begin, end: clock, names: active.map(r => r.name).join(','), step, cachedTokens });
      }
      results.push(...records.map(r => ({ ...r, ttft: r.emitted[0] - r.arrival, latency: r.emitted.at(-1) - r.arrival, wait: start - r.arrival })));
    }
    // Two layers, one KV head, head dimension four, FP16 keys and values.
    return { results, events, finish: clock, throughput: sum(jobs.map(r => r.tokens)) / clock * 1000, peakKVTokens, kvBytes: peakKVTokens * 2 * 2 * 1 * 4 * 2 };
  }

  const content = {};
  function lesson(kind, title, summary, why, interview, details, formula, annotations, code, question, right, wrong, explanation) {
    content[kind] = { title, summary, what: summary, why, interview, details, math: { title: 'Mechanism and notation', formula, note: details[0], annotations }, code: { title: 'Minimal computation', lang: 'python', snippet: code }, quiz: { prompt: question, options: [{ text: right, correct: true, explanation }, { text: wrong, correct: false, explanation }] }, controls: [], presets: [], geometry: null };
  }
  lesson('learning-rate', 'Learning rate: stability is set by the sharpest direction',
    'Follow gradient descent on a two-parameter quadratic. Crossing the optimum is not the same as diverging.',
    'A ranking model can improve slowly along flat directions while oscillating along sharp ones.',
    'For a local quadratic, each Hessian eigenmode contracts only when |1 - eta lambda| < 1.',
    ['The example has curvature 1 and 8; a constant step converges only for 0 < eta < 0.25. At exactly 0.25 the sharp coordinate oscillates without decaying.', 'Real networks are nonconvex and stochastic. This exact local model explains a failure mechanism, not a universal learning-rate prescription.'],
    ['L(\\theta)=\\tfrac12(\\theta_1^2+8\\theta_2^2)', '\\theta_{t+1}=(I-\\eta H)\\theta_t,\\quad 0<\\eta<2/\\lambda_{max}'],
    [['H', 'Hessian of the local loss', 'diag(1, 8)'], ['1-\\eta\\lambda', 'Per-step multiplier in one curvature direction', 'eta=0.22 gives -0.76 in the sharp direction']],
    'theta = np.array([2., 1.])\nH = np.diag([1., 8.])\nfor step in range(8):\n    grad = H @ theta\n    theta = theta - 0.22 * grad\n    loss = 0.5 * theta @ H @ theta',
    'At eta = 0.22, the sharp coordinate changes sign every step. Is training unstable?', 'No: its magnitude contracts by 0.76 each step.', 'Yes: every sign change means divergence.', 'Stability depends on the magnitude of 1 - eta lambda, not its sign.');
  lesson('initialization', 'Initialization: inspect the activations, not a promised variance curve',
    'Propagate an eight-dimensional vector through six fixed-seed ReLU layers. Change only the weight scale.',
    'Collapsed feature vectors can silently limit a retrieval or ranking tower before useful learning begins.',
    'He initialization preserves the second moment in expectation under independence and symmetric preactivation assumptions.',
    ['RMS is the square root of the activation second moment, not the centered standard deviation.', 'Width eight is intentionally small: the He trace fluctuates. The expectation is not a guarantee for one seed or every layer.'],
    ['h_l=\\operatorname{ReLU}(W_lh_{l-1})', '\\operatorname{Var}(W_{ij})=2/fan_{in}\\quad\\text{(He)}'],
    [['fan_{in}', 'Inputs per output unit', '8'], ['h_l', 'Actual post-ReLU activation vector', 'Negative preactivations become exactly zero'], ['\\operatorname{RMS}(h)', 'sqrt(mean(h squared))', 'Measures signal magnitude including a nonzero mean']],
    'rng = np.random.default_rng(4)\nh = np.array([1., -1.] * 4)\nfor layer in range(6):\n    W = rng.normal(0, np.sqrt(2 / 8), (8, 8))\n    z = W @ h\n    h = np.maximum(z, 0)\n    rms = np.sqrt(np.mean(h ** 2))\n# JS uses an LCG + Box-Muller PRNG; samples differ.',
    'Why does this finite-width He network not have exactly unit RMS at every layer?', 'The variance rule is an expectation, not an exact per-layer invariant.', 'The He rule requires every ReLU unit to be positive.', 'Finite samples and correlations cause fluctuations; ReLU also creates a nonzero mean.');
  lesson('gradient-flow', 'Gradient flow: multiply the actual layer Jacobians',
    'Inspect a six-layer vector network in both directions. Saturated tanh and dead ReLU gates change its derivatives.',
    'Gradient norms alone hide which features are gated off and how weight matrices rotate credit.',
    'Backprop multiplies transposed Jacobians; their singular directions and activation derivatives matter, not just weight size.',
    ['The terminal objective is L = h6[0] + h6[1], so the incoming gradient is exactly [1, 1]. All activation choices use the same input [2, -2].', 'Weights are tied across six layers to isolate the mechanism; real networks usually have distinct matrices.'],
    ['J_l=\\operatorname{diag}(\\phi\'(z_l))W_l', 'g_{l-1}=J_l^Tg_l'],
    [['J_l', 'Local output-to-input sensitivity', 'Each row of W is multiplied by its activation derivative'], ['g_l', 'Gradient with respect to the layer output', '[1, 1] at layer 6']],
    'h = x\ncache = []\nfor W in weights:\n    z = W @ h\n    h = np.tanh(z)\n    J = np.diag(1 - h*h) @ W\n    cache.append(J)\ng = np.ones(2)\nfor J in reversed(cache):\n    g = J.T @ g',
    'Can large weights still produce vanishing gradients through tanh?', 'Yes: saturation can make the activation derivatives almost zero.', 'No: every weight above one guarantees exploding gradients.', 'The full Jacobian contains both weights and nonlinear derivatives.');
  lesson('normalization', 'Normalization: the axis decides which examples interact',
    'Select a cell in a three-example, three-feature matrix. Its highlighted group supplies the actual statistics.',
    'Batch coupling matters for small batches, sequence packing, and train/eval consistency.',
    'LayerNorm normalizes a token across features; BatchNorm uses examples per feature; RMSNorm omits centering.',
    ['This is training-mode BatchNorm with population variance. Gamma = 1, beta = 0, epsilon = 1e-5.', 'BatchNorm inference normally uses running statistics; this lesson deliberately recomputes batch statistics.'],
    ['y_i=(x_i-\\mu)/\\sqrt{\\sigma^2+\\epsilon}', '\\operatorname{RMSNorm}(x)_i=x_i/\\sqrt{\\operatorname{mean}(x^2)+\\epsilon}'],
    [['\\mu,\\sigma^2', 'Statistics over the highlighted axis', 'BatchNorm: one column; LayerNorm: one row'], ['\\epsilon', 'Numerical stabilizer', '0.00001']],
    'axis = 0 if mode == "batch" else 1\nmu = 0 if mode == "rms" else x.mean(axis, keepdims=True)\nv = ((x - mu)**2).mean(axis, keepdims=True)\ny = (x - mu) / np.sqrt(v + 1e-5)',
    'Changing only example 3 changes example 1 under which mode here?', 'Training-mode BatchNorm.', 'LayerNorm and RMSNorm, but not BatchNorm.', 'BatchNorm shares column statistics; the other modes use each row independently.');
  lesson('residuals', 'Residuals: add the identity to the Jacobian',
    'Compare F(x) with x + F(x) using the same linear branch. Inspect exact forward vectors and end-to-end sensitivities.',
    'Skip paths change optimization geometry; they do not promise gradients of one.',
    'A residual block has Jacobian I + JF. Near-zero branches preserve information, but a branch near -I cancels the skip.',
    ['The branch is linear so its Jacobian and every stacked sensitivity are exact.', 'Six blocks share one branch. Positive branch gain can still amplify gradients; skip connections are not normalization.'],
    ['y=x+F(x),\\quad J_y=I+J_F', 'J_{stack}=(I+J_F)^6'],
    [['I', 'Identity sensitivity from the skip path', 'A zero branch yields y=x'], ['J_F', 'Sensitivity of the learned branch', 'J_F=-I makes I+J_F=0']],
    'plain = x.copy()\nresidual = x.copy()\nJ_plain = J_res = np.eye(2)\nfor _ in range(6):\n    plain = W @ plain\n    residual = residual + W @ residual\n    J_plain = W @ J_plain\n    J_res = (np.eye(2) + W) @ J_res',
    'What happens when F(x) = -x in a residual block?', 'Both its output and local input sensitivity are zero.', 'The identity path guarantees a gradient of one.', 'I + (-I) = 0. A learned branch can cancel the identity path.');
  lesson('flash-attention', 'Flash Attention: preserve the answer while streaming score tiles',
    'Process four keys in two tiles. Rescale the old denominator and value accumulator whenever the running maximum increases.',
    'Long-context attention is constrained by memory traffic as well as arithmetic.',
    'Online softmax gives exact dense attention in real arithmetic without writing an N by N intermediate to HBM.',
    ['A single query row, two-dimensional keys and values, no causal mask or dropout. Floating-point roundoff is reported.', 'This models the online-softmax invariant and IO placement, not a GPU kernel or measured performance. Full Flash Attention also tiles query rows.'],
    ['m\'=\\max(m,\\max s_{tile}),\\quad a=e^{m-m\'}', '\\ell\'=a\\ell+\\sum_j e^{s_j-m\'},\\quad u\'=au+\\sum_j e^{s_j-m\'}v_j', 'o=u/\\ell'],
    [['m', 'Running score maximum', 'Increases when a later tile contains a larger score'], ['\\ell', 'Sum of exponentials in the current scale', 'Old mass must be multiplied by a'], ['u', 'Unnormalized weighted value sum', 'Rescale it with the same a']],
    'm, l, u = -np.inf, 0., np.zeros(2)\nfor scores, values in tiles:\n    m_new = max(m, scores.max())\n    a = np.exp(m - m_new)\n    p = np.exp(scores - m_new)\n    l = a*l + p.sum()\n    u = a*u + p @ values\n    m = m_new\nout = u / l',
    'Why rescale the old accumulator when a new maximum arrives?', 'Old and new exponentials must use the same reference maximum.', 'To approximate attention by discarding low-scoring keys.', 'Rescaling preserves every prior contribution; tiling is not an attention approximation.');
  lesson('pretrain-finetune', 'Pretraining and SFT: inspect exactly which tokens incur loss',
    'The same causal sequence can have different supervised targets. Switch between all-token training and response-only SFT.',
    'An incorrect loss mask can spend the training budget imitating prompts instead of learning responses.',
    'SFT usually changes the data distribution and loss mask, not the autoregressive factorization.',
    ['Each probability is a fixed illustrative model probability of the displayed target conditioned on preceding tokens; natural-log losses are computed exactly.', 'BOS precedes the first token. The assistant delimiter is context-only under this chosen SFT convention; templates differ. Prompt masking does not remove context or block gradients through prompt representations.'],
    ['L=-\\frac{\\sum_t m_t\\log p_\\theta(x_t\\mid x_{<t})}{\\sum_t m_t}'],
    [['m_t', 'Whether the target contributes to the loss', 'Response-only mask: A, then, B, eos'], ['x_{<t}', 'Causal prefix used to predict the target', 'Predict A after the assistant delimiter']],
    'logits = model(ids[:-1])\ntargets = ids[1:]  # shift exactly once\nmask = response_target_mask[1:]\nnll = -logits.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)\nloss = (nll * mask).sum() / mask.sum()',
    'A prompt token has mask zero. Is it removed from the attention context?', 'No: it remains context, but has no direct target loss.', 'Yes: its hidden state is deleted.', 'Loss masking changes the objective, not which tokens the causal model can attend to.');
  lesson('peft', 'PEFT: frozen parameters still transmit gradients',
    'Inspect a two-layer linear model and take one exact SGD step under four trainability masks.',
    'Optimizer-state savings depend on the number of trainable parameters, not the number of forward operations.',
    'Freezing W avoids its parameter gradients and optimizer state, but gradients may still flow through W to an adapter.',
    ['The model has a 4x4 backbone, 2x4 head, and six biases; the adapter adds eight rank-one parameters.', 'Loss is half squared error on one vector. Raw partial derivatives are shown analytically even for frozen parameters; an autograd implementation need not store those parameter gradients.'],
    ['h=Wx+b+B(Ax),\\quad y=Hh+c', 'L=\\tfrac12\\|y-[1,0]\|^2,\\quad\\theta\'=\\theta-0.1M\\odot\\nabla L'],
    [['M', 'Binary trainability mask', 'Bias-only: six trainable scalars'], ['B(Ax)', 'Adapter contribution', 'B begins at zero, so A has zero gradient on the first step']],
    'for name, p in model.named_parameters():\n    p.requires_grad_(name in trainable_names)\noptimizer = torch.optim.SGD(\n    (p for p in model.parameters() if p.requires_grad), lr=0.1)\nloss = 0.5 * ((model(x) - target)**2).sum()\nloss.backward()\noptimizer.step()',
    'Why is A trainable but unchanged on the first adapter step when B = 0?', 'Its gradient contains B transpose and is zero on this step.', 'The optimizer automatically freezes every low-rank matrix.', 'Trainable does not mean nonzero gradient. B learns first, then A can receive a signal.');
  lesson('lora', 'LoRA: multiply the low-rank factors and inspect the correction',
    'Every cell of the merged matrix is W[i,j] plus a dot product between a row of B and a column of A.',
    'Rank bounds the correction matrix rank and controls parameter count, not a guaranteed quality score.',
    'A rank-r delta uses r(din + dout) parameters and can be merged for ordinary linear inference.',
    ['The example uses alpha/r = 1, no dropout, and a frozen 4x3 base matrix.', 'At rank two this tiny adapter has 14 parameters, more than the 12-parameter base. Savings require rank small relative to the dimensions.'],
    ['W\'=W+(\\alpha/r)BA,\\quad y=W x+(\\alpha/r)B(Ax)', '\\operatorname{rank}(BA)\\le r'],
    [['A', 'r by input-dimension projection', '1x3 or 2x3'], ['B', 'output-dimension by r projection', '4x1 or 4x2'], ['\\alpha/r', 'Adapter scaling', 'Fixed to 1 in this example']],
    'delta = B @ A  # alpha/r = 1\ny = W @ x + B @ (A @ x)\nmerged = (W + delta) @ x\nnp.testing.assert_allclose(y, merged)\ntrainable = A.size + B.size',
    'Is rank two always parameter-efficient?', 'No: here 2(3+4)=14 exceeds the 12 base weights.', 'Yes: factorizing any matrix always reduces parameter count.', 'Low rank is only cheaper when r(din+dout) is less than din*dout.');
  lesson('quantization', 'Quantization: outliers determine the grid, clipping spends error',
    'Inspect integer codes and reconstructed weights. Compare the outlier-preserving range with a clipped range.',
    'A weight-memory reduction does not imply a measured speedup or a known downstream accuracy change.',
    'Quantization error depends on scale granularity, clipping and data distribution, not just bit width.',
    ['Symmetric per-tensor quantization uses codes [-qmax, qmax], leaving one signed code unused; ties round toward positive infinity in this JS example.', 'Packed weight bytes plus one FP32 scale are theoretical storage. Alignment, kernels, activations, KV cache and optimizer states are excluded.'],
    ['s=c/(2^{b-1}-1),\\quad q=\\operatorname{clip}(\\operatorname{round}(x/s),-q_{max},q_{max})', '\\hat x=sq'],
    [['c', 'Absolute clipping limit', '5 retains the outlier; 1 clips it'], ['s', 'Distance between reconstruction levels', '4-bit, c=5: s=5/7'], ['\\hat x-x', 'Signed weight reconstruction error', 'Not a task-quality metric']],
    'qmax = 2**(bits - 1) - 1\nscale = clip_limit / qmax\nq = np.floor(np.clip(w, -clip_limit, clip_limit) / scale + 0.5)\nw_hat = scale * q\nmse = np.mean((w_hat - w)**2)\npacked_bytes = (w.size * bits + 7) // 8',
    'Clipping the 5.0 outlier at 1.0 improves the small weights. Must total MSE improve?', 'No: the outlier alone contributes squared error 16.', 'Yes: a finer quantization grid always lowers total error.', 'Clipping trades resolution for saturation error; the evaluation distribution determines whether that trade helps.');
  lesson('distillation', 'Distillation: transfer relative probabilities, not just the winner',
    'Train three student logits against a fixed teacher. Inspect each KL term and the actual update direction.',
    'Soft targets expose relative preference among candidates, useful for compressing ranking and language models.',
    'Temperature changes the target distribution; the T squared factor compensates the gradient scale in the distillation objective.',
    ['This is logit optimization on one example, not an estimate of student architecture quality or compression speed.', 'For a controlled comparison both soft and hard targets use T-squared cross-entropy/KL at the selected T. Ordinary hard-label CE is the T=1 case; no mixed hard/soft objective is used here.'],
    ['p_T=\\operatorname{softmax}(z_t/T),\\quad q_T=\\operatorname{softmax}(z_s/T)', 'L=T^2 KL(p_T\\|q_T),\\quad\\nabla_{z_s}L=T(q_T-p_T)'],
    [['T', 'Temperature', '2 softens teacher logits [3,2,-1]'], ['p_T\\log(p_T/q_T)', 'One class contribution to KL', 'A term may be negative; the sum is nonnegative']],
    'p = softmax(teacher_logits / T)\nq = softmax(student_logits / T)\nloss = T*T * np.sum(p * np.log(p / q))\ngrad = T * (q - p)\nstudent_logits -= 0.5 * grad',
    'Can one class contribute a negative term to KL?', 'Yes, but the complete sum over classes is nonnegative.', 'No: each class term must independently be nonnegative.', 'When p is below q, p log(p/q) is negative; KL nonnegativity applies to the total.');
  lesson('serving-tradeoffs', 'Serving: read a request timeline before optimizing throughput',
    'Three requests arrive at different times. Compare immediate serial service with a fixed batching window.',
    'TTFT, completion latency, and aggregate token throughput answer different product questions.',
    'A scheduler trades queueing delay, shared compute, and active KV memory; benchmark the actual workload and hardware.',
    ['Deterministic illustrative costs, not measured GPU timings: prefill = 2 + 0.5 times prompt tokens ms; each output round = 2 + 0.5 times active requests ms.', 'Prefill computes first-token logits; this scheduler delays their emission by one modeled round. Later rounds consume the preceding output token. KV counts processed inputs, not the newest sampled token. Finished requests leave the batch; overlap, padding, paging and speculative decoding are excluded.'],
    ['TTFT_i=t_{first,i}-t_{arrival,i}', '\\operatorname{throughput}=N_{output}/(t_{last}-t_{first\ arrival})', 'M_{KV}=N_{cached}\\cdot2Lh_{KV}d_{head}b'],
    [['TTFT', 'Arrival-to-first-token time, including queueing', 'Request B arrives at 4ms'], ['N_{cached}', 'Peak active prompt plus already-consumed output tokens', 'Newest sampled token is not cached yet; finished requests are freed'], ['2Lh_{KV}d_{head}b', 'KV bytes per cached token', '2 * 2 layers * 1 head * 4 dimensions * 2 bytes = 32']],
    'ttft = first_token_time - arrival_time\nlatency = last_token_time - arrival_time\nthroughput = total_output_tokens / elapsed_seconds\nkv_bytes = peak_cached_tokens * 2 * layers * kv_heads * head_dim * bytes_per_element\n# Measure these on real hardware; the lesson uses stated costs.',
    'Can batching improve throughput while worsening first-token latency?', 'Yes: queueing and shared prefill can delay the first response.', 'No: higher aggregate throughput means every request is faster.', 'Throughput is an aggregate rate; TTFT includes each request\'s individual waiting time.');

  const mechanisms = {
    'learning-rate': 'At eta = 0.22, the sharp coordinate flips sign but shrinks by 0.76 each step. Inspect the gradient and parameter update at any plotted position.',
    initialization: 'Every ReLU unit exposes its input-weight products. The seed is fixed at 4 across scale choices so differences come from initialization scale, not a new random draw.',
    'gradient-flow': 'The backward table decomposes each input gradient into contributions from two output units. Compare saturation, ReLU gating, and a linear chain.',
    normalization: 'Perturb the last example without touching the first. Only training-mode BatchNorm changes the first row, because its denominator depends on other examples.',
    residuals: 'A branch close to zero nearly erases a plain stack while the residual stack retains input sensitivity. The cancellation case shows why the identity path is not a guarantee.',
    'flash-attention': 'The second tile raises the reference maximum. Both accumulated probability mass and accumulated values must be rescaled before adding the new tile.',
    'pretrain-finetune': 'Select a target token to see its causal prefix, negative log probability, and masked contribution. The reduction divides by supervised targets, not sequence length.',
    peft: 'Compare full, head-only, bias-only, and low-rank adapter updates on the same example. A selected parameter exposes its derivative, binary update mask, and post-step value.',
    lora: 'Select corresponding cells of W, BA, and W + BA. The inspector expands the low-rank dot product; the forward table checks the unmerged and merged paths.',
    quantization: 'The final weight is an outlier. Keeping it expands the shared grid; clipping it makes small weights more precise but adds a large saturation error.',
    distillation: 'At the default T=2, the teacher raises the related second candidate from a uniform student while a one-hot target suppresses it. Change temperature to see when that direction reverses.',
    'serving-tradeoffs': 'Queue segments, prefill blocks, and emitted-token dots share a timeline. Selecting a request separates its first-token wait from full completion latency.'
  };
  const questions = {
    'learning-rate': 'Can an update cross the optimum and still improve?', initialization: 'Does the signal survive another layer?',
    'gradient-flow': 'Where does the backward signal shrink or disappear?', normalization: 'Which cells determine this cell\'s normalization?',
    residuals: 'Can a skip connection keep the signal alive?', 'flash-attention': 'How does tile two preserve tile one\'s contribution?',
    'pretrain-finetune': 'Which tokens are we actually training on?', peft: 'Which weights actually change in an update?',
    lora: 'How can a small adapter change a frozen layer?', quantization: 'What happens when one weight is much larger?',
    distillation: 'What does the teacher teach about the runner-up?', 'serving-tradeoffs': 'Does batching make every request faster?'
  };
  for (const kind of Object.keys(content)) content[kind].what = mechanisms[kind];

  const defaults = {
    'learning-rate': { lr: .22, step: 2 }, initialization: { mode: 'he', layer: 0, cell: 0 },
    'gradient-flow': { mode: 'tanh', gain: 1.3, layer: 5 }, normalization: { mode: 'batch', cell: 0, shifted: false },
    residuals: { mode: 'small', layer: 1, cell: 0 }, 'flash-attention': { query: 0, step: 0, cell: 0 },
    'pretrain-finetune': { mode: 'sft', cell: 5 }, peft: { mode: 'adapter', matrix: 'B', cell: 0, applied: false },
    lora: { rank: 1, cell: 0, zero: true }, quantization: { bits: 4, limit: 5, cell: 7 },
    distillation: { temperature: 2, objective: 'soft', step: 0, cell: 1 }, 'serving-tradeoffs': { policy: 'batch', long: false, cell: 0 }
  };
  const initial = kind => ({ ...defaults[kind] });
  function reduce(kind, state, action, value) {
    if (action === 'reset') return initial(kind);
    const s = { ...state };
    const choices = {
      'learning-rate': { lr: [.05, .1, .22, .28] }, initialization: { mode: ['small', 'xavier', 'he', 'large'] },
      'gradient-flow': { mode: ['tanh', 'relu', 'linear'], gain: [.7, 1.3] }, normalization: { mode: ['batch', 'layer', 'rms'] },
      residuals: { mode: ['small', 'positive', 'cancel'] }, 'flash-attention': { query: [0, 1] },
      'pretrain-finetune': { mode: ['pretrain', 'sft'] }, peft: { mode: ['full', 'head', 'bias', 'adapter'], matrix: ['W', 'H', 'b', 'c', 'A', 'B'] },
      lora: { rank: [1, 2] }, quantization: { bits: [2, 4, 8], limit: [1, 5] },
      distillation: { temperature: [1, 2, 4], objective: ['soft', 'hard'] }, 'serving-tradeoffs': { policy: ['serial', 'batch'] }
    };
    if (choices[kind]?.[action]) {
      const selected = choices[kind][action].find(v => String(v) === String(value));
      if (selected === undefined) return s;
      s[action] = selected;
      if (kind === 'peft') { s.applied = false; s.cell = 0; if (action === 'mode') s.matrix = selected === 'adapter' ? 'B' : selected === 'head' ? 'H' : selected === 'bias' ? 'b' : 'W'; }
      if (kind === 'distillation') s.step = 0;
    }
    const cellMax = { initialization: 7, normalization: 8, residuals: 3, 'flash-attention': 3, 'pretrain-finetune': 8, peft: 15, lora: 11, quantization: 7, distillation: 2, 'serving-tradeoffs': 2 }[kind];
    if (action === 'cell' && Number.isInteger(Number(value))) s.cell = clamp(Number(value), 0, cellMax ?? 0);
    if (action === 'layer' && Number.isInteger(Number(value))) s.layer = clamp(Number(value), 0, kind === 'residuals' ? 6 : 5);
    if (action === 'step' && Number.isInteger(Number(value))) s.step = clamp(Number(value), 0, kind === 'flash-attention' ? 1 : kind === 'distillation' ? 20 : 8);
    if (action === 'next' || action === 'previous') s.step = clamp(s.step + (action === 'next' ? 1 : -1), 0, kind === 'flash-attention' ? 1 : kind === 'distillation' ? 20 : 8);
    if (action === 'toggle' && ['shifted', 'zero', 'long'].includes(value) && value in s) s[value] = !s[value];
    if (action === 'apply' && kind === 'peft') s.applied = !s.applied;
    return s;
  }

  const escape = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const f = (x, digits = 3) => !Number.isFinite(x) ? (x === -Infinity ? '-infinity' : String(x)) : Math.abs(x) > 0 && Math.abs(x) < .0005 ? x.toExponential(2) : Number(x.toFixed(digits)).toString();
  const vector = x => `[${x.map(v => f(v)).join(', ')}]`;
  const btn = (action, value, label, selected = false) => `<button type="button" data-action="${escape(action)}" data-value="${escape(value)}" aria-pressed="${!!selected}" class="${selected ? 'ml-selected' : ''}">${escape(label)}</button>`;
  const controls = (label, html) => `<div class="tl-control"><span class="ml-kicker">${escape(label)}</span><div class="ml-controls">${html}</div></div>`;
  const choices = (action, entries, selected) => entries.map(([value, label]) => btn(action, value, label, value === selected)).join('');
  const stat = (label, value) => `<div class="ml-stat"><span class="ml-kicker">${escape(label)}</span><strong>${escape(value)}</strong></div>`;
  const note = text => `<p class="ml-note">${escape(text)}</p>`;
  const inspector = (title, text) => `<aside class="ml-inspector" aria-live="polite"><h4>${escape(title)}</h4><p class="tl-equation">${escape(text)}</p></aside>`;
  const reveal = (s, panel, html) => `<details class="tl-disclosure" data-panel="${panel}"><summary>${panel === 'calculation' ? 'Inspect the calculation' : 'Change settings'}</summary><div class="tl-disclosure-body">${html}</div></details>`;
  const calculation = (s, html) => reveal(s, 'calculation', html);
  const outcome = text => `<p class="tl-outcome" aria-live="polite">${escape(text)}</p>`;
  const flow = items => `<div class="tl-flow" role="group" aria-label="Computed mechanism">${items.map(([label, value]) => `<div class="tl-flow-node"><span>${escape(label)}</span><strong>${escape(value)}</strong></div>`).join('')}</div>`;
  function primary(kind, s) {
    const actions = {
      'learning-rate': ['step', (s.step + 1) % 9, s.step === 8 ? 'Replay updates' : 'Next update'],
      initialization: ['layer', (s.layer + 1) % 6, s.layer === 5 ? 'Back to layer 1' : 'Next layer'],
      'gradient-flow': ['layer', s.layer > 0 ? s.layer - 1 : 5, s.layer > 0 ? 'Backprop one layer' : 'Replay backward pass'],
      normalization: ['toggle', 'shifted', s.shifted ? 'Restore example 3' : 'Change example 3'],
      residuals: ['layer', (s.layer + 1) % 7, s.layer === 6 ? 'Remove all blocks' : 'Add one block'],
      'flash-attention': ['step', 1 - s.step, s.step ? 'Replay tile 1' : 'Process tile 2'],
      'pretrain-finetune': ['mode', s.mode === 'sft' ? 'pretrain' : 'sft', s.mode === 'sft' ? 'Train on every token' : 'Train on responses only'],
      peft: ['apply', '', s.applied ? 'Undo the update' : 'Apply one update'],
      lora: ['toggle', 'zero', s.zero ? 'Apply example adapter' : 'Zero the adapter'],
      quantization: ['limit', s.limit === 5 ? 1 : 5, s.limit === 5 ? 'Clip the outlier' : 'Keep the outlier'],
      distillation: ['step', (s.step + 1) % 21, s.step === 20 ? 'Replay student training' : 'Train the student once'],
      'serving-tradeoffs': ['policy', s.policy === 'batch' ? 'serial' : 'batch', s.policy === 'batch' ? 'Serve without batching' : 'Batch the requests']
    };
    const [action, value, label] = actions[kind];
    return `<div class="tl-primary-row"><button type="button" class="tl-primary" data-action="${escape(action)}" data-value="${escape(value)}">${escape(label)}</button></div>`;
  }
  function settings(kind, s) {
    const fields = {
      'learning-rate': controls('Learning rate', choices('lr', [[.05, '0.05 slow'], [.1, '0.10 monotone'], [.22, '0.22 oscillating'], [.28, '0.28 divergent']], s.lr)),
      initialization: controls('Initialization', choices('mode', [['small', 'Small'], ['xavier', 'Xavier'], ['he', 'He'], ['large', 'Large']], s.mode)),
      'gradient-flow': controls('Activation', choices('mode', [['tanh', 'tanh'], ['relu', 'ReLU'], ['linear', 'Linear']], s.mode)) + controls('Weight gain', choices('gain', [[.7, '0.7'], [1.3, '1.3']], s.gain)),
      normalization: controls('Normalization axis', choices('mode', [['batch', 'BatchNorm'], ['layer', 'LayerNorm'], ['rms', 'RMSNorm']], s.mode)),
      residuals: controls('Learned branch', choices('mode', [['small', 'Near-zero'], ['positive', 'Amplifying'], ['cancel', 'Cancel the skip']], s.mode)),
      'flash-attention': controls('Query', choices('query', [[0, '[1, 1]'], [1, '[0, 2]']], s.query)),
      'pretrain-finetune': note('Model probabilities stay fixed so only the loss mask changes.'),
      peft: controls('Trainability', choices('mode', [['full', 'Full: 30'], ['head', 'Head: 10'], ['bias', 'Bias: 6'], ['adapter', 'Adapter: 8']], s.mode)),
      lora: controls('Adapter rank', choices('rank', [[1, 'Rank 1'], [2, 'Rank 2']], s.rank)),
      quantization: controls('Precision', choices('bits', [[2, '2 bit'], [4, '4 bit'], [8, '8 bit']], s.bits)),
      distillation: controls('Temperature', choices('temperature', [[1, 'T=1'], [2, 'T=2'], [4, 'T=4']], s.temperature)) + controls('Target', choices('objective', [['soft', 'Teacher'], ['hard', 'One-hot']], s.objective)),
      'serving-tradeoffs': controls('Prompt length', btn('toggle', 'long', 'Use 4x longer prompts', s.long))
    };
    return reveal(s, 'settings', fields[kind] + `<div class="tl-reset">${btn('reset', '', 'Reset experiment')}</div>`);
  }
  function matrix(data, label, selected = -1, action = 'cell', classFor = () => '') {
    return `<div class="tl-matrix-wrap"><h4>${escape(label)}</h4><div class="tl-matrix" style="--tl-cols:${data[0].length}" role="group" aria-label="${escape(label)}">${data.flatMap((row, i) => row.map((v, j) => {
      const index = i * row.length + j;
      return action ? `<button type="button" data-action="${escape(action)}" data-value="${index}" aria-pressed="${index === selected}" aria-label="${escape(label)}, row ${i + 1}, column ${j + 1}: ${escape(f(v))}" class="tl-cell ${index === selected ? 'ml-selected' : ''} ${escape(classFor(i, j))}">${escape(f(v))}</button>` : `<span class="tl-cell ${escape(classFor(i, j))}">${escape(f(v))}</span>`;
    })).join('')}</div></div>`;
  }
  function table(headers, rows) {
    return `<div class="ml-table tl-table" tabindex="0" role="region" aria-label="Computed values, scroll horizontally if needed"><table><thead><tr>${headers.map(h => `<th scope="col">${escape(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(v => `<td>${escape(v)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  }
  function lineChart(series, labels, selected, action = 'step') {
    const all = series.flatMap(s => s.values), lo = Math.min(0, ...all), hi = Math.max(...all, .01), range = hi - lo;
    const x = i => 28 + i / Math.max(1, labels.length - 1) * 444, y = v => 150 - (v - lo) / range * 120;
    return `<div class="tl-chart"><div class="tl-axis-labels"><span>Linear scale: ${escape(f(lo))} to ${escape(f(hi))}</span></div><svg viewBox="0 0 500 160" role="group" aria-label="${escape(series.map(s => s.name).join(' and '))}; select a point to inspect"><line x1="28" x2="472" y1="${y(0)}" y2="${y(0)}" class="tl-axis"/>${series.map((s, si) => `<polyline class="tl-line tl-color-${si}" points="${s.values.map((v, i) => `${x(i)},${y(v)}`).join(' ')}"/>${s.values.map((v, i) => `<g role="button" tabindex="0" data-action="${escape(action)}" data-value="${i}" aria-label="${escape(labels[i])}: ${escape(f(v))}" aria-pressed="${i === selected}"><circle cx="${x(i)}" cy="${y(v)}" r="14" fill="transparent"/><circle cx="${x(i)}" cy="${y(v)}" r="${i === selected ? 6 : 3}" class="tl-point tl-color-${si}"/></g>`).join('')}`).join('')}</svg><div class="tl-axis-labels"><span>${escape(labels[0])}</span><span>${escape(labels.at(-1))}</span></div><div class="tl-legend">${series.map((s, i) => `<span class="tl-legend-${i}">${escape(s.name)}</span>`).join('')}</div></div>`;
  }

  function renderLearning(s) {
    const trace = quadraticTrace(s.lr), row = trace[s.step], next = trace[Math.min(8, s.step + 1)];
    return `<div class="tl-mechanism">${lineChart([{ name: 'Loss', values: trace.map(r => r.loss) }], trace.map((_, i) => `t${i}`), s.step)}${flow([['Update', s.step], ['Parameters', vector(row.x)], ['Loss', f(row.loss)]])}</div>` +
      outcome(`Update ${s.step}: loss ${f(row.loss)}. With learning rate ${s.lr}, the sharp coordinate ${s.lr > .25 ? 'flips and grows' : s.lr > .125 ? 'flips but shrinks' : 'shrinks without flipping'}.`) +
      calculation(s, inspector('Two update multipliers', `Flat: 1 - ${s.lr} = ${f(1 - s.lr)}. Sharp: 1 - 8 * ${s.lr} = ${f(1 - 8 * s.lr)}. Magnitude below 1 means contraction.`) + table(['Coordinate', 'Before', 'Gradient', 'Update -eta*g', 'After'], row.x.map((v, i) => [i + 1, f(v), f(row.grad[i]), f(-s.lr * row.grad[i]), s.step < 8 ? f(next.x[i]) : 'end of trace'])));
  }
  function renderInitialization(s) {
    const trace = initializationTrace(s.mode), row = trace[s.layer], j = s.cell;
    return `<div class="tl-mechanism"><div class="ml-grid"><div>${lineChart([{ name: 'Activation RMS', values: trace.map(r => r.rms) }], trace.map((_, i) => `L${i + 1}`), s.layer, 'layer')}</div><div>${matrix([row.output.slice(0, 4), row.output.slice(4)], `Layer ${s.layer + 1} activations: tap a unit`, j)}${stat('Signal magnitude (RMS)', f(row.rms))}</div></div></div>` +
      outcome(`Unit ${j}: weighted input ${f(row.z[j])} becomes ${f(row.output[j])} after ReLU. ${row.dead} of 8 units are zero on this input.`) +
      calculation(s, inspector(`Layer ${s.layer + 1}: ${s.mode} initialization`, `z[${j}] = W[${j},:] dot h. Weight std = ${f(row.std)}; input = ${vector(row.input)}.`) + table(['Input i', 'h[i]', `W[${j},i]`, 'Product'], row.input.map((v, i) => [i, f(v), f(row.weights[j][i]), f(v * row.weights[j][i])])) + note('Fixed random seed; these are actual activations. He preserves a second moment in expectation, not the RMS of every finite-width layer.'));
  }
  function renderGradient(s) {
    const rows = gradientTrace(s.mode, s.gain), row = rows[s.layer];
    return `<div class="tl-mechanism">${lineChart([{ name: 'Gradient norm at each layer input', values: rows.map(r => norm(r.inGradient)) }], rows.map((_, i) => `L${i + 1}`), s.layer, 'layer')}${flow([['Incoming gradient', vector(row.outGradient)], [`Layer ${s.layer + 1}: ${s.mode} derivatives`, vector(row.derivative)], ['Gradient passed backward', vector(row.inGradient)]])}</div>` +
      outcome(`Across layer ${s.layer + 1}, gradient magnitude goes from ${f(norm(row.outGradient))} to ${f(norm(row.inGradient))}. Backprop travels from layer 6 toward layer 1.`) +
      calculation(s, inspector('Exact local backpropagation', `g_in = J transpose times g_out. Input ${vector(row.input)}; preactivation ${vector(row.z)}. Full-network input gradient ${vector(rows[0].inGradient)}.`) + `<div class="ml-grid">${matrix(row.weights, 'Weights W', -1, null)}${matrix(row.jacobian, 'J = diag(activation derivatives) W', -1, null)}</div>` + table(['Input coordinate', 'From output 0', 'From output 1', 'Sum'], [0, 1].map(i => [i, f(row.jacobian[0][i] * row.outGradient[0]), f(row.jacobian[1][i] * row.outGradient[1]), f(row.inGradient[i])])));
  }
  function renderNormalization(s) {
    const x = [[1, 2, 6], [2, 4, 8], [3, 6, 10]].map((r, i) => r.map(v => v + (s.shifted && i === 2 ? 8 : 0)));
    const out = normalizeMatrix(x, s.mode), i = Math.floor(s.cell / 3), j = s.cell % 3, selected = out[i][j];
    const group = (r, c) => (s.mode === 'batch' ? c === j : r === i) ? 'tl-group' : '';
    return `<div class="tl-mechanism"><div class="ml-grid">${matrix(x, 'Input: one example per row', s.cell, 'cell', group)}${matrix(out.map(r => r.map(c => c.value)), `${{ batch: 'BatchNorm', layer: 'LayerNorm', rms: 'RMSNorm' }[s.mode]} output`, s.cell, 'cell', group)}</div></div>` +
      outcome(`Example ${i + 1}, feature ${j + 1}: ${x[i][j]} becomes ${f(selected.value)}. ${s.mode === 'batch' ? 'Highlighted column shares statistics across examples.' : 'Highlighted row uses only its own features.'}`) +
      calculation(s, inspector('The highlighted cells supply the statistics', `Group ${vector(selected.group)}. ${s.mode === 'rms' ? 'No centering; mean square' : `Mean = ${f(selected.mean)}; population variance`} = ${f(selected.variance)}. (${x[i][j]} - ${f(selected.mean)}) / ${f(selected.denominator, 5)} = ${f(selected.value)}.`) + note('Gamma=1, beta=0, epsilon=1e-5. BatchNorm uses training statistics here, not inference running averages. LayerNorm cancels a shared row shift; RMSNorm does not.'));
  }
  function renderResidual(s) {
    const result = residualTrace(s.mode), row = result.rows[s.layer], i = Math.floor(s.cell / 2), j = s.cell % 2;
    return `<div class="tl-mechanism">${flow([['Input', '[1, -1]'], [`${s.layer} plain blocks`, 'branch only'], ['Output', vector(row.plainX)]])}${flow([['Input', '[1, -1]'], [`${s.layer} residual blocks`, 'input + branch'], ['Output', vector(row.residualX)]])}${matrix(row.residualJ, 'Residual input sensitivity: tap a connection', s.cell)}</div>` +
      outcome(`Input ${j} to output ${i}: plain sensitivity ${f(row.plainJ[i][j])}; residual sensitivity ${f(row.residualJ[i][j])}.`) +
      calculation(s, inspector('Identity changes the Jacobian', `Plain: (W^${s.layer})[${i},${j}] = ${f(row.plainJ[i][j])}. Residual: ((I+W)^${s.layer})[${i},${j}] = ${f(row.residualJ[i][j])}.`) + `<div class="ml-grid">${matrix(result.branch, 'Branch Jacobian W', -1, null)}${matrix(result.residual, 'Residual Jacobian I + W', -1, null)}</div>` + controls('Inspect stack depth', result.rows.map((_, i) => btn('layer', i, i, i === s.layer)).join('')) + note('A small branch preserves signal via identity. A positive branch can amplify it; W=-I cancels it completely.'));
  }
  function renderFlash(s) {
    const a = attentionExample(s.query), t = a.trace[s.step], k = s.cell;
    return `<div class="tl-mechanism"><h4>HBM to SRAM: process one highlighted tile</h4><div class="tl-keytiles">${attentionKeys.map((key, i) => `<button type="button" data-action="cell" data-value="${i}" aria-pressed="${i === k}" class="tl-tile ${i === k ? 'ml-selected' : ''} ${Math.floor(i / 2) === s.step ? 'tl-group' : ''}"><strong>Tile ${Math.floor(i / 2) + 1} / key ${i}</strong><span>score ${f(a.scores[i])}</span></button>`).join('')}</div>${flow([['Previous mass', f(t.oldL)], ['Rescale, then add this tile', `${f(t.alpha)}x + ${f(sum(t.weights))}`], ['New mass', f(t.l)]])}${matrix([t.output], `Output after tile ${s.step + 1}`, -1, null)}</div>` +
      outcome(s.step ? `Both tiles processed. Streaming output matches dense attention: ${vector(a.output)}.` : `Tile 1 is processed. Tile 2 contains the larger score ${f(Math.max(...a.scores.slice(2)))}; the old mass will need rescaling.`) +
      calculation(s, inspector(`Key ${k}: score and final probability`, `K=${vector(attentionKeys[k])}; V=${vector(attentionValues[k])}. dot(${vector(a.q)}, K) / sqrt(2) = ${f(a.scores[k])}; final probability = ${f(a.probabilities[k], 6)}. Running maximum ${f(t.oldM)} -> ${f(t.m)}.`) +
      table(['Accumulator coordinate', 'Rescaled previous u', 'New tile contribution', 'New u'], t.numerator.map((v, j) => [j, `${f(t.alpha)} * ${f(t.oldNumerator[j])}`, t.weights.map((p, i) => `${f(p)} * ${f(attentionValues[t.start + i][j])}`).join(' + '), f(v)])) +
      table(['Computation', 'Output vector'], [['Dense softmax over all 4 keys', vector(a.dense)], ['Streaming after both tiles', vector(a.output)], ['Maximum absolute difference', f(Math.max(...a.output.map((v, i) => Math.abs(v - a.dense[i]))), 12)]]) +
      note('Highlighted tile and running accumulators live in SRAM; K/V are read from HBM. Avoiding a materialized row of 4 float32 scores saves 32 bytes of score writes/reads. A full N x N score buffer alone is 4N squared bytes. Not a complete kernel IO estimate.'));
  }
  function renderPretrain(s) {
    const d = tokenLoss(s.mode), r = d.rows[s.cell];
    return `<div class="tl-mechanism"><div class="tl-tokens" role="group" aria-label="Inspect a target token">${d.rows.map((r, i) => `<button type="button" data-action="cell" data-value="${i}" aria-pressed="${i === s.cell}" class="tl-token ${r.mask ? 'tl-trainable' : 'tl-frozen'} ${i === s.cell ? 'ml-selected' : ''}"><strong>${escape(r.token)}</strong><span>${r.mask ? 'Train target' : 'Context only'}</span></button>`).join('')}</div>${flow([['Supervised targets', `${d.count} / 9`], ['Average target loss', `${f(d.loss)} nats`]])}</div>` +
      outcome(`${r.token}: ${r.mask ? `${f(r.nll)} nats count toward training` : 'no direct target loss, but still available as context'}.`) +
      calculation(s, inspector(`Target ${s.cell}: ${r.token}`, `Prefix: BOS ${tokens.slice(0, s.cell).join(' ')}. p(target | prefix) = ${r.p}. NLL = -ln(${r.p}) = ${f(r.nll)}; masked contribution = ${f(r.mask * r.nll)}.`) + table(['Target', 'Mask', 'Probability', 'Loss contribution'], d.rows.map(r => [r.token, r.mask, r.p, f(r.nll * r.mask)])) + note('Predictions are held fixed to isolate masking. Divide by supervised target count, not sequence length.'));
  }
  function renderPeft(s) {
    const p = peftStep(s.mode, s.applied), key = p.active.includes(s.matrix) ? s.matrix : p.active[0], data = p.updated[key], cell = Math.min(s.cell, data.flat().length - 1), i = Math.floor(cell / data[0].length), j = cell % data[0].length;
    const display = data[0].length === 1 ? [data.map(r => r[0])] : data;
    return `<div class="tl-mechanism"><div class="tl-parameter-map" role="group" aria-label="Model parameter groups; choose one to inspect">${p.active.map(k => `<button type="button" data-action="matrix" data-value="${k}" aria-pressed="${key === k}" class="${p.trainable(k) ? 'tl-trainable' : 'tl-frozen'}"><strong>${k}</strong><span>${p.trainable(k) ? 'Trainable' : 'Frozen'}</span><span>${p.matrices[k].flat().length} weights</span></button>`).join('')}</div><div class="ml-grid">${matrix(display, `${key}${data[0].length === 1 ? ' column shown left to right' : ''}: ${s.applied ? 'after' : 'before'} update`, cell, 'cell', () => p.trainable(key) ? 'tl-trainable' : 'tl-frozen')}${flow([['Loss before', f(p.loss, 6)], ['Loss now', f(p.nextLoss, 6)]])}</div></div>` +
      outcome(`${p.count} parameters may change. ${s.applied ? `Output is now ${vector(p.nextOutput)}; target is [1, 0].` : 'Frozen groups stay fixed when you apply the update.'}`) +
      calculation(s,
      inspector(`${key}[${i},${j}]: trainability is not gradient magnitude`, `Initial ${f(p.matrices[key][i][j])}; initial raw partial derivative ${f(p.gradients[key][i][j])}; mask ${+p.trainable(key)}; after one eligible step ${f(p.matrices[key][i][j] - .1 * +p.trainable(key) * p.gradients[key][i][j])}. The ${p.trainable('H') ? 'trainable' : 'frozen'} head sends initial dh = ${vector(p.dh)} backward.`) +
      stat('FP32 Adam m + v, theoretical', `${p.count * 8} bytes`) + note('Only SGD at learning rate 0.1 is simulated. Adam bytes count two moment tensors, not total memory. Zero-initialized B makes the first A gradient zero even though A is trainable.'));
  }
  function renderLora(s) {
    const d = loraExample(s.rank, s.zero), i = Math.floor(s.cell / 3), j = s.cell % 3;
    return `<div class="tl-mechanism"><div class="tl-branches">${flow([['Frozen path: W x', vector(d.base)]])}<span class="tl-plus" aria-label="plus">+</span>${flow([['Adapter path: B(A x)', vector(d.correction)]])}</div>${matrix(d.merged, 'Merged weights W + BA: tap a weight', s.cell)}${flow([['Same input x', vector(d.x)], ['Combined output', vector(d.output)]])}</div>` +
      outcome(`Weight [${i},${j}]: base ${f(d.W[i][j])} + adapter ${f(d.delta[i][j])} = ${f(d.merged[i][j])}. ${d.parameters} adapter parameters; 12 base weights stay frozen.`) +
      calculation(s, `<div class="ml-grid">${matrix(d.A, 'A (rank x 3)', -1, null)}${matrix(d.B, 'B (4 x rank)', -1, null)}</div>` +
      inspector(`Merged cell [${i},${j}]`, `${f(d.W[i][j])} + (${d.B[i].map((v, k) => `${f(v)} * ${f(d.A[k][j])}`).join(' + ')}) = ${f(d.merged[i][j])}.`) +
      `<div class="ml-grid">${matrix(d.W, 'Frozen W', s.cell)}${matrix(d.delta, 'BA correction', s.cell)}</div>` + table(['Forward path', 'Vector'], [['Input x', vector(d.x)], ['A x', vector(d.ax)], ['W x', vector(d.base)], ['B(A x)', vector(d.correction)], ['Sum = merged output', vector(d.output)]]) + note(`Adapter/base parameters: ${d.parameters}/12. At rank 2 this tiny adapter is not cheaper. A 4096 x 4096 layer at rank ${s.rank} uses ${s.rank * 8192} adapter parameters versus 16,777,216 base weights. No quality improvement is assumed.`));
  }
  function renderQuantization(s) {
    const d = quantize(quantWeights, s.bits, s.limit), row = d.rows[s.cell];
    const px = x => 30 + (x + 1) / 6 * 440;
    const levels = Array.from({ length: 2 * d.qmax + 1 }, (_, i) => (i - d.qmax) * d.scale).filter(v => v >= -1 && v <= 5);
    return `<div class="tl-mechanism"><div class="tl-chart"><svg viewBox="0 0 500 85" role="img" aria-label="Selected original and reconstructed weight; ticks are actual quantization levels"><line x1="30" x2="470" y1="54" y2="54" class="tl-axis"/>${levels.map(v => `<line x1="${px(v)}" x2="${px(v)}" y1="47" y2="61" class="tl-axis"/>`).join('')}<line x1="${px(row.x)}" x2="${px(row.reconstructed)}" y1="54" y2="54" class="tl-line tl-color-1"/><circle cx="${px(row.x)}" cy="40" r="6" class="tl-point tl-color-0"/><circle cx="${px(row.reconstructed)}" cy="68" r="6" class="tl-point tl-color-1"/></svg><div class="tl-axis-labels"><span>-1</span><span>5</span></div><div class="tl-legend"><span class="tl-legend-0">Original</span><span class="tl-legend-1">Reconstructed</span><span>Ticks: available levels</span></div></div><div class="tl-weight-choices" role="group" aria-label="Select a weight">${quantWeights.map((v, i) => btn('cell', i, v, i === s.cell)).join('')}</div>${flow([['Selected weight', row.x], [`${s.bits}-bit integer code`, row.q], ['Reconstructed weight', f(row.reconstructed)]])}</div>` +
      outcome(`Grid spacing ${f(d.scale)}. Selected error ${f(row.error)}. ${s.limit === 1 ? 'Clipping improves small-weight precision but sacrifices the outlier.' : 'Keeping the outlier forces a wider grid.'}`) +
      calculation(s,
      inspector(`Weight ${s.cell}: integer code ${row.q}`, `x=${row.x}; clipped=${row.clipped}; q=round(clipped/${f(d.scale, 6)})=${row.q}; reconstructed=${f(row.reconstructed)}; signed error=${f(row.error)}. Code range [${-d.qmax}, ${d.qmax}].`) +
      table(['Original', 'Integer q', 'Reconstructed', 'Error'], d.rows.map(r => [f(r.x), r.q, f(r.reconstructed), f(r.error)])) + stat('All-weight / non-outlier MSE', `${f(d.mse, 6)} / ${f(d.bulkMse, 6)}`) + stat('Packed weights + FP32 scale', `${d.packedBytes} + 4 = ${d.packedBytes + 4} bytes; FP16 base: 16`) + note('Weight MSE is not task accuracy. Storage excludes alignment and other runtime memory; no throughput estimate is inferred.'));
  }
  function renderDistillation(s) {
    const d = studentTrace(s.temperature, s.objective, s.step), i = s.cell, names = ['Relevant A', 'Related B', 'Irrelevant C'];
    const klTerm = d.target[i] === 0 ? 'KL term = 0 by the zero-target limit convention; log(0) is not evaluated.' : `KL term = ${f(d.target[i])} * ln(${f(d.target[i])}/${f(d.student[i])}) = ${f(d.terms[i])}.`;
    return `<div class="tl-mechanism"><div class="tl-distributions">${names.map((name, j) => `<button type="button" data-action="cell" data-value="${j}" aria-pressed="${j === i}" class="tl-distribution ${j === i ? 'ml-selected' : ''}"><strong>${name}</strong><span class="tl-prob-label">Target ${f(d.target[j])}</span><span class="tl-prob-track"><span class="tl-prob-target" style="width:${100 * d.target[j]}%"></span></span><span class="tl-prob-label">Student ${f(d.student[j])}</span><span class="tl-prob-track"><span class="tl-prob-student" style="width:${100 * d.student[j]}%"></span></span></button>`).join('')}</div>${flow([['Student updates', s.step], ['Mismatch: KL', f(d.kl, 6)]])}</div>` +
      outcome(`${names[i]}: target ${f(d.target[i])}, student ${f(d.student[i])}. The next update ${d.gradient[i] < 0 ? 'raises' : 'lowers'} its logit.`) +
      calculation(s, inspector(names[i], `${klTerm} Logit gradient = ${s.temperature} * (${f(d.student[i])} - ${f(d.target[i])}) = ${f(d.gradient[i])}. Next logit update = ${f(-.5 * d.gradient[i])}.`) + stat('T squared scaled objective', f(d.loss, 6)) + note(`At T=${s.temperature}, the teacher target for B is ${f(d.teacher[1])}, ${d.teacher[1] > 1 / 3 ? 'above' : 'below'} the uniform student probability 1/3. Its first soft-target update therefore ${d.teacher[1] > 1 / 3 ? 'raises' : 'lowers'} B's logit; one-hot training lowers it. Temperature or target changes restart the trace.`));
  }
  function renderServing(s) {
    const d = servingTimeline(s.policy, s.long), selected = d.results[s.cell], scale = t => 65 + t / d.finish * 415;
    const timeline = `<div class="tl-chart"><div class="tl-axis-labels"><span>0 ms</span><span>${f(d.finish)} ms</span></div>${d.results.map((r, i) => `<div class="tl-request-row">${btn('cell', i, r.name, i === s.cell)}<svg viewBox="60 0 430 35" role="img" aria-label="Request ${r.name} timeline; first token at ${r.emitted[0]} ms, last at ${r.emitted.at(-1)} ms"><line class="tl-wait" x1="${scale(r.arrival)}" x2="${scale(r.start)}" y1="18" y2="18"/><rect class="tl-prefill" x="${scale(r.start)}" y="10" width="${scale(r.prefillEnd) - scale(r.start)}" height="16"/><line class="tl-line tl-color-0" x1="${scale(r.prefillEnd)}" x2="${scale(r.emitted.at(-1))}" y1="18" y2="18"/>${r.emitted.map(t => `<circle class="tl-point tl-color-0" cx="${scale(t)}" cy="18" r="5"/>`).join('')}</svg></div>`).join('')}<div class="tl-legend"><span>Dashed: queue</span><span class="tl-legend-1">Block: prefill</span><span class="tl-legend-0">Dots: output tokens</span></div></div>`;
    return `<div class="tl-mechanism">${timeline}${flow([[`Request ${selected.name}: first-token wait`, `${selected.ttft} ms`], ['Total output rate', `${f(d.throughput, 1)} tokens/s`]])}</div>` +
      outcome(`${s.policy === 'batch' ? 'Batch at 10 ms' : 'Immediate serial service'}. Request ${selected.name}: ${selected.prompt} input / ${selected.tokens} output tokens. Timings are a deterministic model, not a benchmark.`) +
      calculation(s,
      inspector(`Request ${selected.name}`, `Arrival ${selected.arrival} ms; queue ${selected.wait} ms; prefill ${selected.start}-${selected.prefillEnd} ms; emitted at ${vector(selected.emitted)} ms. TTFT=${selected.ttft} ms; completion latency=${selected.latency} ms.`) +
      table(['Request', 'Arrival ms', 'Queue ms', 'TTFT ms', 'Completion latency ms'], d.results.map(r => [r.name, r.arrival, r.wait, r.ttft, r.latency])) + stat('Peak live KV payload', `${d.peakKVTokens} tokens * 32 = ${d.kvBytes} bytes`) +
      note('Toy costs: prefill = 2 + 0.5 * total prompt tokens ms; each output round = 2 + 0.5 * active requests ms. First output is delayed one round after prefill logits. KV excludes each newest sampled token until consumed, and excludes allocation overhead. These are not hardware benchmarks.'));
  }
  const renderers = { 'learning-rate': renderLearning, initialization: renderInitialization, 'gradient-flow': renderGradient, normalization: renderNormalization, residuals: renderResidual, 'flash-attention': renderFlash, 'pretrain-finetune': renderPretrain, peft: renderPeft, lora: renderLora, quantization: renderQuantization, distillation: renderDistillation, 'serving-tradeoffs': renderServing };
  function render(kind, state) {
    return `<div class="tl-lab"><header class="ml-heading"><h3 class="ml-question">${escape(questions[kind])}</h3></header>${primary(kind, state)}${renderers[kind](state)}${settings(kind, state)}</div>`;
  }
  const api = { quadraticTrace, initializationTrace, gradientTrace, normalizeMatrix, residualTrace, onlineAttention, attentionExample, tokenLoss, peftStep, loraExample, quantize, distillation, studentTrace, servingTimeline, softmax, mv, mm, transpose, dot, content, initial, reduce, render, escape };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window !== 'undefined') {
    window.AtelierLab.createModule({ id: 'training-labs', content, initial, render, reduce });
  }
})();
