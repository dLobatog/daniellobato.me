/* Finite, reproducible datasets: every displayed statistic is derived below. */
(() => {
  'use strict';
  const sum = xs => xs.reduce((a, b) => a + b, 0);
  const mean = xs => sum(xs) / xs.length;
  const mse = (a, b) => mean(a.map((v, i) => (v - b[i]) ** 2));
  const sigmoid = x => 1 / (1 + Math.exp(-x));
  const escape = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = (v, places = 3) => Number(v).toFixed(places);
  const pct = v => `${fmt(v * 100, 1)}%`;

  function betaEstimate(k, n, alpha = 3, beta = 9) {
    if (!(n > 0 && k >= 0 && k <= n && alpha >= 1 && beta >= 1)) throw new RangeError('Invalid binomial counts or prior');
    const a = k + alpha, b = n - k + beta;
    return { mle: k / n, map: (a - 1) / (a + b - 2), posteriorMean: a / (a + b), a, b,
      dataWeight: n / (n + alpha + beta - 2) };
  }
  function relativeBeta(x, a, b) {
    const mode = (a - 1) / (a + b - 2);
    if (a === 1 && b === 1) return 1;
    const kernel = p => (a === 1 ? 0 : (a - 1) * Math.log(p)) + (b === 1 ? 0 : (b - 1) * Math.log1p(-p));
    return Math.exp(kernel(x) - kernel(mode));
  }

  const sampleX = [-1, 0, 1];
  const samples = Array.from({ length: 8 }, (_, s) => sampleX.map((x, i) => x * x + (s & (1 << i) ? 0.5 : -0.5)));
  // Gray-code traversal changes exactly one training label on every primary click.
  const sampleOrder = [0, 1, 3, 2, 6, 7, 5, 4];
  function fitPolynomial(y, degree) {
    if (degree === 0) return [mean(y)];
    if (degree === 1) return [mean(y), (y[2] - y[0]) / 2];
    return [y[1], (y[2] - y[0]) / 2, (y[0] + y[2]) / 2 - y[1]];
  }
  const predictPolynomial = (w, x) => sum(w.map((v, i) => v * x ** i));
  const predictionInfluence = (degree, x) => sampleX.map((_, j) => predictPolynomial(fitPolynomial(sampleX.map((__, i) => Number(i === j)), degree), x));
  function biasVariance(degree, x) {
    const predictions = samples.map(y => predictPolynomial(fitPolynomial(y, degree), x));
    const average = mean(predictions), truth = x * x;
    const bias2 = (average - truth) ** 2, variance = mean(predictions.map(p => (p - average) ** 2));
    const risk = mean(predictions.flatMap(p => [-0.5, 0.5].map(e => (truth + e - p) ** 2)));
    return { predictions, average, truth, bias2, variance, noise: 0.25, risk };
  }

  const design = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
  const regularizationY = design.map(([a, b]) => 2 * a + 0.4 * b);
  function regularizedFit(penalty, lambda) {
    const ols = [0, 1].map(j => mean(design.map((x, i) => x[j] * regularizationY[i])));
    const weights = ols.map(w => penalty === 'l1' ? Math.sign(w) * Math.max(0, Math.abs(w) - lambda) : w / (1 + lambda));
    const predictions = design.map(x => sum(x.map((v, j) => v * weights[j])));
    return { weights, predictions, trainMSE: mse(regularizationY, predictions),
      holdoutMSE: mse(design.map(x => 2 * x[0]), predictions) };
  }

  const treeRows = [0, 0, 1, 0, 1, 1, 1, 1].map((y, i) => ({ id: `R${i + 1}`, x: i + 1, y }));
  function gini(rows) {
    if (!rows.length) return 0;
    const p = mean(rows.map(r => r.y));
    return 2 * p * (1 - p);
  }
  function splitStats(rows, threshold) {
    const left = rows.filter(r => r.x <= threshold), right = rows.filter(r => r.x > threshold);
    const parent = gini(rows), weighted = (left.length * gini(left) + right.length * gini(right)) / rows.length;
    return { left, right, parent, weighted, gain: parent - weighted };
  }
  const boostRows = [0, 0, 1, 1, 3, 3, 4, 4].map((y, i) => ({ id: `R${i + 1}`, x: i + 1, y }));
  function fitStump(rows, residuals) {
    const candidates = [...new Set(rows.map(r => r.x))].sort((a, b) => a - b);
    let best = null;
    candidates.slice(0, -1).forEach((x, i) => {
      const threshold = (x + candidates[i + 1]) / 2;
      const left = residuals.filter((_, j) => rows[j].x <= threshold);
      const right = residuals.filter((_, j) => rows[j].x > threshold);
      const leftMean = mean(left), rightMean = mean(right);
      const predictions = rows.map(r => r.x <= threshold ? leftMean : rightMean);
      const sse = sum(residuals.map((r, j) => (r - predictions[j]) ** 2));
      if (!best || sse < best.sse - 1e-12) best = { threshold, leftMean, rightMean, predictions, sse };
    });
    return best;
  }
  function boostTrace(rounds = 4, rate = 0.5) {
    const y = boostRows.map(r => r.y), initial = mean(y);
    let predictions = y.map(() => initial);
    const trace = [{ predictions, mse: mse(y, predictions), initial }];
    for (let round = 1; round <= rounds; round++) {
      const before = predictions, residuals = y.map((v, i) => v - before[i]);
      const stump = fitStump(boostRows, residuals);
      predictions = before.map((p, i) => p + rate * stump.predictions[i]);
      trace.push({ before, predictions, residuals, stump, mse: mse(y, predictions), rate });
    }
    return trace;
  }

  const leakageRows = [0, 1, 1, 0, 1, 0].map((y, i) => {
    const prediction = 10 + i * 10;
    return { id: `Q${i + 1}`, prediction, y, fields: {
      history: { value: [0, 1, 0, 0, 1, 1][i], event: prediction - 3, available: prediction - 2 },
      late: { value: y, event: prediction - 1, available: prediction + 1 },
      outcome: { value: y, event: prediction + 2, available: prediction + 2 },
    } };
  });
  function leakageAudit(feature, split) {
    const rows = leakageRows.map((r, i) => {
      const field = r.fields[feature], available = field.event <= r.prediction && field.available <= r.prediction;
      const test = split === 'time' ? i >= 4 : i % 2 === 1;
      return { ...r, field, available, test, offline: field.value, replay: available ? field.value : 0 };
    });
    const test = rows.filter(r => r.test);
    return { rows, offlineAccuracy: mean(test.map(r => Number(r.offline === r.y))),
      replayAccuracy: mean(test.map(r => Number(r.replay === r.y))), unavailable: rows.filter(r => !r.available).length };
  }

  const featureBatches = {
    reference: [0, 1, 2, 3, 4, 5, 6, null],
    missing: [0, null, null, 3, null, 5, null, null],
    range: [0, 2, 4, 6, 8, 10, 12, null],
  };
  function featureSummary(values, policy = 'explicit') {
    const transformed = values.map(x => x === null && policy === 'zero' ? 0 : x);
    const counts = [0, 0, 0, 0];
    transformed.forEach(x => counts[x === null ? 0 : x === 0 ? 1 : x < 4 ? 2 : 3]++);
    const observed = values.filter(x => x !== null);
    const predictions = transformed.map(x => sigmoid(-2 + 0.4 * (x ?? 0) - 0.8 * Number(x === null)));
    return { counts, transformed, predictions, observedMean: mean(observed), missingRate: values.filter(x => x === null).length / values.length,
      visibleMissingRate: counts[0] / values.length, averageScore: mean(predictions) };
  }

  const servingRows = [0, 1, 4, 9, null].map((count, i) => ({ id: `U${i + 1}`, count, stale: [0, 0, 1, 3, null][i] }));
  function servingPair(row, mode) {
    const offline = Math.log1p(row.count ?? 2);
    const online = mode === 'raw' ? (row.count ?? 2) : mode === 'stale' ? Math.log1p(row.stale ?? 2) : Math.log1p(row.count ?? (mode === 'null' ? 0 : 2));
    return { offline, online, offlineScore: sigmoid(-1.5 + 1.2 * offline), onlineScore: sigmoid(-1.5 + 1.2 * online), delta: online - offline };
  }

  const cohorts = [{ name: 'Returning', baseline: 16, candidate: 17, n: 20 }, { name: 'New', baseline: 8, candidate: 5, n: 20 }];
  const mixes = { offline: [0.9, 0.1], live: [0.2, 0.8], balanced: [0.5, 0.5] };
  function cohortScore(weights) {
    const baseline = sum(cohorts.map((c, i) => weights[i] * c.baseline / c.n));
    const candidate = sum(cohorts.map((c, i) => weights[i] * c.candidate / c.n));
    return { baseline, candidate, delta: candidate - baseline,
      contributions: cohorts.map((c, i) => weights[i] * (c.candidate - c.baseline) / c.n) };
  }

  // [negative labels, positive labels] within low/high-score feature buckets.
  const driftBatches = { reference: [[8, 2], [2, 8]], stable: [[8, 2], [2, 8]],
    covariate: [[4, 1], [3, 12]], concept: [[2, 8], [8, 2]], label: [[4, 3], [1, 12]] };
  function driftStats(batch) {
    const total = sum(batch.flat()), counts = batch.map(sum), distribution = counts.map(n => n / total);
    const labelRate = sum(batch.map(c => c[1])) / total;
    const accuracy = (batch[0][0] + batch[1][1]) / total;
    const brier = sum(batch.map((c, i) => c[0] * [0.2, 0.8][i] ** 2 + c[1] * (1 - [0.2, 0.8][i]) ** 2)) / total;
    return { total, counts, distribution, labelRate, accuracy, brier,
      conditional: batch.map((c, i) => c[1] / counts[i]),
      tv: 0.5 * sum(distribution.map(p => Math.abs(p - 0.5))) };
  }

  const content = {};
  function lesson(kind, title, summary, why, interview, details, formula, annotations, note, snippet, prompt, options) {
    content[kind] = { title, summary, what: '', why, interview, details,
      math: { title: 'Compute The Mechanism', formula, annotations, note },
      code: { title: 'Reproduce The Calculation', lang: 'python', snippet },
      quiz: { prompt, options: options.map(([text, correct, explanation]) => ({ text, correct, explanation })) },
      controls: [], presets: [], geometry: null };
  }
  lesson('mle-map', 'CTR estimates: how much evidence earns trust?',
    'Two clicks from three impressions is not the same evidence as 200 from 300. Inspect the posterior mode for each item.',
    'Sparse item statistics need calibrated shrinkage before becoming ranking features.',
    'Specify the likelihood and prior; distinguish a posterior mode from a posterior predictive mean.',
    ['This fixture treats impressions as independent Bernoulli trials with a stationary CTR.', 'A category prior should be estimated without evaluation leakage; selection and position bias remain unresolved.'],
    ['p \\mid D \\sim \\operatorname{Beta}(k+\\alpha,n-k+\\beta)', '\\hat p_{MAP}=\\frac{k+\\alpha-1}{n+\\alpha+\\beta-2},\\quad \\hat p_{MLE}=k/n'],
    [['k,n', 'Observed clicks and impressions', '2 clicks / 3 impressions'], ['alpha,beta', 'Beta prior parameters, not observed counts', 'Beta(3,9) has mode 0.2'], ['MAP', 'Posterior mode, not posterior mean', '(2+2)/(3+10) = 0.3077']],
    'For alpha,beta > 1, MAP blends the prior mode with MLE using effective prior strength alpha+beta-2. The mean uses alpha+beta instead.',
    'k, n = 2, 3\na, b = 3, 9\nmle = k / n\nmap_ctr = (k + a - 1) / (n + a + b - 2)\npredictive_ctr = (k + a) / (n + a + b)',
    'If the prior keeps its 20% mode but becomes stronger, what happens to MAP for the same 2/3 clicks?',
    [['It moves toward 20%; MLE stays at 2/3.', true, 'Beta(3,9) gives 4/13; Beta(11,41) gives 12/53. Only prior strength changed.'], ['It moves toward 2/3 because there are more observations.', false, 'Prior strength is not additional observed impressions.'], ['Only the posterior mean changes; MAP cannot move.', false, 'The posterior mode depends on the prior as well as the likelihood.']]);
  lesson('bias-variance', 'Fit all eight noisy datasets, not an invented U-curve',
    'Enumerate every +/-0.5 noise assignment at three fixed training inputs. Inspect how the fitted prediction moves at one test input.',
    'Instability is a property of the training procedure across datasets, not a property of a single fitted curve.',
    'At a fixed x, average over training datasets; a separate future label contributes irreducible noise.',
    ['The eight datasets are equally likely under independent symmetric binary noise.', 'This is fixed-design regression with known truth f(x)=x squared. It is not a universal complexity curve.'],
    ['\\mathbb E_{D,\\epsilon}[(f(x)+\\epsilon-\\hat f_D(x))^2]=(\\mathbb E_D[\\hat f_D(x)]-f(x))^2+\\operatorname{Var}_D(\\hat f_D(x))+0.25'],
    [['D', 'One of eight equally likely training datasets', 'Three independent +/-0.5 noises'], ['f(x)', 'Known synthetic response function', 'f(0.5)=0.25'], ['Variance', 'Population variance across fitted predictions', 'Divide by 8, not 7']],
    'All 8 training-noise patterns and both independent test-noise outcomes are enumerated, so the decomposition is exact, not a Monte Carlo estimate.',
    'import itertools\nimport numpy as np\nx = np.array([-1., 0., 1.])\nnoise = np.array(list(itertools.product([-.5, .5], repeat=3)))\nY = x*x + noise\ndegree, probe = 2, .5\nX = np.vander(x, degree+1, increasing=True)\nW = np.linalg.lstsq(X, Y.T, rcond=None)[0]\npred = probe ** np.arange(degree+1) @ W\nbias2 = (pred.mean() - probe**2)**2\nrisk = bias2 + pred.var() + .25',
    'If only the center training label rises by 1, how does the quadratic prediction at x=0.5 change?',
    [['It rises by 0.75, although training error remains zero.', true, 'The center label coefficient is 1-x squared = 0.75. The refit still interpolates all three labels.'], ['It cannot change because both fits have zero training error.', false, 'Interpolation does not imply stable predictions between training inputs.'], ['It rises by exactly 1 everywhere.', false, 'The changed label has a different influence at different test inputs.']]);
  lesson('regularization', 'Which feature survives the penalty?',
    'Solve L1 and L2 exactly on a centered, orthogonal two-feature design. Inspect the actual feature contributions to a relevance score.',
    'Shrinkage changes the learned coefficients; it is not a guaranteed improvement in held-out performance.',
    'Penalty scale and feature scale matter. L1 soft-thresholding is exact here because X transpose X / n is identity.',
    ['The weak second feature has an association only in this deliberately constructed training set.', 'Holdout labels omit that association. This illustrates one failure mode, not evidence that the feature is causally irrelevant.'],
    ['L_1=\\frac{\\|y-Xw\\|^2}{2n}+\\lambda\\|w\\|_1,\\quad w_j=\\operatorname{sign}(b_j)(|b_j|-\\lambda)_+', 'L_2=\\frac{\\|y-Xw\\|^2}{2n}+\\frac{\\lambda}{2}\\|w\\|_2^2,\\quad w_j=\\frac{b_j}{1+\\lambda}'],
    [['b', 'Unpenalized coefficient X transpose y / n', '[2.0, 0.4]'], ['lambda', 'Penalty in the displayed normalized objective', 'At 0.5, L1 gives [1.5,0]'], ['x_j w_j', 'Contribution to this row score', 'x1=-1 and w1=1.5 gives -1.5']],
    'Centered inputs and targets make the intercept zero. These closed forms do not apply to a general correlated design.',
    'import numpy as np\nX = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])\ny = X @ np.array([2., .4])\nb = X.T @ y / len(y)\nlam = .5\nl1 = np.sign(b) * np.maximum(abs(b)-lam, 0)\nl2 = b / (1+lam)\ntrain_mse = np.mean((y-X@l1)**2)\nholdout_mse = np.mean((2*X[:,0]-X@l1)**2)',
    'If lambda rises from 0.25 to 0.5, what happens to the weak coefficient with OLS value 0.4?',
    [['L1 goes from 0.15 to 0; L2 goes from 0.32 to about 0.267.', true, 'L1 subtracts the threshold. L2 divides by 1+lambda in this orthogonal design.'], ['Both penalties make the coefficient exactly zero.', false, 'L2 shrinks this nonzero coefficient but does not threshold it.'], ['The held-out error must improve.', false, 'The useful first coefficient also shrinks; the net quality change is not guaranteed.']]);
  lesson('tree-split', 'Choose the split from actual class counts',
    'Eight impressions, one numeric feature, five clicks. Move the cut and inspect the weighted impurity; then restrict the search to histogram boundaries.',
    'Histogram binning accelerates split search by aggregating counts, but it restricts candidate thresholds.',
    'Compare size-weighted child impurity, enforce leaf-size constraints, and separate training gain from generalization.',
    ['Click labels and features are fixed; moving the split never changes the data.', 'This is binary Gini splitting, not the gradient/Hessian gain formula used by second-order boosted trees.'],
    ['G(S)=2p_S(1-p_S)', '\\Delta G=G(S)-\\frac{n_L}{n}G(L)-\\frac{n_R}{n}G(R)'],
    [['p_S', 'Click fraction inside a node', 'Parent: 5/8'], ['n_L/n', 'Weight of the left node', 'At 4.5: 4/8'], ['Delta G', 'Reduction in weighted Gini', '0.46875-0.1875=0.28125']],
    'Counts in each histogram bin are sufficient for this one-dimensional classification split search.',
    'rows = list(zip(range(1,9), [0,0,1,0,1,1,1,1]))\ndef gini(rows):\n    p = sum(y for x,y in rows)/len(rows)\n    return 2*p*(1-p)\ndef gain(t):\n    L = [r for r in rows if r[0] <= t]\n    R = [r for r in rows if r[0] > t]\n    return gini(rows)-(len(L)*gini(L)+len(R)*gini(R))/len(rows)\nbest = max([i+.5 for i in range(1,8)], key=gain)',
    'If every training row is duplicated equally, does the Gini gain of a fixed split change?',
    [['No; class fractions and child-size proportions stay the same.', true, 'Both n_child and n double. The normalized impurity reduction is unchanged.'], ['It doubles because there are twice as many rows.', false, 'This is normalized Gini gain, not an unnormalized loss sum.'], ['The smaller child automatically becomes the best split.', false, 'Uniform duplication changes neither candidate partition fractions nor their ordering.']]);
  lesson('boosting', 'Fit the residuals, then add the leaf mean',
    'Trace four real squared-loss boosting rounds. Each stump searches every split and fits mean residuals in its two leaves.',
    'Stagewise additions update the existing ensemble instead of replacing it or refitting the original labels.',
    'Residual fitting is the negative-gradient rule for squared loss; other losses need their own gradients.',
    ['Training MSE is calculated after each addition. No synthetic overfitting score is shown.', 'Depth-one trees, no subsampling or leaf penalty, deterministic first-threshold tie breaking.'],
    ['r_{im}=y_i-F_{m-1}(x_i)', 'h_m=\\arg\\min_{h\\in\\mathrm{stumps}}\\sum_i(r_{im}-h(x_i))^2', 'F_m(x)=F_{m-1}(x)+\\eta h_m(x)'],
    [['F0', 'Initial constant: mean label', 'Mean of [0,0,1,1,3,3,4,4] is 2'], ['h1', 'Mean residual in each best leaf', 'At x<=4.5: -1.5; otherwise +1.5'], ['eta', 'Fraction of fitted residual added', '0.5 * -1.5 = -0.75']],
    'For one-half squared error, y-F is the negative derivative with respect to the prediction. Logistic boosting uses y-sigmoid(F), not y-F.',
    'import numpy as np\nx = np.arange(1,9)\ny = np.array([0,0,1,1,3,3,4,4], dtype=float)\nF = np.full(8, y.mean())\nfor _ in range(4):\n    r = y-F\n    candidates = []\n    for t in np.arange(1.5,8,1):\n        h = np.where(x<=t, r[x<=t].mean(), r[x>t].mean())\n        candidates.append((((r-h)**2).sum(), h))\n    _, h = min(candidates, key=lambda item: item[0])\n    F += .5*h',
    'Hold the first fitted tree fixed. If its learning rate rises from 0.5 to 1, what is R1\'s new prediction?',
    [['0.5: start at 2 and add the full -1.5 leaf mean.', true, 'At rate 0.5 it is 1.25; at rate 1 it is 0.5. Later trees would then see different residuals.'], ['1.25; changing the rate does not change a fitted tree.', false, 'The tree stays fixed, but the fraction added to the ensemble changes.'], ['0; a full step must fit every row perfectly.', false, 'The leaf mean fits a group, not each target individually.']]);
  lesson('feature-leakage', 'A temporal split cannot repair a future-aware join',
    'Audit event time and availability time for each request. Compare a hindsight snapshot with a prediction-time replay of the same fixed rule.',
    'A feature can describe a past event but still be unavailable when the scoring request arrives.',
    'Validate both information availability and train/test boundaries; time splitting alone is not a point-in-time join.',
    ['The rule predicts the selected binary feature value. It is intentionally fixed, not a trained model.', 'The replay uses an explicit missing-feature fallback of zero. The small held-out accuracy is an audit example, not a deployment estimate.'],
    ['\\mathrm{eligible}(v,q)=[t_{event}(v)\\le t_q]\\land[t_{available}(v)\\le t_q]'],
    [['t_q', 'Request prediction time', 'Q5 at minute 50'], ['t_available', 'First time serving could read the value', 'A late event arrives at minute 51'], ['Split', 'Which request rows are held out', 'Temporal split: Q5 and Q6']],
    'A post-outcome field is unavailable even if it is present in a later warehouse snapshot. A late-arriving past event also fails the availability check.',
    'def replay(field, prediction_time):\n    eligible = (field["event"] <= prediction_time\n                and field["available"] <= prediction_time)\n    return field["value"] if eligible else 0\n# Apply per request BEFORE scoring held-out rows.\n# A chronological row split does not change eligibility.',
    'A request scores at minute 50; an event is readable at 51. If its event timestamp changes from 49 to 48, is it now valid?',
    [['No; it is still unavailable at scoring time.', true, 'The event-time test passes in both cases; the availability-time test still fails.'], ['Yes; moving the event earlier repairs the join.', false, 'Earlier occurrence does not make the stored value readable sooner.'], ['Yes, if held-out requests are later than training requests.', false, 'The request split does not alter per-request feature availability.']]);
  lesson('feature-shift', 'Missing is not zero, and a schema is not a distribution',
    'Inspect eight raw values from each snapshot. Compare a missingness-aware transform with silently replacing nulls by zero.',
    'A default value can hide an upstream outage and alter the model score while the schema still passes.',
    'Monitor raw missingness before imputation, bucket counts after transformation, and downstream quality once labels mature.',
    ['These fixed snapshots show missingness and range changes, not the cause of those changes.', 'The score is an illustrative fixed logistic model. No labels are supplied, so its accuracy is unknown.'],
    ['\\hat p=\\sigma(-2+0.4x_{filled}-0.8\\mathbf 1_{missing})', '\\mathrm{missing\ rate}=\\frac{\\#\\{x_i=\\mathrm{null}\\}}{n}'],
    [['x_filled', 'Use zero only for numeric computation', 'A missing input has x_filled=0'], ['1_missing', 'Preserves whether zero was imputed', 'Null scores sigmoid(-2.8), real zero sigmoid(-2)'], ['Observed mean', 'Mean over non-null raw values', 'Reference: 21/7 = 3']],
    'A raw null-to-zero coercion destroys the missingness indicator. It is not equivalent to a fitted missingness-aware preprocessing step.',
    'import math\ndef features(x, preserve_missing=True):\n    missing = x is None\n    return (0 if missing else x), int(missing and preserve_missing)\ndef score(x, preserve_missing=True):\n    value, missing = features(x, preserve_missing)\n    z = -2 + .4*value - .8*missing\n    return 1/(1+math.exp(-z))',
    'If the missing-flag coefficient were zero, would erasing that flag still change this model score?',
    [['No, but post-transform missingness monitoring would still lose the signal.', true, 'The flag no longer contributes to the logit; raw nulls still exist and must be monitored before coercion.'], ['Yes; every missingness change must alter every score.', false, 'A zero coefficient makes this particular flag irrelevant to the fixed model score.'], ['No, so the input pipeline must be healthy.', false, 'An unchanged score does not establish that the raw feed recovered.']]);
  lesson('serving-skew', 'Replay the same request through both feature paths',
    'The model artifact is fixed. Compare log transforms, stale counters, and null defaults on paired requests.',
    'Paired parity tests isolate implementation differences that population-level histograms can miss.',
    'Same schema is insufficient: compare feature values, source timestamps, transformation versions, and scores for the same request.',
    ['Offline uses log1p(count), with count=2 for null. The model coefficients are fixed for this fixture.', 'The stale scenario replays explicitly stored older counter values; it is separate from a changed user population.'],
    ['x_{offline}=\\log(1+c),\\quad p=\\sigma(-1.5+1.2x)', '\\Delta x_i=x_{serve,i}-x_{offline,i}'],
    [['c', 'Recent interaction count, null filled with 2', 'U4 count=9'], ['x', 'Transformed model input', 'log1p(9)=2.303, not 9'], ['Delta x', 'Paired difference on the same example', 'Raw-count bug: 9-2.303=6.697']],
    'An identical feature name and numeric type can hide a large semantic mismatch. A near-zero difference is the parity target, not a model-quality guarantee.',
    'import math\ndef offline(c):\n    return math.log1p(2 if c is None else c)\ndef buggy_online(c):\n    return 2 if c is None else c  # missing log1p\nfor c in [0,1,4,9,None]:\n    expected = offline(c)\n    actual = buggy_online(c)\n    print(c, actual-expected)\n# Repair: version and reuse the same transform, then replay.',
    'If every replay count is zero, will the missing-log1p bug necessarily fail the parity test?',
    [['No; log1p(0)=0. Include positive counts and null cases.', true, 'Passing a degenerate fixture does not prove equivalent transformations.'], ['Yes; different source code always produces different values.', false, 'These transforms agree at zero despite disagreeing elsewhere.'], ['No, which proves the pipelines are equivalent.', false, 'Test coverage must exercise values where the implementations differ.']]);
  lesson('online-offline', 'The candidate wins offline, but which population?',
    'Keep observed cohort success rates fixed. Change only evaluation weights to see the aggregate comparison reverse.',
    'Observed candidate-minus-baseline rates can differ by cohort; the logged mix may not match the launch population.',
    'Standardize both models to the same target cohort mix. Treat transport assumptions separately from causal identification.',
    ['Counts here are a small illustrative observational replay, not a randomized experiment.', 'The live-mix result is a projection assuming stable within-cohort rates, not a measured launch effect. Randomized online evaluation still needs uncertainty and guardrails.'],
    ['M_A(w)=\\sum_c w_c\\frac{s_{Ac}}{n_c},\\quad \\Delta(w)=M_B(w)-M_A(w)'],
    [['w_c', 'Common target weight for a cohort', 'Offline: returning 0.9, new 0.1'], ['s/n', 'Observed success rate in that cohort', 'New: baseline 8/20, candidate 5/20'], ['Delta(w)', 'Weighted candidate minus baseline', '0.9*0.05+0.1*(-0.15)=+0.03']],
    'This is a cohort-weight reversal with heterogeneous differences, not proof of a causal treatment effect or a pure Simpson paradox with same-sign within-cohort effects.',
    'baseline = [16/20, 8/20]\ncandidate = [17/20, 5/20]\nfor weights in ([.9,.1], [.2,.8]):\n    a = sum(w*r for w,r in zip(weights, baseline))\n    b = sum(w*r for w,r in zip(weights, candidate))\n    print(a, b, b-a)\n# The projection does not remove exposure/selection bias.',
    'If the candidate were better in every cohort, could common nonnegative cohort weights make its aggregate worse?',
    [['No; a weighted sum of positive differences cannot be negative.', true, 'This example reverses because the within-cohort differences have opposite signs.'], ['Yes; any traffic shift can reverse any comparison.', false, 'Using the same weights for both models preserves same-sign cohort differences.'], ['No, so observational reweighting proves causal benefit.', false, 'The algebra does not remove exposure or selection bias.']]);
  lesson('data-drift', 'No feature alert. Is the model still healthy?',
    'Inspect exact joint counts for two score buckets. Delay outcome labels to separate what input monitoring sees from what performance monitoring can know.',
    'Unchanged input histograms can coexist with a reversed label relationship and sharply worse predictions.',
    'Separate P(x), P(y), P(y|x), and observed loss; label shift specifically assumes stable P(x|y).',
    ['Each scenario is an exact finite batch, not an estimated alert probability.', 'Delayed labels hide quality metrics rather than substituting a drift score for accuracy. Real alerts also need sample-size and delay-aware uncertainty.'],
    ['TV(P,Q)=\\frac12\\sum_x|P(x)-Q(x)|', '\\mathrm{Brier}=\\frac1n\\sum_i(p_i-y_i)^2'],
    [['TV', 'Input bucket distribution difference', 'Concept-change batch: TV=0'], ['P(y=1|x)', 'Positive fraction within each bucket', 'Low bucket changes from 2/10 to 8/10'], ['Accuracy', 'Threshold p at 0.5', 'Concept batch: (2+2)/20 = 0.2']],
    'The concept-change batch keeps both P(x) and P(y) fixed. The label-shift batch preserves P(x|y) while changing prevalence; these distribution descriptions are not mutually exclusive in general.',
    'reference = [(8,2),(2,8)]  # (negative, positive) by bucket\nlive = [(2,8),(8,2)]\nN = sum(a+b for a,b in live)\ntv = .5*sum(abs((a+b)/N-.5) for a,b in live)\n# Only calculate the following once labels are available.\naccuracy = (live[0][0]+live[1][1])/N\nbrier = sum(a*p*p+b*(1-p)**2\n            for (a,b),p in zip(live,[.2,.8]))/N',
    'If input mix and overall click rate stay fixed but clicks switch buckets, can accuracy fall from 80% to 20%?',
    [['Yes; the conditional label relationship can reverse.', true, 'This batch swaps 2/10 and 8/10 positive rates within the buckets without changing either marginal.'], ['No; stable marginals imply a stable joint distribution.', false, 'Marginals do not determine P(x,y).'], ['Input drift alone establishes causal harm.', false, 'It neither measures prediction error nor identifies a causal effect.']]);

  const initial = kind => ({
    'mle-map': { item: 0, prior: 'category' }, 'bias-variance': { degree: 2, sample: 0, probe: 0.5 },
    regularization: { penalty: 'l1', lambda: 0.5, row: 0, feature: 1 }, 'tree-split': { mode: 'exact', threshold: 4.5, row: 2 },
    boosting: { step: 1, rate: 0.5, row: 0 }, 'feature-leakage': { field: 'outcome', split: 'time', row: 4, replay: false },
    'feature-shift': { batch: 'missing', policy: 'explicit', row: 1 }, 'serving-skew': { mode: 'raw', row: 3 },
    'online-offline': { mix: 'offline', cohort: 1 }, 'data-drift': { scenario: 'concept', labels: false, bucket: 0 },
  }[kind]);
  function reduce(kind, state, action, value) {
    if (action === 'reset') return initial(kind);
    const s = { ...state };
    if (kind === 'bias-variance' && action === 'flip-label' && ['0', '1', '2'].includes(String(value))) {
      action = 'sample'; value = s.sample ^ (1 << Number(value));
    }
    if (action === 'primary') {
      // A stable action/value identity lets the shared core restore keyboard focus.
      const next = {
        'mle-map': ['item', (s.item + 1) % 3], 'bias-variance': ['sample', sampleOrder[(sampleOrder.indexOf(s.sample) + 1) % 8]],
        regularization: ['penalty', s.penalty === 'l1' ? 'l2' : 'l1'],
        'tree-split': ['threshold', s.threshold === 4.5 ? 2.5 : 4.5],
        boosting: [s.step < 4 ? 'next' : 'reset', ''], 'feature-leakage': ['replay', ''],
        'feature-shift': ['policy', s.policy === 'explicit' ? 'zero' : 'explicit'],
        'serving-skew': ['mode', s.mode === 'fixed' ? 'raw' : 'fixed'],
        'online-offline': ['mix', s.mix === 'live' ? 'offline' : 'live'], 'data-drift': ['labels', ''],
      }[kind];
      if (!next) return state;
      [action, value] = next;
      if (action === 'reset') return initial(kind);
    }
    const allowed = {
      'mle-map': { item: [0, 1, 2], prior: ['flat', 'category', 'strong'] },
      'bias-variance': { degree: [0, 1, 2], sample: [0, 1, 2, 3, 4, 5, 6, 7], probe: [0, 0.5, 1.5] },
      regularization: { penalty: ['l1', 'l2'], lambda: [0, 0.25, 0.5, 1], row: [0, 1, 2, 3], feature: [0, 1] },
      'tree-split': { mode: ['exact', 'hist'], threshold: [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], row: [0, 1, 2, 3, 4, 5, 6, 7] },
      boosting: { rate: [0.25, 0.5, 1], row: [0, 1, 2, 3, 4, 5, 6, 7] },
      'feature-leakage': { field: ['history', 'late', 'outcome'], split: ['time', 'alternating'], row: [0, 1, 2, 3, 4, 5] },
      'feature-shift': { batch: ['reference', 'missing', 'range'], policy: ['explicit', 'zero'], row: [0, 1, 2, 3, 4, 5, 6, 7] },
      'serving-skew': { mode: ['raw', 'stale', 'null', 'fixed'], row: [0, 1, 2, 3, 4] },
      'online-offline': { mix: ['offline', 'live', 'balanced'], cohort: [0, 1] },
      'data-drift': { scenario: ['stable', 'covariate', 'concept', 'label'], bucket: [0, 1] },
    }[kind] || {};
    if (allowed[action]) {
      const next = allowed[action].find(v => String(v) === String(value));
      if (next !== undefined) s[action] = next;
    }
    if (kind === 'tree-split' && s.mode === 'hist' && ![2.5, 4.5, 6.5].includes(s.threshold)) s.threshold = 4.5;
    if (kind === 'boosting' && action === 'next') s.step = Math.min(4, s.step + 1);
    if (kind === 'boosting' && action === 'previous') s.step = Math.max(0, s.step - 1);
    if (kind === 'data-drift' && action === 'labels') s.labels = !s.labels;
    if (kind === 'data-drift' && action === 'scenario') s.labels = false;
    if (kind === 'feature-leakage' && action === 'replay') s.replay = !s.replay;
    if (!['row', 'feature', 'cohort', 'bucket'].includes(action) && JSON.stringify(baseState(s)) !== JSON.stringify(baseState(state))) s._previous = baseState(state);
    return s;
  }
  function baseState(state) {
    const { _previous, ...base } = state;
    return base;
  }

  const button = (action, value, label, selected, disabled = false) => `<button type="button" data-action="${escape(action)}" data-value="${escape(value)}"${selected === undefined ? '' : ` aria-pressed="${Boolean(selected)}"`}${disabled ? ' disabled' : ''}>${escape(label)}</button>`;
  const controls = (label, html) => `<div class="tl-controls"><span class="ml-kicker">${escape(label)}</span><div class="ml-controls" role="group" aria-label="${escape(label)}">${html}</div></div>`;
  const header = (question, note) => `<header class="tl-heading"><h3 class="ml-question">${escape(question)}</h3><p class="ml-note">${escape(note)}</p></header>`;
  const stat = (label, value) => `<div class="ml-stat"><span>${escape(label)}</span><strong>${escape(value)}</strong></div>`;
  const note = text => `<p class="tl-observation" aria-live="polite">${escape(text)}</p>`;
  const table = (caption, headings, rows) => `<div class="ml-table tl-table" tabindex="0" role="region" aria-label="${escape(caption)}"><table><caption>${escape(caption)}</caption><thead><tr>${headings.map(h => `<th scope="col">${escape(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(cell => `<td>${cell}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  const panel = (title, body) => `<section class="tl-panel"><h4>${escape(title)}</h4>${body}</section>`;
  const bar = (label, value, color = 'blue') => `<div class="tl-bar-row"><span>${escape(label)}</span><div class="tl-bar-track"><i class="tl-${color}" style="width:${Math.max(0, Math.min(100, value * 100))}%"></i></div><strong>${escape(pct(value))}</strong></div>`;
  const rowsControls = (action, n, selected, prefix = 'R') => controls('Inspect a row', Array.from({ length: n }, (_, i) => button(action, i, `${prefix}${i + 1}`, selected === i)).join(''));
  const primary = (label, context = '') => `<div class="tl-action">${button('primary', '', label).replace('<button ', '<button class="tl-primary" ')}${context ? `<span class="ml-note">${escape(context)}</span>` : ''}</div>`;
  const fold = (s, name, title, body) => `<details class="tl-disclosure" data-disclosure="${escape(name)}"><summary>${escape(title)}</summary><div class="tl-disclosure-body">${body}</div></details>`;
  const inspect = (s, body) => fold(s, 'calculation', 'Inspect the calculation', body);
  const settings = (s, body) => fold(s, 'settings', 'Try other settings', body);
  const mechanism = body => `<div class="tl-mechanism">${body}</div>`;
  const flow = (nodes, focus = -1) => `<div class="tl-flow">${nodes.map(([label, value], i) => `<div class="tl-flow-node${focus >= 0 && i !== focus ? ' tl-muted-data' : ''}"><span>${escape(label)}</span><strong>${escape(value)}</strong></div>`).join('')}</div>`;
  const tile = (action, value, label, detail, selected, extra = '') => `<button type="button" class="tl-tile ${escape(extra)}" data-action="${escape(action)}" data-value="${escape(value)}" aria-pressed="${Boolean(selected)}"><span>${escape(label)}</span><strong>${escape(detail)}</strong></button>`;
  const linked = (label, parts) => `<p class="tl-linked-equation"><span>${escape(label)}</span><code>${parts.map(part => typeof part === 'object' ? `<mark>${escape(part.text)}</mark>` : escape(part)).join('')}</code></p>`;
  const changed = (label, before, after, reason) => `<p class="tl-causal-change" aria-live="polite"><span>${escape(label)}: </span><span class="tl-before">${escape(before)}</span><span> to </span><strong>${escape(after)}</strong><span class="tl-causal-reason">${escape(reason)}</span></p>`;

  function renderMap(s) {
    const items = [[2, 3], [20, 30], [200, 300]], priors = { flat: [1, 1], category: [3, 9], strong: [11, 41] };
    const [k, n] = items[s.item], [a, b] = priors[s.prior], result = betaEstimate(k, n, a, b);
    const previous = s._previous;
    const before = previous ? betaEstimate(...items[previous.item], ...priors[previous.prior]).map : result.mle;
    const equation = linked('MAP: highlighted counts are observations', ['(', {text:k}, ` + ${a - 1}) / (`, {text:n}, ` + ${a + b - 2}) = ${fmt(result.map, 4)}`]);
    const curves = [[k + 1, n - k + 1, 'blue'], [a, b, 'muted'], [result.a, result.b, 'orange']];
    const svg = `<svg class="tl-plot" viewBox="0 0 360 190" role="img" aria-label="Relative likelihood, prior, and posterior over click probability. Each curve has its own peak scaled to one."><path class="tl-axis" d="M30 18V178H340"/>${curves.map(([aa, bb, color]) => `<path class="tl-line tl-stroke-${color}" d="${Array.from({ length: 151 }, (_, i) => { const x = 0.001 + i / 150 * 0.998; return `${i ? 'L' : 'M'}${30 + x * 310},${178 - relativeBeta(x, aa, bb) * 150}`; }).join(' ')}"/>`).join('')}</svg><div class="tl-axis-labels"><span>0%</span><span>Click probability</span><span>100%</span></div>`;
    return header(`How much should we trust ${k} clicks from ${n} impressions?`, 'The observed rate stays the same. More observations give it more weight.') +
      primary(s.item < 2 ? 'See 10x more observations' : 'Return to 3 impressions') +
      mechanism(flow([['Prior mode', s.prior === 'flat' ? 'No preferred rate' : '20.0%'], [`Observed: ${k}/${n}`, pct(result.mle)], ['MAP estimate', pct(result.map)]]) +
        `<div class="tl-mix-labels"><span>Prior weight ${pct(1 - result.dataWeight)}</span><span>Data weight ${pct(result.dataWeight)}</span></div><div class="tl-evidence-mix" role="img" aria-label="MAP weight: prior ${pct(1 - result.dataWeight)}, observed data ${pct(result.dataWeight)}"><span class="tl-mix-prior" style="width:${100 * (1 - result.dataWeight)}%"></span><span class="tl-blue" style="width:${100 * result.dataWeight}%"></span></div>` +
        equation + changed(previous ? 'MAP before / now' : 'Observed rate / MAP', pct(before), pct(result.map), s.prior === 'flat' ? 'The flat prior contributes zero mode pseudo-counts: no shrinkage.' : `${n} observations compete with ${a + b - 2} effective prior counts. The prior mode remains 20%.`)) +
      settings(s, controls('Prior', [['flat', 'Flat: Beta(1,1)'], ['category', 'Category: Beta(3,9)'], ['strong', 'Stronger: Beta(11,41)']].map(([v, l]) => button('prior', v, l, s.prior === v)).join('')) + controls('Item evidence', items.map(([c, impressions], i) => button('item', i, `${c}/${impressions} clicks`, s.item === i)).join(''))) +
      inspect(s, `<p class="tl-equation">(${k} + ${a - 1}) / (${n} + ${a + b - 2}) = ${fmt(result.map)}</p><p class="ml-note">Posterior: Beta(${result.a}, ${result.b}). Its predictive mean is ${pct(result.posteriorMean)}, not the mode.</p><div class="tl-legend"><span class="tl-text-blue">Likelihood</span><span>Prior (dashed)</span><span class="tl-text-orange">Posterior</span></div>${svg}<p class="ml-note">Each curve is normalized to its own peak, not its area. Heights across curves are not evidence.</p>`);
  }
  function renderBias(s) {
    const result = biasVariance(s.degree, s.probe), weights = samples.map(y => fitPolynomial(y, s.degree));
    const previous = s._previous || {...s, sample:s.sample ^ 1};
    const beforeWeights = fitPolynomial(samples[previous.sample], previous.degree);
    const beforePrediction = predictPolynomial(beforeWeights, previous.probe);
    const influence = predictionInfluence(s.degree, s.probe);
    const changedLabels = sampleX.map((_, i) => samples[s.sample][i] !== samples[previous.sample][i]);
    const sameEstimator = previous.degree === s.degree && previous.probe === s.probe;
    const terms = influence.flatMap((v, i) => [i ? ' + ' : '', changedLabels[i] || !sameEstimator ? {text:`${fmt(v)} * (${fmt(samples[s.sample][i], 1)})`} : `${fmt(v)} * (${fmt(samples[s.sample][i], 1)})`]);
    const reason = sameEstimator ? `${changedLabels.filter(Boolean).length} label(s) changed; the input locations and fitted family stayed fixed.` : 'The estimator or test input changed; compare the displayed coefficients, not only training error.';
    const grid = Array.from({ length: 61 }, (_, i) => -1.2 + i * 2.9 / 60);
    const all = weights.flatMap(w => grid.map(x => predictPolynomial(w, x))).concat(grid.map(x => x * x), grid.map(x => predictPolynomial(beforeWeights, x)));
    const lo = Math.min(-0.6, ...all), hi = Math.max(3, ...all), px = x => 30 + (x + 1.2) / 2.9 * 310, py = y => 180 - (y - lo) / (hi - lo) * 160;
    const path = f => grid.map((x, i) => `${i ? 'L' : 'M'}${px(x)},${py(f(x))}`).join(' ');
    const svg = `<svg class="tl-plot" viewBox="0 0 360 192" role="group" aria-label="Before and current fits. Bright training points changed; tap any training point to flip only that label. Probe x=${s.probe}."><path class="tl-axis" d="M30 15V180H340"/>${weights.map(w => `<path class="tl-line tl-stroke-faint" d="${path(x => predictPolynomial(w, x))}"/>`).join('')}<path class="tl-line tl-stroke-muted" d="${path(x => predictPolynomial(beforeWeights, x))}"/><path class="tl-line tl-stroke-orange" d="${path(x => predictPolynomial(weights[s.sample], x))}"/><path class="tl-line tl-stroke-blue" stroke-dasharray="5 4" d="${path(x => x * x)}"/><path class="tl-axis" stroke-dasharray="3 4" d="M${px(s.probe)} 18V180"/>${sampleX.map((x, i) => `<g role="button" tabindex="0" data-action="flip-label" data-value="${i}" aria-label="Flip label at x=${x}; current y=${samples[s.sample][i]}"><circle cx="${px(x)}" cy="${py(samples[s.sample][i])}" r="18" fill="transparent"/>${changedLabels[i] ? `<path class="tl-stroke-muted" d="M${px(x)} ${py(samples[previous.sample][i])}V${py(samples[s.sample][i])}"/><circle class="tl-before-point" cx="${px(x)}" cy="${py(samples[previous.sample][i])}" r="5"/>` : ''}<circle class="tl-fill-orange${changedLabels[i] ? '' : ' tl-dim-point'}" cx="${px(x)}" cy="${py(samples[s.sample][i])}" r="5"/></g>`).join('')}<circle class="tl-fill-orange" cx="${px(s.probe)}" cy="${py(result.predictions[s.sample])}" r="5"/></svg><div class="tl-axis-labels"><span>x: -1.2</span><span>Probe: ${s.probe}</span><span>1.7</span></div><p class="tl-axis-note">Vertical range: ${fmt(lo, 1)} to ${fmt(hi, 1)}. Tap a training dot to flip its label.</p>`;
    const trainMSE = mse(samples[s.sample], sampleX.map(x => predictPolynomial(weights[s.sample], x)));
    return header('Would one changed label move this prediction?', 'Truth: y=x^2. Same three inputs; each label has -0.5 or +0.5 noise.') +
      primary('Change one label and refit', `Dataset ${s.sample + 1} of 8`) +
      mechanism(`<div class="tl-legend"><span class="tl-text-blue">Truth (dashed)</span><span class="tl-text-orange">Current fit</span><span>Before fit (gray dashed)</span></div><div class="tl-fit-layout"><div>${svg}</div><div class="tl-stats">${stat(`Prediction at x=${s.probe}`, fmt(result.predictions[s.sample]))}${stat('True response', fmt(result.truth))}${stat('This fit: training MSE', fmt(trainMSE))}</div></div>` + linked(`Prediction at ${s.probe}: label influence times label`, [...terms, ` = ${fmt(result.predictions[s.sample])}`]) + changed(s._previous ? 'Before / after refit' : 'One-label counterfactual / current fit', fmt(beforePrediction), fmt(result.predictions[s.sample]), reason)) +
      settings(s, controls('Fitted family', ['Constant', 'Linear', 'Quadratic'].map((v, i) => button('degree', i, v, s.degree === i)).join('')) + controls('Probe input', [0, 0.5, 1.5].map(x => button('probe', x, `x=${x}${x === 1.5 ? ' (extrapolation)' : ''}`, s.probe === x)).join(''))) +
      inspect(s, `<div class="tl-stats">${stat('Bias squared', fmt(result.bias2))}${stat('Variance across 8 fits', fmt(result.variance))}${stat('Independent label noise', fmt(result.noise))}${stat('Total test risk', fmt(result.risk))}</div><p class="ml-note">Risk = bias squared + variance + independent noise. Displayed values are rounded. Mean prediction=${fmt(result.average)}.</p>` + table('Select a dataset to inspect its fit (all are equally likely)', ['Dataset', 'y at -1', 'y at 0', 'y at 1', `Prediction at ${s.probe}`], samples.map((ys, i) => [button('sample', i, `D${i + 1}`, s.sample === i), ...ys.map(v => escape(fmt(v, 1))), escape(fmt(result.predictions[i]))])) + `<p class="ml-note">D${s.sample + 1} coefficients [${weights[s.sample].map(w => fmt(w)).join(', ')}] in increasing powers of x. Variance uses all eight predictions, not points within one dataset.</p>`);
  }
  function renderRegularization(s) {
    const result = regularizedFit(s.penalty, s.lambda), row = design[s.row];
    const names = ['Semantic match', 'Training-only correlate'], feature = s.feature ?? 1;
    const beforeWeights = s._previous ? regularizedFit(s._previous.penalty, s._previous.lambda).weights : [2, .4];
    const coefficient = [2, .4][feature];
    const solved = s.penalty === 'l1' ? `max(${coefficient} - ${s.lambda}, 0)` : `${coefficient} / (1 + ${s.lambda})`;
    return header('Which feature survives the penalty?', `Two standardized features. ${s.penalty.toUpperCase()} penalty, strength ${s.lambda}. Tap a weight to trace it.`) +
      primary(s.penalty === 'l1' ? 'Compare L2 shrinkage' : 'Apply L1 instead') +
      mechanism(`<div class="tl-weight-key"><span>${s._previous ? 'Previous solution' : 'Unpenalized fit'} (gray)</span><span>${s.penalty.toUpperCase()} solution (orange)</span></div>${beforeWeights.map((before, j) => `<button type="button" class="tl-weight-lane${feature === j ? '' : ' tl-muted-data'}" data-action="feature" data-value="${j}" aria-pressed="${feature === j}"><span class="tl-weight-label"><span>x${j + 1}: ${escape(names[j])}</span><strong>${fmt(before)} to ${fmt(result.weights[j])}</strong></span><span class="tl-weight-track" aria-hidden="true"><i class="tl-weight-before" style="width:${before / 2 * 100}%"></i><i class="tl-weight-after" style="width:${result.weights[j] / 2 * 100}%"></i></span></button>`).join('')}` + linked(`Solve the selected weight x${feature + 1}`, [`w${feature + 1} = `, {text:solved}, ` = ${fmt(result.weights[feature])}`]) + linked(`Selected feature's contribution to R${s.row + 1}`, [`${row[feature]} * `, {text:fmt(result.weights[feature])}, ` = ${fmt(row[feature] * result.weights[feature])}; total score = ${fmt(result.predictions[s.row])}`]) + changed(`x${feature + 1} weight`, fmt(beforeWeights[feature]), fmt(result.weights[feature]), s.penalty === 'l1' ? `The ${s.lambda} threshold is subtracted from OLS magnitude ${coefficient}; anything below zero is clipped to zero.` : `Division by ${1 + s.lambda} shrinks this weight without a hard cutoff.`)) +
      settings(s, controls('Exact penalty values', [0, 0.25, 0.5, 1].map(l => button('lambda', l, `lambda=${l}`, s.lambda === l)).join('')) + controls('Penalty', ['l1', 'l2'].map(p => button('penalty', p, p.toUpperCase(), s.penalty === p)).join('')) + rowsControls('row', 4, s.row)) +
      inspect(s, `<p class="ml-note">Centered targets y=2*x1+0.4*x2; orthogonal columns have mean square one.</p><p class="tl-equation">${row[0]} * ${fmt(result.weights[0])} + ${row[1]} * ${fmt(result.weights[1])} = ${fmt(result.predictions[s.row])}</p><div class="tl-stats">${stat('Train MSE', fmt(result.trainMSE))}${stat('Constructed holdout MSE', fmt(result.holdoutMSE))}</div>` + table('Complete design and target sets', ['Row', 'x1', 'x2', 'Train y', 'Holdout y'], design.map((x, i) => [`R${i + 1}`, ...x.map(escape), escape(fmt(regularizationY[i], 1)), escape(2 * x[0])])) + '<p class="ml-note">The constructed holdout reuses these feature combinations but sets y=2*x1. This is not an independent estimate of deployment quality.</p>');
  }
  function renderTree(s) {
    const thresholds = s.mode === 'hist' ? [2.5, 4.5, 6.5] : [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];
    const r = splitStats(treeRows, s.threshold), selected = treeRows[s.row];
    const beforeThreshold = s._previous?.threshold ?? (s.threshold === 4.5 ? 2.5 : 4.5);
    const before = splitStats(treeRows, beforeThreshold);
    const moved = treeRows.filter(row => (row.x <= beforeThreshold) !== (row.x <= s.threshold));
    const leftTerm = `(${r.left.length}/8) * ${fmt(gini(r.left), 4)}`, rightTerm = `(${r.right.length}/8) * ${fmt(gini(r.right), 4)}`;
    const best = thresholds.reduce((a, t) => splitStats(treeRows, t).gain > splitStats(treeRows, a).gain ? t : a);
    const child = (label, rows) => `<section class="tl-tree-leaf"><h4>${escape(label)}</h4><strong>${sum(rows.map(row => row.y))}/${rows.length} clicks</strong><span class="tl-leaf-gini">Gini ${fmt(gini(rows), 4)}</span><div class="tl-leaf-rows">${rows.map(row => tile('row', treeRows.indexOf(row), `${row.id}: x=${row.x}`, `${row.y}${moved.includes(row) ? ' moved' : ''}`, selected === row, `${row.y ? 'tl-positive' : ''} ${moved.includes(row) ? 'tl-moved-row' : selected === row ? '' : 'tl-muted-data'}`)).join('')}</div></section>`;
    return header('Which cut makes the click labels less mixed?', 'Eight fixed impressions. Move the boundary, not the data. Click a row to follow its route.') +
      primary(s.threshold === 4.5 ? 'Try a weaker cut' : 'Apply the best cut') +
      mechanism(`<div class="tl-legend"><span>x: prior interactions</span><span>1=click; 0=no click</span><span>Bright rows crossed the cut</span></div><div class="tl-tree-root"><span>Cut ${beforeThreshold} to ${s.threshold}; labels unchanged</span><strong>x <= ${s.threshold}?</strong></div><svg class="tl-tree-fork" viewBox="0 0 300 34" preserveAspectRatio="none" aria-hidden="true"><path class="tl-axis" d="M150 0V12H75V34M150 12H225V34"/></svg><div class="tl-tree-children">${child('Yes: left', r.left)}${child('No: right', r.right)}</div>` + linked('Weighted child impurity; selected row\'s child highlighted', [selected.x <= s.threshold ? {text:leftTerm} : leftTerm, ' + ', selected.x > s.threshold ? {text:rightTerm} : rightTerm, ` = ${fmt(r.weighted, 4)}; gain = ${fmt(r.parent, 4)} - ${fmt(r.weighted, 4)} = ${fmt(r.gain, 4)}`]) + changed('Gini gain', fmt(before.gain, 4), fmt(r.gain, 4), `${moved.length ? moved.map(row => row.id).join(' and ') + ' crossed the cut.' : 'No rows crossed this cut.'} ${selected.id} now routes ${selected.x <= s.threshold ? 'left' : 'right'} because ${selected.x} ${selected.x <= s.threshold ? '<=' : '>'} ${s.threshold}.`)) +
      settings(s, controls('Candidate search', button('mode', 'exact', 'All observed gaps', s.mode === 'exact') + button('mode', 'hist', 'Four histogram bins', s.mode === 'hist')) + controls('Choose threshold', thresholds.map(t => button('threshold', t, `x <= ${t}`, s.threshold === t)).join(''))) +
      inspect(s, `<p class="ml-note">Left contribution: ${r.left.length}/8 * ${fmt(gini(r.left))}; right: ${r.right.length}/8 * ${fmt(gini(r.right))}. Best allowed cut=${best}.</p>` + table('Histogram sufficient statistics', ['Bin', 'Rows', 'No click', 'Click'], [0, 1, 2, 3].map(i => { const bin = treeRows.slice(i * 2, i * 2 + 2); const yes = sum(bin.map(row => row.y)); return [`[${1 + 2 * i}, ${2 + 2 * i}]`, escape(bin.map(row => row.id).join(', ')), escape(2 - yes), escape(yes)]; })) + table('Search all allowed cuts', ['Threshold', 'Left size', 'Right size', 'Gain'], thresholds.map(t => { const a = splitStats(treeRows, t); return [escape(t), escape(a.left.length), escape(a.right.length), escape(fmt(a.gain))]; })) + '<p class="ml-note">Histogram mode removes within-bin cuts. It happens to preserve the best split for these rows.</p>');
  }
  function renderBoost(s) {
    const trace = boostTrace(4, s.rate), r = trace[s.step], i = s.row;
    const correction = s.step ? s.rate * r.stump.predictions[i] : 0;
    const inLeaf = j => !s.step || (boostRows[j].x <= r.stump.threshold) === (boostRows[i].x <= r.stump.threshold);
    const leafIndices = boostRows.map((_, j) => j).filter(inLeaf);
    const meanParts = s.step ? ['(', ...leafIndices.flatMap((j, n) => [n ? ' + ' : '', j === i ? {text:fmt(r.residuals[j])} : fmt(r.residuals[j])]), `) / ${leafIndices.length} = ${fmt(r.stump.predictions[i])}`] : [{text:'mean([0,0,1,1,3,3,4,4])'}, ' = 2'];
    return header('What does the next tree change?', 'Each small tree fits the errors left by the current ensemble. Tap an example to follow its update.') +
      primary(s.step < 4 ? 'Fit the next residual tree' : 'Restart the trace', `Round ${s.step} of 4`) +
      mechanism(`<div class="tl-legend"><span class="tl-text-blue">Relevance target</span><span class="tl-text-orange">Current prediction</span><span>Before: gray tick; height 0 to 4</span></div><div class="tl-example-bars">${boostRows.map((row, j) => `<button type="button" class="${inLeaf(j) ? '' : 'tl-muted-data'}" data-action="row" data-value="${j}" aria-pressed="${s.row === j}" aria-label="${row.id}: target ${row.y}, prediction ${fmt(r.predictions[j])}${inLeaf(j) ? ', in selected leaf' : ''}"><span class="tl-bar-pair"><i class="tl-blue" style="height:${row.y / 4 * 100}%"></i><i class="tl-orange" style="height:${r.predictions[j] / 4 * 100}%"></i><i class="tl-old-prediction" style="bottom:${(s.step ? r.before[j] : r.predictions[j]) / 4 * 100}%"></i></span><span>${row.id}</span></button>`).join('')}</div>` + linked(s.step ? `Selected leaf: mean residual (R${i + 1} highlighted)` : 'Initial prediction', meanParts) + linked(`R${i + 1}: add only the scaled tree output`, [`${fmt(s.step ? r.before[i] : r.predictions[i])} + `, {text:`${s.rate} * (${fmt(s.step ? r.stump.predictions[i] : 0)})`}, ` = ${fmt(r.predictions[i])}`]) + changed(`R${i + 1} prediction`, fmt(s.step ? r.before[i] : r.predictions[i]), fmt(r.predictions[i]), s.step ? `Only the ${leafIndices.length} bright rows determine this leaf's correction. Training MSE is ${fmt(r.mse)}; held-out quality is not measured.` : 'Start from the mean target before fitting any residual tree.')) +
      settings(s, controls('Learning rate', [.25, .5, 1].map(v => button('rate', v, `eta=${v}`, s.rate === v)).join('')) + controls('Trace navigation', button('previous', '', 'Previous tree', undefined, s.step === 0))) +
      inspect(s, `<p class="ml-note">Regression fixture: x is an engagement feature; y is a relevance target with mean 2. ${s.step ? `Best split x<=${r.stump.threshold}; leaf means ${fmt(r.stump.leftMean)} and ${fmt(r.stump.rightMean)}. Residual-fit SSE=${fmt(r.stump.sse)}.` : 'F0=mean(y)=2.'}</p>` + table('Select a row and follow its residual', ['Row / x', 'y', 'Before', 'Residual', 'Tree h', 'After'], boostRows.map((row, j) => [button('row', j, `${row.id} / ${row.x}`, s.row === j), escape(row.y), escape(fmt(s.step ? r.before[j] : r.predictions[j])), escape(fmt(row.y - (s.step ? r.before[j] : r.predictions[j]))), s.step ? escape(fmt(r.stump.predictions[j])) : 'Not fit', escape(fmt(r.predictions[j]))])) + `<p class="ml-note">Training MSE path: ${trace.slice(0, s.step + 1).map(t => fmt(t.mse)).join(' -> ')}. Validation performance is not measured.</p>`);
  }
  function renderLeakage(s) {
    const result = leakageAudit(s.field, s.split), row = result.rows[s.row];
    const labels = { history: 'Historical engagement', late: 'Late-arriving event', outcome: 'Post-outcome status' };
    const events = [['Event occurs', row.field.event], ['Value readable', row.field.available]];
    const stamps = before => events.filter(([, t]) => (t <= row.prediction) === before).map(([label, t]) => `<div class="tl-time-event${s.replay && t > row.prediction ? ' tl-blocked-source' : ''}"><span>${escape(label)}</span><strong>${t} min</strong>${s.replay && t > row.prediction ? '<span>Blocked by time gate</span>' : ''}</div>`).join('') || '<span class="ml-note">No event here</span>';
    const activePrediction = s.replay ? row.replay : row.offline;
    return header('Could the model know this value when it scored?', `${row.id}: ${labels[s.field]}. The rule predicts this binary field, using 0 if it is unavailable.`) +
      primary(s.replay ? 'Show the hindsight snapshot' : 'Replay at prediction time', s.replay ? 'Point-in-time replay' : 'Hindsight evaluation') +
      mechanism(`<div class="tl-time-rail"><section><h4>Before scoring</h4>${stamps(true)}</section><section class="tl-time-boundary"><h4>Prediction</h4><strong>${row.prediction} min</strong></section><section><h4>After scoring</h4>${stamps(false)}</section></div>` +
        linked(s.replay ? 'Time gate is enforced' : 'Time gate is currently bypassed', [`usable = (${row.field.event} <= ${row.prediction}) AND (`, {text:`${row.field.available} <= ${row.prediction}`}, `) = ${row.available}`]) +
        flow([['Snapshot value', row.field.value], ['Replay eligibility', row.available ? 'Available' : 'Too late'], [s.replay ? 'Replay prediction' : 'Hindsight prediction', activePrediction]], 1) +
        changed('Snapshot prediction / active prediction', row.offline, activePrediction, s.replay ? row.available ? 'The value passes both timestamp checks; the time gate keeps it.' : 'The gate rejects the unreadable value and substitutes the explicit fallback 0. The request split did not change.' : 'The unchecked snapshot predicts its field value even when the availability comparison fails.') +
        bar('Held-out accuracy', s.replay ? result.replayAccuracy : result.offlineAccuracy, 'orange')) +
      settings(s, controls('Feature to audit', Object.entries(labels).map(([v, l]) => button('field', v, l, s.field === v)).join('')) + controls('Row split', button('split', 'time', 'Chronological: last two held out', s.split === 'time') + button('split', 'alternating', 'Alternating rows held out', s.split === 'alternating'))) +
      inspect(s, table('Select a request to audit its timestamps', ['Request', 'Split', 'Predict t', 'Event / readable t', 'Value / label', 'Eligible?'], result.rows.map((r, i) => [button('row', i, r.id, s.row === i), r.test ? 'Held out' : 'Train', escape(r.prediction), `${r.field.event} / ${r.field.available}`, `${r.field.value} / ${r.y}`, r.available ? 'Yes' : 'No'])) + `<p class="ml-note">${result.unavailable}/6 fields are unavailable at scoring. Both event and availability time must pass. A chronological request split does not repair the feature join.</p>`);
  }
  function renderFeatureShift(s) {
    const raw = featureBatches[s.batch], reference = featureSummary(featureBatches.reference), live = featureSummary(raw, s.policy);
    const i = s.row, x = raw[i];
    const flag = Number(x === null && s.policy === 'explicit');
    const otherPolicy = s.policy === 'explicit' ? 'zero' : 'explicit';
    const other = featureSummary(raw, otherPolicy);
    return header('Did missing data recover, or did we hide it?', 'Eight recent-interaction counts pass through preprocessing. Tap one to trace its score.') +
      primary(s.policy === 'explicit' ? 'Replace nulls with zero' : 'Restore the missing flag') +
      mechanism(`<div class="tl-mix-labels"><span>Raw input to model input</span><span>* = missing flag retained</span></div><div class="tl-input-grid">${raw.map((v, j) => `<button type="button" class="tl-input-node${v === null ? ' tl-missing-node' : i === j ? '' : ' tl-muted-data'}" data-action="row" data-value="${j}" aria-pressed="${i === j}" aria-label="Slot ${j + 1}: raw ${v ?? 'null'}, model value ${v ?? 0}, missing flag ${Number(v === null && s.policy === 'explicit')}"><span>Slot ${j + 1}</span><strong>${v ?? 'null'}</strong><span class="tl-transform-arrow" aria-hidden="true">to</span><strong>${v ?? 0}${v === null && s.policy === 'explicit' ? '*' : ''}</strong></button>`).join('')}</div>` + linked(`Slot ${i + 1}: only the missing-flag term changes`, [`p = sigmoid(-2 + 0.4*${x ?? 0} - `, {text:`0.8*${flag}`}, `) = ${fmt(live.predictions[i], 4)}`]) + changed(s.policy === 'zero' ? 'With flag / without flag' : 'Without flag / with flag', pct(other.predictions[i]), pct(live.predictions[i]), x === null ? 'The raw null and filled value 0 are identical in both cases. Only the indicator changes; prediction quality is unknown without labels.' : 'This is a real observed number. Erasing a missing flag has no effect on this row.') + `<div class="tl-stats">${stat('Raw missing rate', pct(live.missingRate))}${stat('After preprocessing', pct(live.visibleMissingRate))}</div>`) +
      settings(s, controls('Live snapshot', [['reference', 'Unchanged'], ['missing', 'Missing-value spike'], ['range', 'Larger counts']].map(([v, l]) => button('batch', v, l, s.batch === v)).join('')) + controls('Live preprocessing', button('policy', 'explicit', 'Preserve missing indicator', s.policy === 'explicit') + button('policy', 'zero', 'Silently coerce null to zero', s.policy === 'zero'))) +
      inspect(s, `<p class="tl-equation">Slot ${i + 1}: sigmoid(-2 + 0.4*${x ?? 0} - 0.8*${Number(x === null && s.policy === 'explicit')}) = ${pct(live.predictions[i])}</p><p class="ml-note">Observed mean: ${fmt(reference.observedMean, 2)} reference, ${fmt(live.observedMean, 2)} live. Mean score: ${pct(reference.averageScore)} reference, ${pct(live.averageScore)} live. A score shift is not an accuracy measurement.</p>` + table('Transformed buckets', ['Bucket', 'Reference', 'Live'], ['Missing', 'Real/coerced zero', '1 to 3', '4 or more'].map((label, j) => [escape(label), escape(reference.counts[j]), escape(live.counts[j])])) + table('Raw snapshots (not paired users)', ['Slot', 'Reference raw', 'Live raw', 'Model value', 'Live score'], raw.map((v, j) => [button('row', j, `Slot ${j + 1}`, i === j), escape(featureBatches.reference[j] ?? 'null'), escape(v ?? 'null'), escape(live.transformed[j] ?? '0 + missing flag'), escape(pct(live.predictions[j]))])));
  }
  function renderSkew(s) {
    const row = servingRows[s.row], p = servingPair(row, s.mode), pairs = servingRows.map(r => servingPair(r, s.mode));
    const onlineCode = { raw: 'x = fill_null(count, 2)', stale: 'x = log1p(fill_null(stale_count, 2))', null: 'x = log1p(fill_null(count, 0))', fixed: 'x = log1p(fill_null(count, 2))' };
    const before = s._previous ? servingPair(row, s._previous.mode).onlineScore : p.offlineScore;
    const repair = s.mode === 'stale' ? 'Refresh the feature count' : s.mode === 'null' ? 'Use the same null default' : 'Apply the shared feature code';
    return header('Why does the same request get a different score?', `${row.id}: one interaction count, two feature paths, the same fixed model.`) +
      primary(s.mode === 'fixed' ? 'Reintroduce the raw-count bug' : repair) +
      mechanism(`<div class="tl-pipeline"><h4>Offline: log1p, null defaults to 2</h4>${flow([['Raw count', row.count ?? 'null'], ['Model input', fmt(p.offline)], ['Predicted click rate', pct(p.offlineScore)]], 1)}<h4>Online: ${escape({ raw: 'raw count; no log1p', stale: 'log1p of stale count', null: 'log1p; null defaults to 0', fixed: 'same code and fresh count' }[s.mode])}</h4>${flow([['Source count', (s.mode === 'stale' ? row.stale : row.count) ?? 'null'], ['Model input', fmt(p.online)], ['Predicted click rate', pct(p.onlineScore)]], 1)}</div>` + linked(`${row.id}: the model weights stay fixed`, ['p = sigmoid(-1.5 + 1.2*', {text:fmt(p.online, 4)}, `) = ${fmt(p.onlineScore, 4)}`]) + changed(s._previous ? 'Online score before / now' : 'Offline score / online score', pct(before), pct(p.onlineScore), `${pairs.filter(v => Math.abs(v.delta) > 1e-9).length}/5 requests fail parity. Only the feature path changed, not the model artifact.`) + `<div class="tl-row-chips" role="group" aria-label="Inspect a paired request">${servingRows.map((r, i) => tile('row', i, r.id, r.count ?? 'null', s.row === i, s.row === i ? '' : 'tl-muted-data')).join('')}</div>`) +
      settings(s, controls('Serving path', [['raw', 'Missing log1p'], ['stale', 'Stale counter'], ['null', 'Different null default'], ['fixed', 'Shared transform + fresh data']].map(([v, l]) => button('mode', v, l, s.mode === v)).join(''))) +
      inspect(s, `<pre class="tl-code">offline: x = log1p(fill_null(count, 2))\nonline:  ${escape(onlineCode[s.mode])}</pre><p class="ml-note">The model scores sigmoid(-1.5+1.2*x). ${row.id}: offline logit=${fmt(-1.5 + 1.2 * p.offline)}, online logit=${fmt(-1.5 + 1.2 * p.online)}. Parity does not guarantee model quality.</p>` + table('Paired feature parity', ['Request', 'Count / stale', 'Offline x', 'Online x', 'Delta x'], servingRows.map((r, i) => [button('row', i, r.id, s.row === i), `${r.count ?? 'null'} / ${r.stale ?? 'null'}`, escape(fmt(pairs[i].offline)), escape(fmt(pairs[i].online)), escape(fmt(pairs[i].delta))])));
  }
  function renderCohorts(s) {
    const selected = cohortScore(mixes[s.mix]), offline = cohortScore(mixes.offline), c = cohorts[s.cohort], weight = mixes[s.mix][s.cohort];
    const previous = cohortScore(mixes[s._previous?.mix || 'offline']);
    const terms = cohorts.flatMap((row, i) => [i ? ' + ' : '', i === s.cohort ? {text:`${mixes[s.mix][i]} * (${row.candidate}/${row.n} - ${row.baseline}/${row.n})`} : `${mixes[s.mix][i]} * (${row.candidate}/${row.n} - ${row.baseline}/${row.n})`]);
    return header('Can a different audience reverse the winner?', 'Keep observed cohort rates fixed. Change only who counts more in the evaluation.') +
      primary(s.mix === 'live' ? 'Return to the offline audience' : 'Apply the live audience mix', s.mix === 'live' ? 'Projection, not a measured launch' : 'Offline evaluation mix') +
      mechanism(`<div class="tl-legend"><span class="tl-text-blue">Returning audience</span><span class="tl-text-orange">New audience</span><span>pp = percentage points</span></div><div class="tl-cohort-nodes">${cohorts.map((row, i) => `<button type="button" class="tl-cohort-node${s.cohort === i ? '' : ' tl-muted-data'}" data-action="cohort" data-value="${i}" aria-pressed="${s.cohort === i}"><span>${row.name} users</span><strong>${pct(mixes[s.mix][i])} of audience</strong><span>Observed difference: ${(row.candidate - row.baseline) > 0 ? '+' : ''}${fmt((row.candidate - row.baseline) / row.n * 100, 0)} pp</span></button>`).join('')}</div><div class="tl-evidence-mix" role="img" aria-label="Audience weights: returning ${pct(mixes[s.mix][0])}, new ${pct(mixes[s.mix][1])}"><span class="tl-blue" style="width:${mixes[s.mix][0] * 100}%"></span><span class="tl-orange" style="width:${mixes[s.mix][1] * 100}%"></span></div>` + linked('Weighted candidate minus baseline; selected cohort highlighted', [...terms, ` = ${fmt(selected.delta, 3)} (${fmt(selected.delta * 100, 1)} pp)`]) + changed(s._previous ? 'Previous mix / current mix' : 'Offline reference / current mix', `${fmt(previous.delta * 100, 1)} pp`, `${fmt(selected.delta * 100, 1)} pp`, `Observed success: ${c.name.toLowerCase()} users contribute ${fmt(selected.contributions[s.cohort] * 100, 1)} pp at weight ${pct(weight)}. These are not causal effects.`)) +
      settings(s, controls('Common evaluation population', [['offline', 'Offline: 90/10'], ['live', 'Live: 20/80'], ['balanced', 'Balanced: 50/50']].map(([v, l]) => button('mix', v, l, s.mix === v)).join(''))) +
      inspect(s, table('Cohort counts and weighted differences', ['Cohort', 'Baseline successes', 'Candidate successes', 'Weight', 'Contribution'], cohorts.map((row, i) => [button('cohort', i, row.name, s.cohort === i), `${row.baseline}/${row.n}`, `${row.candidate}/${row.n}`, escape(pct(mixes[s.mix][i])), `${fmt(selected.contributions[i] * 100, 1)} pp`])) + `<p class="tl-equation">${c.name}: ${fmt(weight, 2)} * (${c.candidate}/${c.n} - ${c.baseline}/${c.n}) = ${fmt(selected.contributions[s.cohort] * 100, 1)} pp</p><p class="ml-note">Offline reference: +${fmt(offline.delta * 100, 1)} pp. Live-mix projection: ${fmt(cohortScore(mixes.live).delta * 100, 1)} pp. Reweighting assumes stable within-cohort rates; it does not remove selection or exposure bias.</p>`);
  }
  function renderDrift(s) {
    const batch = driftBatches[s.scenario], ref = driftStats(driftBatches.reference), r = driftStats(batch), i = s.bucket;
    const explanations = { stable: 'Joint counts match the reference in this fixture.', covariate: 'The input mix changes, but within-bucket positive rates stay at 20% and 80%. Accuracy stays 80%.', concept: 'Input mix and overall label rate are unchanged. Conditional rates reverse: accuracy falls from 80% to 20%.', label: 'Positive prevalence rises from 50% to 75%, while P(x|y) stays fixed: low-bucket share among positives is 20%, among negatives 80%.' };
    const correct = s.labels ? [String(batch[0][0]), String(batch[1][1])] : ['?', '?'];
    return header('Do these inputs tell us whether the model still works?', 'The model still scores one bucket at 20% and the other at 80%. The outcome labels arrive later.') +
      primary(s.labels ? 'Hide the labels again' : 'Reveal the delayed labels') +
      mechanism(`<div class="tl-legend"><span>1 = click; 0 = no click</span><span>? = label pending</span></div><div class="tl-bucket-grid">${batch.map((counts, j) => `<button type="button" class="tl-bucket-node${i === j ? '' : ' tl-muted-data'}" data-action="bucket" data-value="${j}" aria-pressed="${i === j}"><span>${j ? 'High-score' : 'Low-score'} bucket: ${r.counts[j]} rows</span><strong>Predict ${j ? 'click' : 'no click'}</strong><span class="tl-label-dots" aria-label="${s.labels ? `${counts[j]} correct predictions out of ${r.counts[j]}` : 'Outcome labels pending'}">${Array.from({ length: r.counts[j] }, (_, n) => `<span class="tl-label-dot${s.labels && n >= counts[0] ? ' tl-click-dot' : ''}${s.labels && Number(n >= counts[0]) === j ? ' tl-correct-dot' : ''}">${s.labels ? Number(n >= counts[0]) : '?'}</span>`).join('')}</span><span>${s.labels ? `${counts[j]}/${r.counts[j]} predicted correctly` : 'Correct count unknown'}</span></button>`).join('')}</div>` + linked('Accuracy: correct low + correct high, over all requests', ['(', i === 0 ? {text:correct[0]} : correct[0], ' + ', i === 1 ? {text:correct[1]} : correct[1], `) / ${r.total} = ${s.labels ? fmt(r.accuracy) : 'unknown'}`]) + `<div class="tl-stats">${stat('Input shift (total variation)', pct(r.tv))}${stat('Accuracy: reference / live', s.labels ? `${pct(ref.accuracy)} / ${pct(r.accuracy)}` : '80.0% / unknown')}</div>` + changed('Live accuracy before / after label availability', 'unknown', s.labels ? pct(r.accuracy) : 'unknown', s.labels ? 'The outlined marks are correct predictions. Labels reveal performance; they do not change the inputs or model.' : 'Both correct-count terms are unknown until outcomes arrive. Input histograms cannot supply them.')) +
      (s.labels ? note(explanations[s.scenario]) : '') +
      settings(s, controls('Live batch', [['stable', 'Unchanged'], ['covariate', 'Input-mix shift'], ['concept', 'Relationship reversal'], ['label', 'Label-prior shift']].map(([v, l]) => button('scenario', v, l, s.scenario === v)).join(''))) +
      inspect(s, `<div class="tl-stats">${stat('Positive rate: reference / live', s.labels ? `${pct(ref.labelRate)} / ${pct(r.labelRate)}` : '50.0% / unknown')}${stat('Brier: reference / live', s.labels ? `${fmt(ref.brier)} / ${fmt(r.brier)}` : '0.160 / unknown')}</div>` + table('Joint counts after labels arrive', ['Bucket', 'Reference y=0 / y=1', 'Live n', 'Live y=0 / y=1', 'Live P(y=1|x)'], batch.map((counts, j) => [button('bucket', j, j ? 'High score' : 'Low score', i === j), driftBatches.reference[j].join(' / '), escape(r.counts[j]), s.labels ? counts.join(' / ') : 'Pending', s.labels ? escape(pct(r.conditional[j])) : 'Unknown'])) + `<p class="ml-note">${i ? 'High' : 'Low'} bucket contains ${r.counts[i]}/${r.total} live requests. Decision threshold=0.5. ${s.labels ? `${batch[i][1]}/${r.counts[i]} have label 1.` : 'Accuracy, prevalence, and conditional drift require matured labels.'}</p>`);
  }
  const renderers = { 'mle-map': renderMap, 'bias-variance': renderBias, regularization: renderRegularization, 'tree-split': renderTree,
    boosting: renderBoost, 'feature-leakage': renderLeakage, 'feature-shift': renderFeatureShift, 'serving-skew': renderSkew, 'online-offline': renderCohorts, 'data-drift': renderDrift };
  const render = (kind, state) => `<div class="tl-lab">${renderers[kind](state)}<footer class="tl-footer"><span class="ml-note">Fixed teaching data. Inspect the arithmetic; do not infer deployment performance.</span>${button('reset', '', 'Reset lesson')}</footer></div>`;
  const api = { betaEstimate, relativeBeta, samples, sampleX, sampleOrder, predictionInfluence, baseState, fitPolynomial, predictPolynomial, biasVariance, regularizedFit, design, regularizationY,
    treeRows, gini, splitStats, boostRows, fitStump, boostTrace, leakageRows, leakageAudit, featureBatches, featureSummary,
    servingRows, servingPair, cohorts, mixes, cohortScore, driftBatches, driftStats, content, initial, reduce, render, escape };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window !== 'undefined') {
    window.AtelierLab.createModule({ id: 'tabular-labs', content, initial, render, reduce });
  }
})();
