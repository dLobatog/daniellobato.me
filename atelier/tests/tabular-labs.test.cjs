const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const lab = require('../tabular-labs.js');
const close = (a, b, tol = 1e-10) => assert.ok(Math.abs(a - b) < tol, `${a} != ${b}`);

test('Beta-binomial MAP is not the posterior mean; evidence dominates prior', () => {
  const a = lab.betaEstimate(2, 3);
  close(a.mle, 2/3); close(a.map, 4/13); close(a.posteriorMean, 5/15);
  close(lab.betaEstimate(2, 3, 1, 1).map, 2/3);
  assert.ok(lab.betaEstimate(200, 300).map > lab.betaEstimate(20, 30).map);
  close(lab.betaEstimate(0, 3).map, 2/13);
  close(lab.betaEstimate(3, 3).map, 5/13);
  close(lab.relativeBeta(.2, 3, 9), 1);
  close(lab.relativeBeta(.8, 1, 1), 1);
  assert.throws(() => lab.betaEstimate(4, 3), RangeError);
});

test('All eight independent noise patterns are unique and centered at the truth', () => {
  assert.equal(new Set(lab.samples.map(x => x.join(','))).size, 8);
  lab.sampleX.forEach((x, i) => close(lab.samples.reduce((a, ys) => a + ys[i], 0)/8, x*x));
});

test('Bias variance decomposition matches exhaustive test-label loss for all probes and families', () => {
  for (const degree of [0, 1, 2]) for (const x of [0, .5, 1.5]) {
    const r = lab.biasVariance(degree, x);
    close(r.risk, r.bias2 + r.variance + .25);
    if (degree === 2) close(r.bias2, 0);
  }
  close(lab.biasVariance(0, .5).variance, 1/12);
  close(lab.biasVariance(2, .5).variance, .1796875);
  for (const y of lab.samples) {
    const w = lab.fitPolynomial(y, 2);
    lab.sampleX.forEach((x, i) => close(lab.predictPolynomial(w, x), y[i]));
  }
});

test('L1 threshold and L2 shrinkage solve the specified orthogonal objectives', () => {
  const l1 = lab.regularizedFit('l1', .5), l2 = lab.regularizedFit('l2', .5);
  close(l1.weights[0], 1.5); close(l1.weights[1], 0);
  close(l2.weights[0], 4/3); close(l2.weights[1], .4/1.5);
  close(l1.trainMSE, .41); close(l1.holdoutMSE, .25);
  close(lab.regularizedFit('l1', 0).trainMSE, 0);
  for (const penalty of ['l1', 'l2']) for (const lambda of [0, .25, .5, 1]) {
    const fit = lab.regularizedFit(penalty, lambda);
    fit.weights.forEach((w, j) => {
      const gradient = lab.design.reduce((v, x, i) => v + x[j]*(fit.predictions[i]-lab.regularizationY[i]), 0)/4;
      if (penalty === 'l2') close(gradient + lambda*w, 0);
      else if (Math.abs(w) > 1e-12) close(gradient + lambda*Math.sign(w), 0);
      else assert.ok(Math.abs(gradient) <= lambda + 1e-12);
    });
  }
});

test('Gini split derives impurity from actual labels and weights child sizes', () => {
  const r = lab.splitStats(lab.treeRows, 4.5);
  assert.equal(r.left.length, 4); assert.equal(r.right.length, 4);
  close(r.parent, .46875); close(r.weighted, .1875); close(r.gain, .28125);
  for (let t=1.5; t<8; t++) assert.ok(lab.splitStats(lab.treeRows, t).gain <= r.gain + 1e-12);
  close(lab.gini([]), 0); close(lab.gini(r.right), 0);
});

test('Boosting fits residual leaf means and updates the existing ensemble', () => {
  const trace = lab.boostTrace(4, .5);
  close(trace[0].initial, 2); close(trace[0].mse, 2.5);
  close(trace[1].stump.threshold, 4.5);
  close(trace[1].stump.leftMean, -1.5); close(trace[1].stump.rightMean, 1.5);
  close(trace[1].predictions[0], 1.25); close(trace[1].mse, .8125);
  for (const rate of [.25, .5, 1]) {
    const steps = lab.boostTrace(4, rate);
    steps.slice(1).forEach((r, j) => {
      assert.ok(r.mse <= steps[j].mse + 1e-12);
      r.predictions.forEach((p, i) => {
        close(r.residuals[i], lab.boostRows[i].y-r.before[i]);
        close(p, r.before[i] + rate*r.stump.predictions[i]);
      });
    });
  }
});

test('Leakage checks both event and availability time regardless of row split', () => {
  for (const split of ['time', 'alternating']) {
    const future = lab.leakageAudit('outcome', split), late = lab.leakageAudit('late', split);
    close(future.offlineAccuracy, 1); assert.equal(future.unavailable, 6);
    assert.equal(late.unavailable, 6);
    assert.ok(late.rows.every(r => r.field.event < r.prediction && r.field.available > r.prediction));
    assert.equal(lab.leakageAudit('history', split).unavailable, 0);
  }
  close(lab.leakageAudit('outcome', 'time').replayAccuracy, .5);
});

test('Feature missingness remains observable before coercion and affects the actual score', () => {
  const explicit = lab.featureSummary(lab.featureBatches.missing);
  const zero = lab.featureSummary(lab.featureBatches.missing, 'zero');
  close(explicit.missingRate, 5/8); close(zero.missingRate, 5/8);
  close(zero.visibleMissingRate, 0); close(explicit.visibleMissingRate, 5/8);
  assert.deepEqual(explicit.counts, [5, 1, 1, 1]); assert.deepEqual(zero.counts, [0, 6, 1, 1]);
  assert.ok(explicit.predictions[1] < zero.predictions[1]);
  close(lab.featureSummary(lab.featureBatches.reference).observedMean, 3);
});

test('Paired serving transforms expose log, stale, and null bugs without changing rows', () => {
  for (const row of lab.servingRows) close(lab.servingPair(row, 'fixed').delta, 0);
  const p = lab.servingPair(lab.servingRows[3], 'raw');
  close(p.offline, Math.log(10)); close(p.online, 9);
  assert.ok(p.onlineScore > p.offlineScore);
  assert.ok(lab.servingPair(lab.servingRows[3], 'stale').delta < 0);
  lab.servingRows.slice(0,4).forEach(row => close(lab.servingPair(row, 'null').delta, 0));
  close(lab.servingPair(lab.servingRows[4], 'null').delta, -Math.log(3));
});

test('Common cohort reweighting reverses the aggregate comparison', () => {
  const offline = lab.cohortScore(lab.mixes.offline), live = lab.cohortScore(lab.mixes.live);
  close(offline.baseline, .76); close(offline.candidate, .79); close(offline.delta, .03);
  close(live.baseline, .48); close(live.candidate, .37); close(live.delta, -.11);
  close(live.contributions.reduce((a,b) => a+b, 0), live.delta);
});

test('Scene claims match uncentered boosting targets and observational cohort comparisons', () => {
  const average = lab.boostRows.reduce((total, row) => total + row.y, 0)/lab.boostRows.length;
  close(average, 2);
  const boost = lab.render('boosting', lab.initial('boosting'));
  assert.ok(boost.includes(`relevance target with mean ${average}`));
  assert.ok(!boost.includes('centered relevance target'));
  const cohorts = lab.render('online-offline', lab.initial('online-offline'));
  assert.ok(cohorts.includes('Observed success:'));
  assert.ok(cohorts.includes('not causal effects'));
});

test('Concept reversal changes loss with unchanged input and label marginals', () => {
  const ref = lab.driftStats(lab.driftBatches.reference), live = lab.driftStats(lab.driftBatches.concept);
  close(live.tv, 0); close(live.labelRate, ref.labelRate);
  close(ref.accuracy, .8); close(live.accuracy, .2);
  close(ref.brier, .16); close(live.brier, .52);
  const covariate = lab.driftStats(lab.driftBatches.covariate);
  close(covariate.tv, .25); close(covariate.accuracy, .8); close(covariate.brier, .16);
  assert.deepEqual(covariate.conditional, ref.conditional);
});

test('Label-shift fixture preserves P(x|y), not just an arbitrary changed label rate', () => {
  const a = lab.driftBatches.reference, b = lab.driftBatches.label;
  for (const y of [0, 1]) close(a[0][y]/(a[0][y]+a[1][y]), b[0][y]/(b[0][y]+b[1][y]));
  close(lab.driftStats(b).labelRate, .75);
});

test('Ten complete content overrides retain substantive math/code/quiz and remove legacy controls', () => {
  assert.equal(Object.keys(lab.content).length, 10);
  for (const [kind, c] of Object.entries(lab.content)) {
    assert.ok(c.title && c.summary && c.why && c.interview);
    assert.ok(c.math.formula.length && c.math.annotations.length >= 3);
    assert.ok(c.code.snippet.length > 100);
    assert.equal(c.quiz.options.filter(o => o.correct).length, 1);
    assert.deepEqual(c.controls, []); assert.deepEqual(c.presets, []); assert.equal(c.geometry, null);
    const html = lab.render(kind, lab.initial(kind));
    assert.ok(html.includes('data-action="reset"'));
    assert.ok(!/NaN|Infinity|undefined/.test(html), kind);
  }
});

test('Reducers are immutable, constrained, bounded, and resettable', () => {
  for (const kind of Object.keys(lab.content)) {
    const state = lab.initial(kind), old = JSON.stringify(state);
    lab.reduce(kind, state, 'row', '<script>');
    assert.equal(JSON.stringify(state), old);
    assert.deepEqual(lab.reduce(kind, state, 'reset'), lab.initial(kind));
  }
  assert.equal(lab.reduce('boosting', {step:4}, 'next').step, 4);
  assert.equal(lab.reduce('boosting', {step:0}, 'previous').step, 0);
  assert.equal(lab.reduce('tree-split', {mode:'exact',threshold:1.5}, 'mode','hist').threshold, 4.5);
  assert.equal(lab.reduce('data-drift', {scenario:'concept',labels:true}, 'scenario','stable').labels, false);
  assert.equal(lab.escape('<img "a" & \'b\'>'), '&lt;img &quot;a&quot; &amp; &#39;b&#39;&gt;');
});

test('Delayed-label rendering withholds live performance until explicit reveal', () => {
  let state = lab.initial('data-drift');
  const hidden = lab.render('data-drift', state);
  assert.ok(hidden.includes('80.0% / unknown'));
  assert.ok(!hidden.includes('80.0% / 20.0%'));
  state = lab.reduce('data-drift', state, 'labels');
  const revealed = lab.render('data-drift', state);
  assert.ok(revealed.includes('80.0% / 20.0%'));
});

test('Browser registers the agreed module contract; Node requires no window', () => {
  let module;
  const context = {window: {AtelierLab: {createModule: m => {module = m;}}}};
  vm.runInNewContext(fs.readFileSync(require.resolve('../tabular-labs.js'), 'utf8'), context);
  assert.equal(module.id, 'tabular-labs');
  assert.equal(typeof module.render, 'function'); assert.equal(typeof module.reduce, 'function');
  assert.equal(Object.keys(module.content).length, 10);
});

test('Every reachable button state renders finite, escaped HTML and bounded geometry', () => {
  let states = 0;
  for (const kind of Object.keys(lab.content)) {
    const queue = [lab.initial(kind)], visited = new Set();
    while (queue.length) {
      const state = queue.shift(), key = JSON.stringify(state);
      if (visited.has(key)) continue;
      visited.add(key); states++;
      const html = lab.render(kind, state);
      assert.ok(!/NaN|Infinity|undefined/.test(html), `${kind}: ${key}`);
      assert.equal((html.match(/<svg /g) || []).length, (html.match(/<\/svg>/g) || []).length);
      for (const match of html.matchAll(/<button\b([^>]*)>/g)) {
        const attrs = match[1];
        if (/\bdisabled\b/.test(attrs)) continue;
        const action = attrs.match(/data-action="([^"]*)"/)[1];
        const value = attrs.match(/data-value="([^"]*)"/)[1];
        const next = lab.reduce(kind, state, action, value);
        if (!visited.has(JSON.stringify(next))) queue.push(next);
      }
    }
    assert.ok(visited.size > 5, `${kind} must offer substantive reachable states`);
  }
  assert.equal(states, 475);
});

test('Every first view has one question, one primary action, and a visible mechanism, not a spreadsheet', () => {
  const front = html => html.replace(/<details\b[\s\S]*?<\/details>/g, '');
  for (const kind of Object.keys(lab.content)) {
    const state = lab.initial(kind), html = lab.render(kind, state), visible = front(html);
    assert.match(visible, /<h3 class="ml-question">[^<]*\?<\/h3>/, kind);
    assert.equal((visible.match(/class="tl-primary"/g) || []).length, 1, kind);
    assert.ok(visible.includes('class="tl-mechanism"'), kind);
    assert.ok(!visible.includes('<table'), `${kind}: table should be secondary`);
    assert.ok(!visible.includes('class="tl-equation"'), `${kind}: derivation should be secondary`);
    assert.ok(!visible.includes('class="ml-controls"'), `${kind}: alternative settings should be secondary`);
    assert.ok(html.includes('<summary>Inspect the calculation</summary>'), kind);
    assert.ok(!/<details[^>]*\bopen\b/.test(html), `${kind}: disclosure starts closed`);
    const next = lab.reduce(kind, state, 'primary', '');
    assert.notDeepEqual(next, state, `${kind}: primary action must change state`);
    const nextVisible = front(lab.render(kind, next));
    assert.notEqual(nextVisible, visible, `${kind}: consequence must be visible without details`);
    assert.match(nextVisible, /class="tl-primary"[^>]*data-action="primary" data-value=""/, `${kind}: stable focus identity`);
  }
});

test('Primary actions produce the promised numerical changes', () => {
  const after = kind => lab.reduce(kind, lab.initial(kind), 'primary', '');
  assert.equal(after('mle-map').item, 1);
  assert.equal(after('bias-variance').sample, 1);
  assert.equal(after('regularization').penalty, 'l2');
  close(lab.regularizedFit(after('regularization').penalty, .5).weights[1], .4/1.5);
  close(lab.splitStats(lab.treeRows, after('tree-split').threshold).gain, .26041666666666663);
  assert.equal(after('boosting').step, 2);
  assert.equal(after('feature-leakage').replay, true);
  close(lab.leakageAudit('outcome', 'time').replayAccuracy, .5);
  assert.equal(after('feature-shift').policy, 'zero');
  close(lab.featureSummary(lab.featureBatches.missing, 'zero').visibleMissingRate, 0);
  assert.equal(after('serving-skew').mode, 'fixed');
  assert.equal(after('online-offline').mix, 'live');
  close(lab.cohortScore(lab.mixes.live).delta, -.11);
  assert.equal(after('data-drift').labels, true);
});

test('All formulas parse in the existing bundled KaTeX', () => {
  const katex = require('../vendor/katex/katex.js');
  for (const c of Object.values(lab.content)) for (const formula of c.math.formula) {
    assert.doesNotThrow(() => katex.renderToString(formula, {throwOnError:true}));
  }
});

// Opt-in standalone layout test. It launches its own headless process, never a shared browser.
test('Isolated browser: all lessons at 320/390/720/1440, keyboard and cross-root state', {
  skip: !process.env.TABULAR_BROWSER_TEST, timeout: 120000,
}, async () => {
  const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
  const path = require('node:path');
  const dir = path.resolve(__dirname, '..');
  const browser = await chromium.launch({headless:true, ...(process.env.CHROME_PATH ? {executablePath:process.env.CHROME_PATH} : {})});
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const styles = ['lab-core.css', 'tabular-labs.css'].map(name => fs.readFileSync(path.join(dir, name), 'utf8')).join('\n');
    await page.setContent(`<style>:root{--lesson-ink:#202526;--lesson-muted:#566364;--lesson-surface:#fff;--lesson-soft:#f3f5f5;--lesson-border:#ccd3d3;--lesson-blue:#126582;--lesson-orange:#a74921;--sans:Georgia,serif;--mono:monospace}body{margin:0;padding:16px}#first,#second{max-width:1100px;margin:auto;min-width:0}</style><style>${styles}</style><div id="first"></div><div id="second" hidden></div>`);
    await page.addScriptTag({path:path.join(dir, 'lab-core.js')});
    await page.addScriptTag({path:path.join(dir, 'tabular-labs.js')});
    for (const width of [320,390,720,1440]) {
      await page.setViewportSize({width,height:900});
      for (const kind of Object.keys(lab.content)) {
        await page.evaluate(kind => {
          const module = window.AtelierLessonModules.find(m => m.has(kind));
          module.render(kind, document.querySelector('#first'));
          module.render(kind, document.querySelector('#second'));
        }, kind);
        await page.locator('#first [data-action="reset"]').click();
        const layout = await page.evaluate(() => ({width:innerWidth, scroll:document.documentElement.scrollWidth,
          height:document.querySelector('#first').getBoundingClientRect().height}));
        assert.ok(layout.scroll <= width + 1, `${kind} at ${width}: overflow ${layout.scroll}`);
        assert.ok(layout.height > 100, `${kind} is actually laid out`);
        const action = page.locator('#first .tl-primary');
        await action.focus();
        await page.keyboard.press('Enter');
        assert.equal(await page.locator('#first').innerHTML(), await page.locator('#second').innerHTML());
        assert.equal(await page.evaluate(() => document.activeElement?.dataset.action), 'primary');
        assert.equal(await page.locator('#first svg text').count(), 0, 'Plot labels must remain HTML-sized');
      }
    }
    assert.deepEqual(errors, []);
  } finally { await browser.close(); }
});
