const { test } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const os = require('node:os');
const api = require('../optimization-lessons.js');
const { loss, gradient, initialState, optimizerStep, trajectory, scheduleRate, clipSegment, START, BUDGET } = api;
const close = (actual, expected, tolerance = 1e-10) => assert.ok(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${actual} != ${expected}`);
const vectorClose = (actual, expected, tolerance) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, i) => close(value, expected[i], tolerance));
};

test('the fixed loss has the advertised start and minimum', () => {
  close(loss(START), 10.98);
  close(loss([0, 0]), 0);
  vectorClose(gradient([0, 0]), [0, 0]);
  close(loss([2, 1]), loss([-2, -1]));
});

test('analytic gradients agree with central finite differences', () => {
  const epsilon = 1e-5;
  for (const point of [[0, 0], START, [2.25, -1.7], [-0.01, 0.2], [8, -7], [-3.1, -0.5]]) {
    const numerical = point.map((_, i) => {
      const left = [...point], right = [...point];
      left[i] -= epsilon;
      right[i] += epsilon;
      return (loss(right) - loss(left)) / (2 * epsilon);
    });
    vectorClose(gradient(point), numerical, 1e-8);
  }
});

test('gradient-descent examples use the actual update and loss', () => {
  const first = optimizerStep(initialState(), { lr: 0.12 });
  vectorClose(first.gradient, [-3, 10.8]);
  vectorClose(first.delta, [0.36, -1.296]);
  vectorClose(first.point, [-2.64, -0.096]);
  close(first.loss, 3.526272);
  assert.ok(first.point[1] < 0 && first.loss < loss(START), 'crossing zero does not imply failure');
  const hot = optimizerStep(initialState(), { lr: 0.26 });
  assert.ok(hot.loss > loss(START));
  const small = optimizerStep(initialState(), { lr: 0.02 });
  assert.ok(small.loss > first.loss && small.loss < loss(START));
});

test('SGD paths obey the closed-form coordinate contractions', () => {
  for (const lr of [0, 0.02, 0.12, 0.18, 0.26]) {
    const result = trajectory({ lr });
    assert.equal(result.points.length, BUDGET + 1);
    for (const point of result.points) {
      vectorClose(point.point, [START[0] * (1 - lr) ** point.step, START[1] * (1 - 9 * lr) ** point.step], 1e-8);
      close(point.loss, loss(point.point));
    }
  }
});

test('stable SGD lowers loss each step; the hot vertical error grows', () => {
  for (const lr of [0.02, 0.12, 0.18, 0.22]) {
    const { points } = trajectory({ lr });
    points.slice(1).forEach((point, i) => assert.ok(point.loss < points[i].loss));
  }
  const hot = trajectory({ lr: 0.26 }).points;
  hot.slice(1).forEach((point, i) => {
    assert.ok(Math.abs(point.point[1]) > Math.abs(hot[i].point[1]));
    assert.ok(point.point[1] * hot[i].point[1] < 0);
  });
});

test('momentum uses the documented unnormalized recurrence', () => {
  const first = optimizerStep(initialState(), { method: 'momentum', lr: 0.12 });
  vectorClose(first.velocity, [-3, 10.8]);
  const second = optimizerStep(first, { method: 'momentum', lr: 0.12 });
  vectorClose(second.velocity, [-5.34, 8.856]);
  vectorClose(second.point, [-1.9992, -1.15872]);
  const path = trajectory({ method: 'momentum' }).points;
  path.slice(1).forEach((point, i) => vectorClose(point.velocity, gradient(path[i].point).map((g, k) => 0.9 * path[i].velocity[k] + g)));
});

test('Adam corrects both zero-initialized moments, including the first update', () => {
  const first = optimizerStep(initialState(), { method: 'adam', lr: 0.12 });
  vectorClose(first.m, [-0.3, 1.08]);
  vectorClose(first.v, [0.009, 0.11664]);
  vectorClose(first.mHat, [-3, 10.8]);
  vectorClose(first.vHat, [9, 116.64]);
  vectorClose(first.point, [-2.88, 1.08], 1e-8);
  const zero = optimizerStep(initialState([0, 0]), { method: 'adam' });
  vectorClose(zero.point, [0, 0]);
  assert.ok(zero.direction.every(Number.isFinite));
});

test('Adam agrees with an independent scalar recurrence for both coordinates', () => {
  for (const lr of [0.02, 0.12, 0.26]) {
    const points = trajectory({ method: 'adam', lr }).points;
    for (let axis = 0; axis < 2; axis++) {
      let x = START[axis], mean = 0, square = 0;
      for (let step = 1; step <= BUDGET; step++) {
        const slope = (axis === 0 ? 1 : 9) * x;
        mean = 0.9 * mean + 0.1 * slope;
        square = 0.999 * square + 0.001 * slope * slope;
        const correctedMean = mean / (1 - 0.9 ** step);
        const correctedSquare = square / (1 - 0.999 ** step);
        x -= lr * correctedMean / (Math.sqrt(correctedSquare) + 1e-8);
        close(points[step].point[axis], x);
      }
    }
  }
});

test('all optimizer paths share start, count, and loss; comparison is not an Adam victory story', () => {
  const paths = ['sgd', 'momentum', 'adam'].map(method => trajectory({ method }));
  for (const path of paths) {
    assert.equal(path.points.length, 25);
    assert.deepEqual(path.points[0].point, [...START]);
    path.points.forEach((point, i) => { assert.equal(point.step, i); close(point.loss, loss(point.point)); });
  }
  assert.ok(paths[0].points.at(-1).loss < paths[2].points.at(-1).loss);
  assert.ok(paths[1].points.some((point, i) => i > 0 && point.loss > paths[1].points[i - 1].loss));
});

test('step decay switches at update 13; cosine includes both endpoints', () => {
  for (let index = 0; index < 24; index++) {
    close(scheduleRate('constant', index), 0.18);
    close(scheduleRate('step', index), index < 12 ? 0.18 : 0.018);
    assert.ok(scheduleRate('cosine', index) >= 0 && scheduleRate('cosine', index) <= 0.18);
    if (index) assert.ok(scheduleRate('cosine', index) <= scheduleRate('cosine', index - 1));
  }
  close(scheduleRate('cosine', 0), 0.18);
  close(scheduleRate('cosine', 23), 0);
});

test('schedule trajectories keep SGD fixed and consume the same budget', () => {
  const constant = trajectory({ lr: 0.18, schedule: 'constant' }).points;
  const step = trajectory({ lr: 0.18, schedule: 'step' }).points;
  const cosine = trajectory({ lr: 0.18, schedule: 'cosine' }).points;
  assert.equal(constant.length, step.length);
  assert.equal(step.length, cosine.length);
  for (let i = 0; i <= 12; i++) vectorClose(constant[i].point, step[i].point);
  vectorClose(step[13].gradient, constant[13].gradient);
  vectorClose(step[13].delta, constant[13].delta.map(value => value / 10));
  vectorClose(cosine[24].point, cosine[23].point);
  assert.ok(gradient(cosine[23].point).some(value => Math.abs(value) > 0), 'zero LR is not a zero gradient');
  assert.ok(constant.at(-1).loss < step.at(-1).loss);
  assert.ok(constant.at(-1).loss < cosine.at(-1).loss);
});

test('pure operations do not mutate their input or previously returned steps', () => {
  const start = initialState();
  const copy = JSON.stringify(start);
  for (const method of ['sgd', 'momentum', 'adam']) optimizerStep(start, { method });
  assert.equal(JSON.stringify(start), copy);
  const point = [...START];
  trajectory({ start: point });
  assert.deepEqual(point, [...START]);
});

test('unsafe updates stop before non-finite values are exposed', () => {
  for (const lr of [5, 1e100, Number.MAX_VALUE]) {
    const result = trajectory({ lr, budget: 200 });
    assert.equal(result.stopped, true);
    assert.ok(result.reason);
    for (const point of result.points) {
      assert.ok(point.point.every(Number.isFinite));
      assert.ok(Number.isFinite(point.loss));
      for (const key of ['gradient', 'delta', 'velocity', 'm', 'v']) if (point[key]) assert.ok(point[key].every(Number.isFinite));
    }
  }
});

test('invalid numerical input is rejected instead of contaminating the plot', () => {
  assert.throws(() => initialState([Infinity, 1]));
  assert.throws(() => initialState([1e200, 1]));
  assert.throws(() => gradient([NaN, 1]));
  assert.throws(() => optimizerStep(initialState(), { lr: -1 }));
  assert.throws(() => optimizerStep(initialState(), { method: 'imaginary' }));
  assert.throws(() => optimizerStep(initialState(), { beta1: 1 }));
  assert.throws(() => scheduleRate('cosine', 24));
  assert.throws(() => trajectory({ budget: 0 }));
});

test('clipping preserves real segments without making giant SVG coordinates', () => {
  const bounds = [-4, 4, -2.5, 2.5];
  assert.deepEqual(clipSegment([-1, 0], [1, 0], bounds), [[-1, 0], [1, 0]]);
  const wide = clipSegment([-1e6, 0], [1e6, 0], bounds);
  vectorClose(wide[0], [-4, 0], 1e-8);
  vectorClose(wide[1], [4, 0], 1e-8);
  assert.deepEqual(clipSegment([5, 0], [6, 0], bounds), null);
  assert.deepEqual(clipSegment([0, 0], [0, 0], bounds), [[0, 0], [0, 0]]);
  for (const p of trajectory({ lr: 0.26 }).points) {
    const clipped = clipSegment([0, 0], p.point, bounds);
    assert.ok(clipped);
    clipped.forEach(([x, y]) => { assert.ok(x >= -4 - 1e-8 && x <= 4 + 1e-8); assert.ok(y >= -2.5 - 1e-8 && y <= 2.5 + 1e-8); });
  }
});

test('all integration records include annotated real math, code, and one correct quiz answer', () => {
  for (const kind of ['gradient-descent', 'optimizers', 'lr-schedule']) {
    assert.ok(api.has(kind));
    const content = api.content[kind];
    for (const field of ['title', 'summary', 'what', 'why', 'interview']) assert.equal(typeof content[field], 'string');
    assert.deepEqual(content.controls, []);
    assert.deepEqual(content.presets, []);
    assert.ok(content.math.formula.length);
    assert.ok(content.math.annotations.every(row => row.length === 3 && row.every(Boolean)));
    assert.equal(content.code.lang, 'python');
    assert.equal(content.quiz.options.filter(option => option.correct).length, 1);
  }
  assert.equal(api.has('not-a-lesson'), false);
  assert.equal(api.render('not-a-lesson', null), null);
});

// Optional DOM checks, using the parent's Playwright installation without adding a dependency.
// ATELIER_BROWSER_TESTS=1 PLAYWRIGHT_MODULE=/path/to/playwright-core node --test ...
test('browser: responsive markup, direct inspection, shared state, focus, and lazy unmount', { skip: !process.env.ATELIER_BROWSER_TESTS, timeout: 90000 }, async () => {
  const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
  const browser = await chromium.launch({ headless: true, ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}) });
  const page = await browser.newPage({ viewport: { width: 1280, height: 960 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  try {
    await page.setContent('<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><main><div id="inline"></div><div id="focus"></div></main>');
    await page.addStyleTag({ content: 'html{background:#f2f5f7}body{margin:0;padding:20px;font-family:"Source Sans 3",sans-serif}main{max-width:1200px;margin:auto}#focus{margin-top:24px}' });
    await page.addStyleTag({ path: path.join(__dirname, '../optimization-lessons.css') });
    await page.addScriptTag({ path: path.join(__dirname, '../optimization-lessons.js') });
    const mount = kind => page.evaluate(kind => {
      window.AtelierOptimization.render(kind, document.getElementById('inline'));
      window.AtelierOptimization.render(kind, document.getElementById('focus'));
    }, kind);
    await mount('gradient-descent');
    assert.equal(await page.evaluate(() => window.AtelierLessonModules.includes(window.AtelierOptimization)), true);
    // Ten renders must not attach ten click handlers.
    await page.evaluate(() => { for (let i = 0; i < 10; i++) window.AtelierOptimization.render('gradient-descent', document.getElementById('inline')); });
    await page.locator('#inline [data-opt-action="next"]').click();
    assert.match(await page.locator('#inline .opt-step-count').innerText(), /Step 1 of 24/);
    assert.match(await page.locator('#focus .opt-step-count').innerText(), /Step 1 of 24/);
    assert.equal(await page.evaluate(() => document.activeElement.dataset.optAction), 'next');
    await page.locator('#inline [data-opt-action="finish"]').click();
    await page.locator('#inline .opt-point-selected').focus();
    await page.keyboard.press('Home');
    assert.match(await page.locator('#inline .opt-step-count').innerText(), /Step 0 of 24/);
    await page.keyboard.press('ArrowRight');
    assert.match(await page.locator('#inline .opt-step-count').innerText(), /Step 1 of 24/);
    assert.equal(await page.evaluate(() => document.activeElement.dataset.optKey), 'point:1');
    await page.locator('#inline [data-opt-action="preset"][data-opt-value="hot"]').click();
    await page.locator('#inline [data-opt-action="finish"]').click();
    assert.ok(await page.locator('#inline .opt-outside-step').count());
    await page.locator('#inline .opt-outside-step').focus();
    await page.keyboard.press('ArrowLeft');
    assert.equal(await page.evaluate(() => document.activeElement.dataset.optKey), 'point:23');
    await page.evaluate(() => window.AtelierOptimization.unmount(document.getElementById('focus')));
    await page.locator('#inline [data-opt-action="reset"]').click();
    assert.match(await page.locator('#focus .opt-step-count').innerText(), /Step 23 of 24/);
    await mount('gradient-descent');
    assert.match(await page.locator('#focus .opt-step-count').innerText(), /Step 0 of 24/);

    const screenshotDir = process.env.ATELIER_SCREENSHOTS || path.join(os.tmpdir(), 'atelier-optimization-qa');
    fs.mkdirSync(screenshotDir, { recursive: true });
    for (const kind of ['gradient-descent', 'optimizers', 'lr-schedule']) {
      await mount(kind);
      if (kind === 'gradient-descent') await page.locator('#inline [data-opt-action="preset"][data-opt-value="useful"]').click();
      for (const width of [1280, 720, 390, 320]) {
        await page.setViewportSize({ width, height: 960 });
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        const geometry = await page.locator('#inline').evaluate(root => ({
          overflow: root.scrollWidth > root.clientWidth + 1,
          badSvg: [...root.querySelectorAll('svg path, svg polyline')].some(element => /NaN|Infinity/.test(element.getAttribute('d') || element.getAttribute('points') || '')),
          svgText: root.querySelectorAll('svg text').length,
        }));
        assert.equal(geometry.overflow, false, `${kind} overflow at ${width}`);
        assert.equal(geometry.badSvg, false);
        assert.equal(geometry.svgText, 0);
        if ([1280, 390].includes(width)) await page.locator('#inline').screenshot({ path: path.join(screenshotDir, `${kind}-${width}.png`) });
      }
      if (kind === 'optimizers') {
        for (const method of ['momentum', 'adam', 'sgd']) await page.locator(`#inline [data-opt-action="active"][data-opt-value="${method}"]`).click();
        await page.locator('#inline [data-opt-action="active"][data-opt-value="adam"]').click();
        assert.match(await page.locator('#inline .opt-inspector').innerText(), /Corrected mean/);
        assert.match(await page.locator('#focus .opt-inspector').innerText(), /Corrected mean/);
      }
      if (kind === 'lr-schedule') {
        await page.locator('#inline [data-opt-action="scheduled"][data-opt-value="13"]').focus();
        await page.keyboard.press('End');
        assert.match(await page.locator('#inline .opt-step-count').innerText(), /Step 24 of 24/);
        assert.equal(await page.evaluate(() => document.activeElement.dataset.optKey), 'scheduled:24');
        await page.locator('#inline [data-opt-action="active"][data-opt-value="cosine"]').click();
        assert.match(await page.locator('#inline .opt-result').innerText(), /rate is zero/);
      }
    }
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
});
