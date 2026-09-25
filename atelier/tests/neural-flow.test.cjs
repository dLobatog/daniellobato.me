const { test } = require('node:test');
const assert = require('node:assert/strict');
const {
  EXAMPLE, has, render, content, forward, loss, derive, chain, trainStep,
  finiteDifference, createState, reduceState,
} = require('../neural-flow.js');

const near = (actual, expected, tolerance = 1e-9) => assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);
const keys = ['distance', 'stops', 'bias'];

test('All four lessons have the full content override contract and no legacy controls', () => {
  for (const kind of ['neuron', 'forward-pass', 'chain-rule', 'backprop']) {
    assert.ok(has(kind));
    for (const field of ['title', 'summary', 'what', 'why', 'interview']) assert.equal(typeof content[kind][field], 'string');
    assert.ok(content[kind].details.length);
    assert.ok(content[kind].math.formula.length);
    for (const annotation of content[kind].math.annotations) {
      assert.equal(annotation.length, 3);
      assert.ok(annotation.every(part => typeof part === 'string' && part.length > 0));
      assert.ok(!annotation[2].startsWith('Example:'), 'Host owns any Example prefix');
    }
    assert.equal(content[kind].code.lang, 'python');
    assert.ok(content[kind].code.snippet.length);
    assert.equal(content[kind].quiz.options.filter(option => option.correct).length, 1);
    assert.ok(content[kind].quiz.options.every(option => option.explanation));
    assert.deepEqual(content[kind].controls, []);
    assert.deepEqual(content[kind].presets, []);
  }
  assert.equal(has('transformer'), false);
  assert.equal(render('neuron', null), false);
});

test('The shared example adds real contributions with an identity output', () => {
  const actual = forward();
  assert.deepEqual(actual.contributions, { distance: 6, stops: 4, bias: 5 });
  near(actual.prediction, 15);
  near(loss(actual.prediction, 19), 8);
  near(derive().error, -4);
  assert.deepEqual(derive().gradients, { distance: -8, stops: -4, bias: -4 });
});

test('Changing an input is not training: one extra kilometer adds exactly 3 minutes', () => {
  const farther = forward({ ...EXAMPLE.inputs, distance: 3 });
  near(farther.prediction, 18);
  near(farther.contributions.distance, 9);
  near(farther.contributions.stops, 4);
  assert.deepEqual(EXAMPLE.parameters, { distance: 3, stops: 4, bias: 5 });
});

test('Gradients agree with central finite differences for all parameters and varied inputs', () => {
  const examples = [
    [EXAMPLE.inputs, EXAMPLE.parameters, EXAMPLE.target],
    [{ distance: 0, stops: 0 }, { distance: 1, stops: 2, bias: 3 }, 1],
    [{ distance: 3.25, stops: 2 }, { distance: -0.5, stops: 1.7, bias: -1 }, 4.2],
    [{ distance: -1.2, stops: 0.2 }, { distance: 2.2, stops: -0.8, bias: 5 }, -3],
    [EXAMPLE.inputs, EXAMPLE.parameters, 15],
  ];
  for (const [inputs, parameters, target] of examples) {
    for (const key of keys) {
      const actual = chain(key, inputs, parameters, target);
      near(actual.gradient, actual.predictionToLoss * actual.parameterToPrediction);
      near(actual.gradient, finiteDifference(key, inputs, parameters, target), 1e-7);
    }
  }
});

test('The 0.01 nudge demonstrates a local approximation, not an exact finite-change claim', () => {
  const before = derive();
  const after = derive(EXAMPLE.inputs, { ...EXAMPLE.parameters, distance: 3.01 });
  near(after.prediction, 15.02);
  near(after.loss, 7.9202);
  near(after.loss - before.loss, -0.0798);
  near(before.gradients.distance * 0.01, -0.08);
  assert.ok(Math.abs(after.loss - before.loss - before.gradients.distance * 0.01) > 0);
});

test('One simultaneous update lowers loss and does not mutate the old parameters', () => {
  const parameters = { ...EXAMPLE.parameters };
  const inputs = { ...EXAMPLE.inputs };
  const step = trainStep(inputs, parameters, 19, 0.1);
  assert.deepEqual(parameters, EXAMPLE.parameters);
  assert.deepEqual(inputs, EXAMPLE.inputs);
  near(step.after.parameters.distance, 3.8);
  near(step.after.parameters.stops, 4.4);
  near(step.after.parameters.bias, 5.4);
  near(step.updates.distance, 0.8);
  near(step.updates.stops, 0.4);
  near(step.updates.bias, 0.4);
  near(step.after.prediction, 17.4);
  near(step.after.loss, 1.28);
  assert.ok(step.after.loss < step.before.loss);
});

test('Zero error yields zero gradients; zero LR leaves parameters alone; large LR can fail', () => {
  const exact = trainStep(EXAMPLE.inputs, EXAMPLE.parameters, 15);
  for (const key of keys) near(exact.before.gradients[key], 0);
  near(exact.after.loss, 0);
  assert.deepEqual(trainStep(EXAMPLE.inputs, EXAMPLE.parameters, 19, 0).after.parameters, EXAMPLE.parameters);
  const unstable = trainStep(EXAMPLE.inputs, EXAMPLE.parameters, 19, 1);
  assert.ok(unstable.after.loss > unstable.before.loss);
});

test('Repeated sane steps converge on this sample without claiming generalization', () => {
  let parameters = { ...EXAMPLE.parameters }, oldLoss = derive().loss;
  for (let step = 0; step < 12; step++) {
    const next = trainStep(EXAMPLE.inputs, parameters, EXAMPLE.target);
    assert.ok(next.after.loss < oldLoss);
    parameters = next.after.parameters;
    oldLoss = next.after.loss;
  }
  assert.ok(oldLoss < 1e-8);
});

test('Navigation is bounded and replay does not accumulate parameter updates', () => {
  for (const kind of ['neuron', 'forward-pass', 'chain-rule', 'backprop']) {
    let state = createState(kind);
    assert.equal(reduceState(kind, state, 'previous').step, 0);
    const length = kind === 'neuron' ? 3 : 4;
    for (let i = 1; i < length; i++) {
      state = reduceState(kind, state, 'next');
      assert.equal(state.step, i);
    }
    assert.equal(reduceState(kind, state, 'next').step, 0);
    assert.deepEqual(reduceState(kind, state, 'reset'), createState(kind));
  }
});

test('Inspectors and the meaningful input intervention are bounded', () => {
  const state = createState('neuron');
  const farther = reduceState('neuron', state, 'extra-km');
  assert.equal(state.extraKm, false);
  assert.equal(farther.extraKm, true);
  assert.equal(reduceState('neuron', farther, 'extra-km').extraKm, false);
  assert.equal(reduceState('backprop', state, 'extra-km'), state);
  assert.equal(reduceState('chain-rule', state, 'parameter', 'bias').parameter, 'bias');
  assert.equal(reduceState('chain-rule', state, 'parameter', 'unknown'), state);
  assert.equal(reduceState('neuron', state, 'inspect', 'stops').selected, 'stops');
  assert.equal(reduceState('neuron', state, 'inspect', '<script>'), state);
});

test('Invalid math inputs fail explicitly rather than leaking NaN into labels', () => {
  assert.throws(() => forward({ distance: NaN, stops: 1 }), TypeError);
  assert.throws(() => loss(15, Infinity), TypeError);
  assert.throws(() => trainStep(EXAMPLE.inputs, EXAMPLE.parameters, 19, -0.1), RangeError);
  assert.throws(() => finiteDifference('unknown'), RangeError);
  assert.throws(() => finiteDifference('distance', EXAMPLE.inputs, EXAMPLE.parameters, 19, 0), RangeError);
});

// Optional real-browser contract test. No browser dependency is added to the static site.
// PLAYWRIGHT_MODULE=/path/to/playwright-core CHROME_PATH=/path/to/chrome node --test ...
test('Browser: shared roots, keyboard focus, lazy unmount, bounded layout and theme tokens', {
  skip: !process.env.PLAYWRIGHT_MODULE,
  timeout: 60000,
}, async () => {
  const path = require('node:path');
  const { chromium } = require(process.env.PLAYWRIGHT_MODULE);
  const browser = await chromium.launch({ headless: true, ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}) });
  try {
    const page = await browser.newPage({ viewport: { width: 1200, height: 900 }, reducedMotion: 'reduce' });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.setContent('<main style="max-width:1000px;margin:auto;font-family:Source Sans 3,sans-serif"><div id="inline"></div><div id="focus"></div></main>');
    await page.addStyleTag({ path: path.resolve(__dirname, '../neural-flow.css') });
    await page.addScriptTag({ path: path.resolve(__dirname, '../neural-flow.js') });
    await page.evaluate(() => {
      const api = window.AtelierNeuralFlow;
      const inline = document.querySelector('#inline');
      api.render('neuron', inline);
      api.render('neuron', inline); // Re-rendering must not register another delegated handler.
      api.render('neuron', document.querySelector('#focus'));
      window.bubbledActivations = 0;
      document.querySelector('main').addEventListener('click', () => window.bubbledActivations++);
      document.querySelector('main').addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') window.bubbledActivations++;
      });
    });
    assert.equal(await page.evaluate(() => window.AtelierLessonModules.filter(module => module === window.AtelierNeuralFlow).length), 1);
    await page.locator('#inline [data-nf-focus="next"]').focus();
    await page.keyboard.press('Enter');
    assert.match(await page.locator('#inline .nf-step-count').innerText(), /2 \/ 3/);
    assert.match(await page.locator('#focus .nf-step-count').innerText(), /2 \/ 3/);
    assert.equal(await page.evaluate(() => document.activeElement.dataset.nfFocus), 'next');
    await page.locator('#focus [data-nf-focus="extra-km"]').click();
    assert.match(await page.locator('#inline .nf-prediction .nf-big').innerText(), /18/);
    assert.equal(await page.evaluate(() => window.bubbledActivations), 0, 'Lesson controls must not open a parent viewer');
    await page.locator('#inline [data-nf-focus="inspect-stops"]').focus();
    await page.keyboard.press('Space');
    assert.equal(await page.locator('#inline [data-nf-focus="inspect-stops"]').getAttribute('aria-pressed'), 'true');
    assert.equal(await page.evaluate(() => document.activeElement.dataset.nfFocus), 'inspect-stops');
    await page.evaluate(() => document.querySelector('#focus').replaceChildren());
    await page.locator('#inline [data-nf-focus="next"]').click();
    assert.equal(await page.locator('#focus').innerHTML(), '', 'An emptied lazy root must stay empty');
    await page.evaluate(() => window.AtelierNeuralFlow.render('neuron', document.querySelector('#focus')));
    assert.match(await page.locator('#focus .nf-step-count').innerText(), /3 \/ 3/);
    await page.locator('#focus [data-nf-focus="previous"]').click();
    assert.match(await page.locator('#inline .nf-step-count').innerText(), /2 \/ 3/);

    // Same element can be reused for another visualization, with no stale event binding.
    await page.evaluate(() => {
      document.querySelector('#focus').replaceChildren();
      window.AtelierNeuralFlow.render('backprop', document.querySelector('#inline'));
    });
    for (let i = 0; i < 2; i++) await page.locator('#inline [data-nf-focus="next"]').click();
    assert.match(await page.locator('#inline [data-nf-focus="next"]').innerText(), /Apply one update/);
    assert.match(await page.locator('#inline .nf-update-rule').innerText(), /No update applied yet/);
    await page.locator('#inline [data-nf-focus="next"]').click();
    assert.match(await page.locator('#inline .nf-after').innerText(), /17.4/);
    assert.match(await page.locator('#inline .nf-after').innerText(), /1.28/);

    for (const width of [1200, 720, 390, 320]) {
      await page.setViewportSize({ width, height: 900 });
      for (const kind of ['neuron', 'forward-pass', 'chain-rule', 'backprop']) {
        await page.evaluate(kind => window.AtelierNeuralFlow.render(kind, document.querySelector('#inline')), kind);
        const steps = kind === 'neuron' ? 3 : 4;
        for (let step = 0; step < steps; step++) {
          const failures = await page.locator('#inline').evaluate(root => {
            const bounds = root.getBoundingClientRect();
            return [...root.querySelectorAll('*')].flatMap(node => {
              if (node.closest('.nf-sr-only, svg')) return [];
              const rect = node.getBoundingClientRect();
              const overflow = rect.width && (rect.left < bounds.left - 1 || rect.right > bounds.right + 1);
              const textTooSmall = node.tagName !== 'SUP' && [...node.childNodes].some(child => child.nodeType === 3 && child.textContent.trim()) && parseFloat(getComputedStyle(node).fontSize) < 15.9;
              return overflow || textTooSmall ? [{ tag: node.tagName, class: node.className, overflow, textTooSmall }] : [];
            });
          });
          assert.deepEqual(failures, [], `${kind}, step ${step}, width ${width}`);
          await page.locator('#inline [data-nf-focus="next"]').click();
        }
      }
    }
    await page.evaluate(() => document.documentElement.style.setProperty('--lesson-ink', '#e9eef3'));
    assert.equal(await page.locator('.nf-lesson').first().evaluate(node => getComputedStyle(node).color), 'rgb(233, 238, 243)');
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
});
