const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const api = require('../neural-functions.js');
const close = (a, b, epsilon = 1e-10) => assert.ok(Math.abs(a - b) < epsilon, `${a} != ${b}`);

test('Registration and content expose only the assigned lessons', () => {
  const context = { window: {} };
  vm.runInNewContext(fs.readFileSync(require.resolve('../neural-functions.js'), 'utf8'), context);
  assert.equal(context.window.AtelierLessonModules.length, 1);
  assert.equal(context.window.AtelierLessonModules[0], context.window.AtelierNeuralFunctions);
  assert.deepEqual(Object.keys(api.content), ['activation-basics', 'output-functions']);
  assert.equal(api.has('toString'), false);
  for (const kind of Object.keys(api.content)) {
    assert.equal(api.has(kind), true);
    const lesson = api.content[kind];
    for (const key of ['title', 'summary', 'what', 'why', 'interview']) assert.ok(lesson[key]);
    assert.deepEqual(lesson.controls, []);
    assert.deepEqual(lesson.presets, []);
    assert.equal(lesson.quiz.options.filter(option => option.correct).length, 1);
    assert.ok(lesson.quiz.options.every(option => option.explanation));
    assert.ok(lesson.math.annotations.every(row => row.length === 3 && row.every(Boolean)));
    assert.equal(lesson.code.lang, 'python');
  }
});

test('Sigmoid is stable for extreme finite scores and symmetric', () => {
  close(api.sigmoid(0), 0.5);
  assert.equal(api.sigmoid(1000), 1);
  assert.equal(api.sigmoid(-1000), 0);
  for (const z of [-6, -4, -0.25, 0, 1, 4, 6]) close(api.sigmoid(z), 1 - api.sigmoid(-z));
});

test('Activation derivatives agree with centered numerical derivatives away from the ReLU hinge', () => {
  const h = 1e-5;
  for (const kind of ['sigmoid', 'tanh', 'relu', 'linear']) {
    for (const z of [-6, -4, -0.25, 0.25, 1, 4, 6]) {
      const numeric = (api.activation(kind, z + h).value - api.activation(kind, z - h).value) / (2 * h);
      close(api.activation(kind, z).derivative, numeric, 1e-8);
    }
  }
});

test('ReLU at zero distinguishes the undefined derivative from the backprop convention', () => {
  assert.deepEqual(api.activation('relu', 0), { value: 0, derivative: null, backwardMultiplier: 0 });
  assert.equal(api.activation('relu', -1).derivative, 0);
  assert.equal(api.activation('relu', 1).derivative, 1);
  assert.throws(() => api.activation('unknown', 0), RangeError);
  assert.throws(() => api.activation('sigmoid', NaN), TypeError);
});

test('Saturation has a small local gradient even with a large output', () => {
  close(api.activation('sigmoid', 0).derivative, 0.25);
  close(api.activation('tanh', 0).derivative, 1);
  assert.ok(api.activation('sigmoid', 4).value > 0.98);
  assert.ok(api.activation('sigmoid', 4).derivative < 0.02);
  assert.ok(api.activation('tanh', 4).derivative < 0.002);
  assert.equal(api.activation('relu', 4).derivative, 1);
});

test('Softmax is normalized, shift invariant, and stable for large scores', () => {
  const base = api.softmax([2, 2, -1]);
  close(base.reduce((a, b) => a + b, 0), 1);
  api.softmax([10002, 10002, 9999]).forEach((p, i) => close(p, base[i]));
  api.softmax([-998, -998, -1001]).forEach((p, i) => close(p, base[i]));
  assert.deepEqual(api.softmax([1000, -1000]), [1, 0]);
  for (const invalid of [[], [NaN], [Infinity], 'scores']) assert.throws(() => api.softmax(invalid), TypeError);
});

test('Sigmoid(z) is exactly binary softmax([0,z])[1]', () => {
  for (const z of [-1000, -6, -2, 0, 2, 6, 1000]) close(api.sigmoid(z), api.softmax([0, z])[1]);
  close(api.softmax([4, 7])[1], api.sigmoid(7 - 4));
});

test('Raising dog alone leaves cat sigmoid fixed but decreases its softmax share', () => {
  const before = api.outputProbabilities([2, 2, -1]);
  const after = api.outputProbabilities([2, 4, -1]);
  close(before.independent[0], after.independent[0]);
  close(before.independent[2], after.independent[2]);
  assert.ok(after.independent[1] > before.independent[1]);
  assert.ok(after.exclusive[0] < before.exclusive[0]);
  assert.ok(after.exclusive[2] < before.exclusive[2]);
  assert.ok(after.exclusive[1] > before.exclusive[1]);
});

test('Equal scores are independent halves versus exclusive thirds', () => {
  const result = api.outputProbabilities([0, 0, 0]);
  result.independent.forEach(p => close(p, 0.5));
  result.exclusive.forEach(p => close(p, 1 / 3));
  close(result.independent.reduce((a, b) => a + b, 0), 1.5);
});
