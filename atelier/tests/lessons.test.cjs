const { test } = require('node:test');
const assert = require('node:assert/strict');
const { bayes, entropy, crossEntropy, expectation } = require('../lessons.js');
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-10, `${actual} != ${expected}`);

test('Different dice can have the same expected value', () => {
  close(expectation([1,1,1,1,1,1]), 3.5);
  close(expectation([1,0,0,0,0,1]), 3.5);
  close(expectation([1,1,1,1,2,4]), 4.4);
});

test('Bayes counts conserve people and positive results', () => {
  for (const prior of [.01, .1, .5]) {
    const b = bayes(prior);
    close(b.sick + b.healthy, 10000);
    close(b.positive, b.truePositive + b.falsePositive);
    close(b.posterior, b.truePositive / b.positive);
  }
});
test('Rare-event default produces 99 real detections and 99 false alarms', () => {
  const b = bayes(.01);
  close(b.truePositive, 99);
  close(b.falsePositive, 99);
  close(b.posterior, .5);
  assert.ok(bayes(.1).posterior > b.posterior);
});
test('Entropy is average surprise, highest for a fair binary coin', () => {
  close(entropy(.5), 1);
  close(entropy(.9), entropy(.1));
  assert.ok(entropy(.99) < entropy(.9));
  assert.ok(entropy(.9) < entropy(.5));
});
test('Every teaching preset has nonnegative KL and a minimum at a matching model', () => {
  for (const p of [.5, .9, .99]) {
    close(crossEntropy(p, p), entropy(p));
    for (const q of [.1, .5, p]) assert.ok(crossEntropy(p, q) >= entropy(p) - 1e-10);
    close(crossEntropy(p, .5), 1);
  }
});
