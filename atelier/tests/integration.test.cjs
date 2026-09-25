const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..');
const read = name => fs.readFileSync(path.join(root, name), 'utf8');
const chapters = [...read('index.html').matchAll(/class="route-link" href="\.\/([^"]+)"/g)].map(match => match[1]);

test('All public chapter assets exist and use the same reading shell', () => {
  assert.equal(chapters.length, 16);
  for (const file of ['index.html', ...chapters]) {
    const html = read(file);
    assert.match(html, /reading-theme\.js\?v=/, file);
    assert.match(html, /reading-theme\.css\?v=/, file);
    assert.doesNotMatch(html, /id="chapter-root"[^>]*aria-live/, 'Only targeted results should be live regions');
    for (const [, reference] of html.matchAll(/(?:src|href)="((?:\.\/|\.\.\/)[^"#]+)"/g)) {
      const local = reference.split(/[?#]/)[0];
      assert.ok(fs.existsSync(path.resolve(root, local)), `${file}: missing ${reference}`);
    }
  }
});

test('Native module registrations expose coherent content, not duplicate legacy controls', () => {
  const sandbox = {window:{}};
  for (const file of ['neural-flow.js', 'neural-functions.js', 'optimization-lessons.js', 'lessons.js']) vm.runInNewContext(read(file), sandbox, {filename:file});
  const facade = sandbox.window.AtelierLessons;
  for (const kind of ['neuron','activation-basics','output-functions','forward-pass','chain-rule','backprop','gradient-descent','optimizers','lr-schedule']) {
    assert.ok(facade.has(kind), kind);
    const content = facade.getContent(kind);
    assert.equal(content.controls.length, 0, kind);
    assert.equal(content.presets.length, 0, kind);
    assert.ok(content.summary && content.math.annotations.length && content.code.snippet, kind);
    assert.equal(content.quiz.options.filter(option => option.correct).length, 1, kind);
    for (const [symbol, meaning, example] of content.math.annotations) assert.ok(symbol && meaning && example, kind);
  }
  assert.ok(facade.has('bayes'));
  assert.equal(facade.has('unknown'), false);
});

function themeFixture(saved, unavailable = false) {
  const events = {}, attributes = {}, buttons = [];
  const rootNode = {dataset:{}};
  let stored = saved;
  const button = {setAttribute:(key,value) => attributes[key] = value, addEventListener:(key,handler) => events[key] = handler};
  const document = {
    documentElement:rootNode,
    addEventListener: (key,handler) => events[key] = handler,
    querySelector: () => ({append: node => buttons.push(node)}),
    createElement: () => button,
  };
  const localStorage = {
    getItem() { if (unavailable) throw Error('disabled'); return stored; },
    setItem(_,value) { if (unavailable) throw Error('disabled'); stored = value; },
  };
  vm.runInNewContext(read('reading-theme.js'), {document,localStorage});
  events.DOMContentLoaded();
  return {rootNode,button,attributes,buttons,events,stored:()=>stored};
}

test('Reading theme defaults light, toggles accessibly, and persists a choice', () => {
  const t = themeFixture();
  assert.equal(t.rootNode.dataset.readingTheme, 'light');
  assert.equal(t.button.textContent, 'Dark reading');
  t.events.click();
  assert.equal(t.rootNode.dataset.readingTheme, 'dark');
  assert.equal(t.attributes['aria-label'], 'Switch to light reading theme');
  assert.equal(t.stored(), 'dark');
  assert.equal(themeFixture('dark').rootNode.dataset.readingTheme, 'dark');
});

test('Reading theme does not require local storage', () => {
  const t = themeFixture(null, true);
  assert.doesNotThrow(() => t.events.click());
  assert.equal(t.rootNode.dataset.readingTheme, 'dark');
});
