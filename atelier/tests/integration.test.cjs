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
  for (const file of ['lab-core.js', 'neural-mechanisms.js', 'neural-functions.js', 'optimization-lessons.js', 'foundations-labs.js', 'transformer-labs.js', 'ranking-labs.js', 'tabular-labs.js', 'decision-labs.js', 'training-labs.js', 'lessons.js']) vm.runInNewContext(read(file), sandbox, {filename:file});
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
  const allContent = sandbox.window.AtelierLessonModules.flatMap(module => Object.keys(module.content));
  assert.equal(new Set(allContent).size, allContent.length, 'Each renderer has one owner');
  assert.equal(allContent.length, 67, '66 existing mechanisms plus GRPO; bandit appears in two chapters');
  for (const kind of allContent) {
    const c = facade.getContent(kind);
    assert.ok(c.summary && c.math.annotations.length && c.code.snippet, kind);
    assert.equal(c.controls.length, 0, kind);
    assert.equal(c.presets.length, 0, kind);
    assert.equal(c.quiz.options.filter(option => option.correct).length, 1, kind);
  }
});

test('Every public section resolves to a native mechanism, including ranking objectives and GRPO', () => {
  const sandbox = {window:{}, document:{addEventListener(){}}, setTimeout(){}};
  for (const file of ['lab-core.js', 'neural-mechanisms.js', 'neural-functions.js', 'optimization-lessons.js', 'foundations-labs.js', 'transformer-labs.js', 'ranking-labs.js', 'tabular-labs.js', 'decision-labs.js', 'training-labs.js', 'lessons.js', 'studio.js']) vm.runInNewContext(read(file), sandbox, {filename:file});
  const chapterData = vm.runInNewContext('chapters', sandbox);
  let count = 0;
  for (const chapter of Object.values(chapterData)) for (const section of chapter.sections) {
    const c = sandbox.window.AtelierLessons.getContent(section.viz, section.id);
    assert.ok(c, section.id);
    assert.ok(sandbox.window.AtelierLessons.has(c.viz || section.viz), section.id);
    count++;
  }
  assert.equal(count, 67);
  assert.equal(sandbox.window.AtelierLessons.getContent('ranking-metrics', 'rank-objectives').viz, 'rank-objectives');
  assert.ok(sandbox.window.AtelierLessons.getContent('grpo'));
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
