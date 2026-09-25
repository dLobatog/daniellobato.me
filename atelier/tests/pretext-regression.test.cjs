const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname,'../studio.js'),'utf8');
const helpers = source.slice(source.indexOf('function getPretextPrepared('), source.indexOf('function renderMathLines('));

function fixture(text) {
  let html = text;
  const measured = [];
  const node = {
    dataset: {},
    get innerHTML() { return html; },
    set innerHTML(value) { html = value; },
    get textContent() { return html.replace(/<[^>]*>/g, ''); },
    set textContent(value) { html = value; },
    getBoundingClientRect: () => ({width:300}),
  };
  const context = {
    pretextPreparedCache: new Map(), pretextRenderedNodes: new WeakMap(),
    escapeHtml: x => x,
    getComputedStyle: () => ({fontStyle:'normal',fontWeight:'400',fontSize:'20px',fontFamily:'Source Sans 3',lineHeight:'30px'}),
    AtelierPretext: {
      prepareWithSegments(text, font) { measured.push({text,font}); return text; },
      layoutWithLines(text) { const words = text.split(' '); return {lines:[{text:words.slice(0,2).join(' ')},{text:words.slice(2).join(' ')}]}; },
    },
  };
  vm.createContext(context);
  vm.runInContext(helpers, context);
  return {node,measured,render: () => context.renderPretextLines(node)};
}

test('Pretext uses a newly rendered takeaway instead of restoring the first state', () => {
  const f = fixture('The prediction is too small.');
  f.render();
  f.node.textContent = 'The prediction is too large.';
  f.render();
  assert.equal(f.node.dataset.pretextSource, 'The prediction is too large.');
  assert.match(f.node.textContent, /too large/);
  assert.equal(f.measured.length, 2);
});

test('Pretext reflow retains word boundaries and measures the actual display font', () => {
  const f = fixture('The prediction is too small.');
  f.render();
  f.render();
  assert.equal(f.node.dataset.pretextSource, 'The prediction is too small.');
  assert.equal(f.measured.length, 1);
  assert.equal(f.measured[0].font, 'normal 400 20px Source Sans 3');
});
