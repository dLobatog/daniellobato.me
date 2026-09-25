// Run with PLAYWRIGHT_MODULE pointing to an installed playwright-core package.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const base = process.env.ATELIER_URL || 'http://127.0.0.1:4174/atelier/';
const output = process.env.ATELIER_SCREENSHOTS || '/tmp/atelier-clarity-qa';
fs.mkdirSync(output, {recursive:true});

(async () => {
  const browser = await chromium.launch({headless:true, ...(process.env.CHROME_PATH ? {executablePath:process.env.CHROME_PATH} : {})});
  const errors = [];
  const checks = [];
  try {
    const page = await browser.newPage({viewport:{width:1440,height:900}, reducedMotion:'reduce'});
    page.on('pageerror', error => errors.push(`${page.url()}: ${error.message}`));
    page.setDefaultTimeout(10000);
    async function open(file = '') {
      await page.goto(base + file, {waitUntil:'domcontentloaded'});
      await page.evaluate(() => document.fonts.ready);
      await page.addStyleTag({content:'html { scroll-behavior: auto !important; }'});
    }
    async function capture(name) { await page.screenshot({path:path.join(output, name + '.png')}); }
    async function scroll(selector) {
      await page.locator(selector).first().evaluate(el => el.scrollIntoView({block:'start', behavior:'instant'}));
      await page.waitForTimeout(200);
    }
    async function noPageOverflow(label) {
      const size = await page.evaluate(() => ({width:innerWidth, scroll:document.documentElement.scrollWidth}));
      checks.push({label, ...size});
      assert.ok(size.scroll <= size.width + 1, `${label}: page overflow ${JSON.stringify(size)}`);
    }
    await open();
    const chapters = await page.locator('.route-link').evaluateAll(nodes => nodes.map(node => node.getAttribute('href').replace('./','')));
    assert.equal(chapters.length, 16);

    for (const width of [1440,720,390]) {
      await page.setViewportSize({width, height:width === 390 ? 844 : 900});
      await open();
      await page.locator('[data-guess="99"]').click();
      assert.match(await page.locator('.opening-answer').innerText(), /99 of the 198/);
      await capture(`home-${width}`);
      await noPageOverflow(`home-${width}`);
      await page.locator('#chapter-search').fill('softmax');
      assert.equal(await page.locator('.route-card:visible').count(), 1);
      await page.locator('#chapter-search').fill('nonexistent123');
      assert.equal(await page.locator('.route-card:visible').count(), 0);
      await page.locator('#chapter-search').fill('');
      assert.equal(await page.locator('.route-card:visible').count(), 16);
      await scroll('#chapter-library');
      await capture(`library-${width}`);

      await open('foundations.html');
      for (const id of ['distribution','expectation','bayes','entropy','loss']) {
        await scroll(`#${id} .viz-panel`);
        await page.locator(`#${id} [data-stage-root]`).locator('svg,canvas,.lesson').first().waitFor({state:'visible'});
        await capture(`${id}-${width}`);
        await noPageOverflow(`${id}-${width}`);
      }
      await scroll('#expectation .viz-panel');
      const expectation = page.locator('#expectation .lesson');
      await expectation.locator('[data-action="shape"][data-value="high"]').click();
      assert.equal(await expectation.locator('.surprise-value').innerText(), '4.4');
      await expectation.locator('[data-action="shape"][data-value="extremes"]').click();
      assert.equal(await expectation.locator('.surprise-value').innerText(), '3.5');
      await expectation.locator('[data-action="roll"][data-value="100"]').click();
      assert.match(await expectation.locator('.sampling-experiment').innerText(), /100 rolls/);
      await capture(`expectation-samples-${width}`);
      await scroll('#bayes .viz-panel');
      const bayes = page.locator('#bayes .lesson');
      await bayes.locator('[data-action="next"]').click();
      assert.equal(await bayes.locator('.test-branch strong').first().innerText(), '99');
      await bayes.locator('[data-action="inspect"][data-value="healthy"]').click();
      assert.match(await bayes.locator('.lesson-observation').innerText(), /9,900 × 0.01 = 99/);
      await capture(`bayes-test-${width}`);
      await bayes.locator('[data-action="next"]').click();
      assert.equal(await bayes.locator('.posterior-equation > strong').innerText(), '50%');
      await capture(`bayes-posterior-${width}`);
      await bayes.locator('[data-action="prior"][data-value="0.1"]').click();
      assert.equal(await bayes.locator('.posterior-equation > strong').innerText(), '91.7%');
      assert.equal(await page.locator('.viz-lightbox:visible').count(), 0, 'Lesson buttons must not open the viewer');
      await page.locator('#bayes .viz-expand-button').click();
      const dialog = page.locator('.viz-lightbox:not([hidden])');
      assert.equal(await dialog.locator('.posterior-equation > strong').innerText(), '91.7%');
      await capture(`bayes-focus-${width}`);
      await noPageOverflow(`bayes-focus-${width}`);
      await dialog.locator('[data-action="prior"][data-value="0.01"]').click();
      await page.keyboard.press('Escape');
      assert.equal(await page.locator('.viz-lightbox:visible').count(), 0);
      assert.equal(await bayes.locator('.posterior-equation > strong').innerText(), '50%');
      assert.equal(await page.locator('#bayes .viz-expand-button').evaluate(el => el === document.activeElement), true);
      await page.locator('#bayes .population-strip').click();
      await dialog.locator('.viz-lightbox__close').click();
      assert.equal(await page.locator('.viz-lightbox:visible').count(), 0);
      if (width > 500) {
        await page.locator('#bayes .viz-expand-button').click();
        await page.mouse.click(3,3);
        assert.equal(await page.locator('.viz-lightbox:visible').count(), 0, 'Backdrop closes focus view');
      }
      await scroll('#entropy .viz-panel');
      const ent = page.locator('#entropy .lesson');
      await ent.locator('[data-action="p"][data-value="0.99"]').click();
      await ent.locator('[data-action="outcome"][data-value="tails"]').click();
      assert.match(await ent.locator('.surprise-value').innerText(), /0.081/);
      assert.match(await ent.locator('.lesson-observation').innerText(), /6.644/);
      await capture(`entropy-rare-${width}`);
      await ent.locator('[data-action="mode"][data-value="model"]').click();
      await ent.locator('[data-action="q"][data-value="0.1"]').click();
      assert.match(await ent.locator('.surprise-value').innerText(), /3.290/);
      await page.locator('#entropy .viz-expand-button').click();
      await capture(`entropy-focus-${width}`);
      await dialog.locator('.viz-lightbox__dialog').evaluate(el => { el.scrollTop = el.scrollHeight; });
      await capture(`entropy-focus-bottom-${width}`);
      const closeBox = await dialog.locator('.viz-lightbox__close').boundingBox();
      assert.ok(closeBox.y >= 0 && closeBox.y < (width === 390 ? 844 : 900), 'Close remains visible after scrolling');
      await dialog.locator('.viz-lightbox__close').click();
      for (const id of ['bayes','entropy']) {
        const formula = page.locator(`#${id} .study-tool`).filter({has:page.locator('summary', {hasText:/^Formula$/})});
        await formula.locator('summary').click();
        assert.equal(await formula.locator('.katex-error').count(), 0);
        assert.ok(await formula.locator('.katex').count() > 0);
        await scroll(`#${id} .study-tool[open]`);
        await capture(`${id}-formula-${width}`);
        await noPageOverflow(`${id}-formula-${width}`);
        const clippedCopy = await formula.locator('.formula-copy').evaluateAll(nodes => nodes.filter(node => node.scrollWidth > node.clientWidth + 1).length);
        assert.equal(clippedCopy, 0, `${id} formula annotations must wrap`);
      }
    }

    // Shared-layout smoke check: mount every concept, not just page headings.
    for (const chapter of chapters) {
      await page.setViewportSize({width:1440,height:900});
      await open(chapter);
      const concepts = await page.locator('.concept-card').count();
      assert.ok(concepts > 0, `${chapter} is blank`);
      for (let i = 0; i < concepts; i++) {
        const card = page.locator('.concept-card').nth(i);
        await card.locator('.viz-panel').evaluate(el => el.scrollIntoView({block:'start',behavior:'instant'}));
        await card.locator('[data-stage-root]').locator('svg,canvas,.lesson').first().waitFor({state:'visible'});
      }
      await scroll('.viz-panel');
      await capture(`${chapter}-desktop`);
      await noPageOverflow(`${chapter}-desktop`);
      await page.setViewportSize({width:390,height:844});
      await scroll('.viz-panel');
      await capture(`${chapter}-mobile`);
      await noPageOverflow(`${chapter}-mobile`);
      console.log('RENDERED', chapter, concepts, 'concepts');
    }
    assert.deepEqual(errors, [], 'No runtime errors');
    fs.writeFileSync(path.join(output,'results.json'),JSON.stringify({checks, errors},null,2));
    console.log('PASS', checks.length, 'responsive checks; screenshots:',output);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
