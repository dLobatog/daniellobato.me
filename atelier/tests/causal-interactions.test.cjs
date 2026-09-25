const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

test('browser: causal neural actions stay readable at full, half-screen, and phone widths', {skip:!process.env.ATELIER_BROWSER_TESTS, timeout:120000}, async()=>{
  const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
  const browser = await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
  const out=process.env.ATELIER_SCREENSHOTS || '/tmp/atelier-causal-interactions';
  const errors=[];
  try {
    const page=await browser.newPage({reducedMotion:'reduce'});
    page.on('pageerror', e=>errors.push(e.message));
    for(const width of [1440,720,390]) {
      await page.setViewportSize({width,height:900});
      await page.goto((process.env.ATELIER_BASE_URL || 'http://127.0.0.1:4174/atelier/')+'neural-network-basics.html',{waitUntil:'networkidle'});
      const pixels=await page.locator('#neuron .nn-contributions').first().boundingBox();
      const result=await page.locator('#neuron .nn-causal-result').first().boundingBox();
      if(width>=720) assert.ok(result.x>=pixels.x+pixels.width,`Keep the calculation beside the pixels at ${width}`);
      else assert.ok(result.y>=pixels.y+pixels.height,`Stack the calculation below the pixels at ${width}`);
      const open=async id=>{
        await page.locator(`#${id} .viz-expand-button`).click();
        const dialog=page.locator('.viz-lightbox:not([hidden]) .viz-lightbox__dialog');
        await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
        const header=await dialog.locator('h3').first().boundingBox();
        const toolbar=await dialog.locator('.viz-lightbox__toolbar').boundingBox();
        assert.ok(header.y>=toolbar.y+toolbar.height-1,`${id}: heading hidden under close toolbar at ${width}`);
        return dialog;
      };
      const finish=async(id,dialog)=>{
        await dialog.evaluate(el=>{el.scrollTop=0;});
        fs.mkdirSync(path.join(out,String(width)),{recursive:true});
        await page.screenshot({path:path.join(out,String(width),`${id}.png`)});
        await dialog.locator('.viz-lightbox__close').click();
        assert.equal(await page.locator('.viz-lightbox:not([hidden])').count(),0);
      };
      let d=await open('neuron');
      await d.locator('.nn-contribution[data-value="1"]').click();
      assert.match(await d.locator('.nn-change').innerText(),/2.3 → 1.5/);
      assert.match(await d.locator('.nn-causal-result').innerText(),/Pixel 2 adds nothing/);
      await finish('neuron',d);
      d=await open('activation-basics');
      await d.locator('.fn-first-action button').click();
      assert.match(await d.locator('.fn-gradient-comparison').textContent(),/0.018/);
      await finish('activation-basics',d);
      d=await open('output-functions');
      await d.locator('.fn-first-action button').click();
      assert.match(await d.locator('.fn-takeaway').innerText(),/cat sigmoid stays 88.1%/);
      assert.match(await d.locator('.fn-linked-equation').innerText(),/exp\(4.0\)/);
      await d.locator('.fn-comparison-row[data-fn-value="0"]:visible,.fn-independent .fn-outcome[data-fn-value="0"]:visible').click();
      assert.equal(await d.locator('.fn-options').evaluate(el=>el.open),true);
      await finish('output-functions',d);
      d=await open('forward-pass');
      await d.locator('[data-action="next"]').click();
      await d.locator('[data-action="next"]').click();
      await d.locator('.nn-hidden[data-value="1"]').click();
      assert.match(await d.innerText(),/Inside neuron 2/);
      assert.match(await d.locator('.nn-output').innerText(),/73.1%/);
      await finish('forward-pass',d);
      d=await open('chain-rule');
      await d.locator('button[data-action="weight"]:visible').first().click();
      assert.match(await d.locator('.nn-gradient-answer').innerText(),/One zero factor/);
      await d.locator('button[data-action="step"]').click();
      assert.match(await d.locator('.nn-probe:visible').first().innerText(),/Actual Δloss 0/);
      await finish('chain-rule',d);
      d=await open('backprop');
      for(let i=0;i<3;i++) await d.locator('button[data-action="next"]').click();
      assert.match(await d.locator('.nn-training-result').innerText(),/Pixels and label stayed fixed/);
      assert.match(await d.locator('.nn-change').first().innerText(),/0.8 → 0.8269/);
      await finish('backprop',d);
    }
    assert.deepEqual(errors,[]);
  } finally { await browser.close(); }
});
