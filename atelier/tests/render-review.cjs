/* Run against the local static server. Images stay outside the public repository. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.ATELIER_BASE_URL || 'http://127.0.0.1:4174/atelier/';
const out = process.env.ATELIER_SCREENSHOTS || '/tmp/atelier-render-review';
const root = path.resolve(__dirname, '..');
const pages = [...fs.readFileSync(path.join(root,'index.html'),'utf8').matchAll(/class="route-link" href="\.\/([^"]+)"/g)].map(x=>x[1]);
const widths = (process.env.ATELIER_WIDTHS || '1440,720,390').split(',').map(Number);

(async()=>{
  fs.mkdirSync(out,{recursive:true});
  const browser = await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
  const report = {checks:[],errors:[]};
  try {
    const page = await browser.newPage({reducedMotion:'reduce'});
    page.setDefaultTimeout(10000);
    page.on('pageerror',error=>report.errors.push({url:page.url(),message:error.message}));
    for(const width of widths) {
      await page.setViewportSize({width,height:900});
      for(const file of pages) {
        await page.goto(new URL(file,base).href,{waitUntil:'networkidle'});
        await page.evaluate(()=>document.fonts.ready);
        const ids = await page.locator('.chapter-map a').evaluateAll(xs=>xs.map(x=>x.hash.slice(1)));
        assert.ok(ids.length,`${file}: missing chapter`);
        for(const id of ids) {
          try {
          const card = page.locator(`#${id}`);
          await page.locator(`.chapter-map a[href="#${id}"]`).click();
          await card.locator('.viz-expand-button').click();
          const dialog = page.locator('.viz-lightbox:not([hidden]) .viz-lightbox__dialog');
          const lesson = dialog.locator('.lesson');
          await lesson.waitFor({state:'visible'});
          await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
          const measurement = await lesson.evaluate(el=>({
            stageWidth:el.clientWidth,scroll:el.scrollWidth,height:el.getBoundingClientRect().height,
            documentWidth:document.documentElement.scrollWidth,viewport:innerWidth,
            dialogWidth:el.closest('.viz-lightbox__dialog').clientWidth,
            dialogScroll:el.closest('.viz-lightbox__dialog').scrollWidth,
            collapsedCards:[...el.querySelectorAll('button,.dl-flow-node,.nn-hidden')].filter(node=>node.getBoundingClientRect().width>0&&node.scrollHeight>node.clientHeight+2).map(node=>node.textContent.trim().slice(0,100)),
          }));
          const entry={file,id,width,...measurement};
          report.checks.push(entry);
          if(measurement.scroll>measurement.stageWidth+1||measurement.dialogScroll>measurement.dialogWidth+1||measurement.documentWidth>width+1) {
            report.errors.push({file,id,width,reason:'horizontal overflow',measurement});
          }
          if(measurement.collapsedCards.length) report.errors.push({file,id,width,reason:'content exceeds card height',cards:measurement.collapsedCards});
          if(width!==720) {
            const dir=path.join(out,String(width),file.replace('.html',''));
            fs.mkdirSync(dir,{recursive:true});
            await page.screenshot({path:path.join(dir,`${id}.png`)});
          }
          const primary=lesson.locator('button.fl-primary:not(:disabled):visible,button.tl-primary:not(:disabled):visible,button.rk-primary:not(:disabled):visible,button.dl-primary:not(:disabled):visible,button.opt-primary:not(:disabled):visible,button[data-action="next"]:not(:disabled):visible').first();
          const internalDetails=lesson.locator('details');
          for(let i=0;i<await internalDetails.count();i++) {
            const detail=internalDetails.nth(i);
            if(!await detail.locator(':scope > summary').isVisible()) continue;
            const wasOpen=await detail.evaluate(el=>el.open);
            await detail.locator(':scope > summary').click();
            assert.equal(await detail.evaluate(el=>el.open),!wasOpen,`${id}: disclosure did not toggle`);
            await detail.locator(':scope > summary').click();
            assert.equal(await detail.evaluate(el=>el.open),wasOpen,`${id}: disclosure did not close`);
          }
          const action=await primary.count()?primary:lesson.locator('button[aria-pressed="false"]:not(:disabled):visible').first();
          if(await action.count()) {
            const before=await lesson.innerText();
            await action.click();
            entry.changed=(await lesson.innerText())!==before;
            entry.interaction='click';
          }
          await page.keyboard.press('Escape');
          assert.equal(await page.locator('.viz-lightbox:not([hidden])').count(),0,`${id}: Escape failed`);
          const folds=card.locator('.study-tool');
          for(let i=0;i<await folds.count();i++) {
            const fold=folds.nth(i);
            await fold.locator('summary').click();
            assert.equal(await card.locator('.katex-error').count(),0,`${id}: invalid formula`);
            const choice=fold.locator('.quiz-option').first();
            if(await choice.count()) {
              await choice.click();
              assert.ok((await fold.locator('.quiz-feedback').innerText()).trim(),`${id}: no quiz feedback`);
            }
            await fold.locator('summary').click();
          }
          if(id===ids[0]) {
            await card.locator('.viz-stage .lesson h3').first().click();
            await page.locator('.viz-lightbox:not([hidden]) .viz-lightbox__close').click();
            assert.equal(await page.locator('.viz-lightbox:not([hidden])').count(),0,`${id}: close button failed`);
            if(width>650) {
              await card.locator('.viz-expand-button').click();
              await page.mouse.click(1,1);
              assert.equal(await page.locator('.viz-lightbox:not([hidden])').count(),0,`${id}: backdrop failed`);
            }
          }
          console.log(`${width} ${file} #${id}: ${measurement.stageWidth}/${measurement.scroll}`);
          } catch(error) {
            report.errors.push({file,id,width,reason:error.message});
            console.error(`${width} ${file} #${id}: ${error.message}`);
            await page.goto(new URL(file,base).href,{waitUntil:'networkidle'});
          }
        }
      }
    }
  } finally {
    fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));
    await browser.close();
  }
  console.log(JSON.stringify({checks:report.checks.length,errors:report.errors},null,2));
  if(report.errors.length) process.exitCode=1;
})().catch(error=>{console.error(error);process.exitCode=1;});
