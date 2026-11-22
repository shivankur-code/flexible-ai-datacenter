const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage();
  await p.goto('file://' + __dirname + '/brief.html', { waitUntil: 'networkidle' });
  const foot = `<div style="width:100%;font-family:Inter,Helvetica,sans-serif;font-size:7px;color:#6b6b70;padding:0 0.62in;display:flex;justify-content:space-between">
    <span>Shivankur Sharma · Research brief for Prof. Priya Donti</span><span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span></div>`;
  await p.pdf({ path: __dirname + '/brief.pdf', format: 'Letter', printBackground: true, displayHeaderFooter: true,
    headerTemplate: '<span></span>', footerTemplate: foot, margin: { top: '0.5in', bottom: '0.5in', left: '0.62in', right: '0.62in' } });
  console.log('Inter loaded:', await p.evaluate(() => document.fonts.check('12px Inter')));
  await b.close();
})();
