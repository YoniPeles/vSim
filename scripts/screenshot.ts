// Dev helper: screenshot the running app with headless Chromium (software WebGL).
// Usage: node scripts/screenshot.ts [url] [out.png] [width] [height] [waitMs]
// CLICK="Button text" clicks a button after load (e.g. to start the simulation).
import { chromium } from '@playwright/test';

const [url = 'http://localhost:5173/', out = 'shot.png', w = '1600', h = '1000', wait = '3500'] = process.argv.slice(2);
const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: Number(w), height: Number(h) } });
const logs: string[] = [];
page.on('console', (m) => {
  if (m.type() === 'error' || m.type() === 'warning') logs.push(`[${m.type()}] ${m.text()}`);
});
page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
await page.goto(url);
if (process.env['CLICK']) {
  await page.waitForTimeout(1500);
  await page.getByRole('button', { name: new RegExp(process.env['CLICK']) }).first().click();
}
await page.waitForTimeout(Number(wait));
await page.screenshot({ path: out });
console.log(logs.slice(0, 20).join('\n') || 'no console errors');
await browser.close();
