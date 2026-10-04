import { expect, test } from '@playwright/test';

test('renders the default deployment without console errors', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  await page.goto('/');
  await expect(page.getByText('vllm serve deepseek-ai/DeepSeek-V3')).toBeVisible();
  await expect(page.locator('canvas')).toBeVisible();
  await expect(page.getByText('KV capacity per replica')).toBeVisible();
  await page.waitForTimeout(1500);
  expect(errors).toEqual([]);
});

test('flag changes flow into the command line and the readout', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('radiogroup', { name: 'Tensor parallel size' }).getByRole('radio', { name: '4' }).click();
  await expect(page.getByText('--tensor-parallel-size 4')).toBeVisible();
  // DeepSeek-V3 FP8 does not fit on 4 H200s: the readout must say why.
  await expect(page.getByRole('alert').or(page.getByText(/no room for KV cache/))).toBeVisible();
});

test('loads a model that is not bundled from the Hugging Face Hub', async ({ page }) => {
  test.skip(!!process.env['OFFLINE'], 'needs network');
  await page.goto('/');
  await page.getByLabel('Hugging Face repo id').fill('Qwen/Qwen3-14B');
  await page.getByRole('button', { name: 'Load' }).click();
  await expect(page.getByText('vllm serve Qwen/Qwen3-14B')).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText(/40 × GQA/)).toBeVisible();
});

test('runs the scheduler simulation in a worker and streams frames', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('/');
  await page.getByRole('button', { name: /Simulate .* users/ }).click();
  await expect(page.getByRole('button', { name: 'Pause' })).toBeVisible();
  await expect(page.getByText(/KV blocks, replica of GPU/)).toBeVisible({ timeout: 10_000 });
  // Sim clock advances and requests start completing.
  await expect(page.getByText(/t = [1-9]\d*\.\d s/)).toBeVisible({ timeout: 15_000 });
  await expect(page.getByText(/\d+ requests done/)).toBeVisible({ timeout: 15_000 });
  await page.getByRole('button', { name: 'Trace one step' }).click();
  await expect(page.getByText(/GPU time, .* slower than real/)).toBeVisible();
  await page.getByRole('button', { name: 'Pause' }).first().click();
  expect(errors).toEqual([]);
});
