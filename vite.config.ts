import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  base: './',
  plugins: [react(), tailwindcss()],
  worker: { format: 'es' },
  // three + drei + postprocessing form one lazily loaded ~1.1 MB chunk; that is expected.
  build: { chunkSizeWarningLimit: 1300 },
  test: {
    include: ['src/**/*.test.ts'],
  },
});
