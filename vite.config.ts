import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5178, strictPort: false },
  build: { target: 'es2022', chunkSizeWarningLimit: 1200 },
});
