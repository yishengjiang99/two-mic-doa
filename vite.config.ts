import { defineConfig } from 'vite';
// assetsInlineLimit 0: the worklet ships as its own hashed file, not a data: URL.
export default defineConfig({ base: '/two-mic-doa/', build: { assetsInlineLimit: 0 } });
