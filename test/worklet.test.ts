import { describe, it, expect } from 'vitest';

// Regression: main.ts expects { ch: Float32Array[nMics], t } messages of `hop` samples.
// A stale cached worklet posting the old { a, b } shape left the waveforms blank.
describe('capture worklet', () => {
  it('posts hop-sized, sample-aligned chunks of every channel', async () => {
    const posted: { ch: Float32Array[]; t: number }[] = [];
    let Proc: any;
    Object.assign(globalThis, {
      AudioWorkletProcessor: class { port = { postMessage: (m: any) => posted.push(m) }; },
      registerProcessor: (_: string, c: any) => { Proc = c; },
      currentTime: 0, sampleRate: 48000,
    });
    await import('../src/capture-worklet.js');
    const p = new Proc({ processorOptions: { channels: 3, hop: 256 } });
    let n = 0;
    for (let q = 0; q < 5; q++) {
      const quantum = [0, 1, 2].map((c) => Float32Array.from({ length: 128 }, (_, i) => c * 1000 + n + i));
      expect(p.process([quantum])).toBe(true);
      n += 128;
    }
    expect(posted.length).toBe(2);
    for (const [k, m] of posted.entries()) {
      expect(m.ch.length).toBe(3);
      m.ch.forEach((x, c) => { expect(x.length).toBe(256); expect(x[0]).toBe(c * 1000 + k * 256); expect(x[255]).toBe(c * 1000 + k * 256 + 255); });
    }
  });
});
