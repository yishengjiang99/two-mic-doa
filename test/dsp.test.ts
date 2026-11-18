import { describe, it, expect } from 'vitest';
import { fft, gccPhat, angleDeg, speedOfSound, fitLine } from '../src/dsp';

// Seeded PRNG so tests are deterministic.
function rng(seed: number) {
  return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32) * 2 - 1;
}

/** White noise x and y = x delayed by `delay` samples (fractional via FFT phase shift). */
function delayedPair(n: number, delay: number, snrNoise = 0, seed = 1) {
  const m = n * 4;
  const r = rng(seed);
  const re = new Float64Array(m), im = new Float64Array(m);
  for (let i = 0; i < m; i++) re[i] = r();
  const x = Float64Array.from(re);
  fft(re, im);
  for (let k = 0; k < m; k++) {
    const f = k <= m / 2 ? k : k - m;
    const ph = (-2 * Math.PI * f * delay) / m;
    const c = Math.cos(ph), s = Math.sin(ph);
    const a = re[k], b = im[k];
    re[k] = a * c - b * s; im[k] = a * s + b * c;
  }
  if (m % 2 === 0) im[m / 2] = 0; // keep Nyquist real
  fft(re, im, true);
  const off = n; // take a middle segment, away from circular wrap
  const a = new Float64Array(n), b = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    a[i] = x[off + i] + snrNoise * r();
    b[i] = re[off + i] / m + snrNoise * r();
  }
  return { a, b };
}

describe('gccPhat', () => {
  for (const d of [0, 3, -3, 11, -17]) {
    it(`integer delay ${d}`, () => {
      const { a, b } = delayedPair(2048, d);
      const r = gccPhat(a, b, 30);
      expect(r.lag).toBeCloseTo(d, 1);
      expect(r.peak).toBeGreaterThan(0.5);
    });
  }
  for (const d of [2.3, -2.3, 5.5, -7.75, 0.4]) {
    it(`fractional delay ${d}`, () => {
      const { a, b } = delayedPair(2048, d);
      expect(Math.abs(gccPhat(a, b, 30).lag - d)).toBeLessThan(0.25);
    });
  }
  it('fractional delay with additive noise', () => {
    const { a, b } = delayedPair(4096, -4.6, 0.3, 7);
    expect(Math.abs(gccPhat(a, b, 30).lag + 4.6)).toBeLessThan(0.3);
  });
  it('searches around a calibration offset', () => {
    const { a, b } = delayedPair(4096, 250.4);
    const r = gccPhat(a, b, 30, 245);
    expect(Math.abs(r.lag - 250.4)).toBeLessThan(0.25);
    expect(r.corr.length).toBe(61);
  });
  it('uncorrelated noise has low peak', () => {
    const r1 = rng(3), r2 = rng(99);
    const a = Float64Array.from({ length: 2048 }, r1);
    const b = Float64Array.from({ length: 2048 }, r2);
    expect(gccPhat(a, b, 30).peak).toBeLessThan(0.2);
  });
});

describe('geometry', () => {
  const c = speedOfSound(20);
  it('speed of sound ~343 m/s at 20°C', () => expect(c).toBeCloseTo(343.2, 0));
  it('broadside, endfire, clamp', () => {
    expect(angleDeg(0, 0.2, c)).toBeCloseTo(90);
    expect(angleDeg(0.2 / c, 0.2, c)).toBeCloseTo(0);
    expect(angleDeg(-1, 0.2, c)).toBeCloseTo(180);
    expect(angleDeg((0.1 / c), 0.2, c)).toBeCloseTo(60);
  });
  it('fitLine recovers drift', () => {
    const t = [0, 2, 4, 6], y = t.map((v) => 100 + 0.5 * v);
    const { a, b } = fitLine(t, y);
    expect(a).toBeCloseTo(100); expect(b).toBeCloseTo(0.5);
  });
});
