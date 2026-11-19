import { describe, it, expect } from 'vitest';
import { maxLagSamples, corrAt, heat2, srp, ageAlpha, fft, gccPhat, angleDeg, speedOfSound, fitLine, HighPass, analyze, snrMask, cross, gccFromCross, NoiseFloor, solveDoa, tdoa, azVec, trianglePositions, type Vec2 } from '../src/dsp';

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
    expect(r.corr.length).toBe(241);
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

/** Band-limited white noise (band in fractions of Nyquist), length m, via FFT masking. */
function bandNoise(m: number, lo: number, hi: number, seed: number) {
  const r = rng(seed), re = Float64Array.from({ length: m }, r), im = new Float64Array(m);
  fft(re, im);
  for (let k = 0; k < m; k++) { const f = Math.min(k, m - k) / (m / 2); if (f < lo || f > hi) re[k] = im[k] = 0; }
  fft(re, im, true);
  return re.map((v) => v / m);
}
/** Channels of one source with fractional delays (samples), FFT phase shift on a long buffer. */
function delayedSet(n: number, delays: number[], seed = 5) {
  const m = n * 4, r = rng(seed), src = Float64Array.from({ length: m }, r), si = new Float64Array(m);
  fft(src, si);
  return delays.map((dl) => {
    const re = new Float64Array(m), im = new Float64Array(m);
    for (let k = 0; k < m; k++) {
      const f = k <= m / 2 ? k : k - m, ph = (-2 * Math.PI * f * dl) / m, c = Math.cos(ph), s = Math.sin(ph);
      re[k] = src[k] * c - si[k] * s; im[k] = src[k] * s + si[k] * c;
    }
    im[m / 2] = 0;
    fft(re, im, true);
    return Float64Array.from({ length: n }, (_, i) => re[n + i] / m);
  });
}
const rmsOf = (x: ArrayLike<number>, from = 0) => { let s = 0; for (let i = from; i < x.length; i++) s += x[i] ** 2; return Math.sqrt(s / (x.length - from)); };
const sine = (n: number, f: number, fs: number, a = 1, dc = 0) => Float64Array.from({ length: n }, (_, i) => dc + a * Math.sin((2 * Math.PI * f * i) / fs));

describe('high-pass', () => {
  const fs = 48000, n = 48000, skip = 4800; // judge after 100 ms of settling
  it('removes DC and 60 Hz hum (> 20 dB)', () => {
    const y = new HighPass(120, fs).process(sine(n, 60, fs, 1, 0.5));
    expect(rmsOf(y, skip)).toBeLessThan(0.1 * rmsOf(sine(n, 60, fs), skip));
  });
  it('passes 1 kHz within 0.5 dB', () => {
    const x = sine(n, 1000, fs), y = new HighPass(120, fs).process(x);
    expect(Math.abs(20 * Math.log10(rmsOf(y, skip) / rmsOf(x, skip)))).toBeLessThan(0.5);
  });
  it('chunked processing matches one pass', () => {
    const x = sine(4096, 300, fs, 1, 0.2), one = new HighPass(120, fs).process(x);
    const hp = new HighPass(120, fs), parts = [hp.process(x.subarray(0, 1000)), hp.process(x.subarray(1000))];
    const two = new Float32Array(4096); two.set(parts[0]); two.set(parts[1], 1000);
    expect(Math.max(...one.map((v, i) => Math.abs(v - two[i])))).toBeLessThan(1e-6);
  });
});

describe('SNR weighting and boundary', () => {
  it('snrMask = max(0, 1 - N/|X|^2)', () => {
    expect([...snrMask(Float64Array.of(4, 1, 0.5), Float64Array.of(1, 1, 1))]).toEqual([0.75, 0, 0]);
    expect([...snrMask(Float64Array.of(2, 3))]).toEqual([1, 1]);
  });
  it('mask from a learned noise floor lifts confidence when independent noise fills most bins', () => {
    const n = 4096, m = 4 * n, off = n, D = 6;
    const sig = bandNoise(m, 0, 0.25, 1); // shared low band
    const noiseA = (s: number) => bandNoise(m, 0.3, 1, s).map((v) => 8 * v); // loud, independent, upper band
    const na = noiseA(11), nb = noiseA(12);
    const a = Float64Array.from({ length: n }, (_, i) => sig[off + i] + na[off + i]);
    const b = Float64Array.from({ length: n }, (_, i) => sig[off + i - D] + nb[off + i]);
    const nfA = new NoiseFloor(0.5), nfB = new NoiseFloor(0.5);
    for (let s = 0; s < 6; s++) { // noise-only frames
      nfA.update(analyze(noiseA(100 + s).subarray(off, off + n)).pow);
      nfB.update(analyze(noiseA(200 + s).subarray(off, off + n)).pow);
    }
    const A = analyze(a), B = analyze(b);
    const plain = gccFromCross(cross(A, B), 30);
    const masked = gccFromCross(cross(A, B, snrMask(A.pow, nfA.psd), snrMask(B.pow, nfB.psd)), 30);
    expect(Math.abs(masked.lag - D)).toBeLessThan(0.3);
    expect(masked.peak).toBeGreaterThan(0.6);
    expect(masked.peak).toBeGreaterThan(2 * plain.peak);
  });
  it('flags a peak on the search boundary', () => {
    // Low-band source + faint white noise; the SNR mask keeps only the band, so the main lobe is wide
    // and still rising at the boundary when the true lag is outside the search window.
    const n = 2048, sig = bandNoise(4 * n, 0, 0.1, 3), r1 = rng(21), r2 = rng(22), r3 = rng(23);
    const a = Float64Array.from({ length: n }, (_, i) => sig[n + i] + 1e-3 * r1());
    const b = Float64Array.from({ length: n }, (_, i) => sig[n + i - 13] + 1e-3 * r2());
    const nf = new NoiseFloor(0.2);
    for (let s = 0; s < 20; s++) nf.update(analyze(Float64Array.from({ length: n }, () => 1e-3 * r3())).pow);
    const A = analyze(a), B = analyze(b), x = cross(A, B, snrMask(A.pow, nf.psd), snrMask(B.pow, nf.psd));
    expect(gccFromCross(x, 10).edge).toBe(true);
    const r = gccFromCross(x, 30);
    expect(r.edge).toBe(false);
    expect(Math.abs(r.lag - 13)).toBeLessThan(0.2);
  });
  it('4x upsampled lag is within 0.1 sample', () => {
    for (const dl of [2.3, -5.6, 7.15]) {
      const { a, b } = delayedPair(4096, dl);
      expect(Math.abs(gccPhat(a, b, 30).lag - dl)).toBeLessThan(0.1);
    }
  });
});

describe('3-mic solver', () => {
  const c = speedOfSound(20), fs = 48000;
  const pairs = [[0, 1], [0, 2], [1, 2]] as const;
  const angErr = (x: number, y: number) => Math.abs(((x - y + 540) % 360) - 180);
  const taus = (pos: Vec2[], az: number, noise = 0, r = rng(9)) =>
    pairs.map(([i, j]) => ({ i, j, tau: tdoa(pos[i], pos[j], azVec(az), c) + (noise * r()) / fs }));
  it('matches the 2-mic convention', () => {
    const [A, B] = trianglePositions(0.2);
    for (const az of [0, 30, 90, 150, 180]) expect(angleDeg(tdoa(A, B, azVec(az), c), 0.2, c)).toBeCloseTo(az, 6);
  });
  it('round-trips every azimuth (equilateral)', () => {
    const pos = trianglePositions(0.2);
    for (let az = 0; az < 360; az += 7) expect(angErr(solveDoa(pos, taus(pos, az), c).az, az)).toBeLessThan(1e-6);
  });
  it('within 1° with ±0.25-sample lag noise, arbitrary triangle', () => {
    const pos: Vec2[] = [[-0.15, 0], [0.12, 0.02], [0.03, 0.25]];
    for (let az = 3; az < 360; az += 11) expect(angErr(solveDoa(pos, taus(pos, az, 0.25), c).az, az)).toBeLessThan(1);
  });
  it('collinear mics give NaN', () => {
    const pos: Vec2[] = [[0, 0], [0.1, 0], [0.2, 0]];
    expect(solveDoa(pos, taus(pos, 40), c).az).toBeNaN();
  });
  it('end to end: GCC-PHAT on 3 synthetic channels recovers the azimuth within 2°', () => {
    const pos = trianglePositions(0.3);
    for (const az of [20, 135, 250, 315]) {
      const t = pos.map((p) => tdoa([0, 0], p, azVec(az), c) * fs); // arrival delay of each mic vs origin, samples
      const ch = delayedSet(4096, t);
      const est = pairs.map(([i, j]) => ({ i, j, tau: gccPhat(ch[i], ch[j], 40).lag / fs }));
      expect(angErr(solveDoa(pos, est, c).az, az)).toBeLessThan(2);
    }
  });
});

describe('direction plot', () => {
  const c = speedOfSound(20), fs = 48000, d = 0.2;
  const argmax = (h: Float64Array) => h.indexOf(Math.max(...h));
  it('corrAt interpolates on the lag grid', () => {
    const r = { corr: Float64Array.of(0, 1, 0.5), lo: -1, up: 2 }; // lags -1, -0.5, 0
    expect(corrAt(r, -0.75)).toBeCloseTo(0.5);
    expect(corrAt(r, -0.25)).toBeCloseTo(0.75);
    expect(corrAt(r, 0)).toBeCloseTo(0.5);
    expect(corrAt(r, 3)).toBe(0);
  });
  it('2-mic heat maps lag to angle by acos(c·tau/d)', () => {
    for (const lag of [0, 10, -10, 20, -25]) {
      // synthetic correlation peaked at `lag` on the app's ¼-sample grid, ±30 samples
      const lo = -30, up = 4, corr = Float64Array.from({ length: 60 * up + 1 }, (_, i) => Math.max(0, 1 - Math.abs(lo + i / up - lag) / 3));
      const expected = angleDeg(lag / fs, d, c);
      expect(Math.abs(argmax(heat2({ corr, lo, up }, 0, d, c, fs, 1)) - expected)).toBeLessThanOrEqual(1);
      // a calibration offset shifts the lag axis, not the angle
      const shifted = { corr, lo: lo + 7, up };
      expect(Math.abs(argmax(heat2(shifted, 7, d, c, fs, 1)) - expected)).toBeLessThanOrEqual(1);
    }
  });
  it('2-mic heat peaks at the GCC-PHAT angle', () => {
    const { a, b } = delayedPair(4096, 12.4);
    const r = gccPhat(a, b, maxLagSamples(d, c, fs) + 2);
    expect(Math.abs(argmax(heat2(r, 0, d, c, fs)) - angleDeg(r.lag / fs, d, c))).toBeLessThanOrEqual(1);
  });
  it('3-mic steered response peaks at the true azimuth', () => {
    const pos = trianglePositions(0.3);
    for (const az of [40, 200, 300]) {
      const ch = delayedSet(4096, pos.map((p) => tdoa([0, 0], p, azVec(az), c) * fs));
      const rs = ([[0, 1], [0, 2], [1, 2]] as const).map(([i, j]) => ({ r: gccPhat(ch[i], ch[j], 40), i, j, center: 0 }));
      const h = srp(rs, pos, c, fs, 2), err = Math.abs(((argmax(h) * 2 - az + 540) % 360) - 180);
      expect(err).toBeLessThanOrEqual(3);
    }
  });
  it('age alpha fades linearly to 0 at the lifetime', () => {
    expect(ageAlpha(0)).toBe(1);
    expect(ageAlpha(1)).toBeCloseTo(0.5);
    expect(ageAlpha(2)).toBe(0);
    expect(ageAlpha(5)).toBe(0);
    expect(ageAlpha(-1)).toBe(1);
    expect(ageAlpha(0.5, 1)).toBeCloseTo(0.5);
  });
});
