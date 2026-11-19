// GCC-PHAT time-delay estimation and DOA geometry. Pure functions, no DOM.

/** In-place iterative radix-2 FFT. n = re.length must be a power of two. inverse => unscaled IFFT. */
export function fft(re: Float64Array, im: Float64Array, inverse = false): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((inverse ? 2 : -2) * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci;
        const ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
        const t = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = t;
      }
    }
  }
}

const hannCache = new Map<number, Float64Array>();
export function hann(n: number): Float64Array {
  let w = hannCache.get(n);
  if (!w) {
    w = new Float64Array(n);
    for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
    hannCache.set(n, w);
  }
  return w;
}

export interface GccResult {
  /** Sub-sample lag of b relative to a. Positive => b is delayed (sound reached A first). */
  lag: number;
  /** Weighted-PHAT peak height, normalized by the total bin weight: ~1 for a clean single path, ~0 for noise. */
  peak: number;
  /** Correlation on the upsampled lag grid, center-maxLag..center+maxLag. */
  corr: Float64Array;
  /** True when the peak sits on the search boundary: the real peak is outside the window, so the lag is unreliable. */
  edge: boolean;
}

/** One channel's analysis frame: Hann-windowed FFT zero-padded to m = 2n, raw power, and RMS-matching gain. */
export interface Frame { n: number; re: Float64Array; im: Float64Array; pow: Float64Array; gain: number }

export function analyze(x: ArrayLike<number>): Frame {
  const n = x.length;
  let m = 1;
  while (m < 2 * n) m <<= 1; // zero-pad so circular correlation doesn't wrap
  const w = hann(n), re = new Float64Array(m), im = new Float64Array(m);
  let e = 0;
  for (let i = 0; i < n; i++) { re[i] = x[i] * w[i]; e += x[i] * x[i]; }
  fft(re, im);
  const pow = new Float64Array(m);
  for (let k = 0; k < m; k++) pow[k] = re[k] * re[k] + im[k] * im[k];
  return { n, re, im, pow, gain: 1 / (Math.sqrt(e / n) + 1e-12) };
}

/** Wiener-like SNR weight per bin: max(0, 1 - N/|X|^2). No noise estimate => all ones. */
export function snrMask(pow: Float64Array, noise?: Float64Array | null): Float64Array {
  const w = new Float64Array(pow.length).fill(1);
  if (noise) for (let k = 0; k < pow.length; k++) w[k] = Math.max(0, 1 - noise[k] / (pow[k] + 1e-30));
  return w;
}

/** Slow per-bin EMA of noise power, fed only with non-onset frames. */
export class NoiseFloor {
  psd: Float64Array | null = null;
  constructor(private k = 0.05) {}
  update(pow: Float64Array) {
    if (!this.psd || this.psd.length !== pow.length) { this.psd = Float64Array.from(pow); return; }
    for (let i = 0; i < pow.length; i++) this.psd[i] += this.k * (pow[i] - this.psd[i]);
  }
}

/** Gain-normalized, SNR-weighted cross spectrum conj(A)·B plus its per-bin weight. Sum several before gccFromCross. */
export interface Cross { re: Float64Array; im: Float64Array; w: Float64Array; n: number }

export function cross(A: Frame, B: Frame, wA?: Float64Array, wB?: Float64Array, minFreq = 0): Cross {
  const m = A.re.length, g = A.gain * B.gain, lo = Math.ceil(minFreq * m);
  const re = new Float64Array(m), im = new Float64Array(m), w = new Float64Array(m);
  for (let k = 0; k < m; k++) {
    if (Math.min(k, m - k) < lo) continue; // below the high-pass: hum can't vote after whitening
    const wk = (wA ? wA[k] : 1) * (wB ? wB[k] : 1);
    if (wk <= 0) continue;
    re[k] = g * (A.re[k] * B.re[k] + A.im[k] * B.im[k]);
    im[k] = g * (A.re[k] * B.im[k] - A.im[k] * B.re[k]);
    w[k] = wk;
  }
  return { re, im, w, n: A.n };
}

/** Sum of cross spectra (e.g. the last few onset frames); weights are averaged. */
export function sumCross(xs: Cross[]): Cross {
  const m = xs[0].re.length, re = new Float64Array(m), im = new Float64Array(m), w = new Float64Array(m);
  for (const x of xs) for (let k = 0; k < m; k++) { re[k] += x.re[k]; im[k] += x.im[k]; w[k] += x.w[k] / xs.length; }
  return { re, im, w, n: xs[0].n };
}

/**
 * PHAT-normalize a (summed) cross spectrum, weight by its SNR mask, IFFT zero-padded `up`× for a
 * 1/up-sample lag grid, then search [center - maxLag, center + maxLag] with parabolic interpolation.
 */
export function gccFromCross(x: Cross, maxLag: number, center = 0, up = 4): GccResult {
  const m = x.re.length, M = m * up, n = x.n;
  const re = new Float64Array(M), im = new Float64Array(M);
  let wsum = 0;
  for (let k = 0; k < m; k++) {
    const mag = Math.hypot(x.re[k], x.im[k]);
    if (!mag || !x.w[k]) continue;
    const s = x.w[k] / mag, dst = k < m / 2 ? k : k === m / 2 ? -1 : k + M - m;
    wsum += x.w[k];
    if (dst < 0) { re[k] += (x.re[k] * s) / 2; re[M - k] += (x.re[k] * s) / 2; continue; } // split Nyquist
    re[dst] = x.re[k] * s; im[dst] = x.im[k] * s;
  }
  fft(re, im, true);
  const norm = wsum || 1;
  const at = (j: number) => re[((j % M) + M) % M] / norm; // j on the fine grid: lag = j / up

  const lim = n - 2, c = Math.round(center);
  const lo = Math.max(-lim, c - Math.ceil(maxLag)) * up, hi = Math.min(lim, c + Math.ceil(maxLag)) * up;
  const corr = new Float64Array(hi - lo + 1);
  let best = lo;
  for (let j = lo; j <= hi; j++) {
    corr[j - lo] = at(j);
    if (corr[j - lo] > corr[best - lo]) best = j;
  }
  const y0 = at(best - 1), y1 = at(best), y2 = at(best + 1);
  const den = y0 - 2 * y1 + y2;
  const frac = den < 0 ? Math.max(-0.5, Math.min(0.5, (0.5 * (y0 - y2)) / den)) : 0;
  return { lag: (best + frac) / up, peak: y1, corr, edge: best === lo || best === hi };
}

/** Single-frame GCC-PHAT between equal-length windows a and b (no noise mask, no averaging). */
export function gccPhat(a: ArrayLike<number>, b: ArrayLike<number>, maxLag: number, center = 0, minFreq = 0, up = 4): GccResult {
  return gccFromCross(cross(analyze(a), analyze(b), undefined, undefined, minFreq), maxLag, center, up);
}

/** Speed of sound in air (m/s) at temperature tC (°C). */
export const speedOfSound = (tC: number) => 331.3 * Math.sqrt(1 + tC / 273.15);

/** Max physical lag in samples for baseline d (m). */
export const maxLagSamples = (d: number, c: number, fs: number) => (d / c) * fs;

/**
 * Angle in degrees from the baseline axis pointing toward mic A.
 * 0° = source on A's end, 90° = broadside, 180° = B's end.
 * dt > 0 means sound reached A first.
 */
export function angleDeg(dt: number, d: number, c: number): number {
  const x = Math.max(-1, Math.min(1, (c * dt) / d));
  return (Math.acos(x) * 180) / Math.PI;
}

/** Least-squares line through (t, y); returns slope 0 if too few points or span. */
export function fitLine(t: number[], y: number[], minSpan = 3): { a: number; b: number } {
  const n = t.length;
  const mt = t.reduce((s, v) => s + v, 0) / n;
  const my = y.reduce((s, v) => s + v, 0) / n;
  if (n < 3 || Math.max(...t) - Math.min(...t) < minSpan) return { a: median(y), b: 0 };
  let sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) { sxy += (t[i] - mt) * (y[i] - my); sxx += (t[i] - mt) ** 2; }
  const b = sxy / sxx;
  return { a: my - b * mt, b };
}

export function median(v: number[]): number {
  const s = [...v].sort((x, y) => x - y);
  const h = s.length >> 1;
  return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
}

/** 4th-order Butterworth high-pass (two biquads), stateful across calls so chunked input filters seamlessly. */
export class HighPass {
  private c: number[][] = [];
  private z: Float64Array;
  constructor(fc: number, fs: number) {
    const w = (2 * Math.PI * fc) / fs, cw = Math.cos(w);
    for (const q of [0.5412, 1.3066]) {
      const al = Math.sin(w) / (2 * q), a0 = 1 + al;
      this.c.push([(1 + cw) / 2 / a0, -(1 + cw) / a0, (1 + cw) / 2 / a0, (-2 * cw) / a0, (1 - al) / a0]);
    }
    this.z = new Float64Array(2 * this.c.length);
  }
  process(x: ArrayLike<number>, out = new Float32Array(x.length)): Float32Array {
    const z = this.z;
    for (let i = 0; i < x.length; i++) {
      let v = x[i];
      this.c.forEach(([b0, b1, b2, a1, a2], s) => {
        const y = b0 * v + z[2 * s];
        z[2 * s] = b1 * v - a1 * y + z[2 * s + 1];
        z[2 * s + 1] = b2 * v - a2 * y;
        v = y;
      });
      out[i] = v;
    }
    return out;
  }
}

/** Magnitude spectrum in dBFS (sine at full scale ≈ 0 dB) from an analysis frame, bins 0..m/2 (spacing fs/m). */
export function spectrumDb(f: Frame): Float64Array {
  const m = f.pow.length, out = new Float64Array(m / 2 + 1), g = (4 / f.n) ** 2; // 2/n one-sided, ×2 Hann gain
  for (let k = 0; k <= m / 2; k++) out[k] = 10 * Math.log10(f.pow[k] * g + 1e-24);
  return out;
}

export type Vec2 = [number, number];

/**
 * Unit vector toward a far-field source at azimuth az (deg), x = right, y = ahead.
 * 0° = left (Mic A side), 90° = ahead, 180° = right, 270° = behind; same as the 2-mic angle.
 */
export const azVec = (az: number): Vec2 => { const t = (az * Math.PI) / 180; return [-Math.cos(t), Math.sin(t)]; };
export const azOf = (u: Vec2) => ((Math.atan2(u[1], -u[0]) * 180) / Math.PI + 360) % 360;

/** Plane-wave TDOA t_j - t_i (s) for source direction u. Positive => sound reached mic i first. */
export const tdoa = (pi: Vec2, pj: Vec2, u: Vec2, c: number) => ((pi[0] - pj[0]) * u[0] + (pi[1] - pj[1]) * u[1]) / c;

/** Equilateral triangle, side d: A left, B right, C ahead. */
export const trianglePositions = (d: number): Vec2[] => [[-d / 2, 0], [d / 2, 0], [0, (d * Math.sqrt(3)) / 2]];

export interface PairTdoa { i: number; j: number; tau: number; w?: number }

/**
 * Weighted least squares for the in-plane direction from pairwise TDOAs:
 * c·tau_ij = (p_i - p_j)·u. Returns azimuth (0-360), u (|u| < 1 if the source is off-plane),
 * and RMS residual in seconds. az is NaN when the mics are collinear.
 */
export function solveDoa(pos: Vec2[], pairs: PairTdoa[], c: number): { az: number; u: Vec2; resid: number } {
  let sxx = 0, sxy = 0, syy = 0, bx = 0, by = 0;
  for (const { i, j, tau, w = 1 } of pairs) {
    const dx = pos[i][0] - pos[j][0], dy = pos[i][1] - pos[j][1], r = c * tau;
    sxx += w * dx * dx; sxy += w * dx * dy; syy += w * dy * dy; bx += w * dx * r; by += w * dy * r;
  }
  const det = sxx * syy - sxy * sxy;
  if (Math.abs(det) < 1e-9 * (sxx + syy) ** 2) return { az: NaN, u: [NaN, NaN], resid: NaN };
  const u: Vec2 = [(syy * bx - sxy * by) / det, (sxx * by - sxy * bx) / det];
  let e = 0;
  for (const { i, j, tau } of pairs) e += (tdoa(pos[i], pos[j], u, c) - tau) ** 2;
  return { az: azOf(u), u, resid: Math.sqrt(e / pairs.length) };
}
