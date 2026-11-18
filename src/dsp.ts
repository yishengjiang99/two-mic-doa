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
  /** PHAT peak height, ~1 for a clean single path, ~0 for noise. */
  peak: number;
  /** Correlation for lags center-maxLag..center+maxLag. */
  corr: Float64Array;
}

/**
 * GCC-PHAT between equal-length windows a and b. Searches lags in
 * [center - maxLag, center + maxLag] (center = calibration offset).
 */
export function gccPhat(a: ArrayLike<number>, b: ArrayLike<number>, maxLag: number, center = 0): GccResult {
  const n = a.length;
  let m = 1;
  while (m < 2 * n) m <<= 1; // zero-pad so circular correlation doesn't wrap
  const w = hann(n);
  const ar = new Float64Array(m), ai = new Float64Array(m);
  const br = new Float64Array(m), bi = new Float64Array(m);
  for (let i = 0; i < n; i++) { ar[i] = a[i] * w[i]; br[i] = b[i] * w[i]; }
  fft(ar, ai); fft(br, bi);
  // R = conj(A) * B / |conj(A) * B|  -> peak at +D when b[n] = a[n - D]
  for (let k = 0; k < m; k++) {
    const re = ar[k] * br[k] + ai[k] * bi[k];
    const im = ar[k] * bi[k] - ai[k] * br[k];
    const mag = Math.hypot(re, im) + 1e-12;
    ar[k] = re / mag; ai[k] = im / mag;
  }
  fft(ar, ai, true);
  const at = (lag: number) => ar[((lag % m) + m) % m] / m;

  const lim = n - 2;
  const c = Math.round(center);
  const lo = Math.max(-lim, c - Math.ceil(maxLag));
  const hi = Math.min(lim, c + Math.ceil(maxLag));
  const corr = new Float64Array(hi - lo + 1);
  let best = lo;
  for (let l = lo; l <= hi; l++) {
    corr[l - lo] = at(l);
    if (corr[l - lo] > corr[best - lo]) best = l;
  }
  const y0 = at(best - 1), y1 = at(best), y2 = at(best + 1);
  const den = y0 - 2 * y1 + y2;
  const frac = den < 0 ? Math.max(-0.5, Math.min(0.5, (0.5 * (y0 - y2)) / den)) : 0;
  return { lag: best + frac, peak: y1, corr };
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
