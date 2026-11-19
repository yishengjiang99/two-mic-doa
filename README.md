# two-mic-doa

Browser direction-of-arrival finder for two or three microphones using GCC-PHAT. Two mics give a 0–180° angle; a third mic gives a full 0–360° azimuth.

**Live:** https://yishengjiang99.github.io/two-mic-doa/

## How it works
- Captures 2 or 3 mics with `getUserMedia`, with echo cancellation, noise suppression and AGC off. Each mic can be its own device or an input of one multichannel device (Mic A device inputs 2 and 3).
- The AudioWorklet (`public/capture-worklet.js`) posts sample-aligned hops. The main thread builds 4096-sample windows (8192 above 60 kHz) with a 50% hop.
- Live waveforms are drawn with each channel scaled to its own peak (dBFS shown), next to a log-frequency spectrum per channel.

## Signal chain
1. Capture with no echo cancellation, noise suppression or AGC.
2. 4th-order Butterworth high-pass at 120 Hz per channel, run as a continuous stream to remove DC drift and mains hum.
3. Hann window, then an FFT zero-padded to 2× (this FFT is reused for the spectra).
4. Onset gate: quiet frames update a slow per-bin noise PSD (EMA) and produce no estimate.
5. Gain normalization: each channel is scaled to unit RMS before the cross spectrum.
6. Cross spectrum conj(A)·B, weighted per bin by the SNR masks max(0, 1 − N/|X|²) of both channels. Bins below 120 Hz are dropped.
7. The cross spectra of the last 4 onset frames (≤1.5 s old) are summed.
8. PHAT normalization, then an IFFT zero-padded 4× (¼-sample lag grid).
9. Peak search within ±d/c (+2 samples) around the calibration offset, then parabolic interpolation. A peak on the search boundary is rejected.
10. Confidence is the weighted-PHAT peak, which is the minimum over pairs. Estimates below the threshold are ignored, and the lags are EMA-smoothed.
11. Direction: with 2 mics, angle = acos(clamp(cΔt/d)). With 3 mics, least squares on τ_AB, τ_AC, τ_BC (plane wave: c·τ_ij = (p_i − p_j)·u) gives u, and the azimuth comes from u. c is computed from temperature as 331.3·√(1+T/273.15).

**Angle convention:** 0° = toward Mic A (left), 90° = ahead, 180° = toward Mic B (right), 270° = behind. Two mics can't resolve front from back. Mic positions are entered as `x,y x,y x,y` (x right, y ahead). Leave them blank for an equilateral triangle with side d and C ahead.

## Calibration
Separate USB mics have independent clocks: an unknown fixed offset (often many ms) and slow drift. Click **Calibrate**, then clap (or **Play clicks** from a speaker) at the cal azimuth (default 90°, straight ahead) for 6 s. For each of B and C relative to A, the app searches a wide lag range, subtracts the expected geometric lag, rejects outliers, fits offset + drift (samples/min), and removes it from every later estimate (the BC offset is C − B). Recalibrate periodically. A single multi-input audio interface (shared clock) is far more reliable.

## Dev
```sh
npm install
npm run dev     # http://localhost:5173/two-mic-doa/
npm test        # GCC-PHAT, SNR mask, high-pass and 3-mic solver tests
npm run build
```
Deploys to GitHub Pages from `main` via `.github/workflows/pages.yml`.
