# two-mic-doa

Browser direction-of-arrival finder for two microphones using GCC-PHAT.

**Live:** https://yishengjiang99.github.io/two-mic-doa/

## How it works
- Captures two mics (or L/R of one stereo device) via `getUserMedia` with echo cancellation, noise suppression and AGC off.
- An AudioWorklet (`public/capture-worklet.js`) posts sample-aligned 4096-sample windows (8192 above 60 kHz), 50% hop.
- `src/dsp.ts`: Hann window → FFT (zero-padded 2×) → PHAT weighting → IFFT → peak search within ±d/c around the calibration offset → parabolic sub-sample interpolation. The delay is EMA-smoothed; frames below the confidence threshold are ignored.
- Angle = acos(clamp(cΔt/d)), c from temperature (331.3·√(1+T/273.15)).

**Angle convention:** 0° = toward Mic A (left), 90° = broadside, 180° = toward Mic B (right). Two mics can't resolve front/back.

## Calibration
Two separate USB mics have independent clocks: an unknown fixed offset (often many ms) and slow drift. Click **Calibrate**, then clap (or **Play clicks** from a speaker) at broadside for 6 s. The app searches a wide lag range, rejects outliers, fits offset + drift (samples/min), and subtracts it from every later estimate. Recalibrate periodically. A single 2-input audio interface (shared clock) is far more reliable.

## Dev
```sh
npm install
npm run dev     # http://localhost:5173/two-mic-doa/
npm test        # GCC-PHAT unit tests (integer/fractional, ± delays)
npm run build
```
Deploys to GitHub Pages from `main` via `.github/workflows/pages.yml`.
