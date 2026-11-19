import { heat2, srp, ageAlpha, analyze, snrMask, cross, sumCross, gccFromCross, NoiseFloor, type Cross, type Frame, speedOfSound, maxLagSamples, angleDeg, fitLine, median, HighPass, spectrumDb, solveDoa, tdoa, azVec, trianglePositions, type Vec2 } from './dsp';
// Hashed by Vite so a cached old worklet can never pair with new page code.
import workletUrl from './capture-worklet.js?url';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const devA = $<HTMLSelectElement>('devA'), devB = $<HTMLSelectElement>('devB'), devC = $<HTMLSelectElement>('devC');
const baseline = $<HTMLInputElement>('baseline'), temp = $<HTMLInputElement>('temp'), posIn = $<HTMLInputElement>('positions'), calAz = $<HTMLInputElement>('calAz');
const thresh = $<HTMLInputElement>('thresh'), alpha = $<HTMLInputElement>('alpha');
const btn = { start: $<HTMLButtonElement>('start'), stop: $<HTMLButtonElement>('stop'), cal: $<HTMLButtonElement>('cal'), click: $<HTMLButtonElement>('click'), resetCal: $<HTMLButtonElement>('resetCal') };
const out = { angle: $('angle'), delayMs: $('delayMs'), delaySmp: $('delaySmp'), conf: $('conf'), fs: $('fs'), offset: $('offset'), error: $('error'), cal: $('calStatus') };
const view = $<HTMLCanvasElement>('view'), corrCv = $<HTMLCanvasElement>('corr');
const waveCv = $<HTMLCanvasElement>('wave'), specCv = $<HTMLCanvasElement>('spec');

const RAW = { echoCancellation: false, noiseSuppression: false, autoGainControl: false };
const CAL_SECONDS = 6;
const HP_HZ = 120; // high-pass cutoff: kills DC drift and mains hum before analysis and drawing
const NAMES = ['A', 'B', 'C'];
const AVG_FRAMES = 4, AVG_SECONDS = 1.5; // cross spectra averaged over the last few onset frames
const IN2 = '#2', IN3 = '#3'; // select values: "input 2/3 of the Mic A device"

let ctx: AudioContext | null = null;
let streams: MediaStream[] = [];
let nMics = 2;
let probedCh = 1; // channels the Mic A device delivers
let pairs: [number, number][] = [[0, 1]];
let ema: (number | null)[] = [null]; // per pair: smoothed delay, samples, calibration removed
let lastConf = 0;
let floor = 1e-4; // running RMS noise floor for transient gating
// Calibration per mic k ≥ 1, relative to A: offset_k(t) = a + b * t (samples; t = AudioContext seconds)
let cal: { a: number; b: number }[] | null = null;
let calRun: { until: number; t: number[]; lag: number[][] } | null = null;
let clickTimer = 0;
let specEma: Float64Array[] = [];
let noise: NoiseFloor[] = [];
// Direction plot state: smoothed direction, live correlation heat over angle, and recent confident hits.
let dirAng: number | null = null;
let heat: Float64Array | null = null;
let hits: { ang: number; t: number }[] = [];
let lastHit = -Infinity;
const FADE_S = 2;
const now = () => performance.now() / 1000;
let ring: { t: number; x: Cross[] }[] = []; // recent onset frames: one cross spectrum per pair

const showError = (m: string) => { out.error.textContent = m; };
const num = (el: HTMLInputElement, dflt: number) => (Number.isFinite(+el.value) && +el.value > 0 ? +el.value : dflt);
const css = (v: string) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const colors = () => [css('--acc'), '#f59e0b', '#22c55e'];

/** Mic positions (m). 2 mics: baseline d on the x axis. 3 mics: user "x,y x,y x,y", else equilateral side d. */
function micPositions(n: number, d: number): Vec2[] {
  if (n === 2) return [[-d / 2, 0], [d / 2, 0]];
  const v = posIn.value.trim().split(/[\s;]+/).filter(Boolean).map((s) => s.split(',').map(Number));
  return v.length === 3 && v.every((p) => p.length === 2 && p.every(Number.isFinite)) ? (v as Vec2[]) : trianglePositions(d);
}
const params = () => {
  const d = num(baseline, 0.2);
  const c = speedOfSound(Number.isFinite(+temp.value) ? +temp.value : 20);
  return { d, c, pos: micPositions(nMics, d) };
};
const dist = (p: Vec2, q: Vec2) => Math.hypot(p[0] - q[0], p[1] - q[1]);

for (const [el, o] of [[thresh, 'threshOut'], [alpha, 'alphaOut']] as const)
  el.addEventListener('input', () => { $(o).textContent = (+el.value).toFixed(2); });

let devs: MediaDeviceInfo[] = [];
async function listDevices() {
  devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'audioinput');
  fillSelects();
  // Before permission, Safari/Firefox hide devices and Chrome hides ids and labels.
  const locked = !devs.length || devs.every((d) => !d.deviceId || !d.label);
  allowBtn.classList.toggle('hidden', !locked);
  if (locked) showError('Click "Allow microphones" so the browser lists your mics.');
  else if (out.error.textContent?.startsWith('Click "Allow')) showError('');
}

function fillSelects() {
  const sets: [HTMLSelectElement, string | undefined, [string, string][]][] = [
    [devA, devs[0]?.deviceId, []],
    [devB, devs[1]?.deviceId, probedCh >= 2 ? [['Mic A device, input 2', IN2]] : []],
    [devC, '', [['None (2 mics)', ''], ...(probedCh >= 3 ? [['Mic A device, input 3', IN3]] as [string, string][] : [])]],
  ];
  for (const [sel, dflt, extra] of sets) {
    const prev = sel.value;
    sel.innerHTML = '';
    for (const [label, v] of extra) sel.add(new Option(label, v));
    devs.forEach((d, i) => sel.add(new Option(d.label || `Microphone ${i + 1}`, d.deviceId)));
    const ok = (v: string) => Array.from(sel.options).some((o) => o.value === v);
    sel.value = ok(prev) ? prev : dflt !== undefined && ok(dflt) ? dflt : sel.options[0]?.value ?? '';
  }
  if (!ctx) { nMics = devC.value ? 3 : 2; drawEmpty(); }
}

const allowBtn = $<HTMLButtonElement>('allow');
async function allowMics() {
  showError('');
  try {
    const s = await navigator.mediaDevices.getUserMedia({ audio: RAW });
    s.getTracks().forEach((t) => t.stop());
    await listDevices();
    await probeChannels();
  } catch (e) { showError(describeError(e)); }
}

/** Opens device A asking for 3 channels; offers its inputs 2/3 as Mic B/C if it delivers them. */
async function probeChannels() {
  if (!devA.value || !navigator.mediaDevices) return;
  try {
    const s = await navigator.mediaDevices.getUserMedia({ audio: { deviceId: { exact: devA.value }, channelCount: { ideal: 3 }, ...RAW } });
    probedCh = s.getAudioTracks()[0]?.getSettings().channelCount ?? 1;
    s.getTracks().forEach((t) => t.stop());
  } catch { probedCh = 1; /* probe is best-effort */ }
  fillSelects();
}

async function openMic(id: string, channels: number) {
  return navigator.mediaDevices.getUserMedia({ audio: { deviceId: { exact: id }, channelCount: { exact: channels }, ...RAW } });
}

function describeError(e: unknown): string {
  const n = (e as DOMException)?.name;
  if (n === 'NotAllowedError') return 'Microphone permission denied. Allow mic access and try again.';
  if (n === 'NotFoundError') return 'Selected microphone not found. Re-select devices.';
  if (n === 'OverconstrainedError') return 'Device rejected the requested settings (channel count). Pick separate devices instead.';
  if (n === 'NotReadableError') return 'Microphone is busy or unavailable (in use by another app?).';
  return `Error: ${(e as Error)?.message ?? e}`;
}

/** Which device and channel feeds each mic. */
function sources(): { dev: string; ch: number }[] | string {
  const pick = (v: string) => (v === IN2 ? { dev: devA.value, ch: 1 } : v === IN3 ? { dev: devA.value, ch: 2 } : { dev: v, ch: 0 });
  const s = [{ dev: devA.value, ch: 0 }, pick(devB.value), ...(devC.value ? [pick(devC.value)] : [])];
  const keys = s.map((x) => `${x.dev}#${x.ch}`);
  return new Set(keys).size === keys.length ? s : 'Pick a different device (or input) for each mic.';
}

async function start() {
  showError('');
  if (!window.isSecureContext || !navigator.mediaDevices) return showError('Needs HTTPS (or localhost) for microphone access.');
  if (typeof AudioWorkletNode === 'undefined') return showError('This browser lacks AudioWorklet support.');
  const srcs = sources();
  if (typeof srcs === 'string') return showError(srcs);
  nMics = srcs.length;
  pairs = nMics === 3 ? [[0, 1], [0, 2], [1, 2]] : [[0, 1]];
  ema = pairs.map(() => null);
  noise = Array.from({ length: nMics }, () => new NoiseFloor());
  dirAng = null; heat = null; hits = []; lastHit = -Infinity;
  ring = [];
  if (cal && cal.length !== nMics - 1) cal = null;
  btn.start.disabled = true;
  try {
    ctx = new AudioContext({ latencyHint: 'interactive' });
    await ctx.audioWorklet.addModule(workletUrl);
    const size = ctx.sampleRate > 60000 ? 8192 : 4096, hop = size / 2;
    const node = new AudioWorkletNode(ctx, 'capture', {
      numberOfInputs: 1, numberOfOutputs: 1, channelCount: nMics, channelCountMode: 'explicit', channelInterpretation: 'discrete',
      processorOptions: { channels: nMics, hop },
    });
    const merger = ctx.createChannelMerger(nMics);
    const ids = [...new Set(srcs.map((s) => s.dev))];
    streams = [];
    for (const id of ids) {
      const mine = srcs.map((s, k) => [s, k] as const).filter(([s]) => s.dev === id);
      const chans = Math.max(...mine.map(([s]) => s.ch)) + 1;
      const stream = await openMic(id, chans);
      streams.push(stream);
      const src = ctx.createMediaStreamSource(stream);
      if (chans === 1) src.connect(merger, 0, mine[0][1]);
      else {
        const split = ctx.createChannelSplitter(chans);
        src.connect(split);
        for (const [s, k] of mine) split.connect(merger, s.ch, k);
      }
    }
    merger.connect(node);
    for (const s of streams) for (const t of s.getTracks()) t.onended = () => { showError('A microphone disconnected.'); stop(); };
    const mute = ctx.createGain(); mute.gain.value = 0; // keep the node pulled without monitoring
    node.connect(mute).connect(ctx.destination);
    // High-pass each channel as a continuous stream, then slide it into a size-sample window.
    const fs = ctx.sampleRate;
    const hp = srcs.map(() => new HighPass(HP_HZ, fs));
    const win = srcs.map(() => new Float32Array(size));
    let filled = 0;
    specEma = [];
    node.port.onmessage = (ev) => {
      const { ch, t } = ev.data as { ch?: Float32Array[]; t: number };
      if (!Array.isArray(ch) || ch.length !== nMics) { showError('Audio worklet is out of date. Reload the page.'); return stop(); }
      ch.forEach((x, k) => { win[k].copyWithin(0, x.length); hp[k].process(x, win[k].subarray(size - x.length)); });
      filled = Math.min(size, filled + hop);
      if (filled === size) onWindow(win, t, size);
    };
    await ctx.resume();
    out.fs.textContent = `${fs} Hz`;
    await listDevices(); // labels become available after permission
    btn.stop.disabled = btn.cal.disabled = btn.click.disabled = false;
    btn.resetCal.disabled = !cal;
    if (ids.length > 1 && !cal) out.cal.textContent = 'Separate devices: calibrate now (clap at the calibration azimuth) to remove the clock offsets.';
  } catch (e) {
    showError(describeError(e));
    stop();
  }
}

function stop() {
  streams.forEach((s) => s.getTracks().forEach((t) => { t.onended = null; t.stop(); }));
  streams = [];
  ctx?.close().catch(() => {});
  ctx = null; calRun = null;
  clearInterval(clickTimer); clickTimer = 0;
  btn.start.disabled = false;
  btn.stop.disabled = btn.cal.disabled = btn.click.disabled = true;
  btn.click.textContent = 'Play clicks';
}

function rms(x: Float32Array) { let s = 0; for (const v of x) s += v * v; return Math.sqrt(s / x.length); }
function peakAbs(x: Float32Array) { let m = 0; for (const v of x) m = Math.max(m, Math.abs(v)); return m; }
const calCenter = (k: number, t: number) => (k === 0 || !cal ? 0 : cal[k - 1].a + cal[k - 1].b * t);

function onWindow(chs: Float32Array[], t: number, size: number) {
  if (!ctx) return;
  const fs = ctx.sampleRate, lo = HP_HZ / fs;
  const frames = chs.map(analyze);
  try { drawWave(chs); drawSpectra(frames); } catch (e) { showError(`Drawing failed: ${(e as Error).message}`); }
  const { d, c, pos } = params();
  const level = Math.min(...chs.map(rms));
  const onset = Math.min(...chs.map(peakAbs)) > 6 * floor;
  floor = level < floor ? 0.9 * floor + 0.1 * level : floor * 1.002 + 1e-7; // tracks quiet level
  // Correlation per pair for frames xs, searched ±(d/c + 2) around the calibrated offset.
  const search = (xs: Cross[]) => pairs.map(([i, j], p) => {
    const center = calCenter(j, t) - calCenter(i, t), phys = maxLagSamples(dist(pos[i], pos[j]), c, fs);
    const r = gccFromCross(xs[p], phys + 2, center);
    return { r, i, j, center, rel: Math.max(-phys, Math.min(phys, r.lag - center)) };
  });
  const setHeat = (rs: ReturnType<typeof search>) => { heat = nMics === 2 ? heat2(rs[0].r, rs[0].center, d, c, fs, 1) : srp(rs, pos, c, fs, 2); };
  const angleOf = (lags: number[]) =>
    nMics === 2 ? angleDeg(lags[0] / fs, d, c) : solveDoa(pos, pairs.map(([i, j], p) => ({ i, j, tau: lags[p] / fs })), c).az;

  if (!onset) { // quiet frame: learn the noise floor, keep the last estimate, show raw (unmasked) heat
    frames.forEach((f, k) => noise[k].update(f.pow));
    lastConf = 0; out.conf.textContent = '0.00';
    if (!calRun) setHeat(search(pairs.map(([i, j]) => cross(frames[i], frames[j], undefined, undefined, lo))));
    return drawDir();
  }
  const masks = frames.map((f, k) => snrMask(f.pow, noise[k].psd));
  const xc = (i: number, j: number) => cross(frames[i], frames[j], masks[i], masks[j], lo);

  if (calRun) {
    // Wide search against A, one frame per hit: unknown device offsets can be many ms. Subtract the cal direction's geometry.
    const u = azVec(Number.isFinite(+calAz.value) ? +calAz.value : 90);
    const rs = chs.slice(1).map((_, k) => gccFromCross(xc(0, k + 1), size / 4, 0));
    drawCorr(rs[0].corr);
    if (rs.every((r) => !r.edge && r.peak >= Math.max(+thresh.value, 0.2))) {
      calRun.t.push(t);
      rs.forEach((r, k) => calRun!.lag[k].push(r.lag - tdoa(pos[0], pos[k + 1], u, c) * fs));
    }
    out.cal.textContent = `Calibrating… clap or click at ${calAz.value || 90}°. ${calRun.t.length} hits, ${Math.max(0, calRun.until - t).toFixed(1)} s left`;
    if (t >= calRun.until) finishCalibration();
    return drawDir();
  }

  ring = [...ring.filter((r) => t - r.t < AVG_SECONDS), { t, x: pairs.map(([i, j]) => xc(i, j)) }].slice(-AVG_FRAMES);
  const rs = search(pairs.map((_, p) => sumCross(ring.map((f) => f.x[p]))));
  setHeat(rs);
  drawCorr(rs[0].r.corr);
  const edge = rs.some((x) => x.r.edge);
  lastConf = level > 1e-5 && !edge ? Math.min(...rs.map((x) => x.r.peak)) : 0;
  out.conf.textContent = edge ? 'edge' : lastConf.toFixed(2);
  if (lastConf >= +thresh.value) {
    const k = +alpha.value;
    ema = rs.map((x, p) => (ema[p] === null ? x.rel : k * x.rel + (1 - k) * ema[p]!));
    const hit = angleOf(rs.map((x) => x.rel));
    if (Number.isFinite(hit)) { lastHit = now(); hits = [...hits.filter((h) => lastHit - h.t < FADE_S), { ang: hit, t: lastHit }]; }
  }
  if (ema[0] !== null) {
    const lags = ema as number[];
    out.delayMs.textContent = lags.map((l) => ((l / fs) * 1000).toFixed(3)).join(' / ') + ' ms';
    out.delaySmp.textContent = lags.map((l) => l.toFixed(2)).join(' / ');
    const ang = angleOf(lags);
    if (Number.isNaN(ang)) showError('Mic positions are collinear; 3 mics need a triangle.');
    else { dirAng = ang; out.angle.textContent = `${ang.toFixed(1)}°`; }
  }
  drawDir();
}

function startCalibration() {
  if (!ctx) return;
  calRun = { until: ctx.currentTime + CAL_SECONDS, t: [], lag: pairs.slice(0, nMics - 1).map(() => []) };
  ema = pairs.map(() => null);
  ring = [];
}

function finishCalibration() {
  const run = calRun!;
  calRun = null;
  if (!ctx) return;
  if (run.t.length < 2) { out.cal.textContent = 'Calibration failed: no clear clap/click on every mic. Try louder or closer.'; return; }
  // Per mic: reject outliers (echoes, tonal ambiguity), then fit offset + drift on the inliers.
  const fits: { a: number; b: number }[] = [];
  for (const lags of run.lag) {
    const med = median(lags);
    const keep = lags.map((l, i) => [run.t[i], l] as const).filter(([, l]) => Math.abs(l - med) <= 2);
    if (keep.length < 2 || keep.length < lags.length / 2) { out.cal.textContent = 'Calibration failed: hits disagree (echoes or tonal sound?). Use sharp claps/clicks.'; return; }
    const fit = fitLine(keep.map((k) => k[0]), keep.map((k) => k[1]));
    if (Math.abs(fit.b) > 2e-4 * ctx.sampleRate) fit.b = 0; // >200 ppm isn't clock drift
    fits.push(fit);
  }
  cal = fits;
  ring = [];
  btn.resetCal.disabled = false;
  updateOffset();
  out.cal.textContent = `Calibrated from ${run.t.length} hits.`;
}

function updateOffset() {
  if (!cal || !ctx) { out.offset.textContent = 'none'; return; }
  const now = ctx.currentTime;
  out.offset.textContent = cal.map((f, k) => `${NAMES[k + 1]} ${(f.a + f.b * now).toFixed(1)} smp, ${(f.b * 60).toFixed(2)}/min`).join('; ');
}
setInterval(updateOffset, 1000);

/** Short noise-burst click train through the default output; put the speaker at the calibration azimuth. */
function toggleClicks() {
  if (!ctx) return;
  if (clickTimer) { clearInterval(clickTimer); clickTimer = 0; btn.click.textContent = 'Play clicks'; return; }
  const n = Math.round(ctx.sampleRate * 0.002);
  const buf = ctx.createBuffer(1, n, ctx.sampleRate);
  const ch = buf.getChannelData(0);
  for (let i = 0; i < n; i++) ch[i] = (Math.random() * 2 - 1) * (1 - i / n);
  const fire = () => { if (!ctx) return; const s = ctx.createBufferSource(); s.buffer = buf; s.connect(ctx.destination); s.start(); };
  fire();
  clickTimer = window.setInterval(fire, 400);
  btn.click.textContent = 'Stop clicks';
}

const drawEmpty = () => drawDir();

/**
 * Direction plot. 2 mics: front half-circle; 3 mics: full circle with mics at their positions.
 * Angles follow the app convention (0° = toward A/left, 90° = ahead); screen angle φ = π + az.
 * Layers: live correlation heat band inside the arc, fading dots for recent confident hits, and a ray
 * that fades with time since the last confident hit.
 */
function drawDir() {
  const g = view.getContext('2d')!, W = view.width, H = view.height, full = nMics === 3;
  const cx = W / 2, cy = full ? H / 2 : H - 40, R = full ? H / 2 - 30 : Math.min(W / 2, H) - 60;
  const phi = (az: number) => Math.PI + (az * Math.PI) / 180;
  const P = (az: number, r: number) => [cx + Math.cos(phi(az)) * r, cy + Math.sin(phi(az)) * r] as const;
  const ray = css('--ray'), muted = css('--muted'), t = now();
  g.clearRect(0, 0, W, H);
  g.strokeStyle = muted; g.fillStyle = muted; g.lineWidth = 1; g.font = '12px system-ui';
  g.beginPath(); g.arc(cx, cy, R, Math.PI, full ? 3 * Math.PI : 2 * Math.PI); g.stroke();
  for (let deg = 0; deg <= (full ? 315 : 180); deg += 45) { const [x, y] = P(deg, R + 16); g.fillText(`${deg}°`, x - 10, y + 4); }

  if (heat) { // opacity ∝ (corr / frame max)², overall strength grows with the max so noise stays faint
    const step = (full ? 360 : 180) / (full ? heat.length : heat.length - 1);
    const mx = Math.max(...heat), strength = 0.25 + 0.75 * Math.min(1, Math.max(0, mx) / 0.5);
    g.strokeStyle = ray; g.lineWidth = 12;
    heat.forEach((v, k) => {
      const a = (Math.max(0, v) / Math.max(mx, 0.05)) ** 2 * strength;
      if (a < 0.02) return;
      g.globalAlpha = a;
      g.beginPath(); g.arc(cx, cy, R - 10, phi(k * step - step / 2), phi(k * step + step / 2)); g.stroke();
    });
    g.globalAlpha = 1;
  }

  const col = colors();
  const mics: [number, number][] = full
    ? (() => {
        const pos = params().pos, mx = pos.reduce((s, p) => s + p[0], 0) / 3, my = pos.reduce((s, p) => s + p[1], 0) / 3;
        const span = Math.max(1e-6, ...pos.map((p) => Math.hypot(p[0] - mx, p[1] - my)));
        return pos.map((p) => [cx + ((p[0] - mx) / span) * R * 0.35, cy - ((p[1] - my) / span) * R * 0.35] as [number, number]);
      })()
    : [[cx - R * 0.35, cy], [cx + R * 0.35, cy]];
  mics.forEach(([x, y], k) => {
    g.fillStyle = col[k];
    g.beginPath(); g.arc(x, y, 9, 0, 2 * Math.PI); g.fill();
    g.fillText(`Mic ${NAMES[k]}`, x - 16, y + 24);
  });

  hits = hits.filter((h) => t - h.t < FADE_S);
  g.fillStyle = ray;
  for (const h of hits) {
    const [x, y] = P(h.ang, R);
    g.globalAlpha = ageAlpha(t - h.t, FADE_S);
    g.beginPath(); g.arc(x, y, 6, 0, 2 * Math.PI); g.fill();
  }
  if (dirAng !== null) {
    const [x, y] = P(dirAng, R - 20);
    g.globalAlpha = Math.max(0.12, ageAlpha(t - lastHit, FADE_S));
    g.strokeStyle = ray; g.lineWidth = 4;
    g.beginPath(); g.moveTo(cx, cy); g.lineTo(x, y); g.stroke();
  }
  g.globalAlpha = 1;
}

const LANE = 100; // px per channel lane in the wave and spectrum canvases
const fitLanes = (cv: HTMLCanvasElement, n: number) => { if (cv.height !== LANE * n) cv.height = LANE * n; };

/** One lane per mic, each scaled to its own peak (floor -60 dBFS) so a quiet mic stays visible. */
function drawWave(chs: Float32Array[]) {
  fitLanes(waveCv, chs.length);
  const g = waveCv.getContext('2d')!, W = waveCv.width, H = waveCv.height, lane = H / chs.length;
  g.clearRect(0, 0, W, H);
  chs.forEach((x, k) => {
    const pk = peakAbs(x), max = Math.max(pk, 1e-3), color = colors()[k];
    const mid = lane * k + lane / 2, amp = lane / 2 - 4, per = x.length / W;
    g.strokeStyle = css('--muted'); g.lineWidth = 1; g.beginPath(); g.moveTo(0, mid); g.lineTo(W, mid); g.stroke();
    g.strokeStyle = color; g.beginPath();
    for (let px = 0; px < W; px++) {
      let lo = Infinity, hi = -Infinity;
      for (let i = Math.floor(px * per), e = Math.min(x.length, Math.floor((px + 1) * per) + 1); i < e; i++) { const v = x[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
      g.moveTo(px + 0.5, mid - (hi / max) * amp); g.lineTo(px + 0.5, mid - (lo / max) * amp + 0.5);
    }
    g.stroke();
    const db = pk > 0 ? 20 * Math.log10(pk) : -Infinity;
    g.fillStyle = color; g.font = '15px system-ui';
    g.fillText(`Mic ${NAMES[k]}  ${Number.isFinite(db) ? db.toFixed(1) : '-∞'} dBFS`, 8, lane * k + 20);
  });
}

/** Log-frequency magnitude spectrum per mic (dBFS, EMA-smoothed), high-pass region shaded. */
function drawSpectra(chs: Frame[]) {
  if (!ctx) return;
  fitLanes(specCv, chs.length);
  const fs = ctx.sampleRate, ny = fs / 2, f0 = 20, DB_LO = -120, DB_HI = 0;
  const g = specCv.getContext('2d')!, W = specCv.width, H = specCv.height, lane = H / chs.length;
  const fx = (f: number) => (Math.log(f / f0) / Math.log(ny / f0)) * W;
  g.clearRect(0, 0, W, H);
  chs.forEach((x, k) => {
    const s = spectrumDb(x);
    const prev = specEma[k];
    specEma[k] = prev && prev.length === s.length ? prev.map((v, i) => 0.7 * v + 0.3 * s[i]) : s;
    const sm = specEma[k], top = lane * k, binHz = fs / x.pow.length, color = colors()[k];
    g.fillStyle = 'rgba(127,127,127,0.15)'; g.fillRect(0, top, fx(HP_HZ), lane);
    g.strokeStyle = 'rgba(127,127,127,0.35)'; g.lineWidth = 1; g.fillStyle = css('--muted'); g.font = '11px system-ui';
    for (const [f, l] of [[100, '100'], [1000, '1k'], [10000, '10k']] as const) {
      if (f >= ny) continue;
      g.beginPath(); g.moveTo(fx(f) + 0.5, top); g.lineTo(fx(f) + 0.5, top + lane); g.stroke();
      if (k === chs.length - 1) g.fillText(l, fx(f) + 3, top + lane - 3);
    }
    g.strokeStyle = css('--muted'); g.beginPath(); g.moveTo(0, top + lane - 0.5); g.lineTo(W, top + lane - 0.5); g.stroke();
    g.strokeStyle = color; g.lineWidth = 1.5; g.beginPath();
    let started = false;
    for (let px = 0; px < W; px++) {
      // max over the bins this pixel spans (low end: nearest bin)
      const fa = f0 * (ny / f0) ** (px / W), fb = f0 * (ny / f0) ** ((px + 1) / W);
      let v = -Infinity;
      for (let b = Math.round(fa / binHz), e = Math.max(b, Math.round(fb / binHz) - 1); b <= e && b < sm.length; b++) v = Math.max(v, sm[b]);
      const y = top + 2 + ((DB_HI - Math.max(DB_LO, Math.min(DB_HI, v))) / (DB_HI - DB_LO)) * (lane - 4);
      started ? g.lineTo(px, y) : g.moveTo(px, y); started = true;
    }
    g.stroke();
    g.fillStyle = color; g.font = '12px system-ui'; g.fillText(`Mic ${NAMES[k]}`, W - 44, top + 14);
  });
}

function drawCorr(r: Float64Array) {
  const g = corrCv.getContext('2d')!, W = corrCv.width, H = corrCv.height;
  g.clearRect(0, 0, W, H);
  let max = 1e-9; for (const v of r) max = Math.max(max, Math.abs(v));
  g.strokeStyle = css('--acc'); g.lineWidth = 1.5; g.beginPath();
  r.forEach((v, i) => { const x = (i / (r.length - 1)) * W, y = H / 2 - (v / max) * (H / 2 - 4); i ? g.lineTo(x, y) : g.moveTo(x, y); });
  g.stroke();
  g.strokeStyle = css('--muted'); g.beginPath(); g.moveTo(W / 2, 0); g.lineTo(W / 2, H); g.stroke();
}

allowBtn.onclick = allowMics;
btn.start.onclick = start;
btn.stop.onclick = stop;
btn.cal.onclick = startCalibration;
btn.click.onclick = toggleClicks;
btn.resetCal.onclick = () => { cal = null; ema = pairs.map(() => null); ring = []; dirAng = null; btn.resetCal.disabled = true; updateOffset(); out.cal.textContent = ''; };
devA.onchange = probeChannels;
devC.onchange = posIn.oninput = baseline.oninput = () => { if (!ctx) { nMics = devC.value ? 3 : 2; drawEmpty(); } };
navigator.mediaDevices?.addEventListener('devicechange', () => listDevices().then(probeChannels));
drawEmpty();
if (navigator.mediaDevices) listDevices().then(probeChannels).catch((e) => showError(describeError(e)));
else showError('Needs HTTPS (or localhost) for microphone access.');
