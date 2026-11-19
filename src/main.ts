import { gccPhat, speedOfSound, maxLagSamples, angleDeg, fitLine, median } from './dsp';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const devA = $<HTMLSelectElement>('devA'), devB = $<HTMLSelectElement>('devB');
const stereo = $<HTMLInputElement>('stereo'), stereoWrap = $('stereoWrap');
const baseline = $<HTMLInputElement>('baseline'), temp = $<HTMLInputElement>('temp');
const thresh = $<HTMLInputElement>('thresh'), alpha = $<HTMLInputElement>('alpha');
const btn = { start: $<HTMLButtonElement>('start'), stop: $<HTMLButtonElement>('stop'), cal: $<HTMLButtonElement>('cal'), click: $<HTMLButtonElement>('click'), resetCal: $<HTMLButtonElement>('resetCal') };
const out = { angle: $('angle'), delayMs: $('delayMs'), delaySmp: $('delaySmp'), conf: $('conf'), fs: $('fs'), offset: $('offset'), error: $('error'), cal: $('calStatus') };
const view = $<HTMLCanvasElement>('view'), corrCv = $<HTMLCanvasElement>('corr');

const RAW = { echoCancellation: false, noiseSuppression: false, autoGainControl: false };
const CAL_SECONDS = 6;

let ctx: AudioContext | null = null;
let streams: MediaStream[] = [];
let ema: number | null = null; // smoothed delay, samples, calibration removed
let lastConf = 0;
let floor = 1e-4; // running RMS noise floor for transient gating
// Calibration: offset(t) = a + b * t (samples; t = AudioContext seconds)
let cal: { a: number; b: number } | null = null;
let calRun: { until: number; t: number[]; lag: number[] } | null = null;
let clickTimer = 0;

const showError = (m: string) => { out.error.textContent = m; };
const num = (el: HTMLInputElement, dflt: number) => (Number.isFinite(+el.value) && +el.value > 0 ? +el.value : dflt);
const params = () => {
  const d = num(baseline, 0.2);
  const c = speedOfSound(Number.isFinite(+temp.value) ? +temp.value : 20);
  return { d, c };
};

for (const [el, o] of [[thresh, 'threshOut'], [alpha, 'alphaOut']] as const)
  el.addEventListener('input', () => { $(o).textContent = (+el.value).toFixed(2); });

async function listDevices() {
  const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'audioinput');
  for (const [sel, pick] of [[devA, 0], [devB, 1]] as const) {
    const prev = sel.value;
    sel.innerHTML = '';
    devs.forEach((d, i) => sel.add(new Option(d.label || `Microphone ${i + 1}`, d.deviceId)));
    if (devs.some((d) => d.deviceId === prev)) sel.value = prev;
    else if (devs[pick]) sel.value = devs[pick].deviceId;
  }
  // Before permission, Safari/Firefox hide devices and Chrome hides ids and labels.
  const locked = !devs.length || devs.every((d) => !d.deviceId || !d.label);
  allowBtn.classList.toggle('hidden', !locked);
  if (locked) showError('Click "Allow microphones" so the browser lists your mics.');
  else if (out.error.textContent?.startsWith('Click "Allow')) showError('');
}

const allowBtn = $('allow') as HTMLButtonElement;
async function allowMics() {
  showError('');
  try {
    const s = await navigator.mediaDevices.getUserMedia({ audio: true });
    s.getTracks().forEach((t) => t.stop());
    await listDevices();
    await probeStereo();
  } catch (e) { showError(describeError(e)); }
}

/** Opens device A asking for 2 channels; offers L/R mode if it actually delivers 2. */
async function probeStereo() {
  if (!devA.value || !navigator.mediaDevices) return;
  try {
    const s = await navigator.mediaDevices.getUserMedia({ audio: { deviceId: { exact: devA.value }, channelCount: { ideal: 2 }, ...RAW } });
    const ch = s.getAudioTracks()[0]?.getSettings().channelCount ?? 1;
    s.getTracks().forEach((t) => t.stop());
    stereoWrap.classList.toggle('hidden', ch < 2);
    if (ch < 2) stereo.checked = false;
  } catch { /* probe is best-effort */ }
}

async function openMic(id: string, channels: number) {
  return navigator.mediaDevices.getUserMedia({ audio: { deviceId: { exact: id }, channelCount: { exact: channels }, ...RAW } });
}

function describeError(e: unknown): string {
  const n = (e as DOMException)?.name;
  if (n === 'NotAllowedError') return 'Microphone permission denied. Allow mic access and try again.';
  if (n === 'NotFoundError') return 'Selected microphone not found. Re-select devices.';
  if (n === 'OverconstrainedError') return 'Device rejected the requested settings (channel count). Try mono mode or another device.';
  if (n === 'NotReadableError') return 'Microphone is busy or unavailable (in use by another app?).';
  return `Error: ${(e as Error)?.message ?? e}`;
}

async function start() {
  showError('');
  if (!window.isSecureContext || !navigator.mediaDevices) return showError('Needs HTTPS (or localhost) for microphone access.');
  if (typeof AudioWorkletNode === 'undefined') return showError('This browser lacks AudioWorklet support.');
  const useStereo = stereo.checked && !stereoWrap.classList.contains('hidden');
  if (!useStereo && devA.value === devB.value) return showError('Pick two different devices, or use L/R of a stereo device.');
  btn.start.disabled = true;
  try {
    ctx = new AudioContext({ latencyHint: 'interactive' });
    await ctx.audioWorklet.addModule(`${import.meta.env.BASE_URL}capture-worklet.js`);
    const size = ctx.sampleRate > 60000 ? 8192 : 4096;
    const node = new AudioWorkletNode(ctx, 'capture', {
      numberOfInputs: 1, numberOfOutputs: 1, channelCount: 2, channelCountMode: 'explicit', channelInterpretation: 'discrete',
      processorOptions: { size, hop: size / 2 },
    });
    if (useStereo) {
      streams = [await openMic(devA.value, 2)];
      ctx.createMediaStreamSource(streams[0]).connect(node);
    } else {
      streams = [await openMic(devA.value, 1), await openMic(devB.value, 1)];
      const merger = ctx.createChannelMerger(2);
      ctx.createMediaStreamSource(streams[0]).connect(merger, 0, 0);
      ctx.createMediaStreamSource(streams[1]).connect(merger, 0, 1);
      merger.connect(node);
    }
    for (const s of streams) for (const t of s.getTracks()) t.onended = () => { showError('A microphone disconnected.'); stop(); };
    const mute = ctx.createGain(); mute.gain.value = 0; // keep the node pulled without monitoring
    node.connect(mute).connect(ctx.destination);
    node.port.onmessage = (ev) => onWindow(ev.data.a, ev.data.b, ev.data.t, size);
    await ctx.resume();
    out.fs.textContent = `${ctx.sampleRate} Hz`;
    ema = null;
    await listDevices(); // labels become available after permission
    btn.stop.disabled = btn.cal.disabled = btn.click.disabled = false;
    btn.resetCal.disabled = !cal;
    if (!useStereo && !cal) out.cal.textContent = 'Two separate devices: calibrate now (clap at broadside) to remove the clock offset.';
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

function onWindow(a: Float32Array, b: Float32Array, t: number, size: number) {
  if (!ctx) return;
  drawWave(a, b);
  const fs = ctx.sampleRate;
  const { d, c } = params();
  const phys = maxLagSamples(d, c, fs);
  const level = Math.min(rms(a), rms(b));
  const transient = Math.min(peakAbs(a), peakAbs(b)) > 6 * floor;
  floor = level < floor ? 0.9 * floor + 0.1 * level : floor * 1.002 + 1e-7; // tracks quiet level

  if (calRun) {
    // Wide search: unknown device offset can be many ms.
    const r = gccPhat(a, b, size / 4, 0);
    drawCorr(r.corr);
    if (transient && r.peak >= Math.max(+thresh.value, 0.2)) { calRun.t.push(t); calRun.lag.push(r.lag); }
    out.cal.textContent = `Calibrating… clap or click at broadside (90°). ${calRun.t.length} hits, ${Math.max(0, calRun.until - t).toFixed(1)} s left`;
    if (t >= calRun.until) finishCalibration();
    return;
  }

  const center = cal ? cal.a + cal.b * t : 0;
  const r = gccPhat(a, b, phys + 2, center);
  drawCorr(r.corr);
  lastConf = level > 1e-5 ? r.peak : 0;
  out.conf.textContent = lastConf.toFixed(2);
  if (lastConf >= +thresh.value) {
    const rel = Math.max(-phys, Math.min(phys, r.lag - center));
    const k = +alpha.value;
    ema = ema === null ? rel : k * rel + (1 - k) * ema;
  }
  if (ema !== null) {
    const ang = angleDeg(ema / fs, d, c);
    out.angle.textContent = `${ang.toFixed(1)}°`;
    out.delayMs.textContent = `${((ema / fs) * 1000).toFixed(3)} ms`;
    out.delaySmp.textContent = ema.toFixed(2);
    drawView(ang);
  } else drawView(null);
}

function startCalibration() {
  if (!ctx) return;
  calRun = { until: ctx.currentTime + CAL_SECONDS, t: [], lag: [] };
  ema = null;
}

function finishCalibration() {
  const run = calRun!;
  calRun = null;
  if (!ctx) return;
  if (run.t.length < 2) { out.cal.textContent = 'Calibration failed: no clear clap/click detected. Try louder, closer, at broadside.'; return; }
  // Reject outliers (echoes, tonal ambiguity), then fit offset + drift on the inliers.
  const med = median(run.lag);
  const keep = run.lag.map((l, i) => [run.t[i], l] as const).filter(([, l]) => Math.abs(l - med) <= 2);
  if (keep.length < 2 || keep.length < run.lag.length / 2) { out.cal.textContent = 'Calibration failed: hits disagree (echoes or tonal sound?). Use sharp claps/clicks.'; return; }
  const fit = fitLine(keep.map((k) => k[0]), keep.map((k) => k[1]));
  if (Math.abs(fit.b) > 2e-4 * ctx!.sampleRate) fit.b = 0; // >200 ppm isn't clock drift
  cal = fit;
  btn.resetCal.disabled = false;
  updateOffset();
  out.cal.textContent = `Calibrated from ${run.t.length} hits.`;
}

function updateOffset() {
  if (!cal || !ctx) { out.offset.textContent = 'none'; return; }
  const o = cal.a + cal.b * ctx.currentTime;
  out.offset.textContent = `${o.toFixed(1)} smp, ${(cal.b * 60).toFixed(2)} smp/min`;
}
setInterval(updateOffset, 1000);

/** Short noise-burst click train through the default output; put the speaker at broadside. */
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

const css = (v: string) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

function drawView(ang: number | null) {
  const g = view.getContext('2d')!, W = view.width, H = view.height;
  const cx = W / 2, cy = H - 40, R = Math.min(W / 2, H) - 60;
  g.clearRect(0, 0, W, H);
  g.strokeStyle = css('--muted'); g.fillStyle = css('--muted'); g.lineWidth = 1; g.font = '12px system-ui';
  g.beginPath(); g.arc(cx, cy, R, Math.PI, 0); g.stroke();
  for (const deg of [0, 45, 90, 135, 180]) {
    const th = (deg * Math.PI) / 180;
    g.fillText(`${deg}°`, cx - Math.cos(th) * (R + 14) - 10, cy - Math.sin(th) * (R + 14));
  }
  const mx = R * 0.35;
  g.fillStyle = css('--acc');
  for (const [x, l] of [[cx - mx, 'A'], [cx + mx, 'B']] as const) {
    g.beginPath(); g.arc(x, cy, 9, 0, 2 * Math.PI); g.fill();
    g.fillText(`Mic ${l}`, x - 16, cy + 26);
  }
  if (ang === null) return;
  const th = (ang * Math.PI) / 180;
  g.globalAlpha = lastConf >= +thresh.value ? 1 : 0.35;
  g.strokeStyle = css('--ray'); g.lineWidth = 4;
  g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx - Math.cos(th) * R, cy - Math.sin(th) * R); g.stroke();
  g.globalAlpha = 1;
}

const waveCv = $('wave') as HTMLCanvasElement;
/** Mic A on top, Mic B below, shared scale so level differences show. Min/max per pixel column. */
function drawWave(a: Float32Array, b: Float32Array) {
  const g = waveCv.getContext('2d')!, W = waveCv.width, H = waveCv.height, lane = H / 2;
  g.clearRect(0, 0, W, H);
  const max = Math.max(peakAbs(a), peakAbs(b), 1e-4);
  const lanes: [Float32Array, string, string][] = [[a, css('--acc'), 'Mic A'], [b, '#f59e0b', 'Mic B']];
  lanes.forEach(([x, color, name], k) => {
    const mid = lane * k + lane / 2, amp = lane / 2 - 4, per = x.length / W;
    g.strokeStyle = css('--muted'); g.lineWidth = 1; g.beginPath(); g.moveTo(0, mid); g.lineTo(W, mid); g.stroke();
    g.strokeStyle = color; g.beginPath();
    for (let px = 0; px < W; px++) {
      let lo = Infinity, hi = -Infinity;
      for (let i = Math.floor(px * per), e = Math.min(x.length, Math.floor((px + 1) * per) + 1); i < e; i++) { const v = x[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
      g.moveTo(px + 0.5, mid - (hi / max) * amp); g.lineTo(px + 0.5, mid - (lo / max) * amp + 0.5);
    }
    g.stroke();
    g.fillStyle = color; g.font = '20px system-ui'; g.fillText(name, 8, lane * k + 22);
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
btn.resetCal.onclick = () => { cal = null; ema = null; btn.resetCal.disabled = true; updateOffset(); out.cal.textContent = ''; };
devA.onchange = probeStereo;
stereo.onchange = () => { devB.disabled = stereo.checked; };
navigator.mediaDevices?.addEventListener('devicechange', () => listDevices().then(probeStereo));
drawView(null);
if (navigator.mediaDevices) listDevices().then(probeStereo).catch((e) => showError(describeError(e)));
else showError('Needs HTTPS (or localhost) for microphone access.');
