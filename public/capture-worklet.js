// Buffers sample-aligned windows from a 2-channel input (ch0 = mic A, ch1 = mic B)
// and posts them to the main thread every `hop` samples.
class Capture extends AudioWorkletProcessor {
  constructor(opts) {
    super();
    const { size = 4096, hop = 2048 } = opts.processorOptions || {};
    this.size = size; this.hop = hop;
    this.a = new Float32Array(size); this.b = new Float32Array(size);
    this.fill = 0; this.sinceHop = 0;
  }
  process(inputs) {
    const ch = inputs[0];
    if (!ch || ch.length < 2) return true; // wait until both channels are connected
    const x = ch[0], y = ch[1], n = x.length;
    // shift-in: keep the newest `size` samples
    this.a.copyWithin(0, n); this.b.copyWithin(0, n);
    this.a.set(x, this.size - n); this.b.set(y, this.size - n);
    this.fill = Math.min(this.size, this.fill + n);
    this.sinceHop += n;
    if (this.fill === this.size && this.sinceHop >= this.hop) {
      this.sinceHop = 0;
      const a = this.a.slice(), b = this.b.slice();
      this.port.postMessage({ a, b, t: currentTime }, [a.buffer, b.buffer]);
    }
    return true;
  }
}
registerProcessor('capture', Capture);
