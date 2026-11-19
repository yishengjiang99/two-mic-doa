// Posts each `hop` samples of N channels (ch0 = mic A, ch1 = B, ch2 = C), sample-aligned.
// The main thread high-passes and assembles overlapping analysis windows.
class Capture extends AudioWorkletProcessor {
  constructor(opts) {
    super();
    const { channels = 2, hop = 2048 } = opts.processorOptions || {};
    this.n = channels; this.hop = hop; this.fill = 0;
    this.buf = Array.from({ length: channels }, () => new Float32Array(hop));
  }
  process(inputs) {
    const ch = inputs[0];
    if (!ch || ch.length < this.n) return true; // wait until every channel is connected
    const len = ch[0].length;
    for (let off = 0; off < len; ) {
      const k = Math.min(len - off, this.hop - this.fill);
      for (let c = 0; c < this.n; c++) this.buf[c].set(ch[c].subarray(off, off + k), this.fill);
      this.fill += k; off += k;
      if (this.fill === this.hop) {
        const out = this.buf;
        this.buf = out.map(() => new Float32Array(this.hop));
        this.fill = 0;
        this.port.postMessage({ ch: out, t: currentTime + off / sampleRate }, out.map((b) => b.buffer));
      }
    }
    return true;
  }
}
registerProcessor('capture', Capture);
