// An AudioWorklet that turns the microphone into 16 kHz mono 16-bit PCM in 80 ms chunks, the audio Muse Voice
// Transcribe's realtime stream takes (brief.js sends it on). "flush" posts what is left, marked last.

const CHUNK = 1280;  // 80 ms at 16 kHz

class Pcm16k extends AudioWorkletProcessor {
  constructor() {
    super();
    this.step = sampleRate / 16000;  // input samples per output sample
    this.phase = 0;
    this.sum = 0;
    this.count = 0;
    this.pcm = new Int16Array(CHUNK);
    this.n = 0;
    this.port.onmessage = () => this.post(true);
  }

  process([input]) {
    // Each output sample is the average of the input samples it covers, which also filters out what 16 kHz can't hold.
    for (const v of input[0] || []) {
      this.sum += v;
      this.count += 1;
      this.phase += 1;
      if (this.phase >= this.step) {
        this.phase -= this.step;
        const s = Math.max(-1, Math.min(1, this.sum / this.count));
        this.pcm[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
        this.sum = this.count = 0;
        if (this.n === CHUNK) this.post(false);
      }
    }
    return true;
  }

  post(last) {
    const pcm = this.pcm.slice(0, this.n).buffer;
    this.port.postMessage({ pcm, last }, [pcm]);
    this.n = 0;
  }
}

registerProcessor("pcm16k", Pcm16k);
