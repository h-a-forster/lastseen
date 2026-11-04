// "Start a search here": a brief, typed or spoken (Muse Voice Transcribe), read by Muse Spark into hypotheses on a
// real place's map (app/agent/brief.py). Uses aerialCanvas, heatImage and drawMarker from replay.js.

const B = (id) => document.getElementById(id);
const BRIEF_PX = 600;
const SORTIE_COLOURS = ["#38bdf8", "#f472b6", "#a3e635", "#fb923c"];
const brief = { result: null, aerial: null, showSpark: true };

async function showBriefExample() {
  const response = await fetch("/api/brief/example");
  if (!response.ok) return B("brief").classList.add("hidden");
  const result = await response.json();
  B("brief-text").value = result.brief;
  await showBrief(result);
}

async function showBrief(result, seconds = null) {
  brief.result = result;
  if (!brief.aerial || brief.aerialFor !== result.place.name) {
    brief.aerial = await aerialCanvas(result.place.bounds, BRIEF_PX, BRIEF_PX).catch(() => null);
    brief.aerialFor = result.place.name;
  }
  drawBrief();
  B("brief-hypotheses").innerHTML = result.hypotheses.map((h) => `<li>
      <div class="flex justify-between gap-2"><span class="font-medium">${escapeText(h.name)}</span><span class="font-mono text-amber-300">${Math.round(h.weight * 100)}%</span></div>
      <div class="h-1 bg-slate-800 rounded"><div class="h-1 bg-amber-400 rounded" style="width:${h.weight * 100}%"></div></div>
      <div class="text-xs text-slate-400 mt-1">${escapeText(h.reason)}</div>
      <div class="text-xs text-slate-500 italic">${h.evidence.map((q) => `“${escapeText(q)}”`).join(" · ")}</div>
    </li>`).join("") +
    `<li class="text-xs text-slate-400">None of these (keep the statistics): ${Math.round(result.rest_weight * 100)}%` +
    (result.dropped.length ? ` · ${result.dropped.length} explanation${result.dropped.length === 1 ? "" : "s"} dropped by the checks: ${escapeText(result.dropped.join("; "))}` : "") + "</li>";
  B("brief-witnesses").innerHTML = (result.witnesses || []).map((w) => `<li>
      <span class="grade grade-${w.reliability}">${escapeText(w.reliability)}</span>
      <div><strong>${escapeText(w.who)}</strong>: ${escapeText(w.claim)}
      <div class="witness-why">${escapeText(w.basis)} · ${w.knows_her ? "knows her" : "doesn't know her"} · ${w.backed_up ? "backed up" : "nobody else says so"} (${escapeText(w.why)})</div></div>
    </li>`).join("");
  B("brief-sorties").innerHTML = result.sorties.map((s, i) =>
    `<button data-sortie="${i}" class="rounded px-2 py-1 bg-slate-800 hover:bg-slate-700 text-xs" style="border-left:4px solid ${SORTIE_COLOURS[i % SORTIE_COLOURS.length]}">Drone ${s.drone}: download mission (.plan)</button>`).join(" ");
  B("brief-sorties").querySelectorAll("button").forEach((b) => (b.onclick = () => downloadPlan(result.sorties[+b.dataset.sortie])));
  B("brief-status").textContent = `${result.model || "Muse Spark"} read the brief${result.cached ? " (replayed from cache)" : seconds ? ` in ${seconds} s` : ""} · $${result.usd.toFixed(4)}. ` +
    "In a search the drone flies this on autopilot and LastSeen re-plans after each report and battery. The mission exports for QGroundControl; it hasn't been flown on a real drone yet.";
  B("brief-credit").textContent = `${result.place.title}. Aerial imagery © swisstopo · ${result.place.credit}.`;
}

function drawBrief() {
  const { result } = brief;
  const canvas = B("brief-map");
  canvas.width = canvas.height = BRIEF_PX;
  const ctx = canvas.getContext("2d");
  const s = BRIEF_PX / result.map.cols;
  if (brief.aerial) ctx.drawImage(brief.aerial, 0, 0);
  else { ctx.fillStyle = "#1e293b"; ctx.fillRect(0, 0, BRIEF_PX, BRIEF_PX); }
  const heat = heatImage({ side: result.map.side, values: brief.showSpark ? result.combined : result.statistics_only });
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(heat, 0, 0, BRIEF_PX, BRIEF_PX);
  if (brief.showSpark) {
    result.sorties.forEach((sortie, i) => {
      ctx.strokeStyle = SORTIE_COLOURS[i % SORTIE_COLOURS.length];
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      sortie.route.forEach(([y, x], j) => ctx[j ? "lineTo" : "moveTo"]((x / result.map.cell_m) * s, (y / result.map.cell_m) * s));
      ctx.stroke();
    });
  }
  drawMarker(ctx, result.map.ipp, s, "#ffffff", "last seen · drone base");
}

function downloadPlan(sortie) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(sortie.plan, null, 1)], { type: "application/json" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: `lastseen-drone-${sortie.drone}.plan` });
  a.click();
  URL.revokeObjectURL(url);
}

async function readBrief() {
  B("brief-status").textContent = `${B("brief-open").checked ? "Muse Spark Open" : "Muse Spark"} is reading the brief...`;
  const started = performance.now();
  try {
    const response = await fetch("/api/brief", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ place: brief.result?.place.name || "saint-ursanne", text: B("brief-text").value,
                             open_weights: B("brief-open").checked }) });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || response.statusText);
    await showBrief(body, secondsSince(started));
  } catch (err) {
    B("brief-status").textContent = `Could not read the brief: ${err.message}`;
  }
}

// ---------- voice: Muse Voice Transcribe writes as the visitor speaks, or from the whole recording ----------
// The microphone streams to the server, which relays it to Muse Voice Transcribe's realtime stream and passes the words
// back as they come. If that can't start or finish, the recording is converted to a 16 kHz mono WAV and sent whole.
// A mic is { button, box, say(text), what, idle, listening, done }: the button that records into the text box, how to
// show progress, and its wording. Section 4 (place.js) uses one too.

async function toggleRecording(mic) {
  if (mic.recorder) return mic.recorder.stop();
  // made inside the click: Safari keeps an AudioContext made later suspended. Without one, the bars stay still.
  const audio = window.AudioContext ? new AudioContext() : null;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const live = streamLive(mic, audio, stream);
    // Show "recording" once the stream takes audio, so no first word is lost; one that can't start is handled at stop.
    await within(live.catch(() => {}), 2000);
    const chunks = [];
    mic.recorder = new MediaRecorder(stream);
    mic.recorder.ondataavailable = (e) => chunks.push(e.data);
    const meter = showListening(mic, audio, stream);
    mic.recorder.onstop = async () => {
      meter.stop();
      showWriting(mic);
      const started = performance.now();
      const text = await live.then((relay) => relay.finish()).catch(() => null);
      audio?.close();
      stream.getTracks().forEach((t) => t.stop());
      mic.recorder = null;
      if (text === null) await transcribe(mic, new Blob(chunks));
      else mic.say(`Transcribed live by Muse Voice Transcribe as you spoke (last words in ${secondsSince(started)} s). ${mic.done}`);
      mic.button.disabled = false;
      mic.button.textContent = mic.idle;
      mic.box.placeholder = mic.placeholder;
    };
    mic.recorder.start();
    mic.typed = mic.box.value;  // the box shows only what was said; put this back if transcription fails
    mic.placeholder = mic.box.placeholder;
    mic.box.value = "";
    mic.box.placeholder = "Listening...";
    mic.say(mic.listening);
  } catch (err) {
    audio?.close();
    mic.say(`Couldn't use the microphone (${err.message}); type ${mic.what} instead.`);
  }
}

async function streamLive(mic, audio, stream) {
  // Sends 80 ms chunks of 16 kHz PCM (static/pcm16k.js) to /api/transcribe/live and shows each partial transcript in
  // the box. finish() ends the stream and returns the final transcript, or throws if there isn't one.
  await audio.audioWorklet.addModule("/pcm16k.js?v=1");
  const socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/transcribe/live`);
  const worklet = new AudioWorkletNode(audio, "pcm16k");
  const mute = audio.createGain();
  mute.gain.value = 0;  // the worklet only runs when connected to the speakers; silence it
  audio.createMediaStreamSource(stream).connect(worklet).connect(mute).connect(audio.destination);
  const opened = new Promise((resolve) => { socket.onopen = resolve; socket.onclose = resolve; });
  const closed = new Promise((resolve) => socket.addEventListener("close", resolve));
  const waiting = [];  // audio recorded before the socket opened
  let lastSent;
  const flushed = new Promise((resolve) => (lastSent = resolve));
  worklet.port.onmessage = ({ data }) => {
    if (socket.readyState === WebSocket.OPEN) waiting.splice(0).concat(data.pcm).forEach((pcm) => socket.send(pcm));
    else waiting.push(data.pcm);
    if (data.last) lastSent();
  };
  let text = null, error = "the live transcript didn't finish";
  socket.onmessage = ({ data }) => {
    const event = JSON.parse(data);
    if (event.error) return (error = event.error);
    mic.box.value = event.text;
    if (event.final) text = event.text;
  };
  return {
    async finish() {
      // Each wait has a limit, so a stream that stalls falls back to sending the recording instead of hanging.
      await within(opened, 10000);
      worklet.port.postMessage("flush");
      await within(flushed, 2000);
      if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "endStream" }));
      await within(closed, 15000);
      worklet.disconnect();
      socket.close();
      if (text === null) throw new Error(error);
      return text;
    },
  };
}

function within(promise, ms) {
  return Promise.race([promise, new Promise((resolve) => setTimeout(resolve, ms))]);
}

function showListening(mic, audio, stream) {
  // The button turns into a stop button with the elapsed time and five bars that follow the voice, so a visitor
  // can see the microphone is hearing them.
  const analyser = audio && Object.assign(audio.createAnalyser(), { fftSize: 256, smoothingTimeConstant: 0.6 });
  audio?.createMediaStreamSource(stream).connect(analyser);
  const levels = new Uint8Array(analyser?.frequencyBinCount || 0);
  mic.button.classList.add("mic-on");
  mic.box.classList.add("mic-listening");
  mic.button.innerHTML = `<span class="mic-bars" aria-hidden="true">${"<i></i>".repeat(5)}</span>Stop recording <span class="mic-time">0:00</span>`;
  const bars = mic.button.querySelectorAll(".mic-bars i"), clock = mic.button.querySelector(".mic-time");
  const started = performance.now();
  let frame;
  const draw = () => {
    if (analyser) {
      analyser.getByteFrequencyData(levels);  // bins of ~180 Hz: the bars cover the speech band, ~200 Hz to 3.7 kHz
      bars.forEach((bar, i) => (bar.style.transform = `scaleY(${0.2 + 0.8 * Math.max(...levels.subarray(1 + 4 * i, 5 + 4 * i)) / 255})`));
    }
    const s = Math.floor((performance.now() - started) / 1000);
    clock.textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
    frame = requestAnimationFrame(draw);
  };
  draw();
  return {
    stop() {
      cancelAnimationFrame(frame);
      mic.button.classList.remove("mic-on");
      mic.box.classList.remove("mic-listening");
    },
  };
}

async function typeIn(box, text) {
  // the transcript arrives in one piece; typing it in word by word shows where the box's text came from
  const words = text.split(" ");
  for (let i = 1; i <= words.length; i++) {
    box.value = words.slice(0, i).join(" ");
    await new Promise((resolve) => setTimeout(resolve, 40));
  }
}

function showWriting(mic) {
  mic.button.disabled = true;
  mic.button.innerHTML = `<span class="mic-spin" aria-hidden="true"></span>Transcribing...`;
  mic.box.placeholder = "Muse Voice Transcribe is writing down what you said...";
  mic.say(`Muse Voice Transcribe is writing down ${mic.what}...`);
}

async function transcribe(mic, recording) {
  const started = performance.now();
  try {
    const decoded = await new OfflineAudioContext(1, 16000, 16000).decodeAudioData(await recording.arrayBuffer());
    const response = await fetch("/api/transcribe", { method: "POST", headers: { "Content-Type": "audio/wav" }, body: wav16k(decoded) });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || response.statusText);
    const seconds = secondsSince(started);
    await typeIn(mic.box, body.transcript);
    mic.say(`Transcribed by Muse Voice Transcribe ${body.cached ? "(replayed from cache)" : `in ${seconds} s`}. ${mic.done}`);
  } catch (err) {
    mic.box.value ||= mic.typed;
    mic.say(`Could not transcribe: ${err.message}`);
  }
}

function wav16k(buffer) {
  // decodeAudioData resamples to the context's 16 kHz; average the channels, then write 16-bit PCM with a RIFF header.
  const n = buffer.length, out = new DataView(new ArrayBuffer(44 + 2 * n));
  const channels = Array.from({ length: buffer.numberOfChannels }, (_, c) => buffer.getChannelData(c));
  const text = (offset, s) => [...s].forEach((ch, i) => out.setUint8(offset + i, ch.charCodeAt(0)));
  text(0, "RIFF"); out.setUint32(4, 36 + 2 * n, true); text(8, "WAVE");
  text(12, "fmt "); out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, 1, true);
  out.setUint32(24, 16000, true); out.setUint32(28, 32000, true); out.setUint16(32, 2, true); out.setUint16(34, 16, true);
  text(36, "data"); out.setUint32(40, 2 * n, true);
  for (let i = 0; i < n; i++) {
    const v = Math.max(-1, Math.min(1, channels.reduce((sum, ch) => sum + ch[i], 0) / channels.length));
    out.setInt16(44 + 2 * i, v < 0 ? v * 0x8000 : v * 0x7fff, true);
  }
  return new Blob([out], { type: "audio/wav" });
}

B("brief-read").onclick = readBrief;
B("brief-open").onchange = (e) => (B("brief-read").textContent = e.target.checked ? "Read with Muse Spark Open" : "Read with Muse Spark");
const briefMic = {
  button: B("brief-mic"), box: B("brief-text"), say: (text) => (B("brief-status").textContent = text),
  what: "the brief", idle: "Speak the brief", done: "Check the text, then ask Muse Spark to read it.",
  listening: "Recording... say who is missing, where they were last seen and what witnesses said.",
};
B("brief-mic").onclick = () => toggleRecording(briefMic);
B("brief-spark").onchange = (e) => { brief.showSpark = e.target.checked; drawBrief(); };
showBriefExample().catch(() => B("brief").classList.add("hidden"));
