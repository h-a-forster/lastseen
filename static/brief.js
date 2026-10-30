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

async function showBrief(result) {
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
    (result.dropped.length ? ` · ${result.dropped.length} hypothesis dropped by the checks: ${escapeText(result.dropped.join("; "))}` : "") + "</li>";
  B("brief-witnesses").innerHTML = (result.witnesses || []).map((w) => `<li>
      <span class="grade grade-${w.reliability}">${escapeText(w.reliability)}</span>
      <div><strong>${escapeText(w.who)}</strong>: ${escapeText(w.claim)}
      <div class="witness-why">${escapeText(w.basis)} · ${w.knows_her ? "knows her" : "doesn't know her"} · ${w.backed_up ? "backed up" : "nobody else says so"} (${escapeText(w.why)})</div></div>
    </li>`).join("");
  B("brief-sorties").innerHTML = result.sorties.map((s, i) =>
    `<button data-sortie="${i}" class="rounded px-2 py-1 bg-slate-800 hover:bg-slate-700 text-xs" style="border-left:4px solid ${SORTIE_COLOURS[i % SORTIE_COLOURS.length]}">Drone ${s.drone}: download mission (.plan)</button>`).join(" ");
  B("brief-sorties").querySelectorAll("button").forEach((b) => (b.onclick = () => downloadPlan(result.sorties[+b.dataset.sortie])));
  B("brief-status").textContent = `Muse Spark read the brief${result.cached ? " (replayed from cache)" : ""} · $${result.usd.toFixed(4)}. ` +
    "The first block comes from the probability-map planner. The mission loads in QGroundControl; it hasn't been flown on a real drone yet.";
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
  B("brief-status").textContent = "Muse Spark is reading the brief...";
  try {
    const response = await fetch("/api/brief", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ place: brief.result?.place.name || "saint-ursanne", text: B("brief-text").value }) });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || response.statusText);
    await showBrief(body);
  } catch (err) {
    B("brief-status").textContent = `Could not read the brief: ${err.message}`;
  }
}

// ---------- voice: record, convert to 16 kHz mono WAV in the browser, transcribe on the server ----------

async function toggleRecording() {
  if (brief.recorder) return brief.recorder.stop();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const chunks = [];
    brief.recorder = new MediaRecorder(stream);
    brief.recorder.ondataavailable = (e) => chunks.push(e.data);
    brief.recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      brief.recorder = null;
      B("brief-mic").textContent = "Speak the brief";
      await transcribe(new Blob(chunks));
    };
    brief.recorder.start();
    B("brief-mic").textContent = "Stop recording";
    B("brief-status").textContent = "Recording... say who is missing, where they were last seen and what witnesses said.";
  } catch (err) {
    B("brief-status").textContent = `No microphone: ${err.message}`;
  }
}

async function transcribe(recording) {
  B("brief-status").textContent = "Muse Voice Transcribe is writing down the brief...";
  try {
    const decoded = await new OfflineAudioContext(1, 16000, 16000).decodeAudioData(await recording.arrayBuffer());
    const response = await fetch("/api/transcribe", { method: "POST", headers: { "Content-Type": "audio/wav" }, body: wav16k(decoded) });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || response.statusText);
    B("brief-text").value = body.transcript;
    B("brief-status").textContent = "Transcribed. Check the text, then ask Muse Spark to read it.";
  } catch (err) {
    B("brief-status").textContent = `Could not transcribe: ${err.message}`;
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
B("brief-mic").onclick = toggleRecording;
B("brief-spark").onchange = (e) => { brief.showSpark = e.target.checked; drawBrief(); };
showBriefExample().catch(() => B("brief").classList.add("hidden"));
