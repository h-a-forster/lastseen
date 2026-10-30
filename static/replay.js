// Replay of a recorded race (app/sar/replay.py): every lane is the same scenario flown by a different planner.
// Everything drawn comes from the run file's event log; the truth is hidden unless the viewer reveals it.

const PLANNERS = {
  lawnmower: ["Lawnmower sweep", "Fixed pattern; ignores reports"],
  spiral: ["Expanding square", "Fixed pattern; ignores reports"],
  concentric: ["Concentric rings", "Fixed pattern; ignores reports"],
  sector: ["Sector search", "Fixed pattern; ignores reports"],
  "spiral+spark": ["Expanding square + Muse Spark", "Restarts on Spark's most likely block after each report"],
  greedy: ["Probability map", "Lost-person statistics only"],
  "greedy+spark": ["Probability map + Muse Spark", "Statistics plus Spark's reading of the witness reports"],
  ours: ["Probability map, packed tours", "Lost-person statistics only"],
  "ours+spark": ["Probability map, packed tours + Muse Spark", "Statistics plus Spark's reading of the witness reports"],
  hedge: ["Team planner", "Expanding square; without Spark it never leaves the pattern"],
  "hedge+spark": ["Team planner + Muse Spark", "Expanding square until a sighting, then follows Spark"],
};
const TERRAIN = [
  ["Open", [181, 201, 154]],
  ["Scrub", [138, 154, 91]],
  ["Forest", [58, 90, 64]],
  ["Water", [74, 144, 194]],
  ["Built", [156, 163, 175]],
];
const TRAIL = [233, 216, 166];
const CUE_LABEL = { heading: "Direction of travel", fitness: "Fitness", sighting: "Sighting" };

const $ = (id) => document.getElementById(id);
const state = { run: null, lanes: [], t: 0, endT: 0, playing: false, scale: 4, base: null, aerial: null, feedKey: "" };
// Aerial photos of real places, drawn by the browser straight from swisstopo's public tile service (Web Mercator).
const AERIAL_TILES = "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage/default/current/3857";
const AERIAL_ZOOM = 15; // about 3 m per pixel here: sharper than the 5 m the map is drawn at

// ---------- loading ----------

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || response.statusText);
  return response.json();
}

async function loadRunList(selectId) {
  const runs = await api("/api/runs");
  $("run").innerHTML = runs.map((r) => `<option value="${r.id}">${r.place ? r.place.split(",")[0] + " · " : ""}${r.category} #${r.seed}, ${r.drones} drone${r.drones === 1 ? "" : "s"}: ${Object.keys(r.found_s).map(label).join(" vs ")}</option>`).join("");
  if (!runs.length) return showStatus("No recorded races yet: record one with a seed.");
  $("run").value = selectId || runs[0].id;
  await loadRun($("run").value);
}

async function loadRun(id) {
  pause();
  showStatus("Loading race...", 60000);
  const run = await api(`/api/runs/${id}`);
  $("status").classList.add("hidden");
  const { rows, cols } = run.map;
  state.run = run;
  state.scale = Math.max(1, Math.floor(600 / Math.max(rows, cols)));
  state.base = drawBaseMap(run.map, state.scale);
  state.aerial = null;
  $("aerial-toggle").classList.toggle("hidden", !run.place);
  if (run.place) loadAerial(run).catch(() => showStatus("Aerial imagery unavailable; showing terrain classes."));
  state.lanes = Object.entries(run.lanes).map(([name, lane]) => prepareLane(name, lane, run.map));
  state.endT = Math.max(...state.lanes.map((l) => l.endT));
  $("time").max = state.endT;
  $("seed").value = run.seed;
  buildLaneDom();
  $("result-line").textContent = resultLine();
  seek(0);
}

function prepareLane(name, lane, map) {
  const byType = (type) => lane.events.filter((e) => e.type === type);
  const found = byType("found")[0] || null;
  const searched = document.createElement("canvas");
  searched.width = map.cols;
  searched.height = map.rows;
  return {
    name,
    found,
    endT: found ? found.t_s : state.run.hours * 3600,
    photos: byType("photo"),
    frames: byType("drones"),
    maps: byType("map").map((m) => ({ t_s: m.t_s, image: heatImage(m) })),
    notes: byType("note"),
    lost: byType("drone_lost"),
    sightings: byType("sighting"),
    searched,
    photoIndex: 0,
    alarmCount: 0,
    shownPhoto: false,
  };
}

// ---------- drawing ----------

function drawBaseMap(map, s) {
  const canvas = document.createElement("canvas");
  canvas.width = map.cols * s;
  canvas.height = map.rows * s;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(canvas.width, canvas.height);
  for (let py = 0; py < canvas.height; py++) {
    for (let px = 0; px < canvas.width; px++) {
      const cell = Math.floor(py / s) * map.cols + Math.floor(px / s);
      let rgb = map.trails[cell] === "1" ? TRAIL : TERRAIN[+map.terrain[cell]][1];
      if (map.no_fly[cell] === "1" && (px + py) % 6 < 2) rgb = [220, 38, 38]; // hatched no-fly area
      image.data.set([...rgb, 255], (py * canvas.width + px) * 4);
    }
  }
  ctx.putImageData(image, 0, 0);
  return canvas;
}

async function loadAerial(run) {
  const canvas = await aerialCanvas(run.place.bounds, state.base.width, state.base.height);
  if (state.run !== run) return; // another race was opened meanwhile
  state.aerial = canvas;
  render();
}

async function aerialCanvas({ north, south, west, east }, width, height) {
  // The swisstopo tiles covering these bounds, painted onto a canvas of the given size.
  const n = 2 ** AERIAL_ZOOM;
  const tx = (lon) => ((lon + 180) / 360) * n;
  const ty = (lat) => { const r = (lat * Math.PI) / 180; return ((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * n; };
  const [x0, x1, y0, y1] = [tx(west), tx(east), ty(north), ty(south)];
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const kx = width / (x1 - x0), ky = height / (y1 - y0);
  const jobs = [];
  for (let x = Math.floor(x0); x < x1; x++) {
    for (let y = Math.floor(y0); y < y1; y++) {
      const img = new Image();
      jobs.push(new Promise((resolve, reject) => {
        img.onload = () => resolve(ctx.drawImage(img, (x - x0) * kx, (y - y0) * ky, kx + 0.5, ky + 0.5));
        img.onerror = reject;
      }));
      img.src = `${AERIAL_TILES}/${AERIAL_ZOOM}/${x}/${y}.jpeg`;
    }
  }
  await Promise.all(jobs);
  return canvas;
}

function heatImage(snapshot) {
  // Probability per block, 0-255 relative to the likeliest block: transparent where unlikely, hot where likely.
  const side = snapshot.side;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = side;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(side, side);
  snapshot.values.forEach((v, i) => {
    const f = v / 255;
    image.data.set([255, 60 + 180 * f, 20 + 40 * f, 220 * Math.pow(f, 1.2)], i * 4);
  });
  ctx.putImageData(image, 0, 0);
  return canvas;
}

function render() {
  const { run, scale: s, t } = state;
  const toPx = (m) => (m / run.map.cell_m) * s;
  for (const lane of state.lanes) {
    const ctx = lane.ctx;
    const laneT = Math.min(t, lane.endT);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(state.aerial && $("show-aerial").checked ? state.aerial : state.base, 0, 0);

    const heat = lastAt(lane.maps, laneT);
    if (heat && $("show-heat").checked) {
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(heat.image, 0, 0, lane.canvas.width, lane.canvas.height);
      ctx.imageSmoothingEnabled = false;
    }

    updateSearched(lane, laneT);
    if ($("show-searched").checked) {
      ctx.globalAlpha = 0.45;
      ctx.drawImage(lane.searched, 0, 0, lane.canvas.width, lane.canvas.height);
      ctx.globalAlpha = 1;
    }

    const sameSpot = run.map.ipp.join() === run.map.base.join();
    drawMarker(ctx, run.map.ipp, s, "#ffffff", sameSpot ? "last seen · drone base" : "last seen");
    if (!sameSpot) drawMarker(ctx, run.map.base, s, "#f59e0b", "drone base");
    if ($("show-truth").checked) drawTruth(ctx, laneT, toPx);
    drawDrones(ctx, lane, laneT, toPx);
    if (lane.found && t >= lane.found.t_s) drawFound(ctx, toPx(lane.found.x_m), toPx(lane.found.y_m));
    updateLaneText(lane, t);
  }
  $("clock").textContent = `${clock(t)} in`;
  $("time").value = t;
  $("result-line").classList.toggle("hidden", t < state.endT && !$("show-truth").checked);
  updateFeed();
}

function resultLine() {
  // Every lane's find time from the run file, fastest first; shown once playback ends or the truth is revealed.
  const finds = state.lanes.map((lane) => [state.run.lanes[lane.name].result.found_s, label(lane.name)])
    .sort(([a], [b]) => (a ?? Infinity) - (b ?? Infinity));
  const notFound = `not found in ${clock(state.run.hours * 3600)}`;
  return finds.map(([s, name], i) => (s == null ? `${name} ${notFound}` : i ? `${clock(s)} ${name}` : `Found at ${clock(s)} with ${name}`)).join(" · ");
}

function updateSearched(lane, t) {
  // Photos are painted once onto the lane's own small canvas; going back in time repaints from the start.
  const ctx = lane.searched.getContext("2d");
  if (lane.photoIndex > 0 && lane.photos[lane.photoIndex - 1].t_s > t) {
    ctx.clearRect(0, 0, lane.searched.width, lane.searched.height);
    lane.photoIndex = 0;
    lane.alarmCount = 0;
  }
  ctx.fillStyle = "rgb(125, 211, 252)";
  while (lane.photoIndex < lane.photos.length && lane.photos[lane.photoIndex].t_s <= t) {
    const photo = lane.photos[lane.photoIndex++];
    const [r0, r1, c0, c1] = photo.cells;
    ctx.fillRect(c0, r0, c1 - c0, r1 - r0);
    lane.alarmCount += photo.alarms || 0;
  }
}

function drawMarker(ctx, [row, col], s, colour, text) {
  const x = (col + 0.5) * s, y = (row + 0.5) * s;
  ctx.strokeStyle = colour;
  ctx.lineWidth = 2;
  ctx.strokeRect(x - 5, y - 5, 10, 10);
  ctx.fillStyle = colour;
  ctx.font = "11px system-ui";
  ctx.fillText(text, x + 8, y + 4);
}

function drawDrones(ctx, lane, t, toPx) {
  const i = lastIndexAt(lane.frames, t);
  if (i < 0) return;
  const a = lane.frames[i], b = lane.frames[i + 1];
  const f = b ? Math.min(1, (t - a.t_s) / (b.t_s - a.t_s)) : 0;
  a.drones.forEach(([y, x, returning, lost], k) => {
    if (lost) return drawLost(ctx, toPx(x), toPx(y));
    const [y2, x2] = b ? b.drones[k] : [y, x];
    const px = toPx(x + (x2 - x) * f), py = toPx(y + (y2 - y) * f);
    ctx.beginPath();
    ctx.arc(px, py, 5, 0, 2 * Math.PI);
    ctx.fillStyle = returning ? "#0f172a" : "#fbbf24";
    ctx.strokeStyle = "#fbbf24";
    ctx.lineWidth = 2;
    ctx.fill();
    ctx.stroke();
  });
}

function drawLost(ctx, px, py) {
  ctx.strokeStyle = "#94a3b8";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(px - 5, py - 5); ctx.lineTo(px + 5, py + 5);
  ctx.moveTo(px + 5, py - 5); ctx.lineTo(px - 5, py + 5);
  ctx.stroke();
}

function drawTruth(ctx, t, toPx) {
  const { every_s, track } = state.run.truth;
  const last = Math.min(track.length - 1, Math.floor(t / every_s));
  ctx.strokeStyle = "rgba(244, 63, 94, 0.9)";
  ctx.setLineDash([4, 3]);
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  track.slice(0, last + 1).forEach(([y, x], i) => (i ? ctx.lineTo(toPx(x), toPx(y)) : ctx.moveTo(toPx(x), toPx(y))));
  ctx.stroke();
  ctx.setLineDash([]);
  const [y, x] = track[last];
  ctx.fillStyle = "#f43f5e";
  ctx.beginPath();
  ctx.arc(toPx(x), toPx(y), 4, 0, 2 * Math.PI);
  ctx.fill();
}

function drawFound(ctx, x, y) {
  ctx.strokeStyle = "#22c55e";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(x, y, 14, 0, 2 * Math.PI);
  ctx.stroke();
}

// ---------- text: lanes and feed ----------

function buildLaneDom() {
  $("lanes").innerHTML = state.lanes.map((lane, i) => `
    <article class="bg-slate-900/60 border border-slate-800 rounded-lg p-2">
      <div class="flex items-baseline justify-between gap-2 px-1">
        <h3 class="font-medium text-sm">${label(lane.name)}</h3>
        <span data-status class="text-xs font-mono"></span>
      </div>
      <p class="text-xs text-slate-400 px-1 mb-2">${(PLANNERS[lane.name] || ["", ""])[1]}</p>
      <canvas data-lane="${i}" class="w-full rounded"></canvas>
      <p data-note class="text-xs text-slate-400 px-1 mt-2 min-h-[2.5em]"></p>
      <div data-photo class="px-1"></div>
    </article>`).join("");
  document.querySelectorAll("#lanes article").forEach((el, i) => {
    const lane = state.lanes[i];
    lane.canvas = el.querySelector("canvas");
    lane.canvas.width = state.base.width;
    lane.canvas.height = state.base.height;
    lane.ctx = lane.canvas.getContext("2d");
    lane.statusEl = el.querySelector("[data-status]");
    lane.noteEl = el.querySelector("[data-note]");
    lane.photoEl = el.querySelector("[data-photo]");
  });
  $("imagery").innerHTML = state.run.imagery
    ? "Each photo of her is judged by SAM 3.1 and Muse Spark on a real drone photo of a person her size (under trees, canopy also hides her at an assumed rate). Every planner draws from the same photos, so they share the same luck. Flagged: alarms a reviewer checks."
    : "Detections are sampled from the measured table (no photo pool on this server).";
  if (state.run.place) {
    $("imagery").innerHTML += ` <br>A real place: ${escapeHtml(state.run.place.title)}. She walks over terrain and paths from OpenStreetMap (untick the aerial photo to see them). Aerial imagery © swisstopo · ${escapeHtml(state.run.place.credit)}.`;
  }
  $("legend").innerHTML = [
    ...TERRAIN.map(([name, rgb]) => swatch(`rgb(${rgb})`, name)),
    swatch(`rgb(${TRAIL})`, "Trail"),
    swatch("repeating-linear-gradient(45deg,#dc2626 0 2px,transparent 2px 6px)", "No-fly"),
    swatch("rgb(125,211,252)", "Photographed"),
    swatch("linear-gradient(90deg,rgba(255,70,30,.3),rgb(255,240,30))", "Planner thinks likely"),
  ].join("");
}

function updateLaneText(lane, t) {
  const found = lane.found && t >= lane.found.t_s;
  const ended = !lane.found && t >= lane.endT;
  lane.statusEl.textContent = found ? `FOUND at ${clock(lane.found.t_s)}` : ended ? `not found in ${clock(lane.endT)}` : `${lane.photoIndex} photo${lane.photoIndex === 1 ? "" : "s"} · ${lane.alarmCount} flagged`;
  lane.statusEl.className = `text-xs font-mono ${found ? "text-green-400 font-semibold" : ended ? "text-rose-400" : "text-slate-400"}`;
  const sortie = lastAt(lane.notes.filter((n) => ["sortie", "restart", "hunt"].includes(n.kind)), Math.min(t, lane.endT));
  lane.noteEl.textContent = sortie ? `Last decision: ${sortie.text}.` : "";
  const hit = found && lane.sightings.find((s) => s.found);
  if (Boolean(hit) !== lane.shownPhoto) {
    lane.shownPhoto = Boolean(hit);
    lane.photoEl.innerHTML = hit ? `${photoHtml(hit)}<p class="text-xs text-slate-400 mt-1">${sightingText(hit)} · ${lane.photoIndex} photos taken, ${lane.alarmCount} flagged for review</p>` : "";
  }
}

function feedItems() {
  const items = state.run.reports.map((r) => ({ t_s: r.t_s, html: reportHtml(r) }));
  for (const lane of state.lanes) {
    for (const note of lane.notes) {
      if (note.kind === "hypotheses") items.push({ t_s: note.t_s, html: hypothesesHtml(note) });
      if (note.kind === "spark_failed") items.push({ t_s: note.t_s, html: `<p class="text-amber-300">${note.text}</p>` });
      if (note.kind === "restart" || note.kind === "hunt") items.push({ t_s: note.t_s, html: `<p class="text-sky-300">${label(lane.name)}: ${note.text}</p>` });
    }
    if (lane.found) items.push({ t_s: lane.found.t_s, html: `<p class="text-green-400 font-medium">${label(lane.name)}: person found</p>` });
    if ($("show-truth").checked) {
      for (const s of lane.sightings.filter((s) => !s.found)) {
        items.push({ t_s: s.t_s, html: `<p class="text-xs uppercase tracking-wide text-rose-300">${label(lane.name)}: drone ${s.drone + 1} photographed her, no alarm</p>
          <p class="text-xs text-slate-300">${sightingText(s)}</p>${s.canopy ? "" : photoHtml(s)}` });
      }
    }
  }
  const lost = state.lanes.map((lane) => lane.lost[0]).find(Boolean); // the same failure in every lane
  if (lost) items.push({ t_s: lost.t_s, html: `<p class="text-rose-300 font-medium">Drone ${lost.drone + 1} lost (simulated failure). Each planner carries on with the others.</p>` });
  return items.sort((a, b) => b.t_s - a.t_s);
}

function updateFeed() {
  const visible = feedItems().filter((item) => item.t_s <= state.t);
  const key = `${state.run.id}|${visible.length}|${$("show-truth").checked}`;
  if (key === state.feedKey) return;
  state.feedKey = key;
  $("feed").innerHTML = visible.map((item) => `
    <li class="border-l-2 border-slate-700 pl-3">
      <div class="text-xs font-mono text-slate-500">${clock(item.t_s)}</div>${item.html}
    </li>`).join("") || `<li class="text-slate-500">Press Play.</li>`;
}

function reportHtml(report) {
  let html = `<p class="text-xs uppercase tracking-wide text-slate-400">Report</p><p>${escapeHtml(report.text)}</p>`;
  if ($("show-truth").checked) {
    const cues = state.run.truth.cues.filter((c) => c.t_s === report.t_s);
    html += cues.map((c) => `<p class="text-xs ${c.true ? "text-green-400" : "text-rose-400"}">${CUE_LABEL[c.kind] || c.kind}: ${c.true ? "correct" : "wrong (simulated witness error)"}</p>`).join("");
  }
  return html;
}

function hypothesesHtml(note) {
  const bar = (name, weight, reason) => `
    <div class="mt-1.5">
      <div class="flex justify-between gap-2"><span>${escapeHtml(name)}</span><span class="font-mono text-slate-400">${Math.round(weight * 100)}%</span></div>
      <div class="h-1.5 bg-slate-800 rounded"><div class="h-1.5 bg-amber-400 rounded" style="width:${weight * 100}%"></div></div>
      ${reason ? `<p class="text-xs text-slate-400 mt-0.5">${escapeHtml(reason)}</p>` : ""}
    </div>`;
  return `<p class="text-xs uppercase tracking-wide text-amber-300">Muse Spark: competing explanations</p>
    ${note.hypotheses.map((h) => bar(h.name, h.weight, h.reason)).join("")}
    ${bar("None of these (keep the statistical map)", note.rest_weight, "")}
    ${note.dropped.length ? `<p class="text-xs text-slate-500 mt-1.5">Discarded: ${note.dropped.map(escapeHtml).join("; ")}</p>` : ""}`;
}

function sightingText(s) {
  if (s.found) return `SAM 3.1 flagged her (${Math.round(s.sam * 100)}%), Muse Spark: "${escapeHtml(s.spark)}"`;
  if (s.canopy) return "Under tree cover: hidden from the camera (forest detection is an assumed 40% of open ground).";
  if (s.sam == null) return `SAM 3.1 proposed nothing on a person ${s.size_px} px across.`;
  return `SAM 3.1 flagged her (${Math.round(s.sam * 100)}%), but Muse Spark said "${escapeHtml(s.spark || "not a person")}".`;
}

function photoHtml(s) {
  // A window of the real 1000 px tile around the person, with the person's labelled box.
  const W = 300, H = 200;
  const [x1, y1, x2, y2] = s.box;
  const span = Math.max(200, 5 * s.size_px);
  const k = W / span;
  const left = (x1 + x2) / 2 - span / 2, top = (y1 + y2) / 2 - H / k / 2;
  const colour = s.found ? "border-green-400" : "border-rose-400";
  return `<div class="relative mt-1.5 rounded overflow-hidden bg-slate-800" style="width:${W}px;height:${H}px;max-width:100%">
      <div class="absolute inset-0" style="background:url(/api/tiles/${s.tile}.jpg) no-repeat;background-size:${1000 * k}px;background-position:${-left * k}px ${-top * k}px"></div>
      <div class="absolute border-2 ${colour}" style="left:${(x1 - left) * k - 3}px;top:${(y1 - top) * k - 3}px;width:${(x2 - x1) * k + 6}px;height:${(y2 - y1) * k + 6}px"></div>
    </div>
    <p class="text-[10px] text-slate-500 mt-0.5">Real drone photo: ${escapeHtml(state.run.photo_credit || "")}</p>`;
}

// ---------- playback ----------

let lastFrame = 0;
function tick(now) {
  if (!state.playing) return;
  const speed = +$("speed").value;
  state.t = Math.min(state.endT, state.t + ((now - lastFrame) / 1000) * speed);
  lastFrame = now;
  render();
  if (state.t >= state.endT) return pause();
  requestAnimationFrame(tick);
}

function play() {
  if (state.t >= state.endT) seek(0);
  state.playing = true;
  $("play").textContent = "Pause";
  lastFrame = performance.now();
  requestAnimationFrame(tick);
}

function pause() {
  state.playing = false;
  $("play").textContent = "Play";
}

function seek(t) {
  state.t = Math.max(0, Math.min(state.endT, t));
  render();
}

// ---------- helpers ----------

function lastIndexAt(events, t) {
  let lo = 0, hi = events.length - 1, best = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (events[mid].t_s <= t) { best = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return best;
}
const lastAt = (events, t) => events[lastIndexAt(events, t)] || null;
const label = (name) => (PLANNERS[name] || [name])[0];
const clock = (s) => `${Math.floor(s / 3600)}:${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}`;
const swatch = (bg, text) => `<span class="inline-flex items-center gap-1"><span class="inline-block w-3 h-3 rounded-sm" style="background:${bg}"></span>${text}</span>`;
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function showStatus(text, ms = 4000) {
  $("status").textContent = text;
  $("status").classList.remove("hidden");
  clearTimeout(showStatus.timer);
  showStatus.timer = setTimeout(() => $("status").classList.add("hidden"), ms);
}

// ---------- wiring ----------

$("play").onclick = () => (state.playing ? pause() : play());
$("time").oninput = (e) => { pause(); seek(+e.target.value); };
["show-heat", "show-searched", "show-truth", "show-aerial"].forEach((id) => ($(id).onchange = render));
$("run").onchange = (e) => loadRun(e.target.value).catch((err) => showStatus(err.message));
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && e.target.tagName !== "INPUT") { e.preventDefault(); $("play").click(); }
});
$("new-race").onsubmit = async (e) => {
  e.preventDefault();
  const planners = state.lanes.length ? state.lanes.map((l) => l.name) : ["lawnmower", "greedy", "greedy+spark"];
  showStatus("Recording race...", 60000);
  try {
    const { id } = await api("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ seed: +$("seed").value, planners, place: state.run?.place?.name ?? null,
                             drones: state.run?.fleet.drones ?? 3 }),
    });
    await loadRunList(id);
    showStatus("Race recorded.");
  } catch (err) {
    showStatus(`Could not record: ${err.message}`);
  }
};

async function showLiveControls() {
  // Paid buttons only when the server allows live model calls; otherwise a note that the examples are recorded.
  const live = await api("/api/health").then((h) => h.live_models).catch(() => false);
  document.querySelectorAll("[data-live]").forEach((el) => el.classList.toggle("hidden", !live));
  document.querySelectorAll("[data-live-off]").forEach((el) => el.classList.toggle("hidden", live));
}

loadRunList().then(() => state.run && play()).catch((err) => showStatus(`Could not load races: ${err.message}`));
showLiveControls();
