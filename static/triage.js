// "Is anyone in this photo?": SAM 3.1 proposals and Muse Spark's verdicts on one drone photo (app/perception/triage.py).
// Shows the example photo first; a visitor can check their own photo when the server allows live checks.

const T = (id) => document.getElementById(id);
const THUMB_PX = 96;

async function showExample() {
  const response = await fetch("/api/triage/example");
  if (!response.ok) return T("triage").classList.add("hidden");
  showTriage(await response.json(), "/api/triage/example.jpg");
}

function showTriage(result, imageUrl) {
  const { width: w, height: h } = result;
  const pct = (box) => `left:${(box[0] / w) * 100}%;top:${(box[1] / h) * 100}%;width:${((box[2] - box[0]) / w) * 100}%;height:${((box[3] - box[1]) / h) * 100}%`;
  const labelled = result.people_labelled || [];
  const order = [...result.candidates].sort((a, b) => b.alarm - a.alarm || b.sam - a.sam);
  const missed = labelled.filter((b) => !result.candidates.some((c) => c.alarm && overlaps(c.box, b)));
  T("triage-photo").innerHTML = `<img src="${imageUrl}" alt="Drone photo being checked" class="w-full rounded">` +
    missed.map((b) => `<div class="absolute border-2 border-dashed border-rose-400" style="${pct(pad(b, 18))}" title="missed person"></div>`).join("") +
    order.map((c, i) => `<div class="absolute border-2 ${c.alarm ? "border-green-400" : "border-slate-400/80"}" style="${pct(pad(c.box, 18))}"><span class="absolute -top-5 left-0 text-xs font-mono ${c.alarm ? "text-green-300" : "text-slate-300"}">${i + 1}</span></div>`).join("");
  T("triage-list").innerHTML = order.length
    ? order.map((c, i) => `<li class="flex gap-2 items-start">${thumb(imageUrl, c.box, w, h)}<div>
        <span class="font-mono text-xs text-slate-400">${i + 1}</span>
        <span class="${c.alarm ? "text-green-300 font-medium" : "text-slate-300"}">${c.alarm ? "Alarm: check this spot" : c.spark ? "Dismissed by Spark" : "Not checked (low SAM score)"}</span>
        <div class="text-xs text-slate-400">${c.spark ? `Spark: “${escapeText(c.spark.looks_like)}” (${Math.round(c.spark.confidence * 100)}%) · ` : ""}SAM 3.1: ${Math.round(c.sam * 100)}%</div>
      </div></li>`).join("")
    : `<li class="text-slate-400">SAM 3.1 proposed nobody in this photo.</li>`;
  T("triage-list").innerHTML += missed.map((b) => `<li class="flex gap-2 items-start">${thumb(imageUrl, b, w, h)}<div>
      <span class="text-rose-300">Missed: a labelled person</span>
      <div class="text-xs text-slate-400">${result.candidates.some((c) => overlaps(c.box, b)) ? "SAM 3.1 proposed it; Spark dismissed it" : "SAM 3.1 did not propose it"}</div>
    </div></li>`).join("");
  const found = labelled.length - missed.length;
  T("triage-summary").innerHTML = `${result.tiles} tiles checked · ${result.candidates.length} proposals from SAM 3.1 · ` +
    `<span class="text-green-300 font-medium">${result.alarms} alarm${result.alarms === 1 ? "" : "s"}</span> after Muse Spark` +
    (labelled.length ? ` · finds ${found} of the ${labelled.length} people labelled in this photo (misses dashed red)` : "") +
    ` · model calls $${result.usd.toFixed(3)}${result.cached ? " (replayed from cache)" : ""}`;
  T("triage-credit").textContent = result.credit ? `Example photo: ${result.name}. ${result.credit}` : "Your photo is checked and then discarded; it is not stored.";
}

function pad(box, px) {
  return [box[0] - px, box[1] - px, box[2] + px, box[3] + px];
}

function overlaps(a, b) {
  return a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3];
}

function thumb(url, box, w, h) {
  // A square close-up around the box, cut from the photo with CSS (no extra request).
  const side = Math.max(120, 3 * Math.max(box[2] - box[0], box[3] - box[1]));
  const k = THUMB_PX / side;
  const x = (box[0] + box[2]) / 2 - side / 2, y = (box[1] + box[3]) / 2 - side / 2;
  return `<div class="shrink-0 rounded bg-slate-800" style="width:${THUMB_PX}px;height:${THUMB_PX}px;background:url('${url}') ${-x * k}px ${-y * k}px / ${w * k}px ${h * k}px no-repeat"></div>`;
}

function escapeText(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

T("triage-file").onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  T("triage-summary").textContent = "Checking every tile with SAM 3.1, then each proposal with Muse Spark...";
  try {
    const response = await fetch("/api/triage", { method: "POST", headers: { "Content-Type": file.type || "image/jpeg" }, body: file });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || response.statusText);
    showTriage(body, URL.createObjectURL(file));
  } catch (err) {
    T("triage-summary").textContent = `Could not check the photo: ${err.message}`;
  }
  e.target.value = "";
};

showExample().catch(() => T("triage").classList.add("hidden"));
