// Map-sheet furniture for the page: a neatline, grid ticks with coordinates in the margins, a north arrow and a
// scale bar around every map canvas. Reads the map extent from replay.js (state) and brief.js (brief).

const SHEET_NS = "http://www.w3.org/2000/svg";

function lv95([lat, lon]) {
  // swisstopo's approximate WGS84 -> LV95 formulas (about 1 m accuracy), enough for km ticks.
  const p = (lat * 3600 - 169028.66) / 10000, l = (lon * 3600 - 26782.5) / 10000;
  return {
    e: 2600072.37 + 211455.93 * l - 10938.51 * l * p - 0.36 * l * p * p - 44.54 * l ** 3,
    n: 1200147.07 + 308807.95 * p + 3745.25 * l * l + 76.63 * p * p - 194.56 * l * l * p + 119.79 * p ** 3,
  };
}

function swissKm(m) {
  const km = String(Math.round(m / 1000));
  return km.length > 3 ? `${km.slice(0, -3)}'${km.slice(-3)}` : km;
}

function gridTicks(lo, hi, format) {
  // Whole-kilometre ticks between lo and hi (metres), as fractions of the span.
  const ticks = [];
  for (let v = Math.ceil(lo / 1000) * 1000; v <= hi; v += 1000) ticks.push({ f: (v - lo) / (hi - lo), text: format(v) });
  return ticks;
}

const SWISS_GRID = { south: 45.8, north: 47.85, west: 5.9, east: 10.55 }; // where LV95 coordinates mean something

function inSwissGrid(lat, lon) {
  return lat >= SWISS_GRID.south && lat <= SWISS_GRID.north && lon >= SWISS_GRID.west && lon <= SWISS_GRID.east;
}

function sheetAxes(bounds, widthM) {
  if (!bounds || !inSwissGrid((bounds.north + bounds.south) / 2, (bounds.west + bounds.east) / 2)) {
    const local = (v) => `${v / 1000} km`;
    return { x: gridTicks(0, widthM, local), y: gridTicks(0, widthM, local), datum: "Local grid, 1 km ticks" };
  }
  const midLat = (bounds.north + bounds.south) / 2, midLon = (bounds.west + bounds.east) / 2;
  const w = lv95([midLat, bounds.west]).e, e = lv95([midLat, bounds.east]).e;
  const s = lv95([bounds.south, midLon]).n, n = lv95([bounds.north, midLon]).n;
  const y = gridTicks(s, n, swissKm).map((t) => ({ ...t, f: 1 - t.f })); // north is up
  return { x: gridTicks(w, e, swissKm), y, datum: "Swiss grid LV95, km" };
}

function northArrow() {
  const svg = document.createElementNS(SHEET_NS, "svg");
  svg.setAttribute("viewBox", "0 0 16 28");
  svg.setAttribute("class", "north");
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = '<path d="M8 2 L13 17 L8 14 Z" class="n-dark"/><path d="M8 2 L3 17 L8 14 Z" class="n-light"/>' +
    '<text x="8" y="27" text-anchor="middle">N</text>';
  return svg;
}

function scaleBar(widthM) {
  // A 1 km bar (500 m segments), drawn as a share of the map width so it stays true at any display size.
  const share = (1000 / widthM) * 100;
  const bar = document.createElement("div");
  bar.className = "scalebar";
  bar.innerHTML = `<div class="scalebar-bar" style="width:${share}%"><span></span><span></span></div>
    <div class="scalebar-labels" style="width:${share}%"><span>0</span><span>500</span><span>1 km</span></div>`;
  return bar;
}

function dressFrame(frame, { bounds, widthM }) {
  // (Re)builds the margins of one frame; the canvas inside is left alone.
  frame.querySelectorAll(".margin").forEach((el) => el.remove());
  const axes = sheetAxes(bounds, widthM);
  const tick = (side, t) => `<span class="tick" style="${side === "top" ? "left" : "top"}:${(t.f * 100).toFixed(2)}%">${t.text}</span>`;
  const add = (cls, html) => {
    const el = document.createElement("div");
    el.className = `margin ${cls}`;
    el.innerHTML = html;
    frame.append(el);
    return el;
  };
  add("m-top", axes.x.map((t) => tick("top", t)).join(""));
  add("m-left", axes.y.map((t) => tick("left", t)).join(""));
  const foot = add("m-foot", `<span class="datum">${axes.datum}</span>`);
  foot.prepend(scaleBar(widthM));
  add("m-north", "").append(northArrow());
}

function frameLanes() {
  const run = typeof state !== "undefined" && state.run;
  if (!run) return;
  const extent = { bounds: run.place?.bounds || null, widthM: run.map.cols * run.map.cell_m };
  document.querySelectorAll("#lanes canvas[data-lane]").forEach((canvas) => {
    let frame = canvas.parentElement;
    if (!frame.classList.contains("frame")) {
      frame = document.createElement("div");
      frame.className = "frame";
      canvas.replaceWith(frame);
      frame.append(canvas);
    }
    dressFrame(frame, extent);
  });
}

function frameBrief() {
  const result = typeof brief !== "undefined" && brief.result;
  if (!result) return;
  dressFrame(document.querySelector('[data-frame="brief"]'), {
    bounds: result.place.bounds,
    widthM: result.map.cols * result.map.cell_m,
  });
}

new MutationObserver(frameLanes).observe(document.getElementById("lanes"), { childList: true });
new MutationObserver(frameBrief).observe(document.getElementById("brief-credit"), { childList: true, characterData: true, subtree: true });
