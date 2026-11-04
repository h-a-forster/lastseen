// Try it on any place: pick a point, and the server builds the map from OpenStreetMap and records a race there.
// The visitor may be the witness too: mark where she was seen, and type or say what the witness tells the team.

const PLACE_SIDE_M = 3000;
const COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"];
const place = { map: null, marker: null, square: null, seen: null, seenMarker: null, marking: false };

function placeSay(text) {
  document.getElementById("place-status").textContent = text;
}

function parsePoint(text) {
  // "47.365, 7.155", or a map link that contains "@47.365,7.155".
  const m = text.match(/(-?\d{1,2}(?:\.\d+)?)\s*[, ]\s*(-?\d{1,3}(?:\.\d+)?)/);
  const point = m && [Number(m[1]), Number(m[2])];
  return point && Math.abs(point[0]) <= 90 && Math.abs(point[1]) <= 180 ? point : null;
}

function squareAround([lat, lon]) {
  // the server centres the map on the point rounded to about 100 m (terrain.visitor_place)
  const cLat = Math.round(lat * 1000) / 1000, cLon = Math.round(lon * 1000) / 1000;
  const dLat = PLACE_SIDE_M / 2 / 111195, dLon = dLat / Math.cos((cLat * Math.PI) / 180);
  return [[cLat - dLat, cLon - dLon], [cLat + dLat, cLon + dLon]];
}

function inSquare(start, [lat, lon]) {
  const [[south, west], [north, east]] = squareAround(start);
  return lat > south && lat <= north && lon >= west && lon < east;
}

function showPoint([lat, lon]) {
  document.getElementById("place-coords").value = `${lat.toFixed(4)}, ${lon.toFixed(4)}`;
  if (place.seen && !inSquare([lat, lon], place.seen)) {
    clearSeen();
    placeSay("The witness's mark was outside the new search area, so it was cleared.");
  }
  if (place.seen) describeMark();
  if (!place.map) return;
  const square = squareAround([lat, lon]);
  if (place.square) place.square.setBounds(square);
  else place.square = L.rectangle(square, { color: "#da291c", weight: 2, fill: false }).addTo(place.map);
  if (place.marker) place.marker.setLatLng([lat, lon]);
  else place.marker = L.circleMarker([lat, lon], { radius: 6, color: "#da291c", fillOpacity: 1 }).addTo(place.map);
}

function setMarking(on) {
  place.marking = on;
  const button = document.getElementById("place-mark");
  button.setAttribute("aria-pressed", String(on));
  button.textContent = on ? "Click the map where they saw her" : place.seen ? "Move the mark" : "Mark on the map";
  place.map.getContainer().style.cursor = on ? "crosshair" : "";
}

function markSeen(point) {
  const start = parsePoint(document.getElementById("place-coords").value);
  if (!start) return placeSay("Pick the starting point first, then mark where the witness saw her.");
  if (!inSquare(start, point)) return placeSay("The witness's spot has to be inside the red square; click inside it.");
  place.seen = point;
  place.seenMarker?.remove();
  place.seenMarker = L.circleMarker(point, { radius: 6, color: "#1d4ed8", fillColor: "#ffffff", fillOpacity: 1 })
    .addTo(place.map);
  setMarking(false);
  describeMark();
  document.getElementById("place-unmark").classList.remove("hidden");
  showHeading(false);
  placeSay("Marked: she will walk there. Now type or say what the witness tells the search team.");
}

function describeMark() {
  // how far and which way the mark is from where she left (the square's centre, as the server has it)
  const [[south, west], [north, east]] = squareAround(parsePoint(document.getElementById("place-coords").value));
  const [lat0, lon0] = [(south + north) / 2, (west + east) / 2];
  const dy = (place.seen[0] - lat0) * 111195, dx = (place.seen[1] - lon0) * 111195 * Math.cos((lat0 * Math.PI) / 180);
  const way = COMPASS[Math.round(((Math.atan2(dx, dy) * 180) / Math.PI + 360) / 45) % 8];
  const km = Math.hypot(dy, dx) / 1000;
  document.getElementById("place-seen").textContent = km < 0.2 ? "Close to where she left" : `${km.toFixed(1)} km ${way} of where she left`;
}

function clearSeen() {
  place.seen = null;
  place.seenMarker?.remove();
  place.seenMarker = null;
  if (place.map) setMarking(false);
  document.getElementById("place-seen").textContent = "";
  document.getElementById("place-unmark").classList.add("hidden");
  showHeading(true);
}

function showHeading(on) {
  // with a witness's spot, the server picks the walk by the spot, not by a direction
  document.getElementById("place-heading-row").classList.toggle("hidden", !on);
  document.getElementById("place-heading").disabled = !on;
}

function witnessFields() {
  // the request's witness part, or a sentence saying what is missing
  const words = document.getElementById("place-witness").value.trim();
  if (!place.seen && !words) return { heading: document.getElementById("place-heading").value || null };
  if (!words) return "Type or say what the witness says, or clear the mark.";
  if (!place.seen) return "Mark on the map where the witness saw her, or empty the witness box.";
  if (words.length < 10) return "Say a little more about what the witness saw (at least 10 characters).";
  return { seen_lat: place.seen[0], seen_lon: place.seen[1], witness_text: words };
}

function initPlaceMap() {
  if (!window.L) {
    document.getElementById("place-map").textContent = "The map didn't load; type coordinates instead.";
    return;
  }
  place.map = L.map("place-map").setView([47.365, 7.155], 11);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 17,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(place.map);
  place.map.on("click", (e) => (place.marking ? markSeen : showPoint)([e.latlng.lat, e.latlng.lng]));
}

async function openPlaceRace(id) {
  addVisitorRace(id);
  $("run").value = id;
  await loadRun(id);
  history.replaceState(null, "", `?race=${id}`);
  $("race").scrollIntoView({ behavior: "smooth" });
  play();
}

document.getElementById("place-coords").onchange = (e) => {
  const point = parsePoint(e.target.value);
  if (!point) return;
  showPoint(point);
  place.map?.setView(point, 13);
};

document.getElementById("place-mark").onclick = () => {
  if (!place.map) return placeSay("The map didn't load, so the witness's spot can't be marked.");
  setMarking(!place.marking);
  if (place.marking) placeSay("Click inside the red square where the witness saw her.");
};
document.getElementById("place-unmark").onclick = () => {
  clearSeen();
  placeSay("Mark cleared: the simulator writes the witness report again.");
};

const placeMic = {
  button: document.getElementById("place-mic"), box: document.getElementById("place-witness"), say: placeSay,
  what: "what the witness says", idle: "Speak", done: "Check the words, then run the test search.",
  listening: "Recording... say what the witness saw, where, and how long ago.",
};
document.getElementById("place-mic").onclick = () => toggleRecording(placeMic);

document.getElementById("place-form").onsubmit = async (e) => {
  e.preventDefault();
  const point = parsePoint(document.getElementById("place-coords").value);
  if (!point) return placeSay("Click the map, or type coordinates such as 47.365, 7.155.");
  showPoint(point);
  const witness = witnessFields();
  if (typeof witness === "string") return placeSay(witness);
  const button = e.submitter;
  button.disabled = true;
  const started = Date.now();
  let retrying = "";
  const tick = () => placeSay(`${retrying}Downloading the map from OpenStreetMap, reading the reports with Muse Spark and flying both searches... ${Math.round((Date.now() - started) / 1000)} s`);
  tick();
  const timer = setInterval(tick, 1000);
  const request = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lat: point[0], lon: point[1], ...witness,
                           report_minute: Math.min(180, Math.max(10, Number(document.getElementById("place-minute").value) || 30)) }),
  };
  try {
    const { id } = await api("/api/place-race", request).catch((err) => {
      if (err.status !== 503 || !/OpenStreetMap/.test(err.message)) throw err;
      retrying = "OpenStreetMap's servers were busy, so trying once more. ";  // they often are for a minute
      return api("/api/place-race", request);
    });
    clearInterval(timer);
    placeSay(`Done in ${Math.round((Date.now() - started) / 1000)} s: the test search is playing in section 1. Tick \"Reveal where the person really was\" there to see her real path.`);
    await openPlaceRace(id);
  } catch (err) {
    clearInterval(timer);
    placeSay(`Couldn't search here: ${err.message}`);
  } finally {
    button.disabled = false;
  }
};

initPlaceMap();
