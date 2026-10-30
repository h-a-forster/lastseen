// Search a place you know: pick a point, and the server builds the map from OpenStreetMap and records a race there.

const PLACE_SIDE_M = 3000;
const place = { map: null, marker: null, square: null };

function placeSay(text) {
  document.getElementById("place-status").textContent = text;
}

function parsePoint(text) {
  // "47.365, 7.155", or a map link that contains "@47.365,7.155".
  const m = text.match(/(-?\d{1,2}(?:\.\d+)?)\s*[, ]\s*(-?\d{1,3}(?:\.\d+)?)/);
  const point = m && [Number(m[1]), Number(m[2])];
  return point && Math.abs(point[0]) <= 90 && Math.abs(point[1]) <= 180 ? point : null;
}

function showPoint([lat, lon]) {
  document.getElementById("place-coords").value = `${lat.toFixed(4)}, ${lon.toFixed(4)}`;
  if (!place.map) return;
  const dLat = PLACE_SIDE_M / 2 / 111195, dLon = dLat / Math.cos((lat * Math.PI) / 180);
  const square = [[lat - dLat, lon - dLon], [lat + dLat, lon + dLon]];
  if (place.square) place.square.setBounds(square);
  else place.square = L.rectangle(square, { color: "#da291c", weight: 2, fill: false }).addTo(place.map);
  if (place.marker) place.marker.setLatLng([lat, lon]);
  else place.marker = L.circleMarker([lat, lon], { radius: 6, color: "#da291c", fillOpacity: 1 }).addTo(place.map);
}

function initPlaceMap() {
  if (!window.L) {
    document.getElementById("place-map").textContent = "The map didn't load; type coordinates instead.";
    return;
  }
  place.map = L.map("place-map", { scrollWheelZoom: false }).setView([47.365, 7.155], 11);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 17,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(place.map);
  place.map.on("click", (e) => showPoint([e.latlng.lat, e.latlng.lng]));
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

document.getElementById("place-form").onsubmit = async (e) => {
  e.preventDefault();
  const point = parsePoint(document.getElementById("place-coords").value);
  if (!point) return placeSay("Click the map, or type coordinates such as 47.365, 7.155.");
  showPoint(point);
  const button = e.submitter;
  button.disabled = true;
  const started = Date.now();
  const tick = () => placeSay(`Downloading the map from OpenStreetMap, reading the reports with Muse Spark and flying both searches... ${Math.round((Date.now() - started) / 1000)} s`);
  tick();
  const timer = setInterval(tick, 1000);
  try {
    const { id } = await api("/api/place-race", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lat: point[0], lon: point[1], witness: document.getElementById("place-witness").value.trim() || null,
                             witness_minute: Math.min(180, Math.max(10, Number(document.getElementById("place-minute").value) || 45)) }),
    });
    clearInterval(timer);
    placeSay("Done: the race is playing in section 1.");
    await openPlaceRace(id);
  } catch (err) {
    clearInterval(timer);
    placeSay(`Couldn't search here: ${err.message}`);
  } finally {
    button.disabled = false;
  }
};

initPlaceMap();
