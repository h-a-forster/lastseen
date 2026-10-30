"""The public endpoints refuse what they must: unknown planners and places, the held-out scenarios, and paid photo
checks that are switched off, aren't images, or come too often."""

import io
import json
import os

from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.main import app

client = TestClient(app)


def test_race_endpoint_is_off_by_default_and_refuses_held_out_seeds_and_unknown_planners(monkeypatch):
    assert client.post("/api/runs", json={"seed": 1, "planners": ["spiral"], "drones": 1}).status_code == 403
    assert client.get("/api/health").json()["record_races"] is False
    monkeypatch.setattr(main, "RECORD_RACES", True)
    assert client.post("/api/runs", json={"seed": 900_001, "planners": ["greedy"]}).status_code == 422
    assert client.post("/api/runs", json={"seed": 1, "planners": ["mine"]}).status_code == 422
    assert client.post("/api/runs", json={"seed": 1, "planners": ["greedy"], "place": "atlantis"}).status_code == 422
    assert client.post("/api/runs", json={"seed": 1, "planners": ["greedy"], "drones": 0}).status_code == 422
    assert client.get("/api/runs/..%2Fsecrets").status_code == 404


def test_a_visitors_place_race_is_paid_tells_a_busy_map_server_apart_and_stays_off_the_public_list(tmp_path, monkeypatch):
    body = {"lat": 47.365, "lon": 7.155}
    assert client.post("/api/place-race", json=body).status_code == 403  # live model calls are off by default
    monkeypatch.setattr(main, "LIVE_MODELS", True)
    monkeypatch.setattr(main, "_paid_times", main.defaultdict(main.deque))
    assert client.post("/api/place-race", json={"lat": 89.0, "lon": 7.0}).status_code == 422

    def busy(lat, lon):
        raise main.terrain.MapDataUnavailable("OpenStreetMap's servers are busy; try again in a minute")
    monkeypatch.setattr(main.terrain, "visitor_place", busy)
    assert client.post("/api/place-race", json=body).status_code == 503

    monkeypatch.setattr(main.replay, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(main, "_summaries", {})
    monkeypatch.setattr(main.terrain, "visitor_place", lambda lat, lon: "at-47365n-7155e")
    monkeypatch.setattr(main.replay, "watchable_seed", lambda name, first: 7)
    raced = []
    monkeypatch.setattr(main.replay, "race", lambda *a, **k: raced.append((a, k)) or {"id": "at-47365n-7155e-race"})
    run_id = client.post("/api/place-race", json=body | {"witness": "  Seen by the church. "}).json()["id"]
    (args, kwargs), = raced
    assert args == (7, ["spiral", "hedge+spark"]) and kwargs["witness"].text == "Seen by the church."
    assert client.get(f"/api/runs/{run_id}").status_code == 200
    assert client.get("/api/runs").json() == []  # a visitor's race isn't listed for everyone


def test_photo_checks_are_off_by_default_and_never_spend_on_bad_input_or_past_the_hourly_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(main.triage, "triage", lambda image: calls.append(image.size) or {"candidates": []})
    photo = io.BytesIO()
    Image.new("RGB", (64, 48)).save(photo, "JPEG")
    post = lambda body: client.post("/api/triage", content=body, headers={"Content-Type": "image/jpeg"})
    assert post(photo.getvalue()).status_code == 403
    monkeypatch.setattr(main, "LIVE_MODELS", True)
    monkeypatch.setattr(main, "_paid_times", main.defaultdict(main.deque))
    assert post(b"not an image").status_code == 422
    codes = [post(photo.getvalue()).status_code for _ in range(main.PAID_PER_HOUR)]
    assert codes == [200] * (main.PAID_PER_HOUR - 1) + [429]  # the refused non-image counted towards the limit
    assert calls == [(64, 48)] * (main.PAID_PER_HOUR - 1)


def test_browsers_check_for_a_newer_page_on_every_visit():
    for path in ("/", "/style.css", "/place.js"):
        assert client.get(path).headers["cache-control"] == "no-cache"


def test_health_tells_the_page_whether_live_model_calls_are_on(monkeypatch):
    assert client.get("/api/health").json()["live_models"] is False
    monkeypatch.setattr(main, "LIVE_MODELS", True)
    assert client.get("/api/health").json()["live_models"] is True


def test_runs_list_featured_races_first_in_their_order_then_newest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(main.replay, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(main, "_summaries", {})
    oldest_first = ["old", main.FEATURED_RUNS[0], "new", main.FEATURED_RUNS[2]]  # FEATURED_RUNS[1] is missing
    for age, run_id in enumerate(reversed(oldest_first)):
        path = tmp_path / f"{run_id}.json"
        path.write_text(json.dumps({"id": run_id, "seed": 1, "category": "dementia", "fleet": {"drones": 3},
                                    "lanes": {}}), encoding="utf-8")
        os.utime(path, (1_000_000 - age, 1_000_000 - age))
    listed = [run["id"] for run in client.get("/api/runs").json()]
    assert listed == [main.FEATURED_RUNS[0], main.FEATURED_RUNS[2], "new", "old"]


def test_a_race_can_be_recorded_with_one_drone(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "RECORD_RACES", True)
    monkeypatch.setattr(main.replay, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(main, "_summaries", {})
    run_id = client.post("/api/runs", json={"seed": 1, "planners": ["spiral"], "drones": 1}).json()["id"]
    assert [(run["id"], run["drones"]) for run in client.get("/api/runs").json()] == [(run_id, 1)]
    assert run_id != main.replay.run_id(1, "dementia", ["spiral"])  # doesn't overwrite the three-drone race


def test_a_brief_can_be_read_on_open_weights_and_the_server_switches_back_after(monkeypatch):
    seen = []
    monkeypatch.setattr(main.brief, "read_brief", lambda place, text: seen.append(os.environ["SPARK_BACKEND"]) or {})
    monkeypatch.setattr(main, "LIVE_MODELS", True)
    monkeypatch.setattr(main, "_paid_times", main.defaultdict(main.deque))
    body = {"place": "saint-ursanne", "text": "Seen by the river."}
    assert client.post("/api/brief", json=body | {"open_weights": True}).status_code == 200
    assert os.environ["SPARK_BACKEND"] == "meta"
    assert client.post("/api/brief", json=body).status_code == 200
    assert seen == ["open", "meta"]


def test_the_help_box_only_ever_returns_written_answers(monkeypatch):
    monkeypatch.setattr(main, "LIVE_MODELS", False)
    close = client.post("/api/ask", json={"question": "Does the AI fly the drone?"}).json()
    assert close["matched_by"] == "words" and close["answers"][0]["id"] == "control"
    assert main.faq.instant("Who is it for?")["id"] == "who"  # only common words, but the written question itself
    assert main.faq.instant("what does spark do")["id"] == "spark"  # not the longer question that also names Spark
    offline = client.post("/api/ask", json={"question": "Tell me a joke"}).json()
    assert offline["answers"] == [] and len(offline["suggestions"]) == 3
    answer = main.meta_api.Answer({"ids": ["open", "made-up", "open"]}, 0.001, False, "m")
    monkeypatch.setattr(main.faq, "spark_json", lambda *a, **k: answer)
    assert [e["id"] for e in main.faq.route("Can it run without signal?")["answers"]] == ["open"]

