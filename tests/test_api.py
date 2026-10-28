"""The public endpoints refuse what they must: unknown planners and places, the held-out scenarios, and paid photo
checks that are switched off, aren't images, or come too often."""

import io

from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.main import app

client = TestClient(app)


def test_race_endpoint_refuses_held_out_seeds_and_unknown_planners():
    assert client.post("/api/runs", json={"seed": 900_001, "planners": ["greedy"]}).status_code == 422
    assert client.post("/api/runs", json={"seed": 1, "planners": ["mine"]}).status_code == 422
    assert client.post("/api/runs", json={"seed": 1, "planners": ["greedy"], "place": "atlantis"}).status_code == 422
    assert client.get("/api/runs/..%2Fsecrets").status_code == 404


def test_photo_checks_are_off_by_default_and_never_spend_on_bad_input_or_past_the_hourly_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(main.triage, "triage", lambda image: calls.append(image.size) or {"candidates": []})
    photo = io.BytesIO()
    Image.new("RGB", (64, 48)).save(photo, "JPEG")
    post = lambda body: client.post("/api/triage", content=body, headers={"Content-Type": "image/jpeg"})
    assert post(photo.getvalue()).status_code == 403
    monkeypatch.setattr(main, "LIVE_TRIAGE", True)
    monkeypatch.setattr(main, "_triage_times", main.defaultdict(main.deque))
    assert post(b"not an image").status_code == 422
    codes = [post(photo.getvalue()).status_code for _ in range(main.TRIAGE_PER_HOUR)]
    assert codes == [200] * (main.TRIAGE_PER_HOUR - 1) + [429]  # the refused non-image counted towards the limit
    assert calls == [(64, 48)] * (main.TRIAGE_PER_HOUR - 1)
