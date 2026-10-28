"""The public race endpoint refuses what it must: unknown planners and the held-out scenarios."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_race_endpoint_refuses_held_out_seeds_and_unknown_planners():
    assert client.post("/api/runs", json={"seed": 900_001, "planners": ["greedy"]}).status_code == 422
    assert client.post("/api/runs", json={"seed": 1, "planners": ["mine"]}).status_code == 422
    assert client.get("/api/runs/..%2Fsecrets").status_code == 404
