"""The hosted demo on Modal: the same FastAPI app, with the recorded races and page examples baked into the image.

    uvx --from modal==1.6.2 modal deploy deploy/modal_app.py

Needs a Modal Secret named `lastseen-meta` holding MODEL_API_KEY. Live model calls stay under the spend cap below and
the per-visitor hourly limit in `app/main.py`. The cache and spend ledger live on a Volume and the cap counts the whole
ledger, so it holds across container restarts; one container at most.
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
# What the page and "Record race" read; the API cache and the downloaded dataset stay local.
DATA = ["runs", "briefs", "triage", "tiles", "perception", "places", "pod_table.json", "lost_person_stats.json"]

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_sync(str(ROOT), extra_options="--no-dev")
    .env({"LIVE_MODELS": "1", "USD_CAP": "20", "USD_CAP_INCLUDES_LEDGER": "1", "DATA_DIR": "/vol"})
    .add_local_dir(ROOT / "app", "/root/app", ignore=["**/__pycache__"])
    .add_local_dir(ROOT / "static", "/root/static")
)
for name in DATA:
    path = ROOT / "data" / name
    image = image.add_local_dir(path, f"/root/data/{name}") if path.is_dir() else image.add_local_file(
        path, f"/root/data/{name}")

app = modal.App("lastseen", image=image)
cache = modal.Volume.from_name("lastseen-cache", create_if_missing=True)


@app.function(secrets=[modal.Secret.from_name("lastseen-meta")], volumes={"/vol": cache}, max_containers=1,
              scaledown_window=1200)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web():
    from app.main import app as fastapi_app

    return fastapi_app
