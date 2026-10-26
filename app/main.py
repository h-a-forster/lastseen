from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from app import llm  # noqa: E402

SYSTEM_PROMPT = "You are a helpful assistant. Be concise."  # change per project

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="hackathon-starter")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


@app.get("/api/health")
def health():
    return {"ok": True, "model": llm.current_model()}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *(m.model_dump() for m in req.messages)]

    async def generate():
        try:
            async for text in llm.stream(messages):
                yield text
        except Exception as e:  # surface provider errors in the UI instead of a dead stream
            yield f"\n\n[error: {type(e).__name__}]"
            raise

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


# Serves static/index.html at "/". Keep this last so /api routes win.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
