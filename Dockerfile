# Works on Railway, Render, Fly - they all build from a Dockerfile.
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
ENV PORT=8000
CMD uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
