FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
ENV PORT=8000
CMD uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
