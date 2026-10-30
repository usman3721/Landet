# syntax=docker/dockerfile:1

# ---------- Stage 1: build the virtualenv with uv ----------
FROM python:3.11-slim AS builder

# Pinned to the uv version that wrote uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app
COPY pyproject.toml uv.lock ./
# --locked: fail if uv.lock is out of date; --no-default-groups: runtime packages only
RUN uv sync --locked --no-default-groups --no-install-project

# ---------- Stage 2: slim runtime image ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
# The model is not in the image: docker-compose mounts ./models at runtime.

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
