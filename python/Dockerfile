# 1. Set the base image to Python 3.12
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync

FROM python:3.12-slim AS application
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
ADD ./app/main.py /app/main.py
ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT []

