# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.11.31 AS uv
FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/
WORKDIR /app

FROM base AS build
COPY pyproject.toml uv.lock README.md ./
# `central` is a named build context pointing at the sibling
# ../volleyball-monitoring-ai repository. The SDK's Hatch build includes these
# two contract assets, so preserve the central repository's relative layout.
COPY --from=central sdk /volleyball-monitoring-ai/sdk
COPY --from=central packages/contracts/flatbuffers/overlay.fbs /volleyball-monitoring-ai/packages/contracts/flatbuffers/overlay.fbs
COPY --from=central packages/contracts/fixtures/normal-rally/result.json /volleyball-monitoring-ai/packages/contracts/fixtures/normal-rally/result.json
RUN uv sync --frozen --no-dev --extra cpu --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra cpu

FROM base AS runtime
WORKDIR /app
COPY --from=build /app /app

RUN useradd --create-home --uid 10001 worker \
    && mkdir -p /workspaces \
    && chown -R worker:worker /app /workspaces
USER worker

ENTRYPOINT ["volleyball-analysis-worker"]
