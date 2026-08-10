# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.11.31 AS uv
FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/
WORKDIR /app

FROM base AS dependencies
COPY pyproject.toml uv.lock README.md ./
# `central` is a named build context pointing at the sibling
# ../volleyball-monitoring-ai repository. The SDK's Hatch build includes these
# two contract assets, so preserve the central repository's relative layout.
COPY --from=central sdk /volleyball-monitoring-ai/sdk
COPY --from=central packages/contracts/flatbuffers/overlay.fbs /volleyball-monitoring-ai/packages/contracts/flatbuffers/overlay.fbs
COPY --from=central packages/contracts/fixtures/normal-rally/result.json /volleyball-monitoring-ai/packages/contracts/fixtures/normal-rally/result.json
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra cu130 --extra models \
      --no-install-project --no-editable

FROM base AS runtime
WORKDIR /app
RUN useradd --create-home --uid 10001 worker \
    && mkdir -p /workspaces \
    && chown worker:worker /workspaces

# Keep the large CUDA/PyTorch environment in one stable, reusable layer. Using
# COPY --chown avoids a second multi-gigabyte layer from a recursive chown.
COPY --from=dependencies --chown=10001:10001 /app/.venv /app/.venv
COPY --chown=10001:10001 src /app/src
USER worker

ENTRYPOINT ["python", "-m", "volleyball_analysis_engine", "worker"]
