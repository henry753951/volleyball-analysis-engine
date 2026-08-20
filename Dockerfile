# syntax=docker/dockerfile:1.10
FROM ghcr.io/astral-sh/uv:0.11.31 AS uv
FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/
WORKDIR /app

FROM base AS dependencies
ARG TORCH_EXTRA=cu130
ARG CENTRAL_REPOSITORY=https://github.com/henry753951/volleyball-monitoring-ai.git
ARG CENTRAL_REF=main
COPY --link pyproject.toml uv.lock README.md ./
RUN apt-get update \
    && apt-get install --yes --no-install-recommends git ca-certificates \
    && git clone --depth 1 --branch "${CENTRAL_REF}" "${CENTRAL_REPOSITORY}" /volleyball-monitoring-ai \
    && rm -rf /var/lib/apt/lists/* /volleyball-monitoring-ai/.git
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ${TORCH_EXTRA} --extra models \
      --no-install-project --no-editable

FROM base AS runtime
WORKDIR /app
RUN useradd --create-home --uid 10001 worker \
    && mkdir -p /workspaces /home/worker/.cache/torch/hub \
    && chown -R worker:worker /workspaces /home/worker/.cache

# Keep the large CUDA/PyTorch environment in one stable, reusable layer. Using
# COPY --chown avoids a second multi-gigabyte layer from a recursive chown.
COPY --from=dependencies --chown=10001:10001 /app/.venv /app/.venv
COPY --link --chown=10001:10001 src /app/src
COPY --link --chown=10001:10001 scripts /app/scripts
USER worker

ENTRYPOINT ["python", "-m", "volleyball_analysis_engine", "worker"]
