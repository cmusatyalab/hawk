# SPDX-FileCopyrightText: 2024-2025 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
# SPDX-License-Identifier: GPL-2.0-only

ARG PYTHON_VERSION=3.10

FROM python:${PYTHON_VERSION} AS builder

ARG PYTHON_VERSION

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY LICENSE README.md pyproject.toml uv.lock ./

RUN uv export --format requirements.txt --no-hashes --no-annotate --no-dev \
    --no-emit-project --extra scout -q -o requirements-scout.txt \
 && uv export --format requirements.txt --no-hashes --no-annotate --no-dev \
    --no-emit-project --extra home -q -o requirements-home.txt

COPY src ./src/

RUN uv build --no-cache --wheel -o .

#
# home container (build --target=home)
#
#FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS home
FROM python:${PYTHON_VERSION} AS home

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY --from=builder /app/requirements-home.txt ./

RUN uv venv && uv pip sync --no-cache requirements-home.txt

COPY --from=builder /app/cmuhawk-*-py3-none-any.whl ./

RUN uv pip install --no-cache ./cmuhawk-*-py3-none-any.whl

#
# scout container (build --target=scout)
#
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS scout

ARG PYTHON_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && apt-get clean -y && rm -rf /var/lib/apt/lists/*


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY --from=builder /app/requirements-scout.txt ./

RUN uv venv --python ${PYTHON_VERSION} \
 && uv pip sync --no-cache \
    --index https://storage.cmusatyalab.org/wheels \
    --index https://download.pytorch.org/whl/cu118 \
    --index-strategy unsafe-best-match \
    requirements-scout.txt

COPY --from=builder /app/cmuhawk-*-py3-none-any.whl ./

RUN uv pip install --no-cache ./cmuhawk-*-py3-none-any.whl
