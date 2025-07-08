# SPDX-FileCopyrightText: 2024-2025 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
# SPDX-License-Identifier: GPL-2.0-only

ARG PYTHON_VERSION=3.10 \
    POETRY_VERSION=2.1.3

FROM python:${PYTHON_VERSION} AS builder

ARG POETRY_VERSION

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PATH="/app/venv/bin:${PATH}"

WORKDIR /app

RUN python3 -m venv venv \
 && pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir poetry==${POETRY_VERSION} poetry-plugin-export
#&& venv/bin/poetry self add poetry-plugin-export

COPY LICENSE README.md pyproject.toml poetry.lock ./

RUN poetry export --format=requirements.txt --only=main --without-hashes \
    --extras=scout --output requirements-scout.txt \
 && poetry export --format=requirements.txt --only=main --without-hashes \
    --extras=home --output requirements-home.txt

COPY src ./src/

RUN poetry build --format=wheel --no-ansi --no-interaction --no-cache --output=.

#
# home container (build --target=home)
#
#FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS home
FROM python:${PYTHON_VERSION} AS home

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1 \
    PATH="/app/venv/bin:${PATH}"

WORKDIR /app

COPY --from=builder /app/requirements-home.txt ./

RUN python3 -m venv venv \
 && pip install --no-cache-dir -r requirements-home.txt

COPY --from=builder /app/cmuhawk-*-py3-none-any.whl ./

RUN pip install --no-cache-dir cmuhawk-*-py3-none-any.whl

#
# scout container (build --target=scout)
#
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS scout

ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1 \
    PATH="/app/venv/bin:${PATH}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python3-pip \
    python3-venv \
 && apt-get clean -y && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/requirements-scout.txt ./

RUN python3 -m venv venv \
 && pip install --no-cache-dir -r requirements-scout.txt

COPY --from=builder /app/cmuhawk-*-py3-none-any.whl ./

RUN pip install --no-cache-dir cmuhawk-*-py3-none-any.whl
