# SPDX-FileCopyrightText: 2026 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""A training run that fails on a GPU error is retried, other failures are not."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from hawk.scout.core import model_trainer
from hawk.scout.core.config import ModelTrainerConfig
from hawk.scout.core.model_trainer import ModelTrainer, run_and_tail

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.scout

CUDA_OOM = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20 MiB"


def fails_then_succeeds(tmp_path: Path, failures: int, message: str) -> list[str]:
    """A command that prints message to stderr and fails the first times."""
    counter = tmp_path / "attempts"
    script = (
        "import pathlib, sys\n"
        f"p = pathlib.Path({str(counter)!r})\n"
        "n = int(p.read_text()) if p.exists() else 0\n"
        "p.write_text(str(n + 1))\n"
        f"if n < {failures}:\n"
        f"    sys.stderr.write({message!r} + '\\n')\n"
        "    sys.exit(1)\n"
    )
    return [sys.executable, "-c", script]


def attempts(tmp_path: Path) -> int:
    return int((tmp_path / "attempts").read_text())


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(model_trainer.time, "sleep", slept.append)
    return slept


def run(cmd: list[str], retries: int, delay: float = 60.0) -> None:
    trainer = SimpleNamespace(
        config=SimpleNamespace(train_gpu_retries=retries, train_gpu_retry_delay=delay)
    )
    ModelTrainer.run_training(trainer, cmd)  # type: ignore[arg-type]


def test_success_runs_once(tmp_path: Path, sleeps: list[float]) -> None:
    run(fails_then_succeeds(tmp_path, 0, ""), retries=3)
    assert attempts(tmp_path) == 1
    assert sleeps == []


def test_gpu_error_is_retried(tmp_path: Path, sleeps: list[float]) -> None:
    run(fails_then_succeeds(tmp_path, 2, CUDA_OOM), retries=3, delay=60.0)
    assert attempts(tmp_path) == 3
    assert len(sleeps) == 2
    assert all(30.0 <= s <= 90.0 for s in sleeps)


@pytest.mark.parametrize(
    "message",
    [
        "RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED",
        "RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR",
    ],
)
def test_cudnn_errors_are_retried(
    tmp_path: Path, sleeps: list[float], message: str
) -> None:
    run(fails_then_succeeds(tmp_path, 1, message), retries=1)
    assert attempts(tmp_path) == 2


def test_other_errors_are_not_retried(tmp_path: Path, sleeps: list[float]) -> None:
    cmd = fails_then_succeeds(tmp_path, 1, "FileNotFoundError: train.txt")
    with pytest.raises(subprocess.CalledProcessError) as error:
        run(cmd, retries=3)
    assert attempts(tmp_path) == 1
    assert sleeps == []
    assert "FileNotFoundError" in error.value.stderr


def test_gives_up_after_the_last_retry(tmp_path: Path, sleeps: list[float]) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        run(fails_then_succeeds(tmp_path, 5, CUDA_OOM), retries=2)
    assert attempts(tmp_path) == 3
    assert len(sleeps) == 2


def test_no_retries_by_default(tmp_path: Path, sleeps: list[float]) -> None:
    retries = ModelTrainerConfig().train_gpu_retries
    with pytest.raises(subprocess.CalledProcessError):
        run(fails_then_succeeds(tmp_path, 1, CUDA_OOM), retries=retries)
    assert attempts(tmp_path) == 1


def test_stderr_is_passed_through(capsys: pytest.CaptureFixture[str]) -> None:
    cmd = [sys.executable, "-c", "import sys; sys.stderr.write('a\\nb\\nc\\n')"]
    returncode, tail = run_and_tail(cmd, lines=2)
    assert returncode == 0
    assert tail == "b\nc\n"
    assert capsys.readouterr().err == "a\nb\nc\n"
