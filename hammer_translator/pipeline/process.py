"""Shared cancellable subprocess support for long-running local stages."""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .executor import (
    CancellationCheck,
    ProgressCallback,
    StageCancelledError,
    StageExecutionError,
)


@dataclass(frozen=True)
class LoggedProcessResult:
    return_code: int
    log_tail: tuple[str, ...]
    elapsed_seconds: float


class CancellableProcessRunner:
    def __init__(
        self,
        *,
        poll_interval: float = 0.25,
        heartbeat_interval: float = 5.0,
        progress_sink: Callable[[str], None] | None = None,
    ):
        self.poll_interval = poll_interval
        self.heartbeat_interval = heartbeat_interval
        self.progress_sink = progress_sink

    @staticmethod
    def _terminate(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except OSError:
                pass

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        log_path: Path,
        executor_name: str,
        heartbeat_message: str,
        start_error_message: str,
        start_error_code: str,
        progress: ProgressCallback,
        is_cancel_requested: CancellationCheck,
    ) -> LoggedProcessResult:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=os.name == "posix",
            )
        except OSError as exc:
            raise StageExecutionError(
                start_error_message,
                code=start_error_code,
                retryable=True,
                detail={"error_type": type(exc).__name__},
            ) from exc

        lines: queue.Queue[str | None] = queue.Queue()

        def pump_output() -> None:
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put(None)

        reader = threading.Thread(target=pump_output, daemon=True)
        reader.start()
        started = time.monotonic()
        last_progress = started
        stream_finished = False
        cancelled = False
        tail: list[str] = []
        try:
            with log_path.open("a", encoding="utf-8") as log:
                while process.poll() is None or not stream_finished:
                    if is_cancel_requested():
                        self._terminate(process)
                        cancelled = True
                        break
                    try:
                        line = lines.get(timeout=self.poll_interval)
                    except queue.Empty:
                        line = ""
                    if line is None:
                        stream_finished = True
                    elif line:
                        log.write(line)
                        log.flush()
                        cleaned = line.strip()
                        if cleaned:
                            tail.append(cleaned[-1000:])
                            tail = tail[-20:]
                            elapsed = round(time.monotonic() - started, 1)
                            progress(
                                0,
                                None,
                                cleaned[-500:],
                                {"executor": executor_name, "elapsed_seconds": elapsed},
                            )
                            if self.progress_sink is not None:
                                self.progress_sink(cleaned[-1000:])
                            last_progress = time.monotonic()
                    now = time.monotonic()
                    if now - last_progress >= self.heartbeat_interval:
                        elapsed = round(now - started, 1)
                        heartbeat = f"{heartbeat_message} ({elapsed}s elapsed)"
                        progress(
                            0,
                            None,
                            heartbeat,
                            {"executor": executor_name, "elapsed_seconds": elapsed},
                        )
                        if self.progress_sink is not None:
                            self.progress_sink(heartbeat)
                        last_progress = now
        except KeyboardInterrupt as exc:
            self._terminate(process)
            raise StageExecutionError(
                f"{executor_name} was interrupted",
                code="stage_interrupted",
                retryable=True,
            ) from exc
        except BaseException:
            self._terminate(process)
            raise
        finally:
            reader.join(timeout=1)
            if process.stdout is not None:
                process.stdout.close()
        if cancelled:
            raise StageCancelledError(f"{executor_name} was cancelled")
        return_code = process.wait()
        return LoggedProcessResult(
            return_code=return_code,
            log_tail=tuple(tail),
            elapsed_seconds=round(time.monotonic() - started, 1),
        )
