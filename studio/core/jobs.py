# -*- coding: utf-8 -*-
"""任务（子进程）管理。

worker 与界面的通信协议：worker 的 **stdout** 每行两种形态
- `@@EVT@@{"type": ...}`  结构化事件
- 其它任意文本            普通日志

读取线程把 (job_key, kind, payload) 放进一个 queue，界面用 after() 轮询消费，
避免任何子线程直接操作 tkinter 控件。
"""
from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path

from . import procs

EVT_PREFIX = "@@EVT@@"


class Job:
    def __init__(self, key: str, args: list[str], cwd: str | Path,
                 env: dict[str, str], q: queue.Queue, label: str = "",
                 stdin_pipe: bool = False, stop_file: Path | None = None):
        self.key = key
        self.token = f"{key}-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"
        self.args = args
        self.cwd = cwd
        self.env = env
        self.q = q
        self.label = label or key
        self.stdin_pipe = stdin_pipe
        self.stop_file = stop_file
        self.proc: subprocess.Popen | None = None
        self.started_at = 0.0
        self.ended_at: float | None = None
        self.returncode: int | None = None
        self._soft_stopped = False
        self._killed = False
        self._threads: list[threading.Thread] = []

    # ------------------------------------------------------------ 生命周期
    def start(self) -> None:
        self.proc = procs.popen(self.args, cwd=self.cwd, env=self.env, stdin_pipe=self.stdin_pipe)
        self.started_at = time.time()
        for stream, kind in ((self.proc.stdout, "out"), (self.proc.stderr, "err")):
            t = threading.Thread(target=self._reader, args=(stream, kind), daemon=True)
            t.start()
            self._threads.append(t)
        threading.Thread(target=self._waiter, daemon=True).start()

    def _reader(self, stream, kind: str) -> None:
        try:
            for raw in iter(stream.readline, b""):
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if not line:
                    continue
                if line.startswith(EVT_PREFIX):
                    payload = line[len(EVT_PREFIX):].strip()
                    try:
                        obj = json.loads(payload)
                        if isinstance(obj, dict):
                            obj.setdefault("_job", self.token)
                        self.q.put((self.key, "evt", obj))
                        continue
                    except json.JSONDecodeError:
                        self.q.put((self.key, "log", f"[协议错误] {payload[:400]}"))
                        continue
                self.q.put((self.key, "log", line))
        except Exception as exc:  # noqa: BLE001
            self.q.put((self.key, "log", f"[读取中断] {type(exc).__name__}: {exc}"))
        finally:
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    def _waiter(self) -> None:
        assert self.proc is not None
        code = self.proc.wait()
        # 等读取线程把残留行吐完，最多 1 秒
        for t in self._threads:
            t.join(timeout=1.0)
        self.returncode = code
        self.ended_at = time.time()
        self.q.put((self.key, "exit", {"code": code, "seconds": self.elapsed(),
                                       "killed": self._killed, "soft_stopped": self._soft_stopped,
                                       "_job": self.token}))

    # ------------------------------------------------------------ 控制
    def elapsed(self) -> float:
        end = self.ended_at or time.time()
        return max(0.0, end - self.started_at)

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def send(self, obj: dict) -> bool:
        if not (self.running and self.proc and self.proc.stdin):
            return False
        try:
            self.proc.stdin.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
            self.proc.stdin.flush()
            return True
        except (OSError, ValueError):
            return False

    def stop(self, timeout: float = 0.0) -> bool:
        """优雅停止：优先写停止标记文件（worker 在下一个步边界保存检查点）。"""
        if not self.running:
            return False
        if self.stop_file is not None:
            try:
                self.stop_file.parent.mkdir(parents=True, exist_ok=True)
                self.stop_file.write_text(f"stop {time.time():.0f}", encoding="utf-8")
                self._soft_stopped = True
                return True
            except OSError:
                pass
        return self.send({"cmd": "stop"})

    def kill(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self._killed = True
            procs.kill_tree(self.proc.pid)


class JobManager:
    def __init__(self) -> None:
        self.q: queue.Queue = queue.Queue()
        self.jobs: dict[str, Job] = {}

    def spawn(self, key: str, args: list[str], cwd: str | Path, env: dict[str, str],
              label: str = "", stdin_pipe: bool = False, stop_file: Path | None = None) -> Job:
        old = self.jobs.get(key)
        if old is not None and old.running:
            old.kill()
        job = Job(key, args, cwd, env, self.q, label=label, stdin_pipe=stdin_pipe, stop_file=stop_file)
        self.jobs[key] = job
        job.start()
        return job

    def get(self, key: str) -> Job | None:
        return self.jobs.get(key)

    def running(self, key: str) -> bool:
        j = self.jobs.get(key)
        return bool(j and j.running)

    def kill_all(self) -> None:
        for j in list(self.jobs.values()):
            if j.running:
                j.kill()

    def drain(self, limit: int = 400) -> list[tuple[str, str, object]]:
        items: list[tuple[str, str, object]] = []
        for _ in range(limit):
            try:
                items.append(self.q.get_nowait())
            except queue.Empty:
                break
        return items
