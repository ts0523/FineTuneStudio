"""真实子进程强杀：验证 8 秒回调确实会执行，而不是只检查状态字段。"""
from __future__ import annotations

import queue
import sys
import threading
import time

from studio.core import jobs
from studio.ui.pages.autotrain import AutoTrainPage


class _Button:
    def set_enabled(self, _value: bool) -> None:
        return


class _Stage:
    def configure(self, **_kwargs) -> None:
        return


def test_force_after_eight_seconds_really_kills_stuck_job(tmp_path):
    manager = jobs.JobManager()
    stop_flag = tmp_path / "stop.flag"
    job = manager.spawn(
        "autotrain-train",
        [sys.executable, "-c", "import time; time.sleep(60)"],
        tmp_path,
        {},
        stop_file=stop_flag,
    )
    assert job.running
    assert job.stop()
    assert job.running, "测试子进程必须模拟卡住、忽略优雅停止"

    page = object.__new__(AutoTrainPage)
    page.app = type("App", (), {"jobs": manager})()
    page.btn_start = _Button()
    page.btn_stop = _Button()
    page.stage = _Stage()
    page._force_after = None
    page._final_kind = ""
    page._final_reason = ""

    delays: list[int] = []
    timers: list[threading.Timer] = []

    def after(ms: int, callback):
        delays.append(ms)
        timer = threading.Timer(ms / 1000.0, callback)
        timer.start()
        timers.append(timer)
        return timer

    page.after = after
    started = time.monotonic()
    page._maybe_finalize("stopped", "测试强杀")

    assert delays == [8000]
    deadline = time.time() + 12
    exit_event = None
    while time.time() < deadline:
        try:
            key, kind, payload = manager.q.get(timeout=0.2)
        except queue.Empty:
            continue
        if key == "autotrain-train" and kind == "exit":
            exit_event = payload
            break
    assert exit_event is not None, "强杀后没有收到退出事件"
    assert time.monotonic() - started >= 7.8, "必须在 8 秒等待后执行强杀"
    assert exit_event["killed"] is True
    assert exit_event["soft_stopped"] is True
    assert not job.running
    assert page._final_kind == "killed"
    assert "强杀" in page._final_reason
