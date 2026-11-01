# -*- coding: utf-8 -*-
"""进程内摄像头采集（opencv 随 exe 打包，不再依赖 .venv）。

界面线程只读 latest.jpg 与 frame_count；采集在独立线程里跑，避免摄像头
阻塞 Tk 事件循环。找不到 cv2 / 打不开设备时给出中文错误，调用方可以退回
纯语音模式。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from . import paths


def available() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


class Camera:
    def __init__(self, out_dir: Path | None = None, fps: float = 3.0,
                 width: int = 480, height: int = 360, index: int = 0):
        self.out_dir = Path(out_dir) if out_dir else paths.state_dir() / "camera"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.latest = self.out_dir / "latest.jpg"
        self.fps = max(1.0, float(fps))
        self.width = int(width)
        self.height = int(height)
        self.index = int(index)
        self.frame_count = 0
        self.size: tuple[int, int] = (0, 0)
        self.error = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._opened = False

    @property
    def running(self) -> bool:
        return self._opened and not self._stop.is_set()

    def open(self) -> tuple[bool, str]:
        try:
            import cv2
        except Exception as exc:  # noqa: BLE001
            self.error = f"没有摄像头支持：{exc}"
            return False, self.error
        cap = None
        errors: list[str] = []
        for backend in (cv2.CAP_DSHOW, None):
            try:
                cap = (cv2.VideoCapture(self.index) if backend is None
                       else cv2.VideoCapture(self.index, backend))
                if not cap or not cap.isOpened():
                    errors.append("DirectShow 打不开设备" if backend is not None else "默认后端打不开设备")
                    if cap is not None:
                        cap.release()
                    cap = None
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                ok, frame = False, None
                for _ in range(5):
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        break
                    time.sleep(0.15)
                if not ok or frame is None:
                    errors.append("设备已打开但读不到首帧")
                    cap.release()
                    cap = None
                    continue
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")
                if cap is not None:
                    cap.release()
                cap = None
        if cap is None:
            self.error = (f"打不开摄像头（index={self.index}）：" + "；".join(errors)
                          + "。请检查系统相机权限，以及设备是否被其他程序占用")
            return False, self.error
        self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.width),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.height))
        self.error = ""
        self._opened = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(cap,), daemon=True)
        self._thread.start()
        return True, ""

    def _run(self, cap) -> None:
        import cv2

        interval = 1.0 / self.fps
        while not self._stop.is_set():
            try:
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.error = "读不到画面（设备被占用或断开了）"
                    break
                # 注意：cv2.imwrite 在中文路径下会静默失败，必须自己编码后用 Python 写盘
                okj, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                if not okj:
                    self.error = "JPEG 编码失败"
                    break
                tmp = self.out_dir / "frame.tmp.jpg"
                tmp.write_bytes(buf.tobytes())
                try:
                    tmp.replace(self.latest)
                except OSError:
                    pass
                self.frame_count += 1
                time.sleep(interval)
            except Exception as exc:  # noqa: BLE001
                self.error = f"采集中断：{type(exc).__name__}: {exc}"
                break
        try:
            cap.release()
        except Exception:  # noqa: BLE001
            pass
        self._opened = False

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.2)
        self._thread = None
        self._opened = False
        self.frame_count = 0
