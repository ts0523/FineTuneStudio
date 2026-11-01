# -*- coding: utf-8 -*-
"""摄像头 / 视频帧采集 worker（在 .venv 里跑，因为要用 opencv）。

界面通过 stdin 发指令，worker 用 stdout 的 @@EVT@@ 事件回话：
  {"cmd":"snapshot"}  立刻抓一帧
  {"cmd":"stop"}      结束
事件：
  {"type":"ready","backend":"camera|video","size":[w,h]}
  {"type":"frame","path":".../latest.jpg","n":3,"ts":1.7}
  {"type":"error","message":"..."}
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

EVT = "@@EVT@@"


def emit(obj: dict) -> None:
    sys.stdout.write(EVT + json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    latest = out_dir / "latest.jpg"
    fps = float(cfg.get("fps", 4) or 4)
    width = int(cfg.get("width", 480))
    height = int(cfg.get("height", 360))
    index = int(cfg.get("index", 0))
    video = str(cfg.get("video") or "")

    try:
        import cv2  # type: ignore
    except Exception:  # noqa: BLE001
        emit({"type": "error", "message": "缺少 opencv，无法打开摄像头。"
                                          "到「设置 → AI 功能」点「装摄像头支持」再试。"})
        return 2

    cap = None
    backend = "video" if video else "camera"
    try:
        if video:
            cap = cv2.VideoCapture(video)
        else:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap or not cap.isOpened():
                cap = cv2.VideoCapture(index)
        if not cap or not cap.isOpened():
            emit({"type": "error", "message": f"打不开{'视频文件' if video else '摄像头'}"
                                              f"（index={index}）。检查设备是否被占用/驱动是否正常。"})
            return 3
    except Exception as exc:  # noqa: BLE001
        emit({"type": "error", "message": f"摄像头初始化失败：{type(exc).__name__}: {exc}"})
        return 4

    if not video:
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
        except Exception:  # noqa: BLE001
            pass

    stop = threading.Event()
    force = threading.Event()

    def reader() -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
            except json.JSONDecodeError:
                continue
            if cmd.get("cmd") == "stop":
                stop.set()
                break
            if cmd.get("cmd") == "snapshot":
                force.set()

    threading.Thread(target=reader, daemon=True).start()

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)
    emit({"type": "ready", "backend": backend, "size": [w, h]})

    n = 0
    t0 = time.time()
    interval = 1.0 / max(0.5, fps)
    loop_video = bool(video)
    try:
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                if loop_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.05)
                    continue
                emit({"type": "error", "message": "读不到画面（设备被占用或断开了）"})
                break
            if video and frame.shape[1] > width:
                scale = width / frame.shape[1]
                frame = cv2.resize(frame, (width, max(1, int(frame.shape[0] * scale))))
            # cv2.imwrite 在中文路径下会静默失败，改成先编码再让 Python 写盘
            okj, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not okj:
                emit({"type": "error", "message": "画面编码失败"})
                break
            tmp = out_dir / "frame.tmp.jpg"
            tmp.write_bytes(buf.tobytes())
            try:
                tmp.replace(latest)
            except OSError:
                pass
            n += 1
            if n == 1 or force.is_set() or n % max(1, int(fps)) == 0:
                emit({"type": "frame", "path": str(latest), "n": n,
                      "ts": round(time.time() - t0, 2)})
            force.clear()
            time.sleep(interval if not video else 0.02)
    finally:
        try:
            cap.release()
        except Exception:  # noqa: BLE001
            pass
        emit({"type": "done", "frames": n})
    return 0


if __name__ == "__main__":
    sys.exit(main())
