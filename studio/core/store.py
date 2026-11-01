# -*- coding: utf-8 -*-
"""实验索引 / 预设 / 校准数据的持久化（studio_state/index.json）。"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from typing import Any

from . import paths


def _now() -> float:
    return time.time()


def slugify(text: str, fallback: str = "run") -> str:
    text = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", text or "").strip("-")
    return (text or fallback)[:40]


class Store:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.path = paths.index_path()
        self.data: dict[str, Any] = {"runs": [], "presets": {}, "meta": {}}
        self.load()

    # ------------------------------------------------------------ 基础
    def load(self) -> None:
        p = self.path
        if not p.is_file():
            return
        try:
            raw = json.loads(p.read_text(encoding="utf-8-sig"))
            if not isinstance(raw, dict):
                raise ValueError("索引根节点不是对象")
            with self._lock:
                self.data = {
                    "runs": list(raw.get("runs") or []),
                    "presets": dict(raw.get("presets") or {}),
                    "meta": dict(raw.get("meta") or {}),
                }
        except Exception:  # noqa: BLE001
            try:
                p.replace(p.with_suffix(".broken.json"))
            except OSError:
                pass
            self.data = {"runs": [], "presets": {}, "meta": {}}

    def save(self) -> str:
        """整个快照 + 原子替换在同一临界区，避免并发丢更新。"""
        with self._lock:
            payload = json.dumps(self.data, ensure_ascii=False, indent=1)
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(prefix=".idx-", suffix=".tmp", dir=str(self.path.parent))
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(payload)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self.path)
                return ""
            except OSError as exc:
                return f"索引写入失败：{exc}"

    # ------------------------------------------------------------ 实验记录
    @property
    def runs(self) -> list[dict]:
        with self._lock:
            return list(self.data["runs"])

    def get_run(self, run_id: str) -> dict | None:
        with self._lock:
            for r in self.data["runs"]:
                if r.get("id") == run_id:
                    return dict(r)
        return None

    def new_id(self, name: str) -> str:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = f"{stamp}-{slugify(name)}"
        with self._lock:
            existing = {r.get("id") for r in self.data["runs"]}
        rid, i = base, 1
        while rid in existing:
            i += 1
            rid = f"{base}-{i}"
        return rid

    def add_run(self, rec: dict) -> str:
        rec = dict(rec)
        rec.setdefault("created", _now())
        rec.setdefault("status", "running")
        with self._lock:
            if not rec.get("id"):
                rec["id"] = self._unique_id_locked(rec.get("name", "run"))
            self.data["runs"].insert(0, rec)
            self.save()
        return rec["id"]

    def _unique_id_locked(self, name: str) -> str:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = f"{stamp}-{slugify(name)}"
        existing = {r.get("id") for r in self.data["runs"]}
        rid, i = base, 1
        while rid in existing:
            i += 1
            rid = f"{base}-{i}"
        return rid

    def update_run(self, run_id: str, **fields) -> None:
        with self._lock:
            for r in self.data["runs"]:
                if r.get("id") == run_id:
                    r.update(fields)
                    break
            self.save()

    def remove_run(self, run_id: str) -> dict | None:
        removed = None
        with self._lock:
            keep = []
            for r in self.data["runs"]:
                if r.get("id") == run_id and removed is None:
                    removed = r
                else:
                    keep.append(r)
            self.data["runs"] = keep
            self.save()
        return removed

    def mark_interrupted(self, reason: str = "上次运行未正常结束（启动时标记为中断）") -> int:
        """启动时把遗留的 running 记录标成 interrupted，避免界面显示仍在跑。"""
        n = 0
        with self._lock:
            for r in self.data["runs"]:
                if r.get("status") == "running":
                    r["status"] = "interrupted"
                    r["phase"] = reason
                    r["interrupted_at"] = _now()
                    n += 1
            if n:
                self.save()
        return n

    def mark_closing(self) -> int:
        """应用退出时同步把仍在 running 的记录标成中断。"""
        return self.mark_interrupted("应用退出时任务被中断")

    def recent(self, n: int = 8) -> list[dict]:
        return self.runs[:n]

    def counts(self) -> dict:
        runs = self.runs
        done = [r for r in runs if r.get("status") in ("done", "stopped")]
        losses = [r.get("final_loss") for r in done if isinstance(r.get("final_loss"), (int, float))]
        total_seconds = sum(float(r.get("seconds") or 0) for r in runs)
        return {
            "runs": len(runs),
            "done": len(done),
            "best_loss": min(losses) if losses else None,
            "last_loss": losses[-1] if losses else None,
            "total_seconds": total_seconds,
        }

    # ------------------------------------------------------------ 校准
    def record_speed(self, params_m: float, max_len: int, tokens_per_sec: float) -> None:
        if not tokens_per_sec or tokens_per_sec <= 0:
            return
        with self._lock:
            samples = list(self.data["meta"].get("speed_samples") or [])
            samples.append({"params_m": round(float(params_m), 1), "max_len": int(max_len),
                            "tps": round(float(tokens_per_sec), 1), "ts": _now()})
            self.data["meta"]["speed_samples"] = samples[-24:]
            self.save()

    def speed_samples(self) -> list[dict]:
        with self._lock:
            return list(self.data["meta"].get("speed_samples") or [])

    # ------------------------------------------------------------ 预设
    def presets(self) -> dict[str, dict]:
        with self._lock:
            return dict(self.data["presets"])

    def set_preset(self, name: str, cfg: dict) -> None:
        with self._lock:
            self.data["presets"][name] = dict(cfg)
            self.save()

    def delete_preset(self, name: str) -> None:
        with self._lock:
            self.data["presets"].pop(name, None)
            self.save()


_STORE: Store | None = None


def store() -> Store:
    global _STORE
    if _STORE is None:
        _STORE = Store()
    return _STORE
