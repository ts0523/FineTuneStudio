"""pytest 公共设置：隔离项目根目录，避免测试写进真实 data/models/outputs。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WORKERS = ROOT / "studio" / "workers"
if str(WORKERS) not in sys.path:
    sys.path.insert(0, str(WORKERS))


def pytest_configure() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")


@pytest.fixture
def isolated_root(tmp_path, monkeypatch):
    """把项目根指到临时目录，避免测试污染真实 data/models/outputs/studio_state。"""
    from studio.core import config, paths

    old_cached, old_override = paths._cached_root, paths._override_root
    paths.set_root(tmp_path)
    monkeypatch.setenv("FTS_ROOT", str(tmp_path))
    for name in ("FTS_AI_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    config._CFG = None
    for sub in ("data", "models", "outputs", "logs"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    yield tmp_path
    paths._cached_root, paths._override_root = old_cached, old_override
    config._CFG = None


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT

