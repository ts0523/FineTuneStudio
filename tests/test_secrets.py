"""P0-1/3：Key 不落明文、DPAPI 存储、旧配置迁移。"""
from __future__ import annotations

import json

from studio.core import config, paths, secrets

FAKE_KEY = "sk-or-v1-TESTONLY-000000000000000000000000000000000000000000000000"


def test_dpapi_roundtrip_never_writes_plaintext(isolated_root):
    assert secrets.set_api_key(FAKE_KEY) == ""
    assert secrets.read_stored_key() == FAKE_KEY
    raw = secrets.secret_path().read_bytes()
    assert FAKE_KEY.encode("utf-8") not in raw
    assert b"sk-or-v1" not in raw
    assert secrets.key_source() == "dpapi"


def test_env_key_has_priority(isolated_root, monkeypatch):
    secrets.set_api_key(FAKE_KEY)
    monkeypatch.setenv("FTS_AI_KEY", "sk-env-test")
    assert secrets.get_api_key() == "sk-env-test"
    assert secrets.key_source() == "env"
    note = secrets.set_api_key("sk-another")
    assert "环境变量" in note


def test_legacy_plaintext_migrates_and_clears(isolated_root):
    cfg = config.Config()
    cfg.set("ai.key", FAKE_KEY, save=True)
    cfg_file = paths.config_path()
    assert FAKE_KEY in cfg_file.read_text(encoding="utf-8")

    migrated, note = secrets.migrate_legacy_key(cfg)
    assert migrated is True
    assert "DPAPI" in note
    assert cfg.get("ai.key") == ""
    assert FAKE_KEY not in cfg_file.read_text(encoding="utf-8")
    assert secrets.read_stored_key() == FAKE_KEY
    raw = json.loads(cfg_file.read_text(encoding="utf-8-sig"))
    assert raw["ai"]["key"] == ""
    assert raw.get("theme")


def test_clear_removes_dpapi_and_plaintext(isolated_root):
    cfg = config.Config()
    secrets.set_api_key(FAKE_KEY, cfg)
    cfg.set("ai.key", FAKE_KEY, save=True)
    msg = secrets.clear_api_key(cfg)
    assert "已清除" in msg
    assert not secrets.secret_path().exists()
    assert cfg.get("ai.key") == ""
    assert FAKE_KEY not in paths.config_path().read_text(encoding="utf-8")
