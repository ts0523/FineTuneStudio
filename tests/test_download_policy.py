"""P0-5/6：下载 worker 的代码信任策略与恶意文件名处理（全 mock，不联网）。"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import download_worker
from studio.core import trust


def _run(monkeypatch, cfg_path) -> int:
    monkeypatch.setattr(sys, "argv", ["download_worker.py", "--config", str(cfg_path)])
    return download_worker.main()


def _fake_api(files):
    def api(_url, timeout=30.0):
        return {"siblings": [{"rfilename": n, "size": len(data),
                             "lfs": {"size": len(data), "sha256": ""}}
                             for n, data in files.items()]}
    return api


def _fake_download(files):
    def dl(_endpoint, _repo, name, dest, _size, _overall, _cancel, revision="main"):
        target = Path(dest) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[name])
        return True
    return dl


def test_default_skips_python_and_malicious_names(isolated_root, monkeypatch):
    dest = isolated_root / "models" / "repo"
    files = {
        "../evil.bin": b"evil",
        "model.py": b"print('exec')\n",
        "config.json": b"{}",
        "tokenizer.json": b"{}",
    }
    monkeypatch.setattr(download_worker, "api_json", _fake_api(files))
    monkeypatch.setattr(download_worker, "download_file", _fake_download(files))
    cfg = isolated_root / "dl.json"
    cfg.write_text(json.dumps({"repo": "someone/repo", "dest": str(dest),
                               "endpoint": "https://hf-mirror.com",
                               "verify_sha": False, "cancel_flag": str(dest / ".cancel"),
                               "revision": "main", "allow_code": False}), encoding="utf-8")
    assert _run(monkeypatch, cfg) == 0
    assert (dest / "config.json").is_file()
    assert not (dest / "model.py").exists(), "默认不得下载 *.py"
    assert not (dest / "evil.bin").exists(), "恶意文件名必须被拒绝"
    assert list(dest.rglob("evil.bin")) == []
    assert not (dest / ".fts_trust.json").exists()


def test_custom_code_requires_fixed_revision(isolated_root, monkeypatch):
    dest = isolated_root / "models" / "repo"
    monkeypatch.setattr(download_worker, "api_json", _fake_api({"model.py": b"x"}))
    cfg = isolated_root / "dl.json"
    cfg.write_text(json.dumps({"repo": "someone/repo", "dest": str(dest),
                               "endpoint": "https://hf-mirror.com",
                               "allow_code": True, "revision": "main"}), encoding="utf-8")
    assert _run(monkeypatch, cfg) == 2


def test_custom_code_with_fixed_revision_records_hashes(isolated_root, monkeypatch):
    dest = isolated_root / "models" / "repo"
    files = {"model.py": b"print('trusted by explicit choice')\n", "config.json": b"{}"}
    monkeypatch.setattr(download_worker, "api_json", _fake_api(files))
    monkeypatch.setattr(download_worker, "download_file", _fake_download(files))
    cfg = isolated_root / "dl.json"
    cfg.write_text(json.dumps({"repo": "someone/repo", "dest": str(dest),
                               "endpoint": "https://hf-mirror.com",
                               "verify_sha": False, "allow_code": True,
                               "revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}), encoding="utf-8")
    assert _run(monkeypatch, cfg) == 0
    manifest = json.loads((dest / ".fts_trust.json").read_text(encoding="utf-8"))
    assert manifest["revision"] == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert "model.py" in manifest["code_sha256"]
    ok, _reason = trust.check_code_trust(dest)
    assert ok


def test_code_allowlist_checks_downloaded_bytes(isolated_root, monkeypatch):
    dest = isolated_root / "models" / "repo"
    good = b"print('pinned')\n"
    files = {"model.py": good, "config.json": b"{}"}
    monkeypatch.setattr(download_worker, "api_json", _fake_api(files))
    monkeypatch.setattr(download_worker, "download_file", _fake_download(files))
    cfg = isolated_root / "dl.json"
    cfg.write_text(json.dumps({"repo": "someone/repo", "dest": str(dest),
                               "endpoint": "https://hf-mirror.com",
                               "verify_sha": False, "allow_code": True,
                               "revision": "a" * 40,
                               "code_allowlist": {"model.py": "0" * 64}}), encoding="utf-8")
    assert _run(monkeypatch, cfg) == 8
    assert not (dest / "model.py").exists()
    assert not (dest / ".fts_trust.json").exists()

    cfg.write_text(json.dumps({"repo": "someone/repo", "dest": str(dest),
                               "endpoint": "https://hf-mirror.com",
                               "verify_sha": False, "allow_code": True,
                               "revision": "a" * 40,
                               "code_allowlist": {
                                   "model.py": hashlib.sha256(good).hexdigest()}}),
                   encoding="utf-8")
    assert _run(monkeypatch, cfg) == 0
    assert (dest / ".fts_trust.json").is_file()


def test_builtin_code_hash_mismatch_rejected(isolated_root, monkeypatch):
    dest = isolated_root / "models" / "minimind-3o"
    files = {"config.json": b"{}", "model_minimind.py": b"# fake\n",
             "model_omni.py": b"# fake\n"}
    monkeypatch.setattr(download_worker, "api_json", _fake_api(files))
    monkeypatch.setattr(download_worker, "download_file", _fake_download(files))
    cfg = isolated_root / "dl.json"
    cfg.write_text(json.dumps({"repo": "jingyaogong/minimind-3o", "dest": str(dest),
                               "endpoint": "https://hf-mirror.com",
                               "verify_sha": False}), encoding="utf-8")
    assert _run(monkeypatch, cfg) == 8
    assert not (dest / "model_minimind.py").exists()
    assert not (dest / ".fts_trust.json").exists()
