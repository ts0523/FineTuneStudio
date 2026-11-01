"""P0-6/7：下载路径与用户命名入口的安全规则。"""
from __future__ import annotations

import pytest

import download_worker
from studio.core import fsutil, hubclient


@pytest.mark.parametrize("name", [
    "", "   ", ".", "..", "../evil", "a/../../b", "a\\..\\b",
    "/abs", "\\server\\share", "C:\\evil", "C:/evil", "con", "NUL",
    'bad<name', "bad:name", ".hidden", "\x00evil",
])
def test_safe_component_rejects(name):
    with pytest.raises(fsutil.PathSafetyError):
        fsutil.validate_component(name)


@pytest.mark.parametrize("name", ["model", "Qwen2.5-0.5B-Instruct", "我的模型", "run-01"])
def test_safe_component_accepts(name):
    assert fsutil.validate_component(name) == name.strip()


@pytest.mark.parametrize("rel", [
    "../evil.bin", "/etc/passwd", "C:\\evil.bin", "\\\\srv\\share",
    "a/../../../evil", "", "a//b", ".hidden",
])
def test_download_target_rejects(isolated_root, rel):
    with pytest.raises((fsutil.PathSafetyError, ValueError)):
        fsutil.resolve_download_target(isolated_root / "models", rel)


def test_download_target_stays_inside(isolated_root):
    dest = isolated_root / "models" / "repo"
    target = fsutil.resolve_download_target(dest, "sub/dir/model.safetensors")
    assert dest.resolve() in target.resolve().parents


def test_worker_validator_matches(isolated_root):
    assert download_worker.safe_rel_name("model.safetensors") == "model.safetensors"
    with pytest.raises(ValueError):
        download_worker.safe_rel_name("../evil")


def test_endpoint_protocol_restricted():
    with pytest.raises(ValueError):
        download_worker.validate_endpoint("file:///C:/models")
    with pytest.raises(ValueError):
        download_worker.validate_endpoint("ftp://example.com")
    assert download_worker.validate_endpoint("https://hf-mirror.com").startswith("https://")
    with pytest.raises(hubclient.HubError):
        hubclient.normalize_endpoint("file:///x")


def test_download_dest_must_be_under_models(isolated_root, monkeypatch):
    monkeypatch.setenv("FTS_ROOT", str(isolated_root))
    ok = download_worker.validate_dest(isolated_root / "models" / "x")
    assert ok == (isolated_root / "models" / "x").resolve()
    with pytest.raises(ValueError):
        download_worker.validate_dest(isolated_root / "outputs" / "x")


def test_user_named_outputs_are_sanitized(isolated_root):
    from studio.core import paths

    assert fsutil.sanitize_component("../evil", "run") == "evil"
    out = fsutil.safe_child(paths.outputs_dir(), fsutil.sanitize_component("../evil", "run"))
    assert paths.outputs_dir().resolve() in out.parents
