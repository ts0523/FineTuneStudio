"""P0-5：模型代码信任策略。权重哈希 ≠ 仓库可信。"""
from __future__ import annotations

import shutil
from pathlib import Path

import custom_model
from studio.core import trust

ROOT = Path(__file__).resolve().parents[1]


def _mini_copy(tmp_path: Path) -> Path:
    src = ROOT / "models" / "minimind-3o"
    dst = tmp_path / "minimind-3o"
    dst.mkdir()
    for name in trust.BUILTIN_CODE["jingyaogong/minimind-3o"]:
        shutil.copyfile(src / name, dst / name)
    return dst


def test_builtin_minimind_code_is_trusted(tmp_path):
    d = _mini_copy(tmp_path)
    assert trust.builtin_match(d)
    ok, reason = trust.check_code_trust(d)
    assert ok and "MiniMind-O" in reason
    assert custom_model._code_trust_error(d) == ""


def test_tampered_builtin_code_rejected(tmp_path):
    d = _mini_copy(tmp_path)
    (d / "model_minimind.py").write_text("# tampered\n", encoding="utf-8")
    ok, reason = trust.check_code_trust(d)
    assert not ok
    assert "哈希" in reason
    assert custom_model._code_trust_error(d)


def test_unknown_code_without_manifest_rejected(tmp_path):
    d = tmp_path / "random-repo"
    d.mkdir()
    (d / "model_x.py").write_text("print('would execute')\n", encoding="utf-8")
    ok, reason = trust.check_code_trust(d)
    assert not ok
    assert "信任记录" in reason
    assert custom_model._code_trust_error(d)


def test_manifest_requires_fixed_revision(tmp_path):
    d = tmp_path / "custom"
    d.mkdir()
    f = d / "model_x.py"
    f.write_text("print('x')\n", encoding="utf-8")
    trust.write_manifest(d, "someone/repo", "main")
    ok, reason = trust.check_code_trust(d)
    assert not ok and "完整 commit SHA" in reason

    trust.write_manifest(d, "someone/repo", "a" * 40)
    ok, _ = trust.check_code_trust(d)
    assert ok
    assert custom_model._code_trust_error(d) == ""

    f.write_text("print('changed')\n", encoding="utf-8")
    ok, reason = trust.check_code_trust(d)
    assert not ok and "哈希" in reason


def test_weights_hash_does_not_imply_repo_trust(tmp_path):
    d = tmp_path / "weights-only-claim"
    d.mkdir()
    (d / "model_x.py").write_text("print('x')\n", encoding="utf-8")
    assert trust.repo_verified_by_weights("a/b", {"model.safetensors": "abc"}) is True
    ok, _ = trust.check_code_trust(d)
    assert not ok, "权重 SHA-256 通过不能当成整个仓库可信"


def test_revision_validation():
    for bad in ("", "main", "master", "../evil", "a b", "v1.0.0", "deadbeefcafe"):
        try:
            trust.validate_revision(bad)
        except trust.TrustError:
            continue
        else:
            raise AssertionError(f"{bad} 应被拒绝")
    commit = "A" * 40
    assert trust.validate_revision(commit) == commit.lower()
