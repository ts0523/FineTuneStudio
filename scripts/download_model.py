# -*- coding: utf-8 -*-
r"""从 hf-mirror.com 下载模型到本地 models 目录（安全默认）。

用法:
    python scripts\download_model.py                       # 下载内置可信的 MiniMind-O
    python scripts\download_model.py Qwen/Qwen2.5-0.5B-Instruct
    python scripts\download_model.py 自定义/仓库 --revision <commit> --allow-code   # 明确信任代码

安全说明：
- 默认不下载任何 *.py（模型代码会在加载时执行）。
- MiniMind-O（jingyaogong/minimind-3o）是内置可信例外，代码哈希固定并校验。
- 其他仓库要下载代码，必须显式 --allow-code + 固定 --revision；下载后写入
  .fts_trust.json，加载器会再次校验代码哈希。
- 权重文件的 SHA-256 只证明文件内容，不代表整个仓库可信。
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

DEFAULT_MODEL = "jingyaogong/minimind-3o"
BASE_ALLOW = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", nargs="?", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default="", help="固定 commit/tag；下载代码时必填")
    ap.add_argument("--allow-code", action="store_true", help="明确信任并下载模型自带 *.py")
    args = ap.parse_args()

    from studio.core import fsutil, trust

    repo = args.repo.strip().strip("/")
    try:
        owner, name = repo.split("/")
        fsutil.validate_component(owner, max_len=80)
        fsutil.validate_component(name, max_len=80)
    except (ValueError, fsutil.PathSafetyError) as exc:
        print(f"[错误] 仓库名不安全：{exc}")
        return 2
    local_dir = fsutil.safe_child(MODELS_DIR, name, max_len=80)
    local_dir.mkdir(parents=True, exist_ok=True)

    builtin = trust.builtin_repo(repo)
    revision = args.revision.strip()
    if not builtin and args.allow_code:
        try:
            revision = trust.validate_revision(revision)
        except trust.TrustError as exc:
            print(f"[错误] {exc}")
            return 2
    if builtin:
        revision = revision or "main"

    allow = list(BASE_ALLOW)
    if builtin:
        allow += [f for f in trust.BUILTIN_CODE["jingyaogong/minimind-3o"]]
    elif args.allow_code:
        allow.append("*.py")

    from huggingface_hub import snapshot_download

    print(f"[下载] {repo}（revision {revision or '默认分支'}，{'含代码' if args.allow_code or builtin else '跳过 .py'}）")
    print(f"[镜像] {os.environ['HF_ENDPOINT']}")
    print(f"[目标] {local_dir}")

    try:
        path = snapshot_download(
            repo_id=repo,
            revision=revision or None,
            local_dir=str(local_dir),
            allow_patterns=allow,
            max_workers=4,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[错误] 下载失败：{type(exc).__name__}: {exc}")
        return 3
    print(f"[完成] {path}")

    # 代码信任：内置哈希校验 / 自定义代码写信任清单
    if trust.code_files(local_dir):
        if builtin:
            ok, reason = trust.check_code_trust(local_dir, repo)
            if not ok:
                print(f"[错误] 内置代码校验失败：{reason}")
                return 4
            trust.write_manifest(local_dir, repo, revision)
            print("[信任] MiniMind-O 代码哈希校验通过")
        elif args.allow_code:
            trust.write_manifest(local_dir, repo, revision)
            print(f"[信任] 已记录代码哈希（revision {revision}）")
        else:
            print("[提示] 发现未下载的代码文件被跳过；加载时会拒绝执行未受信任代码")

    total = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())
    print(f"[大小] {total / 1024 / 1024:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
