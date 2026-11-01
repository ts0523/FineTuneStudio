"""AI 请求/造数取消：包含活动 HTTP 连接被真实中断的回归测试。"""
from __future__ import annotations

import http.server
import json
import threading
import time

import pytest

from studio.core import ai, nl


class _FakeCfg:
    def get(self, key, default=None):
        values = {"ai.max_tokens": 1200, "ai.temperature": 0.6}
        return values.get(key, default)


@pytest.fixture
def hanging_ai_server():
    release = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length)
            try:
                payload = json.loads(body or b"{}")
            except ValueError:
                payload = {}
            self.send_response(200)
            if payload.get("stream"):
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(
                    b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')
                self.wfile.flush()
            else:
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "4096")
                self.end_headers()
                self.wfile.write(b'{"choices":[')
                self.wfile.flush()
            release.wait(15)

        def log_message(self, format, *args):  # noqa: A002
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _configure_local_ai(monkeypatch, url: str) -> None:
    monkeypatch.setattr(ai, "api_key", lambda: "sk-test-fake")
    monkeypatch.setattr(ai, "base_url", lambda: url)
    monkeypatch.setattr(ai, "pick_models", lambda *a, **k: [])
    monkeypatch.setattr(ai, "fallbacks", lambda: ["test/model"])
    monkeypatch.setattr(ai, "blocked", lambda: {})
    monkeypatch.setattr(ai, "cfg", lambda: _FakeCfg())
    monkeypatch.setattr(ai, "_bump", lambda *a, **k: None)
    monkeypatch.setattr(ai, "_mark_blocked", lambda *a, **k: None)


def test_ai_chat_cancel_before_request(monkeypatch, isolated_root):
    monkeypatch.setattr(ai, "free_models", lambda *a, **k: [])
    monkeypatch.setattr(ai, "api_key", lambda: "sk-test-fake")
    with pytest.raises(ai.AIError) as exc:
        ai.chat([ai.user_msg("hi")], cancel=lambda: True)
    assert exc.value.code == 499


def test_nl_generate_cancel(monkeypatch, isolated_root):
    monkeypatch.setattr(nl.ai, "enabled", lambda: True)
    with pytest.raises(nl.NLError) as exc:
        nl.generate("写点数据", 4, kind="qa", cancel=lambda: True)
    assert "取消" in str(exc.value)


@pytest.mark.parametrize("streaming", [False, True])
def test_cancel_interrupts_blocking_http(monkeypatch, hanging_ai_server, streaming):
    _configure_local_ai(monkeypatch, hanging_ai_server)
    started = time.monotonic()
    chunks: list[str] = []
    with pytest.raises(ai.AIError) as exc:
        ai.chat([ai.user_msg("hi")], timeout=30,
                stream=(chunks.append if streaming else None),
                cancel=lambda: time.monotonic() - started > 0.35)
    elapsed = time.monotonic() - started
    assert exc.value.code == 499
    assert elapsed < 3.0, f"取消仍被单次 HTTP 阻塞了 {elapsed:.1f}s"
    if streaming:
        assert chunks == ["partial"]
