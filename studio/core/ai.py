# -*- coding: utf-8 -*-
"""AI 能力核心：OpenRouter 客户端。

只用标准库（urllib），所以 exe 里不需要额外依赖，也不占用 .venv。

包含：
- 免费模型目录发现（GET /models，本地缓存）+ 按「需要的能力」挑最优
- 流式对话（SSE）、多模型自动降级（某个免费端点被墙/被限流会自动换下一个）
- 多模态内容片段：文字 / 图片 / 语音
- 结构化输出：从模型回复里稳地掏出 JSON
- Key 校验（GET /key，能看到免费额度还剩多少）
"""
from __future__ import annotations

import base64
import http.client
import json
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from . import config, secrets

BASE_DEFAULT = "https://openrouter.ai/api/v1"
REFERER_DEFAULT = "https://localhost/finetune-studio"
UA = "FineTuneStudio/1.0 (local desktop app)"

# 2026-10 用真实 Key 实测通过的免费模型（会随线上目录刷新，这里只是兜底）
# 注意：这些免费端点大多带思维链，必须传 reasoning=low/exclude，否则 30 个 token
# 全被思维链吃光、正式回答为空——用户看到的就是「所有免费模型都没生效」。
TEXT_MODEL = "openrouter/free"                    # 自动路由，~3s，支持 text+image
OMNI_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"  # 唯一 text+audio+image+video
FAST_MODEL = "liquid/lfm-2.5-2.6b:free"           # 2.6B，~1s，纯文本（部分时段 5xx）
SEED_MODELS = [
    TEXT_MODEL,                                     # 自动路由：实测 1~3s 稳定
    "dots-studio/dots-3-note-preview:free",          # 2.6s，text+image
    "nvidia/nemotron-3-ultra-550b-a55b:free",        # 1.4s，纯文本，实测可答
    "nvidia/nemotron-3.5-super-vl:free",             # text+image+video
    OMNI_MODEL,
]
TEXT_CHAIN = [TEXT_MODEL, "dots-studio/dots-3-note-preview:free",
              "nvidia/nemotron-3-ultra-550b-a55b:free", OMNI_MODEL]

# 统一关掉免费模型的思维链：省 token、降延迟、避免空回复
REASONING = {"effort": "low", "exclude": True}

# 这些错误只说明「这个模型这条请求不行」，换下一个模型还有机会
_RETRY_CODES = {400, 402, 403, 404, 408, 409, 422, 429, 500, 502, 503, 504}


class AIError(Exception):
    def __init__(self, message: str, code: int = 0, model: str = ""):
        super().__init__(message)
        self.message = message
        self.code = code
        self.model = model


_HINTS = (
    ("requires at least $0.50", "云端语音输入要求账户余额 ≥ $0.5；免费 Key 请用本机语音识别（应用已自动处理）"),
    ("no endpoints found that support input audio", "这个免费模型不支持语音输入，已改用本机语音识别"),
    ("no endpoints found", "该免费模型当前没有可用端点，已自动换下一个"),
    ("provider returned error", "免费端点暂时不可用（上游报错），已自动换下一个"),
    ("not available in your region", "该模型在当前地区不可用，已自动换下一个"),
    ("agentic harness", "该模型只允许在官方 Agent 客户端里用，已自动换下一个"),
    ("data policy", "OpenRouter 账户的数据策略不允许该免费模型；到 openrouter.ai/settings/privacy 调整后重试"),
    ("rate limit", "免费额度或速率限制，稍后再试（每天 50 次免费请求）"),
    ("insufficient", "余额不足，免费额度可能已用完"),
)


def friendly_error(message: str) -> str:
    """把 OpenRouter 的英文报错翻成用户能看懂的中文提示。"""
    low = (message or "").lower()
    for key, hint in _HINTS:
        if key in low:
            return hint
    return message


# ---------------------------------------------------------------- 配置读
def cfg() -> Any:
    return config.cfg()


def api_key() -> str:
    """Key 只从环境变量或 Windows DPAPI 读取；不再从配置文件读明文。"""
    return secrets.get_api_key(cfg())


def key_source() -> str:
    return secrets.key_source(cfg())


def enabled() -> bool:
    return bool(api_key())


def base_url() -> str:
    raw = str(cfg().get("ai.base_url", "") or BASE_DEFAULT).strip().rstrip("/")
    if not raw:
        raw = BASE_DEFAULT
    if not raw.startswith("http"):
        raw = "https://" + raw
    if not raw.endswith("/api/v1"):
        raw = raw.rstrip("/") + "/api/v1"
    return raw


def default_model() -> str:
    m = str(cfg().get("ai.model", "") or "").strip()
    return m or TEXT_MODEL


def fallbacks() -> list[str]:
    chain = cfg().get("ai.fallbacks") or SEED_MODELS
    out = [str(x) for x in chain if str(x).strip()]
    return out or list(SEED_MODELS)


def best_for(need: Iterable[str] = ("text",)) -> list[str]:
    """按能力需求返回推荐模型链（已排除被标记不可用的端点）。"""
    return pick_models(tuple(need), limit=3)


def blocked() -> dict:
    b = cfg().get("ai.blocked") or {}
    return b if isinstance(b, dict) else {}


def _mark_blocked(model: str, reason: str) -> None:
    if not model:
        return
    b = dict(blocked())
    b[model] = {"reason": reason[:180], "ts": time.time()}
    cfg().set("ai.blocked", b)


def _bump(success: bool, model: str = "", error: str = "") -> None:
    st = cfg().get("ai.stats") or {}
    if not isinstance(st, dict):
        st = {}
    st["requests"] = int(st.get("requests", 0)) + 1
    if success:
        st["ok"] = int(st.get("ok", 0)) + 1
        st["last_model"] = model
        b = blocked()
        if model and model in b:
            b.pop(model, None)
            cfg().set("ai.blocked", b)
    else:
        st["errors"] = int(st.get("errors", 0)) + 1
        st["last_error"] = error[:300]
    st["last_at"] = time.time()
    cfg().set("ai.stats", st)


def stats() -> dict:
    st = cfg().get("ai.stats") or {}
    return st if isinstance(st, dict) else {}


# ---------------------------------------------------------------- HTTP
def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key()}",
        "Content-Type": "application/json",
        "HTTP-Referer": str(cfg().get("ai.referer", "") or REFERER_DEFAULT),
        "X-Title": str(cfg().get("ai.title", "") or "FineTune Studio"),
        "User-Agent": UA,
        "Accept": "application/json",
    }


class _SocketAbort:
    """保留一个底层 socket 副本；urllib 关闭原连接后仍可 shutdown 打断读取。"""

    def __init__(self, sock: socket.socket):
        self.sock: socket.socket | None = sock

    def _cancel_abort(self) -> None:
        sock, self.sock = self.sock, None
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass

    def close(self) -> None:
        self._cancel_abort()


def _response_socket(response):
    fp = getattr(response, "fp", None)
    raw = getattr(fp, "raw", None)
    return getattr(raw, "_sock", None)


class _ResponseSocketAbort:
    """urllib 会把 HTTPConnection.sock 置空，必须从响应缓冲区反查底层 socket。"""

    def __init__(self, response):
        self.response = response

    def _socket(self):
        return _response_socket(self.response)

    def _cancel_abort(self) -> None:
        sock = self._socket()
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass

    def close(self) -> None:
        self._cancel_abort()


class _CancelState:
    """取消时主动关闭活动连接，让阻塞的 read/readline 立即返回。"""

    def __init__(self, callback: Callable[[], bool] | None):
        self.callback = callback
        self.cancelled = threading.Event()
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._closables: set = set()
        self._response_aborts: dict[int, _ResponseSocketAbort] = {}
        self._watcher: threading.Thread | None = None

    def start(self) -> None:
        if self.callback is None:
            return
        self._watcher = threading.Thread(target=self._watch, daemon=True)
        self._watcher.start()

    def _watch(self) -> None:
        while not self._done.wait(0.05):
            try:
                requested = bool(self.callback and self.callback())
            except Exception:  # noqa: BLE001
                requested = False
            if requested:
                self.cancel()
                self._done.set()
                return

    def cancel(self) -> None:
        self.cancelled.set()
        with self._lock:
            # 只 abort 底层 socket：HTTPResponse.close() 会等待 BufferedReader 锁，
            # 而读取线程正持锁阻塞，先关响应反而无法及时解除阻塞。
            targets = [t for t in self._closables if hasattr(t, "_cancel_abort")]
        for target in targets:
            try:
                target._cancel_abort()
            except Exception:  # noqa: BLE001
                pass

    def track_response(self, resp) -> None:
        abort = _ResponseSocketAbort(resp)
        with self._lock:
            self._response_aborts[id(resp)] = abort
        self.track(abort)

    def untrack_response(self, resp) -> None:
        with self._lock:
            abort = self._response_aborts.pop(id(resp), None)
        if abort is not None:
            self.untrack(abort)

    def track(self, target) -> None:
        with self._lock:
            if not self.cancelled.is_set():
                self._closables.add(target)
                return
        try:
            target.close()
        except Exception:  # noqa: BLE001
            pass

    def untrack(self, target) -> None:
        with self._lock:
            self._closables.discard(target)

    def check(self) -> None:
        try:
            requested = bool(self.callback and self.callback())
        except Exception:  # noqa: BLE001
            requested = False
        if requested:
            self.cancel()
        if self.cancelled.is_set():
            raise AIError("已取消", 499)

    def stop(self) -> None:
        self._done.set()
        if self._watcher is not None:
            self._watcher.join(timeout=0.2)
        with self._lock:
            targets = list(self._closables)
            self._closables.clear()
        for target in targets:
            if self.cancelled.is_set() and not hasattr(target, "_cancel_abort"):
                # 工作线程可能仍卡在 resp.read；此处 close 会等缓冲锁，跳过。
                continue
            try:
                if hasattr(target, "_cancel_abort"):
                    target._cancel_abort()
                else:
                    target.close()
            except Exception:  # noqa: BLE001
                pass


class _TrackedConnection:
    def __init__(self, *args, cancel_state: _CancelState, **kwargs):
        self._cancel_state = cancel_state
        self._fts_sock = None
        self._fts_keep_socket = False
        self._fts_in_getresponse = False
        cancel_state.track(self)
        super().__init__(*args, **kwargs)

    @property
    def sock(self):
        if self._fts_keep_socket:
            # urllib 看到 None 就不会在 do_open 里提前关掉 socket；
            # 真正的 close/shutdown 由取消监听负责。
            return None
        return self._fts_sock

    @sock.setter
    def sock(self, value) -> None:
        self._fts_sock = value

    def getresponse(self):
        self._fts_in_getresponse = True
        try:
            response = super().getresponse()  # type: ignore[misc]
        finally:
            self._fts_in_getresponse = False
        self._fts_keep_socket = True
        sock = self._fts_sock
        if sock is not None:
            try:
                # shutdown 任意一个复制句柄都会作用到同一条连接，能打断阻塞 recv。
                dup = socket.socket(fileno=os.dup(sock.fileno()))
                self._cancel_state.track(_SocketAbort(dup))
            except OSError:
                pass
        return response

    def connect(self) -> None:
        self._cancel_state.check()
        super().connect()  # type: ignore[misc]
        if self._cancel_state.cancelled.is_set():
            self._cancel_abort()
            self._cancel_state.check()

    def _cancel_abort(self) -> None:
        sock = getattr(self, "_fts_sock", None)
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass
        self._fts_sock = None

    def close(self) -> None:
        if self._fts_in_getresponse:
            # http.client 在 will_close 时调用 close，但响应体仍要用同一连接读取。
            self._fts_keep_socket = True
            return
        self._cancel_abort()
        self._fts_keep_socket = False
        self._cancel_state.untrack(self)


class _TrackedHTTPConnection(_TrackedConnection, http.client.HTTPConnection):
    pass


class _TrackedHTTPSConnection(_TrackedConnection, http.client.HTTPSConnection):
    pass


class _TrackedHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, state: _CancelState):
        super().__init__()
        self._state = state

    def http_open(self, req):
        factory = lambda *a, **kw: _TrackedHTTPConnection(  # noqa: E731
            *a, cancel_state=self._state, **kw)
        return self.do_open(factory, req)


class _TrackedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, state: _CancelState):
        super().__init__()
        self._state = state

    def https_open(self, req):
        factory = lambda *a, **kw: _TrackedHTTPSConnection(  # noqa: E731
            *a, cancel_state=self._state, **kw)
        return self.do_open(factory, req, context=getattr(self, "_context", None))


def _raw(method: str, path: str, payload: dict | None = None, timeout: float = 60.0,
         cancel_state: _CancelState | None = None):
    if not api_key():
        raise AIError("还没有配置 AI Key：到「设置 → AI 功能」粘贴 OpenRouter Key", 401)
    if cancel_state is not None:
        cancel_state.check()
    url = base_url() + path
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=_headers(), method=method)
    try:
        if cancel_state is None:
            resp = urllib.request.urlopen(req, timeout=timeout)
        else:
            opener = urllib.request.build_opener(
                _TrackedHTTPHandler(cancel_state), _TrackedHTTPSHandler(cancel_state))
            resp = opener.open(req, timeout=timeout)
            cancel_state.track_response(resp)
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            pass
        msg = ""
        try:
            obj = json.loads(body)
            err = obj.get("error") if isinstance(obj, dict) else None
            msg = str((err or {}).get("message") or body)
        except Exception:  # noqa: BLE001
            msg = body or exc.reason
        raise AIError(msg or f"HTTP {exc.code}", int(exc.code)) from exc
    except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
        if cancel_state is not None and cancel_state.cancelled.is_set():
            raise AIError("已取消", 499) from exc
        if isinstance(exc, TimeoutError):
            raise AIError("请求超时，稍后重试", 0) from exc
        reason = getattr(exc, "reason", exc)
        raise AIError(f"网络不可达：{reason}（检查代理/网络）", 0) from exc
    return resp


def _get_json(path: str, timeout: float = 40.0) -> dict:
    with _raw("GET", path, None, timeout) as resp:
        raw = resp.read().decode("utf-8", "replace")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AIError("返回内容不是 JSON（可能被网络代理拦截）", 0) from exc
    if not isinstance(obj, dict):
        raise AIError("返回格式异常", 0)
    return obj


# ---------------------------------------------------------------- Key 校验
def key_info() -> dict:
    """返回 {label, usage, limit, remaining, free_remaining, is_free_tier}。"""
    data = _get_json("/key", timeout=30)
    inner = data.get("data")
    d: dict = inner if isinstance(inner, dict) else data
    raw_free = d.get("free_model_daily_requests")
    free: dict = raw_free if isinstance(raw_free, dict) else {}
    return {
        "label": str(d.get("label", "")),
        "usage": d.get("usage"),
        "limit": d.get("limit"),
        "limit_remaining": d.get("limit_remaining"),
        "is_free_tier": bool(d.get("is_free_tier")),
        "free_used": free.get("used"),
        "free_limit": free.get("limit"),
        "free_remaining": free.get("remaining"),
        "raw": d,
    }


# ---------------------------------------------------------------- 免费模型目录
def free_models(refresh: bool = False, ttl: float = 6 * 3600) -> list[dict]:
    cat = cfg().get("ai.catalog") or {}
    if not refresh and isinstance(cat, dict) and cat.get("items"):
        if time.time() - float(cat.get("ts", 0)) < ttl:
            return [dict(x) for x in cat["items"]]
    data = _get_json("/models", timeout=45)
    items: list[dict] = []
    for m in data.get("data") or []:
        if not isinstance(m, dict):
            continue
        pr = m.get("pricing") or {}
        try:
            if float(pr.get("prompt", 1)) != 0 or float(pr.get("completion", 1)) != 0:
                continue
        except (TypeError, ValueError):
            continue
        arch = m.get("architecture") or {}
        items.append({
            "id": str(m.get("id", "")),
            "name": str(m.get("name") or m.get("id") or ""),
            "ctx": int(m.get("context_length") or 0),
            "inputs": [str(x) for x in (arch.get("input_modalities") or ["text"])],
            "outputs": [str(x) for x in (arch.get("output_modalities") or ["text"])],
            "created": int(m.get("created") or 0),
        })
    items.sort(key=lambda x: (-len(x["inputs"]), -x["ctx"], -x["created"]))
    if items:
        cfg().set("ai.catalog", {"ts": time.time(), "items": items})
    return items


def _score(m: dict, need: Iterable[str]) -> float:
    need = list(need)
    ins = set(m.get("inputs") or [])
    miss = [n for n in need if n not in ins]
    if miss:
        return -1.0
    score = 20.0 * len(need)
    if "image" in ins:
        score += 4
    if "audio" in ins:
        score += 6
    if "video" in ins:
        score += 3
    score += min(40.0, (m.get("ctx") or 0) / 50000.0)
    name = (m.get("id") or "").lower()
    for bad in ("safety", "moderation", "guard"):
        if bad in name:
            score -= 30
    if "omni" in name or "vl" in name:
        score += 3
    return score


def pick_models(need: Iterable[str] = ("text",), limit: int = 3,
                allow_stale: bool = True) -> list[str]:
    """按能力需求挑免费模型，自动排除已被标记不可用的。"""
    bl = blocked()
    need = list(need)
    try:
        items = free_models()
    except AIError:
        items = []
    if not items:
        items = [{"id": x, "ctx": 0, "inputs": ["text", "image", "audio", "video"]}
                 for x in SEED_MODELS]
    ranked = sorted(items, key=lambda m: -_score(m, need))
    out = [m["id"] for m in ranked
           if _score(m, need) > 0 and m["id"] not in bl][:limit]
    if not out and allow_stale:
        out = [x for x in SEED_MODELS if x not in bl][:limit]
    return out


# ---------------------------------------------------------------- 多模态片段
_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".webp": "image/webp", ".bmp": "image/bmp", ".gif": "image/gif"}


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def image_part(path: str | Path) -> dict:
    p = Path(path)
    mime = _MIME.get(p.suffix.lower(), "image/png")
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def audio_part(path: str | Path) -> dict:
    """OpenRouter 的语音输入格式；免费额度下通常会返回 402（需 $0.5 余额）。"""
    p = Path(path)
    fmt = p.suffix.lstrip(".").lower() or "wav"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}}


def user_msg(parts: list[dict] | str) -> dict:
    return {"role": "user", "content": parts}


# ---------------------------------------------------------------- 对话
@dataclass
class Reply:
    text: str = ""
    reasoning: str = ""
    model: str = ""
    seconds: float = 0.0
    usage: dict | None = None
    attempts: list[dict] | None = None

    def as_dict(self) -> dict:
        return {"text": self.text, "reasoning": self.reasoning, "model": self.model,
                "seconds": round(self.seconds, 2), "usage": self.usage,
                "attempts": self.attempts or []}


def _stream_read_once(payload: dict, stream_cb: Callable[[str], None] | None,
                      timeout: float, cancel_state: _CancelState) -> tuple[str, str, dict]:
    """在工作线程里执行阻塞 HTTP；调用线程用 _stream_once 等待并可立即取消。"""
    body = dict(payload)
    body["stream"] = bool(stream_cb)
    resp = _raw("POST", "/chat/completions", body, timeout, cancel_state)
    text_parts: list[str] = []
    reason_parts: list[str] = []
    usage: dict = {}
    try:
        with resp:
            if not stream_cb:
                raw = resp.read().decode("utf-8", "replace")
                cancel_state.check()
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise AIError("回复不是 JSON（可能被网络代理拦截）", 0) from exc
                if isinstance(obj.get("error"), dict):
                    raise AIError(str(obj["error"].get("message") or obj["error"]), 0)
                choice = (obj.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                return (str(msg.get("content") or ""), str(msg.get("reasoning") or ""),
                        obj.get("usage") or {})
            for raw in iter(resp.readline, b""):
                cancel_state.check()
                line = raw.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk in ("[DONE]", ""):
                    break
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj.get("error"), dict):
                    raise AIError(str(obj["error"].get("message") or obj["error"]), 0)
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("reasoning"):
                        reason_parts.append(str(delta["reasoning"]))
                    piece = delta.get("content")
                    if piece:
                        text_parts.append(str(piece))
                        cancel_state.check()
                        stream_cb(str(piece))
    except AIError:
        raise
    except Exception as exc:  # noqa: BLE001
        if cancel_state.cancelled.is_set():
            raise AIError("已取消", 499) from exc
        raise AIError(f"读取 AI 回复中断：{exc}", 0) from exc
    finally:
        cancel_state.untrack_response(resp)
    return "".join(text_parts), "".join(reason_parts), usage


def _stream_once(payload: dict, stream_cb: Callable[[str], None] | None,
                 timeout: float, cancel_state: _CancelState) -> tuple[str, str, dict]:
    """把单次阻塞 HTTP 隔离到工作线程；取消时调用线程立即返回 499。

    Python 无法安全强杀阻塞在网络 recv 的线程，因此保留该 daemon 线程直到
    socket timeout，但 UI/帮训线程不再等待它；旧结果仍由 session ID 丢弃。
    """
    result: dict = {}
    done = threading.Event()

    def work() -> None:
        try:
            result["value"] = _stream_read_once(payload, stream_cb, timeout, cancel_state)
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc
        finally:
            done.set()

    threading.Thread(target=work, daemon=True).start()
    while not done.wait(0.05):
        cancel_state.check()
    cancel_state.check()
    error = result.get("error")
    if error is not None:
        raise error
    return result["value"]


def chat(messages: list[dict], model: str = "", need: Iterable[str] = ("text",),
         system: str = "", temperature: float | None = None,
         max_tokens: int | None = None, stream: Callable[[str], None] | None = None,
         timeout: float = 90.0, models: list[str] | None = None,
         json_mode: bool = False, cancel: Callable[[], bool] | None = None) -> Reply:
    """发一轮对话；失败会自动在免费模型链上降级重试。

    cancel：可选的取消检查。取消时会关闭活动 HTTP 连接，中断阻塞读取；
    界面仍用 session ID 丢弃已经开始返回的旧结果。
    """
    if system:
        messages = [{"role": "system", "content": system}] + list(messages)
    chain: list[str] = []
    # 有媒体输入时，能力匹配的模型排在最前，避免把语音/图片发给纯文本端点白跑一次
    if tuple(need) != ("text",):
        chain.extend(pick_models(tuple(need), limit=3))
    for m in ([model] if model else []) + (models or []):
        if m and m not in chain:
            chain.append(m)
    if not chain:
        chain.append(default_model())
    chain.extend([m for m in fallbacks() if m not in chain])
    bl = blocked()
    chain = [m for m in chain if m not in bl] or list(SEED_MODELS)
    # 缺少的顺序里再补两个能覆盖所需能力的
    for extra in pick_models(tuple(need), limit=3):
        if extra not in chain:
            chain.append(extra)

    base_tokens = int(cfg().get("ai.max_tokens", 1200) if max_tokens is None
                      else max_tokens)
    payload_base = {
        "temperature": float(cfg().get("ai.temperature", 0.6) if temperature is None
                             else temperature),
        "max_tokens": base_tokens,
        # 免费端点多为推理模型：不关思维链会把 max_tokens 吃光，返回空回答
        "reasoning": REASONING,
    }
    if json_mode:
        payload_base["response_format"] = {"type": "json_object"}

    attempts: list[dict] = []
    t0 = time.time()
    cancel_state = _CancelState(cancel)
    cancel_state.start()
    try:
        for m in chain[:5]:
            cancel_state.check()
            payload = {"model": m, "messages": messages, **payload_base}
            try:
                text, reasoning, usage = _stream_once(payload, stream, timeout, cancel_state)
            except AIError as exc:
                attempts.append({"model": m, "error": friendly_error(exc.message), "code": exc.code})
                if exc.code in (401, 499):
                    _bump(False, m, exc.message)
                    raise
                if exc.code in _RETRY_CODES:
                    low = exc.message.lower()
                    if ("agentic" in low or "location" in low or "not available" in low
                            or "data policy" in low):
                        _mark_blocked(m, exc.message)
                    continue
                _bump(False, m, exc.message)
                raise
            if not text.strip():
                # 思维链把 token 吃光时，同模型加大预算再试一次，别直接放弃
                detail = (usage or {}).get("completion_tokens_details") or {}
                spent = int(detail.get("reasoning_tokens") or 0)
                if base_tokens < 2400 and (spent >= base_tokens * 0.7 or (reasoning and stream)):
                    retry_tokens = min(3000, max(base_tokens * 2, 300))
                    try:
                        text, reasoning, usage = _stream_once(
                            {**payload, "max_tokens": retry_tokens}, stream, timeout, cancel_state)
                    except AIError as exc:
                        attempts.append({"model": m, "error": friendly_error(exc.message),
                                         "code": exc.code})
                        if exc.code == 499:
                            raise
                        continue
                    if text.strip():
                        _bump(True, m)
                        return Reply(text=text.strip(), reasoning=reasoning, model=m,
                                     seconds=time.time() - t0, usage=usage, attempts=attempts)
                attempts.append({"model": m, "error": "只返回推理过程，没有正式回答", "code": 0})
                continue
            _bump(True, m)
            return Reply(text=text.strip(), reasoning=reasoning, model=m,
                         seconds=time.time() - t0, usage=usage, attempts=attempts)
        msg = "；".join(f"{a['model'].split('/')[-1]}: {a['error'][:70]}" for a in attempts[-3:])
        _bump(False, "", msg)
        raise AIError("所有免费模型都没成功。" + (f"最后一次：{msg}" if msg else ""), 429)
    finally:
        cancel_state.stop()


def ask(prompt: str, system: str = "", max_tokens: int | None = None,
        need: Iterable[str] = ("text",), **kw) -> str:
    r = chat([user_msg(prompt)], system=system, need=need, max_tokens=max_tokens, **kw)
    return r.text


_SYS_JSON = ("你只输出一个合法 JSON 对象，不要任何解释、不要 Markdown 代码块。"
             "字符串里不要出现未转义的换行。")


def parse_json(text: str) -> dict | list | None:
    """从回复里掏出 JSON（容忍 ```json 围栏、前后废话、尾逗号）。"""
    if not text:
        return None
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    for cand in (raw,):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
    # 找第一个平衡的 { } 或 [ ]
    for opening, closing in (("{", "}"), ("[", "]")):
        start = raw.find(opening)
        if start < 0:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(raw)):
            ch = raw[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    snippet = raw[start:i + 1]
                    for attempt in (snippet,
                                    re.sub(r",\s*([}\]])", r"\1", snippet)):
                        try:
                            return json.loads(attempt)
                        except json.JSONDecodeError:
                            continue
    return None


def chat_json(messages: list[dict], system: str = "", need: Iterable[str] = ("text",),
              max_tokens: int | None = None, **kw) -> tuple[dict | list | None, Reply]:
    r = chat(messages, system=(system + "\n" + _SYS_JSON).strip(), need=need,
             max_tokens=max_tokens, json_mode=True, **kw)
    obj = parse_json(r.text)
    if obj is None and r.text:
        # 有些免费模型不支持 response_format，会返回带说明的文本，再试一次纯提示
        r2 = chat(list(messages) + [{"role": "user", "content": "只输出 JSON。"}],
                  system=(system + "\n" + _SYS_JSON).strip(), need=need,
                  max_tokens=max_tokens, **kw)
        obj = parse_json(r2.text)
        return obj, r2
    return obj, r


# ---------------------------------------------------------------- 便捷提示词
SYS_ASSIST = ("你是「FineTune Studio」内置的 AI 助手。回答用简体中文，简洁、具体、可执行，"
              "不要空话。涉及本机能力时，只讲这台机器真的能做到的事。")


def describe_status() -> str:
    st = stats()
    if not enabled():
        return "未配置 Key"
    bl = len(blocked())
    return (f"{default_model().split('/')[-1]} · 成功 {st.get('ok', 0)} 次"
            + (f" · 屏蔽 {bl} 个端点" if bl else ""))


def test(model: str = "", need: Iterable[str] = ("text",)) -> dict:
    """连通性测试：真发一条请求，返回 {ok, text, model, seconds, daily_left, error}。"""
    out: dict = {"ok": False, "text": "", "model": model or default_model(), "seconds": 0,
                 "daily_left": None, "error": ""}
    if not enabled():
        out["error"] = "还没有配置 Key"
        return out
    try:
        info = key_info()
        free = info.get("free_remaining")
        out["daily_left"] = free if free is not None else info.get("limit_remaining")
    except AIError as exc:
        out["error"] = f"Key 校验失败：{exc.message}"
        return out
    try:
        r = chat([user_msg("用一句中文回答：连接正常。")],
                 model=model, need=need, max_tokens=320, temperature=0.2, timeout=60)
    except AIError as exc:
        out["error"] = friendly_error(exc.message)
        return out
    out.update({"ok": True, "text": r.text, "model": r.model, "seconds": r.seconds})
    return out


def diagnose(models: list[str] | None = None, timeout: float = 25.0) -> list[dict]:
    """逐个真发一条最小请求，返回哪些免费模型现在能用、耗时多少。

    只测当前配置里的模型 + 内置推荐，最多 5 个，避免把每天 50 次的额度烧光。
    """
    if not enabled():
        return [{"model": "", "ok": False, "seconds": 0, "error": "还没有配置 Key"}]
    cands: list[str] = []
    for m in (models or ([default_model()] + list(SEED_MODELS))):
        m = str(m or "").strip()
        if m and m not in cands:
            cands.append(m)
    out: list[dict] = []
    for m in cands[:5]:
        t0 = time.time()
        state = _CancelState(None)
        try:
            text, reasoning, usage = _stream_once(
                {"model": m, "messages": [user_msg("只回复四个字：连接正常")],
                 "max_tokens": 80, "temperature": 0.1, "reasoning": REASONING},
                None, timeout, state)
            ok = bool(text.strip())
            out.append({"model": m, "ok": ok, "seconds": round(time.time() - t0, 2),
                        "text": text.strip()[:60],
                        "error": "" if ok else friendly_error("只返回推理过程，没有正式回答")})
        except AIError as exc:
            out.append({"model": m, "ok": False, "seconds": round(time.time() - t0, 2),
                        "text": "", "error": friendly_error(exc.message)})
        finally:
            state.stop()
    return out
