"""呼叫 Gemini 並以 structured outputs 取得 JSON。

重試邏輯寫在這裡而不交給 SDK，因為免費額度有兩種上限要分開處理：
每分鐘上限等一下就好；今日額度用完則重試也沒用，應該立刻放棄。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

log = logging.getLogger(__name__)

API_KEY_ENV = "GEMINI_API_KEY"

THINKING_LEVELS = {"minimal", "low", "medium", "high"}

# 觸發每分鐘上限時最多等這麼久；Gemini 通常會在錯誤裡告訴我們要等幾秒
MAX_RETRY_WAIT = 90.0


class LLMError(RuntimeError):
    pass


class QuotaExhausted(LLMError):
    """今天的免費額度用完了，重試也沒用。"""


class _BadOutput(LLMError):
    """輸出被截斷或 JSON 壞掉，值得再試一次。"""

    def __init__(self, message: str, truncated: bool = False):
        super().__init__(message)
        self.truncated = truncated


def make_client():
    """沒有 API key 時回傳 None，讓呼叫端決定要退回哪種模式。"""
    key = os.getenv(API_KEY_ENV)
    if not key:
        return None
    from google import genai
    from google.genai import types
    # SDK 預設不重試；重試由 structured_call 處理。
    # 免費額度實測單次請求可能排隊將近 4 分鐘，逾時設 10 分鐘（單位是毫秒）
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=600_000))


def structured_call(
    client,
    *,
    model: str,
    system: str,
    content: str | list,
    schema: dict,
    max_tokens: int,
    thinking: str | None = None,
    attempts: int = 4,
    sleep=time.sleep,
) -> dict[str, Any]:
    """送出請求並解析 JSON。

    - 每分鐘上限、伺服器錯誤、連線錯誤：等一下再試，最多 attempts 次
    - 今日額度用完：直接丟出 QuotaExhausted
    - 輸出被截斷或 JSON 壞掉：加大上限再試一次
    """
    config = _config(system, schema, max_tokens, thinking)
    redo_left = 1
    failures = 0
    while True:
        try:
            resp = client.models.generate_content(model=model, contents=content, config=config)
        except Exception as exc:
            failures += 1
            error, wait = _classify(exc, failures)
            if wait is None or failures >= attempts:
                raise error from exc
            log.warning("%s；%.0f 秒後重試", error, wait)
            sleep(wait)
            continue

        try:
            return _parse(resp)
        except _BadOutput as exc:
            if not redo_left:
                raise LLMError(str(exc)) from None
            redo_left -= 1
            if exc.truncated:
                # 思考 token 也算在上限內，被截斷通常是想太久
                config = _config(system, schema, config.max_output_tokens * 2, thinking)
            log.warning("%s，重試一次", exc)


def _config(system: str, schema: dict, max_tokens: int, thinking: str | None):
    from google.genai import types

    if thinking and thinking.lower() not in THINKING_LEVELS:
        raise LLMError(f"不支援的思考程度 {thinking!r}，可用：{'、'.join(sorted(THINKING_LEVELS))}")
    return types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_json_schema=schema,
        max_output_tokens=max_tokens,
        thinking_config=types.ThinkingConfig(thinking_level=thinking.upper()) if thinking else None,
    )


def _name(value) -> str:
    return getattr(value, "name", None) or str(value or "")


def _parse(resp) -> dict[str, Any]:
    feedback = getattr(resp, "prompt_feedback", None)
    if feedback is not None and feedback.block_reason:
        raise LLMError(f"輸入被擋下：{_name(feedback.block_reason)}")

    candidate = resp.candidates[0] if resp.candidates else None
    reason = _name(candidate.finish_reason) if candidate else ""
    if reason == "MAX_TOKENS":
        raise _BadOutput("輸出被截斷", truncated=True)
    if reason not in ("", "STOP", "FINISH_REASON_UNSPECIFIED"):
        # SAFETY、RECITATION 等：內容被擋，重試結果也一樣
        raise LLMError(f"模型拒絕回應：{reason}")

    # resp.text 只串接正式輸出，不含思考內容
    try:
        data = json.loads(resp.text or "")
    except json.JSONDecodeError as exc:
        raise _BadOutput(f"JSON 解析失敗：{exc}") from None
    if not isinstance(data, dict):
        raise _BadOutput("回傳的不是 JSON 物件")
    return data


def _classify(exc: Exception, failures: int) -> tuple[LLMError, float | None]:
    """把例外轉成 LLMError，並決定等幾秒再試；None 代表不要重試。"""
    import httpx
    from google.genai import errors

    if isinstance(exc, errors.APIError):
        if exc.code == 429:
            if _is_daily_quota(exc.details):
                return QuotaExhausted(f"今日免費額度已用完：{exc.message}"), None
            wait = _retry_delay(exc.details) or 20.0 * failures
            return LLMError("超過每分鐘請求上限"), min(wait, MAX_RETRY_WAIT)
        if exc.code in (408, 500, 502, 503, 504):
            return LLMError(f"伺服器錯誤 {exc.code}"), 5.0 * 2 ** failures
        # 400 參數錯誤、401/403 金鑰無效、404 模型名稱錯誤，重試也不會好
        return LLMError(f"{exc.code} {exc.status}：{exc.message}"), None
    if isinstance(exc, httpx.TransportError):
        return LLMError(f"連線錯誤：{type(exc).__name__}"), 5.0 * 2 ** failures
    return LLMError(f"{type(exc).__name__}: {exc}"), None


def _error_details(payload) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    items = payload.get("error", payload).get("details", [])
    return [d for d in items if isinstance(d, dict)] if isinstance(items, list) else []


def _is_daily_quota(payload) -> bool:
    return any(
        "PerDay" in str(v.get("quotaId", ""))
        for d in _error_details(payload)
        for v in d.get("violations", []) or []
        if isinstance(v, dict)
    )


def _retry_delay(payload) -> float | None:
    for d in _error_details(payload):
        m = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(d.get("retryDelay", "")))
        if m:
            return float(m.group(1))
    return None
