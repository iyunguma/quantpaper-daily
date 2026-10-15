"""呼叫 Claude 並以 structured outputs 取得保證符合 schema 的 JSON。"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


def make_client():
    """沒有 API key 時回傳 None，讓呼叫端決定要退回哪種模式。"""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    from anthropic import Anthropic
    # SDK 本身會對 429 / 5xx / 連線錯誤自動重試
    return Anthropic(max_retries=4, timeout=300)


def structured_call(
    client,
    *,
    model: str,
    system: str,
    content: str | list[dict],
    schema: dict,
    max_tokens: int,
    effort: str | None = None,
    thinking: dict | None = None,
    attempts: int = 2,
) -> dict[str, Any]:
    """送出請求並解析 JSON。輸出被截斷時自動加大上限再試一次。"""
    output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
    if effort:
        output_config["effort"] = effort

    last_error: Exception | None = None
    for attempt in range(attempts):
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens * (2 ** attempt),
            "system": system,
            "messages": [{"role": "user", "content": content}],
            "output_config": output_config,
        }
        if thinking:
            kwargs["thinking"] = thinking

        try:
            resp = client.messages.create(**kwargs)
        except Exception as exc:  # SDK 已重試過，這裡直接往上丟
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

        if resp.stop_reason == "refusal":
            raise LLMError("模型拒絕回應")
        if resp.stop_reason == "max_tokens":
            last_error = LLMError("輸出被截斷")
            log.warning("輸出超過 max_tokens，加大上限重試")
            continue

        # adaptive thinking 開啟時 content 裡會有 thinking 區塊，只取 text
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            last_error = LLMError(f"JSON 解析失敗：{exc}")

    raise last_error or LLMError("未知錯誤")
