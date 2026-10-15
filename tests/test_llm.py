"""Gemini 呼叫層：重試、免費額度、輸出驗證。不會真的連網。"""

import httpx
import pytest
from google.genai import types

import config
import llm
from conftest import FakeClient, api_error, daily_quota_error, minute_quota_error, text_response

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}},
          "required": ["a"], "additionalProperties": False}


def call(client, sleeps, **kw):
    return llm.structured_call(client, model="m", system="s", content="c", schema=SCHEMA,
                               max_tokens=100, sleep=sleeps.append, **kw)


def sequence(*items):
    it = iter(items)
    return FakeClient(lambda kw: next(it))


def test_per_minute_limit_waits_as_told_then_succeeds():
    sleeps = []
    assert call(sequence(minute_quota_error("37s"), text_response('{"a": 1}')), sleeps) == {"a": 1}
    assert sleeps == [37.0]


def test_daily_quota_gives_up_immediately():
    sleeps, client = [], sequence(daily_quota_error())
    with pytest.raises(llm.QuotaExhausted):
        call(client, sleeps)
    assert sleeps == [] and len(client.calls) == 1


def test_server_and_connection_errors_are_retried():
    sleeps = []
    client = sequence(api_error(503), httpx.ConnectError("boom"), text_response('{"a": 2}'))
    assert call(client, sleeps) == {"a": 2}
    assert len(sleeps) == 2


def test_retries_are_bounded():
    client = FakeClient(lambda kw: api_error(503))
    with pytest.raises(llm.LLMError):
        call(client, [], attempts=3)
    assert len(client.calls) == 3


@pytest.mark.parametrize("code", [400, 403, 404])
def test_client_errors_are_not_retried(code):
    client = FakeClient(lambda kw: api_error(code, message="bad request"))
    with pytest.raises(llm.LLMError, match=str(code)):
        call(client, [])
    assert len(client.calls) == 1


def test_bad_json_is_retried_once():
    assert call(sequence(text_response("{oops"), text_response('{"a": 3}')), []) == {"a": 3}


def test_config_sent_to_gemini():
    client = sequence(text_response('{"a": 1}'))
    call(client, [], thinking="low")
    cfg = client.calls[0]["config"]
    assert cfg.system_instruction == "s"
    assert cfg.response_mime_type == "application/json"
    assert cfg.response_json_schema is SCHEMA
    assert cfg.thinking_config.thinking_level == types.ThinkingLevel.LOW


def test_unknown_thinking_level_fails_before_calling():
    client = sequence()
    with pytest.raises(llm.LLMError, match="思考程度"):
        call(client, [], thinking="max")
    assert client.calls == []


def test_make_client_needs_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert llm.make_client() is None
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert llm.make_client() is not None


def test_empty_variables_fall_back_to_defaults(monkeypatch):
    """GitHub Actions 沒設定的 Variables 會傳進空字串。"""
    monkeypatch.setenv("SOME_SETTING", "")
    assert config._env("SOME_SETTING", "預設") == "預設"
    monkeypatch.setenv("SOME_SETTING", "自訂")
    assert config._env("SOME_SETTING", "預設") == "自訂"
