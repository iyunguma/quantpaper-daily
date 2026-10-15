import json
import threading

import pytest
from google.genai import types

import summarize as sm
from conftest import FakeClient, blocked_response, daily_quota_error, text_response

PAPER = {
    "arxiv_id": "2610.20037", "title": "Data-Driven Dynamic Portfolio Selection",
    "authors": ["A", "B"], "categories": ["q-fin.PM"], "abstract": "An abstract.",
    "pdf_url": "https://arxiv.org/pdf/2610.20037",
}

GOOD = {
    "title_zh": "資料驅動的動態投組選擇", "verdict": "RL 投組在七資產樣本外勝過基準。",
    "problem": "p", "method": "m", "findings": "f", "implication": "i", "caveats": "c",
    "tags": ["強化學習", "投組", " ", "風險", "CVaR", "copula", "多餘"], "practicality": 9,
}


def good_client():
    return FakeClient(lambda kw: text_response(json.dumps(GOOD)))


def test_success_normalizes_output():
    client = good_client()
    rec = sm.summarize_one(client, PAPER)
    s = rec["summary"]

    assert s["ok"] is True and s["source"] == "abstract"
    assert s["practicality"] == 5, "超出 1–5 要夾回範圍"
    assert s["tags"] == ["強化學習", "投組", "風險", "CVaR", "copula"]
    assert not sm.is_failed(s)

    call = client.calls[0]
    config = call["config"]
    assert call["model"] == "gemini-3.8-flash"
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema is sm.SUMMARY_SCHEMA
    assert config.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM
    assert config.max_output_tokens >= 4000, "思考 token 也算在上限內，要留足空間"


def test_truncated_output_retries_with_bigger_budget():
    responses = iter([text_response('{"title_zh": "截', finish_reason="MAX_TOKENS"),
                      text_response(json.dumps(GOOD))])
    client = FakeClient(lambda kw: next(responses))
    assert sm.summarize_one(client, PAPER)["summary"]["ok"]
    first, second = (c["config"].max_output_tokens for c in client.calls)
    assert second == 2 * first


@pytest.mark.parametrize("response", [
    text_response("", finish_reason="SAFETY"),
    blocked_response(),
    text_response(json.dumps({**GOOD, "verdict": ""})),
    text_response("[]"),
    RuntimeError("401 API key not valid"),
])
def test_failures_produce_marked_placeholder(response):
    rec = sm.summarize_one(FakeClient(lambda kw: response), PAPER)
    assert sm.is_failed(rec["summary"])
    assert rec["summary"]["title_zh"] == PAPER["title"]


def test_wrong_field_types_are_tolerated():
    """Gemini 只保證是合法 JSON，欄位型別不對時不能讓整篇失敗。"""
    data = {**GOOD, "tags": "強化學習", "practicality": "很高"}
    s = sm.summarize_one(FakeClient(lambda kw: text_response(json.dumps(data))), PAPER)["summary"]
    assert s["ok"] and s["tags"] == [] and s["practicality"] == 3


def test_legacy_v1_failure_is_detected():
    assert sm.is_failed({"verdict": "自動摘要失敗，請直接看原文。", "practicality": 3})
    assert sm.is_failed(None)


def test_pdf_mode_sends_inline_pdf():
    client = good_client()
    rec = sm.summarize_one(client, PAPER, pdf=b"%PDF-1.7 fake")
    contents = client.calls[0]["contents"]
    assert contents[0]["inline_data"] == {"mime_type": "application/pdf", "data": b"%PDF-1.7 fake"}
    assert rec["summary"]["source"] == "pdf"


def test_pdf_failure_falls_back_to_abstract():
    def respond(kw):
        if isinstance(kw["contents"], list):
            return text_response("", finish_reason="SAFETY")
        return text_response(json.dumps(GOOD))
    rec = sm.summarize_one(FakeClient(respond), PAPER, pdf=b"%PDF-1.7 fake")
    assert rec["summary"]["ok"] and rec["summary"]["source"] == "abstract"


def test_summarize_all_preserves_order():
    papers = [{**PAPER, "arxiv_id": f"id{i}", "title": f"T{i}"} for i in range(6)]
    out = sm.summarize_all(good_client(), papers, source="abstract")
    assert [r["arxiv_id"] for r in out] == [p["arxiv_id"] for p in papers]


def test_summarize_all_runs_in_parallel():
    barrier = threading.Barrier(3, timeout=5)

    def respond(kw):
        barrier.wait()  # 三篇同時在跑才會通過；依序執行會逾時而失敗
        return text_response(json.dumps(GOOD))
    papers = [{**PAPER, "arxiv_id": f"id{i}", "title": f"T{i}"} for i in range(3)]
    out = sm.summarize_all(FakeClient(respond), papers, source="abstract", workers=3)
    assert not any(sm.is_failed(r["summary"]) for r in out)


def test_quota_exhausted_stops_remaining_calls():
    client = FakeClient(lambda kw: daily_quota_error())
    papers = [{**PAPER, "arxiv_id": f"id{i}", "title": f"T{i}"} for i in range(8)]
    out = sm.summarize_all(client, papers, source="abstract", workers=3)
    assert len(client.calls) <= 3, "額度用完後，還沒開始的論文不該再呼叫"
    assert [r["arxiv_id"] for r in out] == [p["arxiv_id"] for p in papers]
    assert all(sm.is_failed(r["summary"]) for r in out)
    assert "額度" in out[-1]["summary"]["error"]
