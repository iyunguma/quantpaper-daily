import json

import pytest

import summarize as sm
from conftest import FakeClient, text_response

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
    assert call["model"] == "claude-sonnet-5-5"
    assert call["output_config"]["effort"] == "medium"
    assert call["output_config"]["format"]["schema"] is sm.SUMMARY_SCHEMA
    assert call["max_tokens"] >= 4000, "要留空間給 adaptive thinking"


def test_truncated_output_retries_with_bigger_budget():
    responses = iter([text_response('{"title_zh": "截', stop_reason="max_tokens"),
                      text_response(json.dumps(GOOD))])
    client = FakeClient(lambda kw: next(responses))
    assert sm.summarize_one(client, PAPER)["summary"]["ok"]
    assert client.calls[1]["max_tokens"] == 2 * client.calls[0]["max_tokens"]


@pytest.mark.parametrize("response", [
    text_response("", stop_reason="refusal"),
    text_response(json.dumps({**GOOD, "verdict": ""})),
    RuntimeError("401 invalid x-api-key"),
])
def test_failures_produce_marked_placeholder(response):
    rec = sm.summarize_one(FakeClient(lambda kw: response), PAPER)
    assert sm.is_failed(rec["summary"])
    assert rec["summary"]["title_zh"] == PAPER["title"]


def test_legacy_v1_failure_is_detected():
    assert sm.is_failed({"verdict": "自動摘要失敗，請直接看原文。", "practicality": 3})
    assert sm.is_failed(None)


def test_pdf_mode_sends_document_block():
    client = good_client()
    rec = sm.summarize_one(client, PAPER, pdf=b"%PDF-1.7 fake")
    content = client.calls[0]["messages"][0]["content"]
    assert content[0]["type"] == "document"
    assert content[0]["source"]["media_type"] == "application/pdf"
    assert rec["summary"]["source"] == "pdf"


def test_pdf_failure_falls_back_to_abstract():
    def respond(kw):
        if isinstance(kw["messages"][0]["content"], list):
            return text_response("", stop_reason="refusal")
        return text_response(json.dumps(GOOD))
    rec = sm.summarize_one(FakeClient(respond), PAPER, pdf=b"%PDF-1.7 fake")
    assert rec["summary"]["ok"] and rec["summary"]["source"] == "abstract"


def test_summarize_all_preserves_order():
    papers = [{**PAPER, "arxiv_id": f"id{i}", "title": f"T{i}"} for i in range(6)]
    out = sm.summarize_all(good_client(), papers, source="abstract")
    assert [r["arxiv_id"] for r in out] == [p["arxiv_id"] for p in papers]
