import json

import pytest
from google.genai import types

import rank
from conftest import NOW, FakeClient, text_response
from fetch_papers import Paper
from llm import LLMError


def make(title, abstract="", cat="q-fin.ST"):
    return Paper("x", title, [], abstract, cat, [cat], "2026-10-14T00:00:00Z", "", "", "")


@pytest.mark.parametrize("text", [
    "an alphabetical index of firms",
    "we compare several options for the estimator",
    "the signal processing community",
    "a stockholm syndrome",
])
def test_no_substring_false_positives(text):
    p = rank.keyword_score(make("Neutral title", text, cat="econ.GN"), NOW)
    assert p.matched_keywords == []


def test_plural_and_title_weighting():
    in_title = rank.keyword_score(make("Limit Order Books under Stress", cat="econ.GN"), NOW)
    in_abstract = rank.keyword_score(make("Neutral", "we model limit order books", cat="econ.GN"), NOW)
    assert "limit order book" in in_title.matched_keywords
    assert in_title.keyword_score > in_abstract.keyword_score > 0


def test_keyword_ranking_on_real_papers(candidates):
    sel = rank.select(candidates, seen=set(), client=None, count=10, now=NOW)
    ids = [p.arxiv_id for p in sel.papers]

    assert sel.method == "keyword"
    assert ids[0] in {"2610.20037", "2609.34004"}, "RL 投組與股票預測應排在最前面"
    assert "2610.19045" not in ids, "疫苗採購論文不能入選"
    assert "2610.18921" not in ids
    assert "2609.11111" not in ids, "超過一週的論文不能入選"


def test_seen_papers_are_excluded(candidates):
    sel = rank.select(candidates, seen={"2610.20037"}, client=None, now=NOW)
    assert "2610.20037" not in [p.arxiv_id for p in sel.papers]


def paper_ids(kwargs):
    return [line.split('"')[1] for line in kwargs["contents"].splitlines()
            if line.startswith("<paper arxiv_id=")]


def rerank_client(scores: dict, missing=()):
    def respond(kwargs):
        rows = [{"arxiv_id": i, "relevance": scores.get(i, 1), "reason": "測試"}
                for i in paper_ids(kwargs) if i not in missing]
        return text_response(json.dumps({"rankings": rows}))
    return FakeClient(respond)


def test_llm_rerank_orders_and_applies_threshold(candidates):
    client = rerank_client({"2610.19428": 9, "2610.20037": 8, "2609.34004": 7, "2610.20366": 15})
    sel = rank.select(candidates, seen=set(), client=client, now=NOW)

    assert sel.method == "llm"
    assert [p.arxiv_id for p in sel.papers] == ["2610.20366", "2610.19428", "2610.20037", "2609.34004"]
    assert sel.papers[0].relevance == 10, "超出範圍的分數要夾回 0–10"
    assert all(p.relevance >= rank.MIN_RELEVANCE for p in sel.papers)

    call = client.calls[0]
    assert call["model"] == "gemini-3.5-flash-lite"
    assert call["config"].response_json_schema is rank.RERANK_SCHEMA
    assert call["config"].thinking_config.thinking_level == types.ThinkingLevel.MINIMAL


def test_llm_failure_falls_back_to_keywords(candidates):
    client = FakeClient(lambda kw: LLMError("down"))
    sel = rank.select(candidates, seen=set(), client=client, now=NOW)
    assert sel.method == "keyword" and sel.papers
    assert all(p.relevance is None for p in sel.papers)


def test_llm_skipping_many_papers_falls_back(candidates):
    all_ids = [p.arxiv_id for p in candidates]
    client = rerank_client({}, missing=all_ids[:5])
    sel = rank.select(candidates, seen=set(), client=client, now=NOW)
    assert sel.method == "keyword"


def test_malformed_rows_count_as_missing(candidates):
    """Gemini 只保證是合法 JSON；分數不是數字的列要當成漏評，而不是讓程式當掉。"""
    def respond(kwargs):
        rows = [{"arxiv_id": i, "relevance": "高", "reason": ""} for i in paper_ids(kwargs)]
        return text_response(json.dumps({"rankings": rows}))
    sel = rank.select(candidates, seen=set(), client=FakeClient(respond), now=NOW)
    assert sel.method == "keyword" and sel.papers


# Gemini structured outputs 支援的 JSON Schema 子集（ai.google.dev/gemini-api/docs/structured-output）
GEMINI_SCHEMA_KEYWORDS = {
    "type", "title", "description", "properties", "required", "additionalProperties",
    "enum", "format", "minimum", "maximum", "items", "prefixItems", "minItems", "maxItems",
}


def test_schema_uses_only_supported_features():
    """只用 Gemini 支援的關鍵字；物件一律 additionalProperties: false 且全部欄位必填。"""
    from summarize import SUMMARY_SCHEMA

    def walk(node):
        assert set(node) <= GEMINI_SCHEMA_KEYWORDS, set(node) - GEMINI_SCHEMA_KEYWORDS
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
            for child in node["properties"].values():
                walk(child)
        if node.get("type") == "array":
            walk(node["items"])

    walk(rank.RERANK_SCHEMA)
    walk(SUMMARY_SCHEMA)
