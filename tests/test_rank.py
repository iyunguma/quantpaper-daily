import json

import pytest

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


def rerank_client(scores: dict, missing=()):
    def respond(kwargs):
        ids = [line.split('"')[1] for line in kwargs["messages"][0]["content"].splitlines()
               if line.startswith("<paper arxiv_id=")]
        rows = [{"arxiv_id": i, "relevance": scores.get(i, 1), "reason": "測試"} for i in ids if i not in missing]
        return text_response(json.dumps({"rankings": rows}), thinking=False)
    return FakeClient(respond)


def test_llm_rerank_orders_and_applies_threshold(candidates):
    client = rerank_client({"2610.19428": 9, "2610.20037": 8, "2609.34004": 7, "2610.20366": 15})
    sel = rank.select(candidates, seen=set(), client=client, now=NOW)

    assert sel.method == "llm"
    assert [p.arxiv_id for p in sel.papers] == ["2610.20366", "2610.19428", "2610.20037", "2609.34004"]
    assert sel.papers[0].relevance == 10, "超出範圍的分數要夾回 0–10"
    assert all(p.relevance >= rank.MIN_RELEVANCE for p in sel.papers)

    call = client.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["thinking"] == {"type": "disabled"}


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


def test_schema_uses_only_supported_features():
    """structured outputs 不支援 minimum / maximum / minLength，物件必須 additionalProperties: false。"""
    from summarize import SUMMARY_SCHEMA

    def walk(node):
        assert not {"minimum", "maximum", "minLength", "maxLength", "maxItems"} & set(node)
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
            for child in node["properties"].values():
                walk(child)
        if node.get("type") == "array":
            walk(node["items"])

    walk(rank.RERANK_SCHEMA)
    walk(SUMMARY_SCHEMA)
