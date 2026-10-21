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
    assert ids.index("2610.20037") < ids.index("2610.20366"), "RL 投組應排在理論論文前面"
    assert "2610.19045" not in ids, "疫苗採購論文不能入選"
    assert "2610.18921" not in ids
    assert "2609.11111" in ids, "一個月前的論文也可以入選"
    assert "2509.22222" not in ids, "超過一年的論文不能入選"


def test_seen_papers_are_excluded(candidates):
    sel = rank.select(candidates, seen={"2610.20037"}, client=None, now=NOW)
    assert "2610.20037" not in [p.arxiv_id for p in sel.papers]


def paper_ids(kwargs):
    return [line.split('"')[1] for line in kwargs["contents"].splitlines()
            if line.startswith("<paper arxiv_id=")]


def rerank_client(scores: dict, missing=()):
    """scores：arxiv_id → (相關度, 內容價值)；沒列出的給 (1, 1)。"""
    def respond(kwargs):
        rows = [{"arxiv_id": i, "relevance": scores.get(i, (1, 1))[0], "value": scores.get(i, (1, 1))[1],
                 "reason": f"理由{i}"}
                for i in paper_ids(kwargs) if i not in missing]
        return text_response(json.dumps({"rankings": rows}))
    return FakeClient(respond)


def test_llm_ranks_by_relevance_plus_value(candidates):
    client = rerank_client({
        "2610.19428": (9, 9), "2610.20037": (8, 6), "2609.34004": (10, 3),
        "2610.20366": (15, 7), "2610.19254": (6, 9),
    })
    sel = rank.select(candidates, seen=set(), client=client, now=NOW)

    assert sel.method == "llm"
    assert [p.arxiv_id for p in sel.papers] == ["2610.19428", "2610.20366", "2610.19254", "2610.20037"]
    assert sel.papers[1].relevance == 10, "超出範圍的分數要夾回 0–10"
    assert "2609.34004" not in [p.arxiv_id for p in sel.papers], "主題再對口，內容價值不夠也不選"
    assert all(p.relevance >= rank.MIN_RELEVANCE and p.value >= rank.MIN_VALUE for p in sel.papers)
    assert sel.papers[0].relevance_reason == "理由2610.19428"

    call = client.calls[0]
    assert call["model"] == "gemini-3.5-flash-lite"
    assert call["config"].response_json_schema is rank.RERANK_SCHEMA
    assert call["config"].thinking_config.thinking_level == types.ThinkingLevel.MEDIUM
    assert "2509.22222" not in paper_ids(call), "超過一年的論文不送去評估"


def test_nothing_good_enough_selects_nothing(candidates):
    """評估成功但沒有夠格的論文時，寧可不選，也不退回關鍵字排序。"""
    sel = rank.select(candidates, seen=set(), client=rerank_client({}), now=NOW)
    assert sel.method == "llm" and sel.papers == []
    assert len(sel.evaluated) == sel.eligible, "評過的都要回報，才能存起來"


def test_comment_and_journal_ref_are_sent():
    p = make("Momentum in Futures")
    p.comment, p.journal_ref = "Accepted at ICAIF 2026; code on GitHub", "Quantitative Finance 26 (2026)"
    client = rerank_client({"x": (8, 8)})
    rank.select([p], seen=set(), client=client, now=NOW)
    content = client.calls[0]["contents"]
    assert "附註：Accepted at ICAIF 2026" in content and "期刊：Quantitative Finance" in content


def test_history_is_reused_not_reevaluated(candidates):
    history = {
        "2610.20037": {"arxiv_id": "2610.20037", "relevance": 9, "value": 9, "reason": "先前評過"},
        "2609.11111": {"arxiv_id": "2609.11111", "relevance": 2, "value": 2, "reason": "先前評過"},
    }
    client = rerank_client({"2610.20366": (8, 7)})
    sel = rank.select(candidates, seen=set(), client=client, now=NOW, history=history)

    sent = paper_ids(client.calls[0])
    assert "2610.20037" not in sent and "2609.11111" not in sent, "評過的論文不再送給模型"
    assert [p.arxiv_id for p in sel.papers] == ["2610.20037", "2610.20366"]
    assert sel.papers[0].relevance_reason == "先前評過"
    assert {p.arxiv_id for p in sel.evaluated} == set(sent), "只回報今天新評的"


def test_everything_already_evaluated_skips_the_call(candidates):
    history = {p.arxiv_id: {"arxiv_id": p.arxiv_id, "relevance": 7, "value": 7, "reason": ""} for p in candidates}
    client = rerank_client({})
    sel = rank.select(candidates, seen=set(), client=client, now=NOW, history=history)
    assert client.calls == [] and sel.method == "llm" and sel.papers


def test_llm_failure_falls_back_to_history(candidates):
    history = {"2610.20366": {"arxiv_id": "2610.20366", "relevance": 8, "value": 8, "reason": ""}}
    sel = rank.select(candidates, seen=set(), client=FakeClient(lambda kw: LLMError("down")),
                      now=NOW, history=history)
    assert sel.method == "llm"
    assert [p.arxiv_id for p in sel.papers] == ["2610.20366"]
    assert sel.evaluated == []


def test_few_missing_scores_stay_unevaluated(candidates):
    eligible = [p for p in candidates if p.arxiv_id != "2509.22222"]
    client = rerank_client({p.arxiv_id: (7, 7) for p in eligible}, missing=["2610.20037"])
    sel = rank.select(candidates, seen=set(), client=client, now=NOW)
    assert sel.method == "llm"
    assert "2610.20037" not in [p.arxiv_id for p in sel.evaluated], "漏評的不能存成假分數，下次要重評"
    assert "2610.20037" not in [p.arxiv_id for p in sel.papers]


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
        rows = [{"arxiv_id": i, "relevance": 8, "value": "高", "reason": ""} for i in paper_ids(kwargs)]
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


def ranked_papers(topics: list[str]) -> list[Paper]:
    out = []
    for i, t in enumerate(topics):
        p = make(f"Paper {i}")
        p.arxiv_id, p.topic = f"p{i}", t
        out.append(p)
    return out


def test_diversify_caps_topic_and_keeps_rank_order():
    ranked = ranked_papers(["造市與微結構"] * 5 + ["Alpha 與因子", "投組與風險"])
    picked = rank.diversify(ranked, count=5, cap=3)
    assert [p.arxiv_id for p in picked] == ["p0", "p1", "p2", "p5", "p6"]


def test_diversify_fills_up_when_other_topics_run_out():
    ranked = ranked_papers(["造市與微結構"] * 5 + ["Alpha 與因子"])
    picked = rank.diversify(ranked, count=5, cap=3)
    assert [p.arxiv_id for p in picked] == ["p0", "p1", "p2", "p3", "p5"], "寧可同主題補滿，也不要無故少選"


def test_topic_is_parsed_and_unknown_topic_is_dropped(candidates):
    def respond(kwargs):
        rows = [{"arxiv_id": i, "topic": "交易執行" if n == 0 else "亂寫的主題", "reason": "",
                 "relevance": 8, "value": 8} for n, i in enumerate(paper_ids(kwargs))]
        return text_response(json.dumps({"rankings": rows}))
    sel = rank.select(candidates, seen=set(), client=FakeClient(respond), now=NOW)
    topics = sorted(p.topic for p in sel.evaluated)
    assert topics.count("交易執行") == 1 and topics.count("") == len(topics) - 1
    assert rank.evaluation_row(sel.evaluated[0])["topic"] in ("交易執行", "")


def test_simulation_only_value_is_capped(candidates):
    """模型不太遵守提示詞裡的上限，由程式保證：只靠模擬的論文內容價值不超過上限。"""
    def respond(kwargs):
        rows = [{"arxiv_id": i, "topic": "造市與微結構", "evidence": "只有模擬資料" if n == 0 else "真實市場資料",
                 "reason": "", "relevance": 9, "value": 9} for n, i in enumerate(paper_ids(kwargs))]
        return text_response(json.dumps({"rankings": rows}))
    sel = rank.select(candidates, seen=set(), client=FakeClient(respond), now=NOW)
    sim = [p for p in sel.evaluated if p.evidence == "只有模擬資料"]
    assert len(sim) == 1 and sim[0].value == rank.SIMULATION_VALUE_CAP
    assert all(p.value == 9 for p in sel.evaluated if p.evidence == "真實市場資料")
    assert rank.evaluation_row(sim[0])["evidence"] == "只有模擬資料"


def test_implausible_performance_is_excluded(candidates):
    def respond(kwargs):
        rows = [{"arxiv_id": i, "topic": "Alpha 與因子", "evidence": "真實市場資料", "implausible": n == 0,
                 "reason": "", "relevance": 9, "value": 9} for n, i in enumerate(paper_ids(kwargs))]
        return text_response(json.dumps({"rankings": rows}))
    client = FakeClient(respond)
    sel = rank.select(candidates, seen=set(), client=client, now=NOW)
    flagged = paper_ids(client.calls[0])[0]
    assert next(p for p in sel.evaluated if p.arxiv_id == flagged).value == rank.IMPLAUSIBLE_VALUE_CAP
    assert flagged not in [p.arxiv_id for p in sel.papers], "績效好到不可信的論文不入選"


def test_ties_prefer_value_and_reason_terms_are_fixed(candidates):
    def respond(kwargs):
        ids = paper_ids(kwargs)
        rows = [{"arxiv_id": i, "topic": "其他", "evidence": "看不出來", "implausible": False,
                 "reason": "最優執行與優化，數據完整", "relevance": 9 if n == 0 else 8, "value": 6 if n == 0 else 7}
                for n, i in enumerate(ids[:2])]
        rows += [{"arxiv_id": i, "topic": "其他", "evidence": "看不出來", "implausible": False,
                  "reason": "", "relevance": 1, "value": 1} for i in ids[2:]]
        return text_response(json.dumps({"rankings": rows}))
    client = FakeClient(respond)
    sel = rank.select(candidates, seen=set(), client=client, now=NOW)
    first, second = paper_ids(client.calls[0])[:2]
    assert [p.arxiv_id for p in sel.papers] == [second, first], "總分同為 15 時，價值 7 排在價值 6 前面"
    assert sel.papers[0].relevance_reason == "最佳執行與最佳化，資料完整"


def test_rerank_asks_for_reason_before_scores():
    """先寫判斷依據再打分數，分數比較不會全部擠在同一個值。"""
    props = list(rank.RERANK_SCHEMA["properties"]["rankings"]["items"]["properties"])
    assert props.index("evidence") < props.index("implausible") < props.index("reason") \
        < props.index("relevance") < props.index("value")
    assert set(rank.RERANK_SCHEMA["properties"]["rankings"]["items"]["properties"]["topic"]["enum"]) == set(rank.TOPICS)


def test_reason_drops_echoed_length_limit():
    row = {"arxiv_id": "x", "relevance": 8, "value": 7, "reason": "十年樣本外，扣成本後夏普 1.2，30 字以內。"}
    assert rank._scores(row)[2] == "十年樣本外，扣成本後夏普 1.2。"
