"""端對端流程：去重、同日重跑、dry-run、失敗門檻、修補。arXiv 與 Gemini 都以假資料替代。"""

import json
from argparse import Namespace

import pytest

import fetch_papers
import rank
import run_daily
import store
from conftest import NOW, FakeClient, text_response
from summarize import is_failed

GOOD = {"title_zh": "中文標題", "verdict": "一句話結論。", "problem": "p", "method": "m",
        "findings": "f", "implication": "i", "caveats": "c", "tags": ["測試"], "practicality": 4}


@pytest.fixture
def env(tmp_path, monkeypatch, candidates):
    """把抓取換成固定資料，時間固定在 NOW。client 回傳的不是評分時，語意評估失敗而退回關鍵字排序。"""
    fetch_calls = []

    def fake_collect():
        fetch_calls.append(1)
        import copy
        return copy.deepcopy(candidates)

    monkeypatch.setattr(fetch_papers, "collect_candidates", fake_collect)
    real_select = rank.select
    monkeypatch.setattr(rank, "select", lambda c, s, client=None, count=10, history=None:
                        real_select(c, s, client, count, now=NOW, history=history))
    return Namespace(dir=tmp_path, evals=tmp_path / "evaluations", fetch_calls=fetch_calls)


def args(**kw):
    base = dict(date="2026-10-15", force=False, dry_run=False, count=3)
    return Namespace(**{**base, **kw})


def ok_client():
    return FakeClient(lambda kw: text_response(json.dumps(GOOD)))


def test_dry_run_writes_nothing(env):
    assert run_daily.run_today(args(dry_run=True), None, env.dir, env.evals) == 0
    assert list(env.dir.glob("*.json")) == []


def test_daily_run_then_same_day_rerun_is_noop(env):
    assert run_daily.run_today(args(), ok_client(), env.dir, env.evals) == 0
    first = store.load_issue("2026-10-15", env.dir)
    assert len(first["papers"]) == 3
    assert first["ranking"] == "keyword"
    assert all(p["summary"]["ok"] for p in first["papers"])

    calls_before = len(env.fetch_calls)
    assert run_daily.run_today(args(), ok_client(), env.dir, env.evals) == 0
    assert len(env.fetch_calls) == calls_before, "同一天重跑不應該再抓"
    assert store.load_issue("2026-10-15", env.dir) == first


def test_next_day_skips_already_published(env):
    run_daily.run_today(args(), ok_client(), env.dir, env.evals)
    run_daily.run_today(args(date="2026-10-16"), ok_client(), env.dir, env.evals)
    day1 = {p["arxiv_id"] for p in store.load_issue("2026-10-15", env.dir)["papers"]}
    day2 = {p["arxiv_id"] for p in store.load_issue("2026-10-16", env.dir)["papers"]}
    assert day2 and not day1 & day2


def test_force_can_reselect_todays_own_papers(env):
    run_daily.run_today(args(), ok_client(), env.dir, env.evals)
    before = [p["arxiv_id"] for p in store.load_issue("2026-10-15", env.dir)["papers"]]
    run_daily.run_today(args(force=True), ok_client(), env.dir, env.evals)
    after = [p["arxiv_id"] for p in store.load_issue("2026-10-15", env.dir)["papers"]]
    assert after == before, "--force 重做時，今天原本的論文不應被當成已收錄而消失"


def llm_client(scores: dict):
    """語意評估依 scores（arxiv_id → (相關度, 內容價值)）回答，摘要一律成功。"""
    def respond(kw):
        if kw["config"].response_json_schema is rank.RERANK_SCHEMA:
            ids = [line.split('"')[1] for line in kw["contents"].splitlines() if line.startswith("<paper arxiv_id=")]
            rows = [{"arxiv_id": i, "relevance": scores.get(i, (1, 1))[0], "value": scores.get(i, (1, 1))[1],
                     "reason": "測試"} for i in ids]
            return text_response(json.dumps({"rankings": rows}))
        return text_response(json.dumps(GOOD))
    return FakeClient(respond)


def rerank_calls(client):
    return [c for c in client.calls if c["config"].response_json_schema is rank.RERANK_SCHEMA]


SCORES = {"2609.11111": (9, 9), "2610.20037": (8, 8), "2610.20366": (7, 7), "2609.34004": (9, 4)}


def test_scores_carry_over_to_next_day(env):
    day1 = llm_client(SCORES)
    assert run_daily.run_today(args(count=2), day1, env.dir, env.evals) == 0
    issue = store.load_issue("2026-10-15", env.dir)
    assert [p["arxiv_id"] for p in issue["papers"]] == ["2609.11111", "2610.20037"]
    assert issue["ranking"] == "llm" and issue["lookback_days"] == 365
    assert issue["papers"][0]["value"] == 9
    evaluated = store.load_evaluations(rank.RUBRIC_VERSION, env.evals)
    assert len(evaluated) == len(rerank_calls(day1)[0]["contents"].split("<paper ")) - 1

    day2 = llm_client({})
    assert run_daily.run_today(args(count=2, date="2026-10-16"), day2, env.dir, env.evals) == 0
    assert rerank_calls(day2) == [], "所有候選都評過了，不必再呼叫模型"
    issue2 = store.load_issue("2026-10-16", env.dir)
    assert [p["arxiv_id"] for p in issue2["papers"]] == ["2610.20366"], "前一天沒選上的好論文隔天入選；價值不夠的不選"


def test_scores_saved_even_when_nothing_qualifies(env):
    client = llm_client({})
    assert run_daily.run_today(args(), client, env.dir, env.evals) == 0
    assert store.load_issue("2026-10-15", env.dir) is None
    assert store.load_evaluations(rank.RUBRIC_VERSION, env.evals), "評過的分數要留下，明天不必重評"


def test_same_day_scores_are_merged(tmp_path):
    store.save_evaluations("2026-10-15", 1, "m", [{"arxiv_id": "a", "relevance": 1, "value": 1, "reason": ""},
                                                  {"arxiv_id": "b", "relevance": 1, "value": 1, "reason": ""}], tmp_path)
    store.save_evaluations("2026-10-15", 1, "m", [{"arxiv_id": "b", "relevance": 9, "value": 9, "reason": ""}], tmp_path)
    scores = store.load_evaluations(1, tmp_path)
    assert set(scores) == {"a", "b"} and scores["b"]["relevance"] == 9


def test_other_rubric_and_excluded_date_are_ignored(tmp_path):
    row = {"arxiv_id": "a", "relevance": 9, "value": 9, "reason": ""}
    store.save_evaluations("2026-10-14", 1, "m", [row], tmp_path)
    store.save_evaluations("2026-10-15", 2, "m", [{**row, "arxiv_id": "b"}], tmp_path)
    assert set(store.load_evaluations(2, tmp_path)) == {"b"}
    assert store.load_evaluations(1, tmp_path, exclude_date="2026-10-14") == {}


def test_too_many_failures_aborts_without_writing(env):
    bad = FakeClient(lambda kw: RuntimeError("404 model not found"))
    assert run_daily.run_today(args(), bad, env.dir, env.evals) == 1
    assert store.load_issue("2026-10-15", env.dir) is None


def test_fetch_failure_returns_error(env, monkeypatch):
    def down():
        raise fetch_papers.FetchError("down")
    monkeypatch.setattr(fetch_papers, "collect_candidates", down)
    assert run_daily.run_today(args(), ok_client(), env.dir, env.evals) == 1


def test_repair_fixes_legacy_failures(tmp_path):
    legacy = {"verdict": "自動摘要失敗，請直接看原文。", "title_zh": "x", "problem": "—", "method": "—",
              "findings": "—", "implication": "—", "caveats": "—", "tags": [], "practicality": 3}
    store.write_issue({"date": "2026-09-03", "scanned": 100, "papers": [
        {"arxiv_id": "a", "title": "A", "authors": [], "categories": [], "abstract": "x", "summary": legacy},
        {"arxiv_id": "b", "title": "B", "authors": [], "categories": [], "abstract": "x", "summary": {**GOOD, "ok": True}},
    ]}, tmp_path)

    client = ok_client()
    assert run_daily.run_repair(client, tmp_path) == 0
    assert len(client.calls) == 1, "只重做失敗的那篇"
    issue = store.load_issue("2026-09-03", tmp_path)
    assert not any(is_failed(p["summary"]) for p in issue["papers"])


def test_seen_ids_ignores_stray_files(tmp_path):
    store.write_issue({"date": "2026-10-01", "papers": [{"arxiv_id": "a"}]}, tmp_path)
    (tmp_path / "notes.json").write_text("{}", encoding="utf-8")
    (tmp_path / "2026-10-02.json").write_text("{broken", encoding="utf-8")
    assert store.seen_ids(tmp_path) == {"a"}


def test_missing_api_key_fails_fast(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(run_daily.build_site, "build", lambda: None)
    assert run_daily.main([]) == 1
