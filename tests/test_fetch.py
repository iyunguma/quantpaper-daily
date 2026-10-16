from types import SimpleNamespace

import pytest
import requests

import fetch_papers as fp
from conftest import NOW, entry_xml, feed_xml, REAL_PAPERS


def test_parse_feed_extracts_fields(real_feed):
    papers, total = fp.parse_feed(real_feed)
    ids = [p.arxiv_id for p in papers]

    assert "2610.20037" in ids
    assert "q-fin/0701001" in ids, "舊式 ID 要保留分類前綴"
    assert len(papers) == len(REAL_PAPERS) + 3, "arXiv 的 Error entry 要略過"

    p = next(p for p in papers if p.arxiv_id == "2610.20037")
    assert p.primary_category == "q-fin.PM"
    assert p.authors == ["Ada Lovelace", "Alan Turing"]
    assert p.abstract.startswith("Vine-copula") and "\n" not in p.abstract
    assert p.pdf_url.startswith("https://arxiv.org/pdf/2610.20037")
    assert p.abs_url == "https://arxiv.org/abs/2610.20037"
    assert p.comment == "" and p.journal_ref == ""


def test_parse_feed_reads_comment_and_journal_ref():
    xml = feed_xml([entry_xml(*REAL_PAPERS[0], comment="12 pages;\n accepted at ICAIF",
                              journal_ref="Quant. Finance 26 (2026)")])
    p = fp.parse_feed(xml)[0][0]
    assert p.comment == "12 pages; accepted at ICAIF"
    assert p.journal_ref == "Quant. Finance 26 (2026)"


@pytest.mark.parametrize("raw,expected", [
    ("http://arxiv.org/abs/2510.01234v12", "2510.01234"),
    ("http://arxiv.org/abs/2510.01234", "2510.01234"),
    ("http://arxiv.org/abs/q-fin/0701001v1", "q-fin/0701001"),
    ("http://arxiv.org/abs/solv-int/9901001v2", "solv-int/9901001"),
    ("http://arxiv.org/api/errors#bad", ""),
])
def test_parse_arxiv_id(raw, expected):
    assert fp.parse_arxiv_id(raw) == expected


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.requests = []

    def get(self, url, params=None, timeout=None):
        self.requests.append(params)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def http(status=200, text="", headers=None):
    resp = SimpleNamespace(status_code=status, text=text, headers=headers or {})

    def raise_for_status():
        if status >= 400:
            raise requests.HTTPError(f"HTTP {status}")
    resp.raise_for_status = raise_for_status
    return resp


def test_query_retries_empty_feed_then_succeeds():
    good = feed_xml([entry_xml(*REAL_PAPERS[0])])
    session = FakeSession([http(text=feed_xml([], total=500)), http(text=good)])
    sleeps = []
    papers, total = fp.query_arxiv(session, "cat:q-fin.TR", 10, sleep=sleeps.append)
    assert [p.arxiv_id for p in papers] == ["2610.20037"] and total == 1
    assert len(sleeps) == 1


def test_query_honours_retry_after_and_gives_up():
    session = FakeSession([http(503, headers={"Retry-After": "42"})] * 3)
    sleeps = []
    assert fp.query_arxiv(session, "q", 10, attempts=3, sleep=sleeps.append) is None
    assert sleeps[0] == 42


def test_query_survives_connection_error():
    good = feed_xml([entry_xml(*REAL_PAPERS[0])])
    session = FakeSession([requests.ConnectionError("boom"), http(text=good)])
    assert fp.query_arxiv(session, "q", 10, sleep=lambda s: None)


def test_collect_dedupes_and_sets_user_agent():
    a = feed_xml([entry_xml(*REAL_PAPERS[0]), entry_xml(*REAL_PAPERS[1])])
    b = feed_xml([entry_xml(*REAL_PAPERS[1], version=2), entry_xml(*REAL_PAPERS[2])])
    session = FakeSession([http(text=a), http(text=b)])
    papers = fp.collect_candidates(session, sleep=lambda s: None)
    assert sorted(p.arxiv_id for p in papers) == ["2609.34004", "2610.20037", "2610.20366"]
    assert "quant-paper-daily" in session.headers["User-Agent"]
    assert all(r["sortBy"] == "submittedDate" for r in session.requests)
    assert all("submittedDate:[" in r["search_query"] for r in session.requests)


def test_build_queries_cover_lookback_window():
    for q in fp.build_queries(NOW):
        assert "submittedDate:[202510151200 TO 202610161200]" in q


def test_fetch_query_pages_until_total(monkeypatch):
    monkeypatch.setattr(fp, "FETCH_PAGE_SIZE", 2)
    pages = [REAL_PAPERS[0:2], REAL_PAPERS[2:3], REAL_PAPERS[3:5]]   # 第二頁 arXiv 只回了一篇
    session = FakeSession([http(text=feed_xml([entry_xml(*p) for p in page], total=5)) for page in pages])
    papers = fp.fetch_query(session, "q", sleep=lambda s: None)
    assert [p.arxiv_id for p in papers] == [p[0] for p in REAL_PAPERS[:5]]
    assert [r["start"] for r in session.requests] == [0, 2, 3], "依實際收到的篇數推進，不跳過論文"


def test_fetch_query_keeps_newer_pages_when_later_page_fails(monkeypatch):
    monkeypatch.setattr(fp, "FETCH_PAGE_SIZE", 2)
    first = http(text=feed_xml([entry_xml(*p) for p in REAL_PAPERS[:2]], total=5))
    session = FakeSession([first] + [http(503)] * 4)
    papers = fp.fetch_query(session, "q", sleep=lambda s: None)
    assert [p.arxiv_id for p in papers] == [p[0] for p in REAL_PAPERS[:2]]


def test_collect_raises_when_arxiv_is_down():
    session = FakeSession([http(503)] * 8)
    with pytest.raises(fp.FetchError):
        fp.collect_candidates(session, sleep=lambda s: None)


def test_cross_query_has_no_bare_alpha():
    _, cross = fp.build_queries()
    assert 'abs:"alpha"' not in cross
