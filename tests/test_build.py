import xml.etree.ElementTree as ET

import pytest

import build_site
import store


def paper(i, **summary):
    s = {"ok": True, "source": "abstract", "title_zh": f"中文標題{i}", "verdict": f"結論{i}",
         "problem": "p", "method": "m", "findings": "f", "implication": "i", "caveats": "c",
         "tags": ["標籤"], "practicality": 4, **summary}
    return {"arxiv_id": f"2610.0000{i}", "title": f"Title {i}", "authors": ["A", "B", "C", "D"],
            "primary_category": "q-fin.TR", "categories": ["q-fin.TR"], "abstract": "",
            "published": "2026-10-14T10:00:00Z", "abs_url": f"https://arxiv.org/abs/2610.0000{i}",
            "pdf_url": f"https://arxiv.org/pdf/2610.0000{i}", "summary": s}


@pytest.fixture
def dirs(tmp_path):
    papers, site = tmp_path / "papers", tmp_path / "site"
    papers.mkdir()
    return papers, site


def test_builds_with_no_data(dirs):
    papers, site = dirs
    build_site.build(papers, site)
    assert "第一期還沒產生" in (site / "index.html").read_text(encoding="utf-8")
    assert "還沒有任何一期" in (site / "archive" / "index.html").read_text(encoding="utf-8")
    ET.fromstring((site / "feed.xml").read_text(encoding="utf-8"))


def test_full_build(dirs, monkeypatch):
    papers, site = dirs
    monkeypatch.setattr(build_site, "SITE_BASE_URL", "https://me.github.io/qpd")
    store.write_issue({"date": "2026-10-13", "eligible": 30, "papers": [paper(1)]}, papers)
    store.write_issue({"date": "2026-10-14", "eligible": 40, "papers": [paper(2), paper(3, ok=False, verdict="失敗")]}, papers)
    store.write_issue({"date": "2026-10-15", "eligible": 50,
                       "papers": [paper(4, title_zh="<script>alert(1)</script>"), paper(5, source="pdf")]}, papers)
    build_site.build(papers, site)

    index = (site / "index.html").read_text(encoding="utf-8")
    assert "2026年10月15日" in index and "星期四" in index
    assert "<script>alert(1)</script>" not in index and "&lt;script&gt;" in index
    assert "依全文" in index
    assert "寧缺勿濫" in index, "少於每日篇數時要說明"
    assert 'href="archive/2026-10-14.html">前一期' in index
    assert "後一期" not in index

    middle = (site / "archive" / "2026-10-14.html").read_text(encoding="utf-8")
    assert "前一期" in middle and "後一期" in middle
    assert 'aria-current="page">歸檔' in middle
    assert middle.count('class="detail"') == 1, "失敗的論文不顯示空白欄位"

    archive = (site / "archive" / "index.html").read_text(encoding="utf-8")
    assert "共 <b>3</b> 期、<b>5</b> 篇" in archive and "2026年10月13日 開始" in archive

    feed = ET.fromstring((site / "feed.xml").read_text(encoding="utf-8"))
    items = feed.findall("./channel/item")
    assert len(items) == 5
    assert items[0].findtext("link") == "https://me.github.io/qpd/archive/2026-10-15.html#p1"


def test_rebuild_removes_stale_pages(dirs):
    papers, site = dirs
    store.write_issue({"date": "2026-10-15", "papers": [paper(1)]}, papers)
    build_site.build(papers, site)
    (papers / "2026-10-15.json").unlink()
    build_site.build(papers, site)
    assert not (site / "archive" / "2026-10-15.html").exists()


def test_legacy_v1_issue_still_renders(dirs):
    papers, site = dirs
    legacy = paper(1)
    for key in ("ok", "source"):
        legacy["summary"].pop(key)
    store.write_issue({"date": "2026-09-03", "scanned": 143, "papers": [legacy]}, papers)
    build_site.build(papers, site)
    assert "中文標題1" in (site / "index.html").read_text(encoding="utf-8")
