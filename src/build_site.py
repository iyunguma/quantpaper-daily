"""把 data/papers/*.json 轉成靜態網站，輸出到 site/。"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from config import (
    CATEGORY_LABELS, DAILY_COUNT, FEED_DAYS, PAPERS_DIR, ROOT, SITE_BASE_URL, SITE_DIR,
    SITE_TAGLINE, SITE_TITLE,
)
from rank import TOPIC_SLUGS
from store import list_issues
from summarize import is_failed

log = logging.getLogger(__name__)

TPE = timezone(timedelta(hours=8))
WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

# 退回關鍵字排序的那幾天沒有主題；主題清單改名後，舊名稱也歸到這裡
UNSORTED, UNSORTED_SLUG = "未分類", "unsorted"


def zh_date(iso: str) -> str:
    d = datetime.strptime(iso, "%Y-%m-%d")
    return f"{d.year}年{d.month}月{d.day}日"


def zh_weekday(iso: str) -> str:
    return WEEKDAYS[datetime.strptime(iso, "%Y-%m-%d").weekday()]


def short_date(iso_ts: str) -> str:
    try:
        d = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        return f"{d.month}/{d.day} 投稿"
    except (ValueError, AttributeError):
        return ""


def window_zh(days: int) -> str:
    return {7: "最近一週", 365: "過去一年"}.get(days, f"最近 {days} 天")


def evaluation_note(p: dict) -> str:
    """評選理由與兩個分數；舊資料沒有內容價值分數時不顯示。"""
    if p.get("relevance") is None or p.get("value") is None:
        return ""
    reason = str(p.get("relevance_reason") or "").strip().rstrip("。.")
    scores = f"相關度 {p['relevance']}、內容價值 {p['value']}（滿分 10）"
    return f"{reason}。{scores}" if reason else scores


def author_line(authors: list[str]) -> str:
    if not authors:
        return "作者未列出"
    if len(authors) <= 3:
        return "、".join(authors)
    return f"{authors[0]} 等 {len(authors)} 人"


def decorate(issues: list[dict]) -> list[dict]:
    """補上模板需要的顯示用欄位，並串起前後期。"""
    for i, issue in enumerate(issues):
        issue["date_zh"] = zh_date(issue["date"])
        issue["weekday"] = zh_weekday(issue["date"])
        issue["newer"] = issues[i - 1]["date"] if i > 0 else None
        issue["older"] = issues[i + 1]["date"] if i + 1 < len(issues) else None
        issue.setdefault("scanned", 0)
        issue.setdefault("eligible", issue.get("scanned", 0))
        # 舊的期數沒記錄 lookback_days，當時都是往回看一週
        issue["window_zh"] = window_zh(issue.get("lookback_days", 7))
        for p in issue["papers"]:
            p.setdefault("summary", {})
            p["failed"] = is_failed(p["summary"])
            p["category_label"] = CATEGORY_LABELS.get(p.get("primary_category", ""), p.get("primary_category", ""))
            p["topic_slug"] = TOPIC_SLUGS.get(p.get("topic") or "")
            p["author_line"] = author_line(p.get("authors", []))
            p["published_short"] = short_date(p.get("published", ""))
            p["from_pdf"] = p["summary"].get("source") == "pdf"
            p["evaluation_note"] = evaluation_note(p)
    return issues


def group_by_topic(issues: list[dict]) -> list[dict]:
    """依 TOPICS 的順序分組，只列有論文的主題。主題內新到舊，同一期依名次。"""
    names = {slug: name for name, slug in TOPIC_SLUGS.items()} | {UNSORTED_SLUG: UNSORTED}
    entries: dict[str, list[dict]] = {slug: [] for slug in names}
    for issue in issues:
        for rank, p in enumerate(issue["papers"], 1):
            entries[p["topic_slug"] or UNSORTED_SLUG].append(
                {"paper": p, "date": issue["date"], "date_zh": issue["date_zh"], "rank": rank})
    return [{"name": names[slug], "slug": slug, "entries": e} for slug, e in entries.items() if e]


def build_feed(issues: list[dict]) -> str:
    base = SITE_BASE_URL or "https://arxiv.org"
    items = []
    for issue in issues[:FEED_DAYS]:
        pub = format_datetime(datetime.strptime(issue["date"], "%Y-%m-%d").replace(hour=7, tzinfo=TPE))
        for i, p in enumerate(issue["papers"], 1):
            s = p["summary"]
            link = f"{SITE_BASE_URL}/archive/{issue['date']}.html#p{i}" if SITE_BASE_URL else p["abs_url"]
            body = "".join(
                f"<p>{escape(s.get(k, ''))}</p>" for k in ("verdict", "findings", "implication") if s.get(k)
            ) + f'<p><a href="{escape(p["abs_url"])}">arXiv 原文</a></p>'
            items.append(
                "<item>"
                f"<title>{escape(s.get('title_zh') or p['title'])}</title>"
                f"<link>{escape(link)}</link>"
                f"<guid isPermaLink=\"false\">arxiv:{escape(p['arxiv_id'])}</guid>"
                f"<pubDate>{pub}</pubDate>"
                f"<description>{escape(body)}</description>"
                "</item>"
            )
    self_link = f'<atom:link href={quoteattr(SITE_BASE_URL + "/feed.xml")} rel="self" type="application/rss+xml"/>' if SITE_BASE_URL else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom"><channel>'
        f"<title>{escape(SITE_TITLE)}</title><link>{escape(base)}</link>"
        f"<description>{escape(SITE_TAGLINE)}</description><language>zh-Hant</language>"
        f"{self_link}{''.join(items)}</channel></rss>\n"
    )


def build(papers_dir: Path = PAPERS_DIR, site_dir: Path = SITE_DIR) -> None:
    issues = decorate(list_issues(papers_dir))

    env = Environment(
        loader=FileSystemLoader(ROOT / "templates"),
        autoescape=select_autoescape(["html", "xml"]),
        undefined=StrictUndefined,     # 模板用到不存在的欄位就直接報錯，不默默輸出空白
        trim_blocks=True,
        lstrip_blocks=True,
    )

    if site_dir.exists():
        shutil.rmtree(site_dir)        # 清掉舊檔，避免刪掉的資料殘留在網站上
    (site_dir / "archive").mkdir(parents=True)
    (site_dir / "topics").mkdir()
    shutil.copy(ROOT / "templates" / "style.css", site_dir / "style.css")

    common = {
        "site_title": SITE_TITLE,
        "site_tagline": SITE_TAGLINE,
        "site_base_url": SITE_BASE_URL,
        "total_papers": sum(len(i["papers"]) for i in issues),
        "issue_count": len(issues),
        "daily_count": DAILY_COUNT,
        "first_date_zh": issues[-1]["date_zh"] if issues else "",
        "generated_at": datetime.now(TPE).strftime("%Y-%m-%d %H:%M"),
    }
    day_tpl = env.get_template("day.html")

    latest = issues[0] if issues else None
    (site_dir / "index.html").write_text(
        day_tpl.render(issue=latest, page="today", root="", **common), encoding="utf-8")

    for issue in issues:
        (site_dir / "archive" / f"{issue['date']}.html").write_text(
            day_tpl.render(issue=issue, page="archive", root="../", **common), encoding="utf-8")

    rows = [{
        "date": i["date"], "date_zh": i["date_zh"], "weekday": i["weekday"],
        "count": len(i["papers"]),
        "peek": "；".join((p["summary"].get("title_zh") or p["title"]) for p in i["papers"][:2]),
    } for i in issues]
    (site_dir / "archive" / "index.html").write_text(
        env.get_template("archive.html").render(issues=rows, page="archive", root="../", **common),
        encoding="utf-8")

    topics = group_by_topic(issues)
    topic_tpl = env.get_template("topic.html")
    for t in topics:
        (site_dir / "topics" / f"{t['slug']}.html").write_text(
            topic_tpl.render(topic=t, topics=topics, page="topics", root="../", **common), encoding="utf-8")
    topic_rows = [{
        "name": t["name"], "slug": t["slug"], "count": len(t["entries"]),
        "peek": "；".join((e["paper"]["summary"].get("title_zh") or e["paper"]["title"]) for e in t["entries"][:2]),
    } for t in topics]
    (site_dir / "topics" / "index.html").write_text(
        env.get_template("topics.html").render(topics=topic_rows, page="topics", root="../", **common),
        encoding="utf-8")

    (site_dir / "feed.xml").write_text(build_feed(issues), encoding="utf-8")
    log.info("網站已產生：%d 期、%d 篇、%d 個主題", len(issues), common["total_papers"], len(topics))
