"""從 arXiv API 抓候選論文並解析成 Paper 物件。只負責抓，不負責排序。"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

import requests

from config import (
    ARXIV_API, ARXIV_DELAY_SECONDS, CROSS_CATEGORIES, CROSS_TERMS, FETCH_MAX,
    FETCH_PAGE_SIZE, LOOKBACK_DAYS, QFIN_CATEGORIES, USER_AGENT,
)

log = logging.getLogger(__name__)

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"

# http://arxiv.org/abs/2510.01234v2 或 http://arxiv.org/abs/q-fin/0701001v1
_ID_RE = re.compile(r"arxiv\.org/abs/(?P<id>.+?)(?:v\d+)?$")


class FetchError(RuntimeError):
    """arXiv 完全無法取得資料。"""


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    primary_category: str
    categories: list[str]
    published: str        # 第一版投稿時間，ISO 8601
    updated: str
    abs_url: str
    pdf_url: str
    comment: str = ""                     # 作者附註，常寫頁數、接受的會議、程式碼連結
    journal_ref: str = ""                 # 已刊登的期刊
    keyword_score: int = 0
    matched_keywords: list[str] = field(default_factory=list)
    relevance: int | None = None          # 語意評估：交易相關度 0–10
    value: int | None = None              # 語意評估：內容價值 0–10
    topic: str = ""                       # 語意評估：主題（rank.TOPICS 之一）
    evidence: str = ""                    # 語意評估：結論的依據（rank.EVIDENCE 之一）
    relevance_reason: str = ""            # 兩個分數的評分理由

    @property
    def published_dt(self) -> datetime | None:
        try:
            return datetime.fromisoformat(self.published.replace("Z", "+00:00"))
        except ValueError:
            return None

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def parse_arxiv_id(raw: str) -> str:
    m = _ID_RE.search(raw.strip())
    return m.group("id") if m else ""


def parse_feed(xml_text: str) -> tuple[list[Paper], int]:
    """回傳（論文清單, arXiv 回報的總筆數）。"""
    root = ET.fromstring(xml_text)
    total = int(root.findtext(f"{OPENSEARCH}totalResults", "0") or 0)
    papers: list[Paper] = []

    for entry in root.findall(f"{ATOM}entry"):
        arxiv_id = parse_arxiv_id(entry.findtext(f"{ATOM}id", ""))
        title = _clean(entry.findtext(f"{ATOM}title"))
        # arXiv 出錯時會回傳一筆 title 為 "Error" 的 entry
        if not arxiv_id or title == "Error":
            continue

        primary = entry.find(f"{ARXIV_NS}primary_category")
        categories = [c.get("term") for c in entry.findall(f"{ATOM}category") if c.get("term")]
        pdf_url = next(
            (l.get("href") for l in entry.findall(f"{ATOM}link") if l.get("title") == "pdf" and l.get("href")),
            f"https://arxiv.org/pdf/{arxiv_id}",
        ).replace("http://", "https://")

        papers.append(Paper(
            arxiv_id=arxiv_id,
            title=title,
            authors=[_clean(a.findtext(f"{ATOM}name")) for a in entry.findall(f"{ATOM}author")],
            abstract=_clean(entry.findtext(f"{ATOM}summary")),
            primary_category=primary.get("term", "") if primary is not None else (categories[0] if categories else ""),
            categories=categories,
            published=entry.findtext(f"{ATOM}published", ""),
            updated=entry.findtext(f"{ATOM}updated", ""),
            abs_url=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_url=pdf_url,
            comment=_clean(entry.findtext(f"{ARXIV_NS}comment")),
            journal_ref=_clean(entry.findtext(f"{ARXIV_NS}journal_ref")),
        ))
    return papers, total


def build_queries(now: datetime | None = None) -> list[str]:
    """兩組查詢，都只取 LOOKBACK_DAYS 天內首次投稿的論文（arXiv 以 UTC 計時）。"""
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(days=LOOKBACK_DAYS)
    window = f"submittedDate:[{since:%Y%m%d%H%M} TO {now + timedelta(days=1):%Y%m%d%H%M}]"
    qfin = "({}) AND {}".format(" OR ".join(f"cat:{c}" for c in QFIN_CATEGORIES), window)
    cross = "({}) AND ({}) AND {}".format(
        " OR ".join(f"cat:{c}" for c in CROSS_CATEGORIES),
        " OR ".join(f'abs:"{t}"' for t in CROSS_TERMS),
        window,
    )
    return [qfin, cross]


def query_arxiv(session: requests.Session, search_query: str, max_results: int, start: int = 0,
                attempts: int = 4, sleep=time.sleep) -> tuple[list[Paper], int] | None:
    """回傳（這一頁的論文, arXiv 回報的總筆數）；失敗時回傳 None。"""
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    for attempt in range(1, attempts + 1):
        wait = ARXIV_DELAY_SECONDS * 2 ** attempt
        try:
            resp = session.get(ARXIV_API, params=params, timeout=120)
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After", "")
                wait = float(retry_after) if retry_after.isdigit() else wait
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            papers, total = parse_feed(resp.text)
            # arXiv 偶爾會回傳 200 但沒有任何 entry；一整年的分類查詢不可能真的是空的
            if not papers:
                raise ValueError(f"回應沒有任何論文（totalResults={total}）")
            return papers, total
        except (requests.RequestException, ET.ParseError, ValueError) as exc:
            if attempt == attempts:
                log.error("arXiv 查詢失敗，已放棄：%s", exc)
                return None
            log.warning("arXiv 查詢失敗（第 %d 次）：%s；%.0f 秒後重試", attempt, exc, wait)
            sleep(wait)
    return None


def fetch_query(session: requests.Session, search_query: str, sleep=time.sleep) -> list[Paper] | None:
    """分頁抓完一組查詢。第一頁就失敗回傳 None；之後的頁失敗則保留已抓到的較新論文。"""
    papers: list[Paper] = []
    while True:
        result = query_arxiv(session, search_query, FETCH_PAGE_SIZE, start=len(papers), sleep=sleep)
        if result is None:
            if papers:
                log.warning("只抓到前 %d 篇，較舊的論文今天不列入候選", len(papers))
            return papers or None
        batch, total = result
        papers.extend(batch)
        # 用實際收到的篇數推進，arXiv 偶爾回傳不足一頁時也不會跳過論文
        if len(papers) >= min(total, FETCH_MAX):
            return papers
        sleep(ARXIV_DELAY_SECONDS)


def collect_candidates(session: requests.Session | None = None, sleep=time.sleep,
                       now: datetime | None = None) -> list[Paper]:
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", USER_AGENT)

    queries = build_queries(now)
    by_id: dict[str, Paper] = {}
    failures = 0
    for i, query in enumerate(queries):
        if i:
            sleep(ARXIV_DELAY_SECONDS)
        batch = fetch_query(session, query, sleep=sleep)
        if batch is None:
            failures += 1
            continue
        log.info("查詢 %d 取得 %d 篇", i + 1, len(batch))
        for p in batch:
            by_id.setdefault(p.arxiv_id, p)

    if failures == len(queries):
        raise FetchError("arXiv 所有查詢都失敗，今天不更新")
    return list(by_id.values())
