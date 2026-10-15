"""從 arXiv API 抓候選論文並解析成 Paper 物件。只負責抓，不負責排序。"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime

import requests

from config import (
    ARXIV_API, ARXIV_DELAY_SECONDS, CROSS_CATEGORIES, CROSS_FETCH,
    CROSS_TERMS, QFIN_CATEGORIES, QFIN_FETCH, USER_AGENT,
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
    keyword_score: int = 0
    matched_keywords: list[str] = field(default_factory=list)
    relevance: int | None = None          # 語意重排分數 0–10
    relevance_reason: str = ""

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
        ))
    return papers, total


def build_queries() -> list[tuple[str, int]]:
    qfin = " OR ".join(f"cat:{c}" for c in QFIN_CATEGORIES)
    cross = "({}) AND ({})".format(
        " OR ".join(f"cat:{c}" for c in CROSS_CATEGORIES),
        " OR ".join(f'abs:"{t}"' for t in CROSS_TERMS),
    )
    return [(qfin, QFIN_FETCH), (cross, CROSS_FETCH)]


def query_arxiv(session: requests.Session, search_query: str, max_results: int,
                attempts: int = 4, sleep=time.sleep) -> list[Paper] | None:
    """失敗時回傳 None（和「查無結果」的空清單區分開來）。"""
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    for attempt in range(1, attempts + 1):
        wait = ARXIV_DELAY_SECONDS * 2 ** attempt
        try:
            resp = session.get(ARXIV_API, params=params, timeout=60)
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After", "")
                wait = float(retry_after) if retry_after.isdigit() else wait
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            papers, total = parse_feed(resp.text)
            # arXiv 偶爾會回傳 200 但沒有任何 entry；依時間排序的分類查詢不可能真的是空的
            if not papers:
                raise ValueError(f"回應沒有任何論文（totalResults={total}）")
            return papers
        except (requests.RequestException, ET.ParseError, ValueError) as exc:
            if attempt == attempts:
                log.error("arXiv 查詢失敗，已放棄：%s", exc)
                return None
            log.warning("arXiv 查詢失敗（第 %d 次）：%s；%.0f 秒後重試", attempt, exc, wait)
            sleep(wait)
    return None


def collect_candidates(session: requests.Session | None = None, sleep=time.sleep) -> list[Paper]:
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", USER_AGENT)

    by_id: dict[str, Paper] = {}
    failures = 0
    for i, (query, n) in enumerate(build_queries()):
        if i:
            sleep(ARXIV_DELAY_SECONDS)
        batch = query_arxiv(session, query, n, sleep=sleep)
        if batch is None:
            failures += 1
            continue
        log.info("查詢 %d 取得 %d 篇", i + 1, len(batch))
        for p in batch:
            by_id.setdefault(p.arxiv_id, p)

    if failures == len(build_queries()):
        raise FetchError("arXiv 所有查詢都失敗，今天不更新")
    return list(by_id.values())
