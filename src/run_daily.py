"""每日主流程：抓論文 → 排序 → 摘要 → 存檔 → 產生網站。

用法：
    python src/run_daily.py               # 產生今天這一期（今天已產生過就只重建網站）
    python src/run_daily.py --dry-run     # 只抓取與排序並列出結果，不摘要、不寫任何檔案
    python src/run_daily.py --force       # 重做今天這一期（今天原本的論文也可以再被選中）
    python src/run_daily.py --repair      # 重新摘要所有摘要失敗的論文
    python src/run_daily.py --build-only  # 只重建網站（調整樣式時用）

結束代碼 0 代表成功；1 代表今天沒有更新（arXiv 連不上、缺 API key、摘要失敗太多），
GitHub Actions 會把這次執行標成失敗並寄信通知。
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import build_site  # noqa: E402
import store  # noqa: E402
from config import (  # noqa: E402
    DAILY_COUNT, MAX_FAILURE_RATIO, PAPERS_DIR, RERANK_ENABLED,
    SUMMARY_MODEL, SUMMARY_SOURCE,
)

log = logging.getLogger("daily")
TPE = timezone(timedelta(hours=8))


def today_tpe() -> str:
    return datetime.now(TPE).strftime("%Y-%m-%d")


def run_today(args, client, papers_dir: Path = PAPERS_DIR) -> int:
    from fetch_papers import FetchError, collect_candidates
    from rank import select
    from summarize import is_failed, summarize_all

    today = args.date or today_tpe()
    if store.load_issue(today, papers_dir) and not (args.force or args.dry_run):
        log.info("%s 這一期已經存在，不重新抓取（要重做請加 --force）", today)
        return 0

    seen = store.seen_ids(papers_dir, exclude_date=today if args.force else None)
    try:
        candidates = collect_candidates()
    except FetchError as exc:
        log.error("%s", exc)
        return 1

    selection = select(
        candidates, seen,
        client=client if RERANK_ENABLED else None,
        count=args.count,
    )

    if args.dry_run:
        for i, p in enumerate(selection.papers, 1):
            rel = f"相關度 {p.relevance:>2}" if p.relevance is not None else ""
            log.info("%2d. [關鍵字 %3d] %s %s  %s", i, p.keyword_score, rel, p.arxiv_id, p.title[:70])
            if p.relevance_reason:
                log.info("      %s", p.relevance_reason)
        log.info("dry-run 結束，沒有產生摘要，也沒有寫入任何檔案")
        return 0

    if not selection.papers:
        log.warning("今天沒有夠相關的新論文，不產生新的一期；網站維持顯示上一期")
        return 0

    records = summarize_all(client, [p.to_dict() for p in selection.papers])
    failed = sum(1 for r in records if is_failed(r["summary"]))
    if failed / len(records) > MAX_FAILURE_RATIO:
        log.error("摘要失敗 %d / %d 篇，超過門檻，今天不發佈（請檢查 API key、額度或模型名稱 %s）",
                  failed, len(records), SUMMARY_MODEL)
        return 1

    path = store.write_issue({
        "date": today,
        "generated_at": datetime.now(TPE).isoformat(timespec="seconds"),
        "scanned": selection.scanned,
        "eligible": selection.eligible,
        "ranking": selection.method,
        "summary_model": SUMMARY_MODEL,
        "summary_source": SUMMARY_SOURCE,
        "papers": records,
    }, papers_dir)
    log.info("已寫入 %s", path.name)
    return 0


def run_repair(client, papers_dir: Path = PAPERS_DIR) -> int:
    from summarize import is_failed, summarize_all

    fixed = remaining = 0
    for issue in store.list_issues(papers_dir):
        broken = [i for i, p in enumerate(issue["papers"]) if is_failed(p.get("summary"))]
        if not broken:
            continue
        log.info("%s：重新摘要 %d 篇", issue["date"], len(broken))
        redone = summarize_all(client, [
            {k: v for k, v in issue["papers"][i].items() if k != "summary"} for i in broken
        ])
        for i, rec in zip(broken, redone):
            issue["papers"][i] = rec
            if is_failed(rec["summary"]):
                remaining += 1
            else:
                fixed += 1
        store.write_issue(issue, papers_dir)
    log.info("修補完成：成功 %d 篇，仍失敗 %d 篇", fixed, remaining)
    # 有修好任何一篇就算成功，讓 GitHub Actions 把修好的部分存回去
    return 1 if remaining and not fixed else 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    ap = argparse.ArgumentParser(description="量化研究日報")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="只抓取與排序，不摘要、不寫檔")
    mode.add_argument("--force", action="store_true", help="重做今天這一期")
    mode.add_argument("--repair", action="store_true", help="重新摘要所有失敗的論文")
    mode.add_argument("--build-only", action="store_true", help="只重建網站")
    ap.add_argument("--count", type=int, default=DAILY_COUNT, help="最多選幾篇")
    ap.add_argument("--date", help=argparse.SUPPRESS)  # 測試用：指定期別日期
    args = ap.parse_args(argv)

    if args.build_only:
        build_site.build()
        return 0

    from llm import API_KEY_ENV, make_client
    client = make_client()
    if client is None and not args.dry_run:
        log.error("找不到環境變數 %s，無法產生摘要", API_KEY_ENV)
        return 1
    if client is None:
        log.info("沒有 API key，dry-run 只用關鍵字排序")

    code = run_repair(client) if args.repair else run_today(args, client)
    if not args.dry_run:
        build_site.build()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
