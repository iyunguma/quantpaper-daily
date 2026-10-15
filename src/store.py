"""每日資料的讀寫。data/papers/YYYY-MM-DD.json 是唯一的資料來源：
哪些論文收錄過，直接從這些檔案推算，不另外維護清單。"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path

from config import PAPERS_DIR

log = logging.getLogger(__name__)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def issue_path(date: str, root: Path = PAPERS_DIR) -> Path:
    return root / f"{date}.json"


def load_issue(date: str, root: Path = PAPERS_DIR) -> dict | None:
    path = issue_path(date, root)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_issues(root: Path = PAPERS_DIR) -> list[dict]:
    """所有每日資料，新到舊。壞掉的檔案會略過並記錄。"""
    issues = []
    for path in sorted(root.glob("*.json"), reverse=True):
        if not _DATE_RE.match(path.stem):
            continue
        try:
            issue = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("略過無法讀取的 %s：%s", path.name, exc)
            continue
        issue.setdefault("date", path.stem)
        issue.setdefault("papers", [])
        issues.append(issue)
    return issues


def seen_ids(root: Path = PAPERS_DIR, exclude_date: str | None = None) -> set[str]:
    return {
        p["arxiv_id"]
        for issue in list_issues(root)
        if issue["date"] != exclude_date
        for p in issue["papers"]
    }


def write_issue(issue: dict, root: Path = PAPERS_DIR) -> Path:
    """先寫暫存檔再改名，程式中途被中斷也不會留下寫一半的檔案。"""
    root.mkdir(parents=True, exist_ok=True)
    path = issue_path(issue["date"], root)
    fd, tmp = tempfile.mkstemp(dir=root, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(issue, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path
