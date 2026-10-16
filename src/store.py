"""每日資料的讀寫。

data/papers/YYYY-MM-DD.json 是網站內容的唯一來源：哪些論文收錄過，直接從這些檔案推算，
不另外維護清單。data/evaluations/YYYY-MM-DD.json 記錄當天語意評估的分數，
讓評過的論文不必重評，沒選上的好論文隔天可以繼續競爭。"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path

from config import EVALUATIONS_DIR, PAPERS_DIR

log = logging.getLogger(__name__)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def issue_path(date: str, root: Path = PAPERS_DIR) -> Path:
    return root / f"{date}.json"


def load_issue(date: str, root: Path = PAPERS_DIR) -> dict | None:
    path = issue_path(date, root)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_dated(root: Path) -> list[tuple[str, dict]]:
    """root 底下以日期命名的 JSON，新到舊。壞掉的檔案會略過並記錄。"""
    out = []
    for path in sorted(root.glob("*.json"), reverse=True):
        if not _DATE_RE.match(path.stem):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("略過無法讀取的 %s：%s", path.name, exc)
            continue
        if isinstance(data, dict):
            out.append((path.stem, data))
    return out


def list_issues(root: Path = PAPERS_DIR) -> list[dict]:
    """所有每日資料，新到舊。"""
    issues = []
    for date, issue in _read_dated(root):
        issue.setdefault("date", date)
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


def load_evaluations(rubric: int, root: Path = EVALUATIONS_DIR,
                     exclude_date: str | None = None) -> dict[str, dict]:
    """arxiv_id → 最近一次的評分。評分標準版本不同的紀錄視同沒評過。"""
    scores: dict[str, dict] = {}
    for date, day in _read_dated(root):
        if date == exclude_date or day.get("rubric") != rubric:
            continue
        rows = day.get("scores")
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict) and row.get("arxiv_id"):
                scores.setdefault(row["arxiv_id"], row)
    return scores


def save_evaluations(date: str, rubric: int, model: str, rows: list[dict],
                     root: Path = EVALUATIONS_DIR) -> Path:
    """寫入當天的評分。同一天跑第二次時併入先前的紀錄，同一篇以新的為準。"""
    path = root / f"{date}.json"
    try:
        earlier = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        earlier = None
    if isinstance(earlier, dict) and earlier.get("rubric") == rubric and isinstance(earlier.get("scores"), list):
        new_ids = {r["arxiv_id"] for r in rows}
        rows = [r for r in earlier["scores"] if isinstance(r, dict) and r.get("arxiv_id") not in new_ids] + rows
    return _write_json(path, {"date": date, "rubric": rubric, "model": model, "scores": rows})


def write_issue(issue: dict, root: Path = PAPERS_DIR) -> Path:
    return _write_json(issue_path(issue["date"], root), issue)


def _write_json(path: Path, data: dict) -> Path:
    """先寫暫存檔再改名，程式中途被中斷也不會留下寫一半的檔案。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path
