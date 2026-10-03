"""Учебные данные в формате JSON Lines"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path

from .config import DATA

CATEGORIES = ["платежи", "возвраты", "доступ", "тарифы", "интеграция", "другое"]


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tickets(
    split: str | None = "dev", limit: int | None = None, seed: int = 0
) -> list[dict]:
    """Обращения из набора

    С limit берётся случайная, но всегда одна и та же часть: файл
    упорядочен по категориям, и первые строки дали бы перекос
    """
    rows = [
        r for r in read_jsonl(DATA / "tickets.jsonl") if split in (None, r["split"])
    ]
    if limit is not None:
        if limit < 0:
            raise ValueError("limit должно быть неотрицательным")
        if limit < len(rows):
            rows = random.Random(seed).sample(rows, limit)
    return rows


def questions() -> list[dict]:
    return read_jsonl(DATA / "questions.jsonl")
