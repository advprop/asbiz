"""Замер кандидатов

    python -m desk.bench --n 5     # пробный запуск
    python -m desk.bench --n 30    # сдаточный замер
    python -m desk.bench --n 30 --only "короткая постановка"

Кандидат это постановка задачи и добавка к запросу. Все кандидаты
проходят одни и те же обращения. Таблица печатается и сохраняется
в runs/s1_bench.md
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import time
from dataclasses import dataclass, field
from math import ceil

from .batch import batches
from .config import RUNS
from .data import CATEGORIES, tickets
from .llm import LLM, Usage

SHORT = (
    "Отнеси обращение в поддержку платёжного сервиса к одной категории: %s. "
    "Ответь одним словом." % ", ".join(CATEGORIES)
)

DETAILED = (
    SHORT
    + """
Правила:
- платежи: платёж не проходит, двойное списание, деньги не дошли, переводы, лимиты;
- возвраты: просьба вернуть деньги за покупку, статус возврата, спор через банк;
- доступ: вход, пароль, смена номера, блокировка, второй фактор;
- тарифы: комиссии, абонентская плата, смена тарифа, вывод выручки;
- интеграция: API, ключи, уведомления о платежах;
- другое: всё остальное; при сомнении выбирай «другое»."""
)

THINKING = {"thinking": {"type": "enabled", "budget_tokens": 1024}}

FLOW_PER_DAY = 50_000


@dataclass
class Candidate:
    name: str
    system: str
    body: dict = field(default_factory=dict)  # добавка к запросу
    max_tokens: int = 16


CANDIDATES = [
    Candidate("короткая постановка", SHORT),
    Candidate("постановка с правилами", DETAILED),
    Candidate("правила + рассуждение", DETAILED, body=THINKING, max_tokens=2048),
    Candidate(
        "правила + два примера",
        DETAILED
        + """

Примеры:
Обращение: Деньги списали дважды за один заказ, P-61842. Ответ: платежи.
Обращение: Когда придёт возврат за отменённый заказ P-61843? Ответ: возвраты.

Теперь ответь одним словом на новое обращение.""",
    ),
]


@dataclass
class Row:
    """Итог одного обращения"""

    id: str
    ok: bool  # категория совпала с эталоном
    failed: bool = False  # вызов не удался
    truncated: bool = False  # ответ оборван
    latency_s: float = 0.0
    ttft_s: float | None = None
    usage: Usage = field(default_factory=Usage)


def prompt(cand: Candidate, row: dict) -> list[dict[str, str]]:
    """Постановка кандидата и текст обращения"""
    return [
        {"role": "system", "content": cand.system},
        {"role": "user", "content": row["text"]},
    ]


def parse_category(text: str) -> str | None:
    """Первая категория, которая встретилась в ответе"""
    low = text.lower()
    found = [(low.find(c), c) for c in CATEGORIES if c in low]
    return min(found)[1] if found else None


def percentile(values: list[float], q: float) -> float:
    """Процентиль q списка значений"""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, ceil(q * len(ordered)) - 1))
    return ordered[idx]


async def run_candidate(
    llm, cand: Candidate, rows: list[dict], concurrency: int = 4
) -> list[Row]:
    """Прогоняет обращения через кандидата, не больше concurrency сразу

    Если вызов не удался, обращение засчитывается как ошибка,
    и замер идёт дальше
    """
    client = llm.variant(extra_body={**llm.cfg.extra_body, **cand.body})

    async def one(row: dict) -> Row:
        try:
            res = await client.astream(prompt(cand, row), max_tokens=cand.max_tokens)
        except Exception:
            return Row(row["id"], ok=False, failed=True)
        return Row(
            row["id"],
            parse_category(res.text) == row["gold"]["category"],
            truncated=res.truncated,
            latency_s=res.total_s,
            ttft_s=res.ttft_s,
            usage=res.usage,
        )

    try:
        results = []
        for batch in batches(rows, concurrency):
            results.extend(await asyncio.gather(*(one(row) for row in batch)))
        return results
    finally:
        if isinstance(client, LLM):
            await client.aclose()


def summarize(name: str, rows: list[Row], flow_per_day: int = FLOW_PER_DAY) -> dict:
    """Строка таблицы для кандидата

    Сбой считается ошибкой. Задержки и токены считаются по удачным вызовам
    """
    done = [r for r in rows if not r.failed]
    total = Usage()
    for r in done:
        total = total + r.usage
    n = max(1, len(done))

    def per(value: float) -> float:
        return value / n

    latencies = [r.latency_s for r in done]
    ttfts = [r.ttft_s for r in done if r.ttft_s is not None]
    accuracy = sum(r.ok for r in rows) / max(1, len(rows))
    monthly = per(total.cost) * flow_per_day * 30
    return {
        "кандидат": name,
        "точность": accuracy,
        "p50, с": percentile(latencies, 0.5),
        "p95, с": percentile(latencies, 0.95),
        "первый токен p50, с": percentile(ttfts, 0.5),
        "токенов на обращение": per(total.total_tokens),
        "входных на обращение": per(
            total.input_tokens + total.cache_read_tokens + total.cache_write_tokens
        ),
        "взвешенных на обращение": per(total.weighted),
        "цена за 1000, у.е.": 1000 * per(total.cost),
        "взвешенных в месяц, млн": per(total.weighted) * flow_per_day * 30 / 1e6,
        "цена в месяц, у.е.": monthly,
        "цена с ошибками в месяц, у.е.": monthly
        + (1 - accuracy) * flow_per_day * 30 * 20,
        "обрезано": sum(r.truncated for r in done),
        "сбоев": len(rows) - len(done),
    }


def table(rows: list[dict]) -> str:
    """Таблица в формате Markdown"""
    heads = list(rows[0])

    def fmt(v: object) -> str:
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    lines += ["| " + " | ".join(fmt(r[h]) for h in heads) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=30, help="сколько обращений из dev взять")
    ap.add_argument("--only", default=None, help="имя одного кандидата")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument(
        "--flow", type=int, default=FLOW_PER_DAY, help="обращений в день для прогноза"
    )
    args = ap.parse_args()
    rows = tickets("dev", args.n)
    chosen = [c for c in CANDIDATES if args.only in (None, c.name)]
    if not chosen:
        ap.error("кандидат не найден: %s" % args.only)
    llm = LLM(cache=False)  # задержку мерим без кэша
    started = time.time()

    async def run_all() -> list[list[Row]]:
        return [
            await run_candidate(llm, cand, rows, args.concurrency) for cand in chosen
        ]

    results, spent = [], 0.0
    try:
        runs = asyncio.run(run_all())
    finally:
        llm.close()
    for cand, done in zip(chosen, runs, strict=True):
        spent += sum(r.usage.weighted for r in done)
        results.append(summarize(cand.name, done, args.flow))
    report = table(results)
    print(report)
    footer = (
        "\nМодель %s, обращений %d, прогноз на %d обращений в день. "
        "Время замера %.0f с, потрачено взвешенных токенов: %.0f."
        % (llm.cfg.model, len(rows), args.flow, time.time() - started, spent)
    )
    print(footer)
    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    (RUNS / "s1_bench.md").write_text(
        "# Замер кандидатов, %s\n\n%s\n%s\n" % (stamp, report, footer), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
