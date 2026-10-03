"""Раунд 1. Запрос к модели без обёртки

    python 1_raw_request.py

Скрипт отправляет несколько запросов прямо через httpx и печатает то,
что обычно прячет SDK: блоки ответа, причину остановки и счёт токенов.
Наблюдения перенесите в отчёт_1.md
"""

from __future__ import annotations

import json
import time

import httpx

from desk.config import Settings, settings
from desk.llm import MAX_RESPONSE_BYTES, READ_CHUNK_BYTES, RateLimit

SYSTEM = (
    "Отнеси обращение в поддержку платёжного сервиса к одной категории: платежи, "
    "возвраты, доступ, тарифы, интеграция, другое. Ответь одним словом."
)
TICKET = "с меня два раза сняли 3490 за один заказ!!! разберитесь"

PAIRS = [
    (
        "Возврат на карту занимает от трёх до десяти рабочих дней "
        "и зависит от банка клиента.",
        "A refund to the card takes three to ten business days "
        "and depends on the client's bank.",
    ),
    (
        "Суточный лимит обнуляется в полночь по московскому времени.",
        "The daily limit resets at midnight Moscow time.",
    ),
]


def headers(cfg: Settings) -> dict[str, str]:
    """Заголовки запроса: версия формата, тип содержимого и ключ"""
    out = {"anthropic-version": "2023-06-01", "content-type": "application/json"}
    if cfg.auth_scheme == "bearer":
        out["authorization"] = "Bearer " + cfg.api_key
    else:
        out["x-api-key"] = cfg.api_key
    return out


def body(
    cfg: Settings,
    messages: list[dict[str, object]],
    *,
    system: str | None = None,
    max_tokens: int = 64,
    temperature: float = 0.0,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Тело запроса к модели"""
    out: dict[str, object] = {
        "model": cfg.model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system:
        out["system"] = system
    out.update(cfg.extra_body)
    out.update(extra or {})
    return out


def send(
    cfg: Settings, payload: dict[str, object], client: httpx.Client | None = None
) -> dict[str, object]:
    """Отправляет запрос и возвращает ответ шлюза

    Если код ответа не 200, бросает ошибку с текстом ответа. Тесты
    передают свой client, чтобы не ходить в сеть
    """
    own = client is None
    client = client or httpx.Client(timeout=cfg.timeout_s, verify=cfg.verify_ssl)
    try:
        started = time.monotonic()
        with client.stream(
            "POST", cfg.base_url + "/v1/messages", headers=headers(cfg), json=payload
        ) as resp:
            chunks = []
            size = 0
            for chunk in resp.iter_bytes(chunk_size=READ_CHUNK_BYTES):
                if time.monotonic() - started > 2 * cfg.timeout_s:
                    raise RuntimeError("общее время запроса истекло")
                size += len(chunk)
                if size > MAX_RESPONSE_BYTES:
                    raise RuntimeError("ответ шлюза превысил допустимый размер")
                chunks.append(chunk)
            content = b"".join(chunks)
            status = resp.status_code
    finally:
        if own:
            client.close()
    if status != 200:
        raise RuntimeError(
            "шлюз ответил %d: %s" % (status, content[:300].decode(errors="replace"))
        )
    return json.loads(content)


def text_of(resp: dict[str, object]) -> str:
    """Текст ответа из всех блоков text"""
    return "".join(
        b.get("text", "") for b in resp.get("content") or [] if b.get("type") == "text"
    ).strip()


def user(text: str) -> list[dict[str, object]]:
    """Сообщение пользователя"""
    return [{"role": "user", "content": text}]


def main() -> None:
    cfg = settings()
    if not cfg.base_url or not cfg.api_key:
        raise SystemExit("впишите LLM_BASE_URL и LLM_API_KEY в .env")
    limit = RateLimit(cfg.rpm)

    def call(payload: dict[str, object]) -> dict[str, object]:
        """Отправка с паузой, если у ключа лимит запросов в минуту"""
        time.sleep(limit.reserve())
        return send(cfg, payload)

    print("1. Обычный запрос: весь ответ как есть\n")
    resp = call(body(cfg, user(TICKET), system=SYSTEM, max_tokens=16))
    print(json.dumps(resp, ensure_ascii=False, indent=2))

    print("\n2. Ответ, оборванный лимитом max_tokens=2\n")
    resp = call(body(cfg, user("Объясни, почему небо голубое."), max_tokens=2))
    print("stop_reason: %s, текст: %r" % (resp.get("stop_reason"), text_of(resp)))

    print("\n3. Токены русского и английского текста с одним смыслом\n")
    for ru, en in PAIRS:
        n_ru = call(body(cfg, user(ru), max_tokens=1))["usage"]["input_tokens"]
        n_en = call(body(cfg, user(en), max_tokens=1))["usage"]["input_tokens"]
        print(
            "по-русски %3d, по-английски %3d, отношение %.2f"
            % (n_ru, n_en, n_ru / max(1, n_en))
        )

    print("\n4. Разброс ответов: пять одинаковых запросов при температуре 0 и 1\n")
    prompt = user("Продолжи фразу одним словом, без точки: «Сегодня погода»")
    for temperature in (0.0, 1.0):
        answers = [
            text_of(call(body(cfg, prompt, max_tokens=8, temperature=temperature)))
            for _ in range(5)
        ]
        print(
            "температура %.1f: различных ответов %d из 5: %s"
            % (temperature, len(set(answers)), answers)
        )

    print("\n5. Сервер не хранит диалог\n")
    first = user("Меня зовут Аня. Запомни это.")
    resp1 = call(body(cfg, first, max_tokens=40))
    alone = call(body(cfg, user("Как меня зовут? Ответь одним словом."), max_tokens=16))
    history = first + [
        {"role": "assistant", "content": text_of(resp1)},
        {"role": "user", "content": "Как меня зовут? Ответь одним словом."},
    ]
    with_history = call(body(cfg, history, max_tokens=16))
    print(
        "без истории: %r, входных токенов %d"
        % (text_of(alone), alone["usage"]["input_tokens"])
    )
    print(
        "с историей:  %r, входных токенов %d"
        % (text_of(with_history), with_history["usage"]["input_tokens"])
    )

    print("\n6. Что будет, если модель начнёт рассуждать, а места мало\n")
    thinking = {"thinking": {"type": "enabled", "budget_tokens": 1024}}
    resp = call(body(cfg, user(TICKET), system=SYSTEM, max_tokens=16, extra=thinking))
    kinds = [b["type"] for b in resp.get("content") or []]
    print(
        "блоки ответа: %s, причина остановки: %s, текст: %r, выходных токенов %d"
        % (
            kinds,
            resp.get("stop_reason"),
            text_of(resp),
            resp["usage"]["output_tokens"],
        )
    )
    print(
        "Поэтому в проекте рассуждение выключено добавкой из .env, а кандидат с "
        "рассуждением в замере получает max_tokens больше бюджета рассуждения."
    )


if __name__ == "__main__":
    main()
