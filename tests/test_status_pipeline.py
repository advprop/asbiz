"""Ответы шлюза проверяются через обе пакетные функции."""

import asyncio
import json

import httpx
import pytest

from desk.bench import CANDIDATES, run_candidate
from desk.config import Settings
from desk.escalation import atriage_many
from desk.llm import LLM, LLMError
from desk.schemas import Ticket


def gateway(recover_500=True):
    calls = {}
    answer = {
        "reasoning": "Проверка статусов шлюза.",
        "category": "другое",
        "severity": 1,
        "quote": "успех",
        "payment_ids": [],
        "amount": None,
        "signals": {
            "refund": None,
            "duplicate": None,
            "fraud": False,
            "threat": False,
            "asks_human": False,
            "tariff_pro": False,
            "merchant_down": False,
            "key_leak": False,
        },
    }

    def handle(request):
        body = json.loads(request.content)
        text = body["messages"][-1]["content"]
        name = next(word for word in ("500", "400", "300", "успех") if word in text)
        calls[name] = calls.get(name, 0) + 1
        if name in ("400", "300") or (
            name == "500" and (not recover_500 or calls[name] == 1)
        ):
            return httpx.Response(int(name), text="ошибка шлюза")
        if body.get("stream"):
            events = [
                ("message_start", {"type": "message_start", "message": {"usage": {}}}),
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": "другое"},
                    },
                ),
                ("message_stop", {"type": "message_stop"}),
            ]
            content = "".join(
                f"event: {event}\ndata: {json.dumps(data)}\n\n"
                for event, data in events
            )
            return httpx.Response(200, text=content)
        draft = {**answer, "quote": "ошибка 500" if name == "500" else "успех"}
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": json.dumps(draft, ensure_ascii=False)}
                ],
                "usage": {},
            },
        )

    return httpx.MockTransport(handle), calls


@pytest.mark.parametrize("recover_500", [True, False])
def test_statuses_do_not_stop_escalation(recover_500):
    transport, calls = gateway(recover_500)
    llm = LLM(
        Settings("https://gw.example", "key"),
        async_transport=transport,
        cache=False,
        max_retries=1,
    )

    async def run():
        try:
            return await atriage_many(
                llm, ["ошибка 500", "ошибка 400", "ошибка 300", "успех"]
            )
        finally:
            await llm.aclose()

    results = asyncio.run(run())
    llm.close()
    assert [isinstance(result, Ticket) for result in results] == [
        recover_500,
        False,
        False,
        True,
    ]
    assert calls == {"500": 2, "400": 1, "300": 1, "успех": 1}


@pytest.mark.parametrize("recover_500", [True, False])
def test_statuses_do_not_stop_benchmark(recover_500):
    transport, calls = gateway(recover_500)
    llm = LLM(
        Settings("https://gw.example", "key"),
        async_transport=transport,
        cache=False,
        max_retries=1,
    )
    rows = [
        {"id": str(i), "text": text, "gold": {"category": "другое"}}
        for i, text in enumerate(("ошибка 500", "ошибка 400", "ошибка 300", "успех"))
    ]
    results = asyncio.run(run_candidate(llm, CANDIDATES[0], rows))
    llm.close()
    assert [row.failed for row in results] == [not recover_500, True, True, False]
    assert [row.ok for row in results] == [recover_500, False, False, True]
    assert calls == {"500": 2, "400": 1, "300": 1, "успех": 1}


@pytest.mark.parametrize("method", ["chat", "stream"])
def test_statuses_do_not_stop_sync_client(method):
    transport, calls = gateway()
    llm = LLM(
        Settings("https://gw.example", "key"),
        transport=transport,
        cache=False,
        max_retries=1,
        sleep=lambda seconds: None,
    )
    send = getattr(llm, method)
    assert send([{"role": "user", "content": "ошибка 500"}])
    for status in ("400", "300"):
        with pytest.raises(LLMError):
            send([{"role": "user", "content": f"ошибка {status}"}])
    assert send([{"role": "user", "content": "успех"}])
    llm.close()
    assert calls == {"500": 2, "400": 1, "300": 1, "успех": 1}
