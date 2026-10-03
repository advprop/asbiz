"""Проверяет удержанную память клиента на подставном шлюзе."""

import gc
import tracemalloc

import httpx

from desk.config import Settings
from desk.llm import LLM

REPLY = {
    "content": [{"type": "text", "text": "другое"}],
    "usage": {"input_tokens": 3, "output_tokens": 1},
}


def main() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=REPLY))
    llm = LLM(Settings("https://gw.example", "test"), transport=transport, cache=False)
    tracemalloc.start()
    points = []
    try:
        for index in range(1000):
            llm.chat([{"role": "user", "content": f"вопрос {index}"}])
            if index in (499, 999):
                gc.collect()
                points.append(tracemalloc.get_traced_memory()[0])
    finally:
        llm.close()
    print(f"После 500: {points[0]} байт")
    print(f"После 1000: {points[1]} байт")
    print(f"Разница: {points[1] - points[0]} байт")
    print(f"Записей в журнале: {len(llm.ledger)}")


if __name__ == "__main__":
    main()
