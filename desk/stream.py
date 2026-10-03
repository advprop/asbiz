"""Потоковая выдача

С "stream": true шлюз присылает ответ по кусочкам, событиями. Событие
состоит из строк event и data, между событиями пустая строка:

    event: message_start
    data: {"type": "message_start", "message": {"usage": {"input_tokens": 25}}}

    event: content_block_delta
    data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Ок"}}

    event: message_delta
    data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}

    event: message_stop
    data: {"type": "message_stop"}

Здесь поток разбирается на события и собирается в ответ
"""

from __future__ import annotations

import json
from codecs import getincrementaldecoder
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from .llm import LLMError, Usage


class LineBuffer:
    """Собирает строки UTF 8 с пределом на весь поток."""

    def __init__(self, limit: int):
        self.limit = limit
        self.size = 0
        self.decoder = getincrementaldecoder("utf-8")()
        self.pending = ""

    def feed(self, chunk: bytes) -> list[str]:
        self.size += len(chunk)
        if self.size > self.limit:
            raise LLMError("поток превысил допустимый размер")
        try:
            self.pending += self.decoder.decode(chunk)
        except UnicodeDecodeError as error:
            raise LLMError("поток содержит неверный UTF 8") from error
        lines = self.pending.split("\n")
        self.pending = lines.pop()
        return [line.removesuffix("\r") for line in lines]

    def finish(self) -> list[str]:
        try:
            self.pending += self.decoder.decode(b"", final=True)
        except UnicodeDecodeError as error:
            raise LLMError("поток содержит неверный UTF 8") from error
        return [self.pending] if self.pending else []


@dataclass
class StreamResult:
    """Ответ, собранный из потока"""

    text: str
    stop_reason: str
    usage: Usage
    ttft_s: float | None  # до первого слова; None, если текста нет
    total_s: float  # до конца потока
    thinking_chars: int = 0  # длина рассуждения в символах

    @property
    def truncated(self) -> bool:
        """Ответ оборван по max_tokens"""
        return self.stop_reason == "max_tokens"


def iter_sse(lines: Iterable[str]) -> Iterator[tuple[str, dict[str, object]]]:
    """Разбирает строки потока на события: имя и данные

    Событие кончается пустой строкой. Несколько строк data одного события
    склеиваются. Строка с двоеточием в начале это комментарий
    """
    event: str | None = None
    data = []

    def decode() -> dict[str, object]:
        try:
            value = json.loads("\n".join(data))
        except ValueError as e:
            raise LLMError("неверный JSON в потоке") from e
        if not isinstance(value, dict):
            raise LLMError("неверное событие в потоке")
        return value

    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:
            if data:
                yield event or "message", decode()
            event, data = None, []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    if data:
        yield event or "message", decode()


def collect(
    events: Iterable[tuple[str, dict[str, object]]],
    clock: Callable[[], float],
    started: float,
    usage_from: Callable[[dict[str, object], float, int], Usage],
    retries: int = 0,
) -> StreamResult:
    """Собирает ответ из событий

    Текст склеивается из кусочков, время до первого слова засекается
    на первом кусочке текста. Событие error обрывает сбор
    """
    usage_raw: dict[str, object] = {}
    parts, ttft, stop, thinking = [], None, "end_turn", 0
    for name, data in events:
        kind = data.get("type", name)
        match kind:
            case "message_start":
                usage_raw.update((data.get("message") or {}).get("usage") or {})
            case "content_block_delta":
                delta = data.get("delta") or {}
                if delta.get("type") == "text_delta":
                    if ttft is None:
                        ttft = clock() - started
                    parts.append(delta.get("text", ""))
                elif delta.get("type") == "thinking_delta":
                    thinking += len(delta.get("thinking", ""))
            case "message_delta":
                stop = (data.get("delta") or {}).get("stop_reason") or stop
                usage_raw.update(data.get("usage") or {})
            case "error":
                err = data.get("error") or {}
                raise LLMError(
                    "поток прерван: %s %s" % (err.get("type"), err.get("message", ""))
                )
    total = clock() - started
    usage = usage_from(usage_raw, total, retries)
    return StreamResult("".join(parts).strip(), stop, usage, ttft, total, thinking)
