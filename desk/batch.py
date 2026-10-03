"""Пачки фиксированного размера."""

from collections.abc import Iterator


def batches[T](items: list[T], size: int) -> Iterator[list[T]]:
    if size < 1:
        raise ValueError("concurrency должно быть положительным")
    for start in range(0, len(items), size):
        yield items[start : start + size]
