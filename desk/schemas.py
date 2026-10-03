"""Схемы ответов

Из схемы строится и проверка ответа модели, и описание формата в постановке
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class Category(StrEnum):
    payments = "платежи"
    refunds = "возвраты"
    access = "доступ"
    tariffs = "тарифы"
    integration = "интеграция"
    other = "другое"


PAYMENT_ID = re.compile(r"^P-\d{5}$")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


class Ticket(BaseModel):
    """Разобранное обращение"""

    reasoning: str = Field(
        description="одна-две фразы: что случилось и почему выбрана категория"
    )
    category: Category = Field(description="категория обращения из закрытого словаря")
    severity: int = Field(ge=1, le=5, description="срочность от 1 до 5 по правилам")
    needs_human: bool = Field(
        description="нужно ли передать обращение человеку по регламенту"
    )
    quote: str = Field(
        min_length=1,
        description="дословный фрагмент обращения, на котором основано решение",
    )
    payment_ids: list[str] = Field(
        default_factory=list,
        validate_default=True,
        description="все идентификаторы платежей вида P-12345",
    )
    amount: int | None = Field(
        None, ge=0, description="сумма операции в рублях или null"
    )

    @field_validator("payment_ids")
    @classmethod
    def ids_look_right_and_come_from_text(
        cls, ids: list[str], info: ValidationInfo
    ) -> list[str]:
        """Номера подходят под шаблон, есть в тексте, и из текста взяты все"""
        source = (info.context or {}).get("source")
        for pid in ids:
            if not PAYMENT_ID.match(pid):
                raise ValueError("идентификатор %r не похож на P-12345" % pid)
            if source is not None and pid not in source:
                raise ValueError(
                    "идентификатора %s нет в обращении, не выдумывай" % pid
                )
        if source is not None:
            for pid in re.findall(r"\bP-\d{5}\b", source):
                if pid not in ids:
                    raise ValueError("в обращении есть номер %s, добавь его" % pid)
        return ids

    @field_validator("quote")
    @classmethod
    def quote_is_verbatim(cls, quote: str, info: ValidationInfo) -> str:
        """Цитата дословно есть в обращении"""
        source = (info.context or {}).get("source")
        if source is not None and _norm(quote) not in _norm(source):
            raise ValueError(
                "цитаты нет в обращении дословно; скопируй фрагмент без изменений"
            )
        return quote


def describe(schema: type[BaseModel]) -> str:
    """Описание формата для постановки: поле, тип и пояснение"""
    lines = []
    for name, f in schema.model_fields.items():
        kind = (
            " | ".join(repr(item.value) for item in Category)
            if f.annotation is Category
            else str(f.annotation)
        )
        lines.append('  "%s": %s  // %s' % (name, kind, f.description or ""))
    return "{\n" + ",\n".join(lines) + "\n}"
