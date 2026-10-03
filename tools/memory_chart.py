"""Рисует сводку Memray из файла stats --json."""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    paths = (
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in paths:
        if path.exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def main(source: str, target: str) -> None:
    stats = json.loads(Path(source).read_text())
    rows = stats["top_modules_by_allocation_size"][:6]
    peak = stats["metadata"]["peak_memory"] / 2**20
    image = Image.new("RGB", (1100, 560), "#f7f5f0")
    draw = ImageDraw.Draw(image)
    draw.text((50, 35), "Профиль памяти Memray", font=font(34), fill="#202c36")
    draw.text(
        (50, 87), f"Пик учтённой памяти: {peak:.1f} МиБ", font=font(25), fill="#225c68"
    )
    draw.text(
        (50, 130), "Сумма выделений по модулям за запуск", font=font(20), fill="#51616b"
    )

    max_size = max(row["total_bytes"] for row in rows)
    for index, row in enumerate(rows):
        y = 185 + index * 55
        size = row["total_bytes"] / 2**20
        draw.text((50, y), row["module"][:22], font=font(20), fill="#202c36")
        draw.rounded_rectangle(
            (290, y + 1, 290 + 650 * row["total_bytes"] / max_size, y + 24),
            radius=5,
            fill="#428b93",
        )
        draw.text((965, y), f"{size:.1f} МиБ", font=font(18), fill="#202c36")

    draw.text(
        (50, 525),
        "Выделения за весь запуск не равны памяти в пике.",
        font=font(16),
        fill="#6b777c",
    )
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    image.save(target)


if __name__ == "__main__":
    main(*sys.argv[1:])
