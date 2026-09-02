"""Labeled PNG rendering with a dependency-free compatibility fallback."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path


def write_bar_png(
    path: Path,
    values: list[float],
    labels: list[str] | None = None,
    *,
    width: int = 900,
    height: int = 480,
) -> None:
    """Write a labeled Pillow chart, or a valid pure-Python bar PNG."""
    labels = labels or [str(index) for index in range(len(values))]
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        _write_fallback(path, values, width=width, height=height)
        return
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 70, 55, width - 25, height - 105
    draw.text((left, 15), "ATTR-RTG-RCMZ H8 NLL", fill="black")
    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    maximum = max(values, default=1.0) or 1.0
    for tick in range(5):
        value = maximum * tick / 4
        y = bottom - (bottom - top) * tick / 4
        draw.line((left - 4, y, left, y), fill="black")
        draw.text((4, y - 6), f"{value:.3f}", fill="black")
    slot = (right - left) / max(1, len(values))
    for index, (label, value) in enumerate(zip(labels, values)):
        x0, x1 = (
            left + index * slot + slot * 0.15,
            left + (index + 1) * slot - slot * 0.15,
        )
        y = bottom - (bottom - top) * max(0.0, value) / maximum
        draw.rectangle((x0, y, x1, bottom), fill="#2a6fbb")
        parts = label.split("/")
        draw.text((x0, bottom + 8), parts[0], fill="black")
        if len(parts) > 1:
            draw.text((x0, bottom + 23), parts[1], fill="black")
    draw.text(
        (left, height - 45),
        "■ H8 NLL    x: arm / training seed    y: H8 NLL",
        fill="#2a6fbb",
    )
    image.save(path, format="PNG")


def _write_fallback(
    path: Path, values: list[float], *, width: int, height: int
) -> None:
    pixels = bytearray([255] * width * height * 3)
    if values:
        maximum = max(max(values), 1e-12)
        slot = width / len(values)
        for index, value in enumerate(values):
            x0, x1 = (
                int(index * slot + slot * 0.18),
                int((index + 1) * slot - slot * 0.18),
            )
            y0 = height - 25 - int((height - 50) * max(0.0, value) / maximum)
            for y in range(max(10, y0), height - 25):
                for x in range(max(0, x0), min(width, x1)):
                    offset = (y * width + x) * 3
                    pixels[offset : offset + 3] = bytes((42, 111, 187))
    raw = b"".join(
        b"\x00" + pixels[y * width * 3 : (y + 1) * width * 3] for y in range(height)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    payload = b"\x89PNG\r\n\x1a\n"
    payload += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    path.write_bytes(
        payload + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    )
