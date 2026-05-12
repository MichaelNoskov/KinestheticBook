"""Small CSV reader tolerant to BOM and Windows encodings."""
from __future__ import annotations

import csv
import io
from pathlib import Path


def decode_csv_bytes(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def normalize_csv_row_keys(row: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in row.items():
        if k is None:
            continue
        nk = str(k).lstrip("\ufeff").strip()
        out[nk] = v
    return out


def read_normalized_csv_rows(path: Path) -> list[dict[str, str]] | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    text = decode_csv_bytes(raw)
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        return None
    return [normalize_csv_row_keys(dict(row)) for row in reader]
