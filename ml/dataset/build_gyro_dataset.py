from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
import threading
from pathlib import Path
from typing import Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.controller import ESP32Controller
from ml.dataset.csv_read import read_normalized_csv_rows

CSV_HEADER = ("t_mono", "pitch", "roll", "yaw", "dp", "dr", "dy")
Row = tuple[float, float, float, float, float, float, float]


def _default_output_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "gyro_timeseries.csv"


def _load_existing_rows(path: Path) -> list[Row]:
    if not path.is_file():
        return []
    raw_rows = read_normalized_csv_rows(path)
    if not raw_rows:
        return []
    rows: list[Row] = []
    for r in raw_rows:
        try:
            rows.append(
                (
                    float(r["t_mono"]),
                    float(r["pitch"]),
                    float(r["roll"]),
                    float(r["yaw"]),
                    float(r["dp"]),
                    float(r["dr"]),
                    float(r["dy"]),
                )
            )
        except (KeyError, ValueError):
            continue
    return rows


def atomic_write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


async def _commit_loop(
    output_path: Path,
    get_rows_snapshot: Callable[[], list[Row]],
    interval: float,
    stop_event: asyncio.Event,
):
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
        snapshot = get_rows_snapshot()
        atomic_write_csv(output_path, snapshot)
    snapshot = get_rows_snapshot()
    atomic_write_csv(output_path, snapshot)


async def run_collection(
    output_path: Path,
    device_name: str,
    commit_interval: float,
) -> None:
    rows_lock = threading.Lock()
    rows: list[Row] = _load_existing_rows(output_path)
    if rows:
        last = rows[-1]
        prev: tuple[float, float, float, float] | None = (last[0], last[1], last[2], last[3])
    else:
        prev = None

    def gyro_cb(t_mono: float, pitch: float, roll: float, yaw: float) -> None:
        nonlocal prev
        if prev is None:
            dp = dr = dy = 0.0
        else:
            _, pp, pr, py = prev
            dp, dr, dy = pitch - pp, roll - pr, yaw - py
        prev = (t_mono, pitch, roll, yaw)
        row: Row = (t_mono, pitch, roll, yaw, dp, dr, dy)
        with rows_lock:
            rows.append(row)

    def get_rows_snapshot() -> list[Row]:
        with rows_lock:
            return list(rows)

    esp = ESP32Controller(name=device_name, gyro_callback=gyro_cb)
    stop_event = asyncio.Event()

    async def watcher():
        try:
            while esp.is_connected:
                await asyncio.sleep(0.5)
        finally:
            stop_event.set()

    await esp.connect()
    commit_task = asyncio.create_task(
        _commit_loop(output_path, get_rows_snapshot, commit_interval, stop_event)
    )
    watcher_task = asyncio.create_task(watcher())
    try:
        await watcher_task
    finally:
        stop_event.set()
        commit_task.cancel()
        try:
            await commit_task
        except asyncio.CancelledError:
            pass
        atomic_write_csv(output_path, get_rows_snapshot())
        await esp.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Сбор гироскопа в CSV (атомарная запись).")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Путь к CSV (по умолчанию: {_default_output_path()})",
    )
    parser.add_argument("--device", type=str, default="ESP32-C3-Motion", help="Имя BLE устройства")
    parser.add_argument(
        "--commit-interval",
        type=float,
        default=5.0,
        help="Интервал сброса буфера на диск, сек (атомарная перезапись всего файла)",
    )
    args = parser.parse_args()
    output_path = args.output or _default_output_path()
    output_path = output_path.resolve()

    try:
        asyncio.run(run_collection(output_path, args.device, args.commit_interval))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
