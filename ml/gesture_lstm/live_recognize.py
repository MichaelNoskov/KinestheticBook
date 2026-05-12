"""Live BLE gesture recognition."""
from __future__ import annotations

import argparse
import asyncio
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.controller import ESP32Controller
from ml.gesture_lstm.dataset import NOISE_LABEL
from ml.gesture_lstm.model import GestureCNNLSTM, GestureLSTM
from ml.gesture_lstm.temporal import align_series_to_uniform_grid


def build_model_from_config(cfg: dict) -> torch.nn.Module:
    common = dict(
        n_features=cfg["n_features"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        num_classes=cfg["num_classes"],
        dropout=cfg.get("dropout", 0.25),
    )
    if cfg.get("model_type", "lstm") == "cnn_lstm":
        return GestureCNNLSTM(
            conv_channels=int(cfg.get("cnn_channels") or 64),
            **common,
        )
    return GestureLSTM(**common)


def load_classifier(checkpoint_path: Path, norm_path: Path, device: torch.device):
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    cfg = ckpt["config"]
    seq_len = cfg.get("seq_len", 64)
    grid_dt = float(cfg.get("grid_dt", 0.05))

    model = build_model_from_config(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    nz = np.load(norm_path)
    mean = torch.from_numpy(np.asarray(nz["mean"], dtype=np.float32)).to(device)
    std = torch.from_numpy(np.asarray(nz["std"], dtype=np.float32)).to(device)

    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = {int(v): k for k, v in label_to_idx.items()}

    return model, mean, std, idx_to_label, int(seq_len), grid_dt


async def run_live(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.checkpoint).resolve()
    norm_path = Path(args.norm).resolve() if args.norm else ckpt_path.parent / "norm.npz"
    if not ckpt_path.is_file():
        raise SystemExit(f"Нет файла модели: {ckpt_path}")
    if not norm_path.is_file():
        raise SystemExit(f"Нет norm.npz (положите рядом с checkpoint или укажите --norm): {norm_path}")

    device = torch.device(args.device)
    model, mean, std, idx_to_label, seq_len_ckpt, grid_dt_ckpt = load_classifier(ckpt_path, norm_path, device)

    seq_len = int(args.seq_len) if args.seq_len is not None else seq_len_ckpt
    grid_dt = float(args.grid_dt) if args.grid_dt is not None else grid_dt_ckpt
    need_span = (seq_len - 1) * grid_dt

    queue: asyncio.Queue[tuple[float, float, float, float]] = asyncio.Queue(maxsize=2048)
    loop = asyncio.get_running_loop()

    def gyro_cb(t_mono: float, pitch: float, roll: float, yaw: float) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, (t_mono, pitch, roll, yaw))
        except Exception:
            pass

    esp = ESP32Controller(name=args.device_name, gyro_callback=gyro_cb, verbose_notify=args.verbose_ble)

    buf: deque[tuple[float, float, float, float]] = deque(maxlen=int(args.buffer_max))
    last_infer_mono = 0.0
    last_print_mono = -1e9

    await esp.connect()
    print("Подключено. Распознавание (Ctrl+C — выход).")
    print(f"Сетка: seq_len={seq_len}, grid_dt={grid_dt}s, окно={(seq_len - 1) * grid_dt:.3f}s")

    try:
        while esp.client and esp.client.is_connected:
            try:
                sample = await asyncio.wait_for(queue.get(), timeout=float(args.tick_sec))
                buf.append(sample)
            except asyncio.TimeoutError:
                pass

            now = loop.time()
            if len(buf) < max(2, int(args.min_events)):
                continue
            if now - last_infer_mono < float(args.infer_interval):
                continue

            data = sorted(buf, key=lambda s: s[0])
            t_np = np.array([s[0] for s in data], dtype=np.float64)
            ori_np = np.array([(s[1], s[2], s[3]) for s in data], dtype=np.float32)
            t_np = t_np - t_np[0]
            span = float(t_np[-1] - t_np[0])
            if span + 1e-9 < need_span:
                continue

            last_infer_mono = now
            try:
                seq = align_series_to_uniform_grid(t_np, ori_np, seq_len, grid_dt).astype(np.float32)
            except ValueError:
                continue

            x = torch.from_numpy(seq).unsqueeze(0).to(device)
            x = (x - mean) / std

            with torch.no_grad():
                logits = model(x)
                prob = torch.softmax(logits, dim=-1)
                conf, pred_idx = prob.max(dim=-1)
                pred_i = int(pred_idx.item())
                confidence = float(conf.item())

            name = idx_to_label[pred_i]

            if name == NOISE_LABEL:
                continue
            if confidence < float(args.conf_threshold):
                continue
            if now - last_print_mono < float(args.cooldown):
                continue

            last_print_mono = now
            print(f"{name}\t(conf={confidence:.2f})")
    finally:
        await esp.disconnect()


def parse_args() -> argparse.Namespace:
    default_ckpt = _REPO_ROOT / "ml" / "gesture_lstm" / "runs" / "notebook" / "checkpoint.pt"
    p = argparse.ArgumentParser(description="BLE + сохранённая модель: распознавание жестов в реальном времени")
    p.add_argument("--checkpoint", type=Path, default=default_ckpt, help="checkpoint.pt из обучения")
    p.add_argument("--norm", type=Path, default=None, help="norm.npz (по умолчанию рядом с checkpoint)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--device-name", type=str, default="ESP32-C3-Motion", help="Имя BLE устройства")
    p.add_argument("--seq-len", type=int, default=None, help="Длина окна на временной сетке (по умолчанию из checkpoint)")
    p.add_argument(
        "--grid-dt",
        type=float,
        default=None,
        help="Шаг временной сетки в секундах (по умолчанию из checkpoint, обычно 0.05)",
    )
    p.add_argument("--buffer-max", type=int, default=512, help="Сколько последних BLE-событий держать")
    p.add_argument("--min-events", type=int, default=8, help="Минимум точек в буфере до инференса")
    p.add_argument("--infer-interval", type=float, default=0.12, help="Интервал между инференсами, сек")
    p.add_argument("--tick-sec", type=float, default=0.05, help="Таймаут ожидания очереди BLE")
    p.add_argument("--cooldown", type=float, default=1.2, help="Пауза между печатью двух жестов, сек")
    p.add_argument("--conf-threshold", type=float, default=0.55, help="Минимум softmax(confidence) для жеста")
    p.add_argument("--verbose-ble", action="store_true", help="Печатать каждое BLE-сообщение от контроллера")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run_live(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
