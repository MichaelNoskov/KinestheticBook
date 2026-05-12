"""Train an LSTM/CNN-LSTM gesture classifier."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.gesture_lstm.dataset import (
    GestureNoiseDataset,
    NormalizedSubset,
    SubsetDataset,
    compute_norm_stats,
    stratified_split,
)
from ml.gesture_lstm.model import GestureCNNLSTM, GestureLSTM


def _default_data_dir() -> Path:
    return _REPO_ROOT / "ml" / "dataset" / "data"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSTM/CNN-LSTM классификация жестов (папки + корневые CSV как шум)")
    p.add_argument("--data-dir", type=Path, default=None, help="Корень данных (подпапки = классы)")
    p.add_argument("--out-dir", type=Path, default=None, help="Куда сохранить веса и label_map.json")
    p.add_argument("--model", choices=("cnn_lstm", "lstm"), default="cnn_lstm")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument(
        "--grid-dt",
        type=float,
        default=0.05,
        help="Шаг равномерной временной сетки (сек), совпадает с логикой forward-fill между событиями",
    )
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--cnn-channels", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--noise-window", type=int, default=48, help="Строк в окне для корневых CSV (шум)")
    p.add_argument("--noise-stride", type=int, default=24)
    p.add_argument("--max-noise", type=int, default=400, help="Макс. окон шума (None = без лимита)")
    p.add_argument(
        "--min-raw-rows",
        type=int,
        default=6,
        help="Мин. строк событий в CSV для одного примера (короткие записи жестов)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_model(args: argparse.Namespace, n_features: int, n_classes: int) -> torch.nn.Module:
    if args.model == "cnn_lstm":
        return GestureCNNLSTM(
            n_features=n_features,
            conv_channels=args.cnn_channels,
            hidden_size=args.hidden,
            num_layers=args.layers,
            num_classes=n_classes,
            dropout=args.dropout,
        )
    return GestureLSTM(
        n_features=n_features,
        hidden_size=args.hidden,
        num_layers=args.layers,
        num_classes=n_classes,
        dropout=args.dropout,
    )


def main() -> None:
    args = parse_args()
    data_dir = (args.data_dir or _default_data_dir()).resolve()
    out_dir = (args.out_dir or (_REPO_ROOT / "ml" / "gesture_lstm" / "runs" / "default")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    max_noise = args.max_noise if args.max_noise > 0 else None
    full_ds = GestureNoiseDataset(
        data_dir,
        seq_len=args.seq_len,
        grid_dt=args.grid_dt,
        noise_window_rows=args.noise_window,
        noise_stride=args.noise_stride,
        max_noise_samples=max_noise,
        seed=args.seed,
        min_raw_rows=args.min_raw_rows,
    )

    train_idx, val_idx = stratified_split(full_ds, val_ratio=args.val_ratio, seed=args.seed)
    if not train_idx:
        raise SystemExit("Пустой train split — мало данных по классам.")

    train_sub = SubsetDataset(full_ds, train_idx, augment=True)
    val_sub = SubsetDataset(full_ds, val_idx, augment=False)
    mean, std = compute_norm_stats(full_ds, train_idx)
    np.savez(out_dir / "norm.npz", mean=mean, std=std)

    train_ds = NormalizedSubset(train_sub, mean, std)
    val_ds = NormalizedSubset(val_sub, mean, std)

    n_classes = full_ds.num_classes()
    model = build_model(args, len(mean), n_classes).to(args.device)

    loader_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    loader_val = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader_train:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        model.eval()
        v_total, v_correct, v_loss = 0, 0, 0.0
        with torch.no_grad():
            for x, y in loader_val:
                x, y = x.to(args.device), y.to(args.device)
                logits = model(x)
                loss = crit(logits, y)
                v_loss += loss.item() * x.size(0)
                pred = logits.argmax(dim=1)
                v_correct += (pred == y).sum().item()
                v_total += x.size(0)

        tr_acc = correct / max(total, 1)
        va_acc = v_correct / max(v_total, 1)
        va_loss = v_loss / max(v_total, 1)
        print(f"epoch {epoch:03d}  train_loss={loss_sum/max(total,1):.4f} acc={tr_acc:.3f}  "
              f"val_loss={va_loss:.4f} acc={va_acc:.3f}")

        if v_total > 0 and va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    ckpt = {
        "model_state": model.state_dict(),
        "config": {
            "model_type": args.model,
            "n_features": int(len(mean)),
            "hidden_size": args.hidden,
            "cnn_channels": args.cnn_channels if args.model == "cnn_lstm" else None,
            "num_layers": args.layers,
            "num_classes": n_classes,
            "dropout": args.dropout,
            "seq_len": args.seq_len,
            "grid_dt": args.grid_dt,
        },
        "label_to_idx": full_ds.label_to_idx,
    }
    torch.save(ckpt, out_dir / "checkpoint.pt")
    full_ds.export_label_map(out_dir / "label_map.json")
    meta = {
        "data_dir": str(data_dir),
        "n_samples": len(full_ds),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "classes": full_ds.label_to_idx,
        "model_type": args.model,
        "grid_dt": args.grid_dt,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Сохранено: {out_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
