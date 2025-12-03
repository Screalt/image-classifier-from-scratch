import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.datamodule import make_dataloaders
from src.models.cnn import SimpleCNN
from src.utils.plotting import plot_training_curves
from src.utils.training import evaluate, train_one_epoch


def parse_args():
    parser = argparse.ArgumentParser(
        description="CNN training with train/val split, metrics and plots."
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Directory to download FashionMNIST.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of training data used for validation.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for train/val split reproducibility."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader workers (set 0 if issues on your OS).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs"),
        help="Base directory for checkpoints and plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = make_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_acc = 0.0
    checkpoints_dir = args.output_dir / "checkpoints"
    plots_dir = args.output_dir / "plots"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"[Epoch {epoch}] "
            f"train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = checkpoints_dir / "best_model.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                },
                checkpoint_path,
            )
            print(f"Saved new best model to {checkpoint_path} (val_acc={val_acc:.4f})")

    plot_training_curves(history, plots_dir)


if __name__ == "__main__":
    main()
