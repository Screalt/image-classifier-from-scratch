import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.data.datamodule import make_test_dataloader
from src.models.cnn import SimpleCNN
from src.utils.training import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the trained CNN on the test set.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints/best_model.pth"),
        help="Path to the checkpoint to evaluate.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Directory containing FashionMNIST (downloaded if missing).",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for eval.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader workers (set 0 if issues on your OS).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model = SimpleCNN().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    test_loader = make_test_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f} - Test acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
