from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def plot_training_curves(history: Dict[str, List[float]], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over epochs")
    plt.legend()
    plt.tight_layout()
    loss_path = plots_dir / "loss_curve.png"
    plt.savefig(loss_path)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over epochs")
    plt.legend()
    plt.tight_layout()
    acc_path = plots_dir / "accuracy_curve.png"
    plt.savefig(acc_path)
    plt.close()

    print(f"Saved plots to {loss_path} and {acc_path}")
