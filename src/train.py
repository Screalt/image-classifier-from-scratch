import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal CNN training loop (step 1).")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to run.")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Directory to download FashionMNIST.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for the optimizer."
    )
    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for step, (images, labels) in enumerate(dataloader, start=1):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % 100 == 0 or step == len(dataloader):
            avg_loss = running_loss / step
            print(f"Step {step:04d}/{len(dataloader)} - Loss: {avg_loss:.4f}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.ToTensor()
    train_dataset = datasets.FashionMNIST(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_one_epoch(model, train_loader, criterion, optimizer, device)


if __name__ == "__main__":
    main()
