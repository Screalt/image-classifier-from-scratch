import argparse
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

from src.models.cnn import SimpleCNN

CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict the class of a custom image with optional inversion or double-pass."
    )
    parser.add_argument(
        "--image_path",
        type=Path,
        required=True,
        help="Path to the image to classify.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints/best_model.pth"),
        help="Path to the checkpoint to use.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert the image (useful when object is dark on light background).",
    )
    parser.add_argument(
        "--double_pass",
        action="store_true",
        help="Run both normal and inverted passes, keep the most confident prediction.",
    )
    return parser.parse_args()


def load_model(ckpt_path: Path, device: torch.device) -> SimpleCNN:
    model = SimpleCNN().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def preprocess(image: Image.Image, invert: bool) -> torch.Tensor:
    tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.Lambda(lambda x: F.invert(x) if invert else x),
            transforms.ToTensor(),
        ]
    )
    return tf(image).unsqueeze(0)


def predict_single(model: torch.nn.Module, x: torch.Tensor) -> Tuple[int, float]:
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
    return pred.item(), conf.item()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    model = load_model(args.checkpoint, device)
    img = Image.open(args.image_path).convert("RGB")

    if args.double_pass:
        x_norm = preprocess(img, invert=False).to(device)
        x_inv = preprocess(img, invert=True).to(device)
        pred_norm, conf_norm = predict_single(model, x_norm)
        pred_inv, conf_inv = predict_single(model, x_inv)
        if conf_inv > conf_norm:
            pred, conf, mode = pred_inv, conf_inv, "inverted"
        else:
            pred, conf, mode = pred_norm, conf_norm, "normal"
        print(f"Prediction: {CLASSES[pred]} (conf={conf:.3f}, mode={mode})")
    else:
        x = preprocess(img, invert=args.invert).to(device)
        pred, conf = predict_single(model, x)
        mode = "inverted" if args.invert else "normal"
        print(f"Prediction: {CLASSES[pred]} (conf={conf:.3f}, mode={mode})")


if __name__ == "__main__":
    main()
