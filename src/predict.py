import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

from src.models.cnn import SimpleCNN
from src.models.resnet import build_resnet18

FASHION_CLASSES = [
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
    parser.add_argument("--invert", action="store_true", help="Invert the image.")
    parser.add_argument(
        "--double_pass",
        action="store_true",
        help="Run both normal and inverted passes, keep the most confident prediction.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fashion_mnist",
        choices=["fashion_mnist", "clothing"],
        help="Dataset the model was trained on (affects preprocessing).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Image size for preprocessing (defaults to 28 for fashion_mnist, 224 for clothing).",
    )
    return parser.parse_args()


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    class_names: List[str] = ckpt.get("class_names", FASHION_CLASSES)
    num_classes = len(class_names)
    model_name = ckpt.get("model", "simple_cnn")

    if model_name == "resnet18":
        model = build_resnet18(num_classes=num_classes).to(device)
    else:
        model = SimpleCNN(num_classes=num_classes, in_channels=1).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, class_names


def preprocess(image: Image.Image, invert: bool, dataset: str, image_size: int) -> torch.Tensor:
    if dataset == "clothing":
        tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Lambda(lambda x: F.invert(x) if invert else x),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        tf = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
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

    model, class_names = load_model(args.checkpoint, device)
    img = Image.open(args.image_path).convert("RGB")
    if args.image_size is None:
        image_size = 28 if args.dataset == "fashion_mnist" else 224
    else:
        image_size = args.image_size

    if args.double_pass:
        x_norm = preprocess(img, invert=False, dataset=args.dataset, image_size=image_size).to(device)
        x_inv = preprocess(img, invert=True, dataset=args.dataset, image_size=image_size).to(device)
        pred_norm, conf_norm = predict_single(model, x_norm)
        pred_inv, conf_inv = predict_single(model, x_inv)
        if conf_inv > conf_norm:
            pred, conf, mode = pred_inv, conf_inv, "inverted"
        else:
            pred, conf, mode = pred_norm, conf_norm, "normal"
        print(f"Prediction: {class_names[pred]} (conf={conf:.3f}, mode={mode})")
    else:
        x = preprocess(img, invert=args.invert, dataset=args.dataset, image_size=image_size).to(device)
        pred, conf = predict_single(model, x)
        mode = "inverted" if args.invert else "normal"
        print(f"Prediction: {class_names[pred]} (conf={conf:.3f}, mode={mode})")


if __name__ == "__main__":
    main()
