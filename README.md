# Image Classifier from Scratch


Train image classifiers on two datasets: a stylized set (FashionMNIST) and a real photo set (clothing-dataset-small). Two models depending on the case, checkpoints + plots, and CLIs for train/eval/predict.

## What’s inside (EN)
- Datasets:
  - `fashion_mnist` (default, 28×28 grayscale).
  - `clothing` (real photos, auto-downloads `clothing-dataset-small`, resizes to your chosen size).
- Models: `simple_cnn` for FashionMNIST, `resnet18` for photos.
- Classes:
  - FashionMNIST: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
  - Clothing: dress, hat, longsleeve, outwear, pants, shirt, shoes, shorts, skirt, t-shirt.
- Train/val split with loss + accuracy metrics.
- Saves best checkpoint: `outputs/checkpoints/best_model.pth`.
- Saves plots: `outputs/plots/loss_curve.png`, `outputs/plots/accuracy_curve.png`.
- Optimizer: Adam + CrossEntropyLoss. Runs on GPU if available, else CPU.

## Code structure
- `src/train.py` — train/val loop + logging + checkpoint + plots.
- `src/evaluate.py` — test set evaluation of a checkpoint.
- `src/predict.py` — inference on custom images (invert/double-pass options).
- `src/models/cnn.py` — model definition.
- `src/data/datamodule.py` — dataset + dataloaders (train/val/test).
- `src/utils/training.py` — train/validate helpers.
- `src/utils/plotting.py` — matplotlib curves.

## Setup
```bash
pip install -r requirements.txt
```

## Train
FashionMNIST (quick, simple CNN):
```bash
python -m src.train --epochs 2 --batch_size 64 --dataset fashion_mnist --model simple_cnn --image_size 28
```
Clothing photos (ResNet18, 224px):
```bash
python -m src.train --epochs 8 --batch_size 64 --lr 1e-3 --dataset clothing --model resnet18 --image_size 224
```
Longer run? Increase `--epochs` (e.g., 12–16). If you have a strong GPU and want 512px, set `--image_size 512` and reduce `--batch_size`.

## Evaluate (test split)
```bash
python -m src.evaluate --checkpoint outputs/checkpoints/best_model.pth --dataset fashion_mnist --image_size 28
# for clothing dataset:
python -m src.evaluate --checkpoint outputs/checkpoints/best_model.pth --dataset clothing --image_size 224
```

## Predict on a custom image
```bash
python -m src.predict --image_path /path/to/image.jpg --checkpoint outputs/checkpoints/best_model.pth --dataset clothing --image_size 224 --double_pass
# FashionMNIST-style (28x28 grayscale, simple CNN):
python -m src.predict --image_path /path/to/image.jpg --checkpoint outputs/checkpoints/best_model.pth --dataset fashion_mnist --image_size 28 --double_pass
```

---

## Ce que contient le projet (FR)
- Jeux de données :
  - `fashion_mnist` (défaut, 28×28 niveau de gris).
  - `clothing` (photos réelles, télécharge automatiquement `clothing-dataset-small`, redimension à la taille choisie).
- Modèles : `simple_cnn` (FashionMNIST) ou `resnet18` (photos).
- Classes :
  - FashionMNIST : T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
  - Clothing : dress, hat, longsleeve, outwear, pants, shirt, shoes, shorts, skirt, t-shirt.
- Split train/val, métriques : loss + accuracy.
- Meilleur modèle : `outputs/checkpoints/best_model.pth`.
- Courbes : `outputs/plots/loss_curve.png`, `outputs/plots/accuracy_curve.png`.
- Optimiseur : Adam + CrossEntropyLoss. GPU si dispo, sinon CPU.

## Structure du code
- `src/train.py` — entraînement/validation + logs + checkpoint + plots.
- `src/evaluate.py` — évaluation test d’un checkpoint.
- `src/predict.py` — inférence sur images custom (options inversion/double-pass).
- `src/models/cnn.py` — définition du modèle.
- `src/data/datamodule.py` — dataset + dataloaders (train/val/test).
- `src/utils/training.py` — helpers train/val.
- `src/utils/plotting.py` — courbes matplotlib.

## Installation
```bash
pip install -r requirements.txt
```

## Entraîner
FashionMNIST (rapide, simple CNN) :
```bash
python -m src.train --epochs 2 --batch_size 64 --dataset fashion_mnist --model simple_cnn --image_size 28
```
Photos réelles (clothing, ResNet18, 224px) :
```bash
python -m src.train --epochs 8 --batch_size 64 --lr 1e-3 --dataset clothing --model resnet18 --image_size 224
```
Plus long ? Augmente `--epochs` (12–16). Pour du 512px, mets `--image_size 512` et réduis `--batch_size`.

## Évaluer (jeu de test)
```bash
python -m src.evaluate --checkpoint outputs/checkpoints/best_model.pth --dataset fashion_mnist --image_size 28
# pour le dataset clothing :
python -m src.evaluate --checkpoint outputs/checkpoints/best_model.pth --dataset clothing --image_size 224
```

## Prédire sur une image custom
```bash
python -m src.predict --image_path /chemin/vers/image.jpg --checkpoint outputs/checkpoints/best_model.pth --dataset clothing --image_size 224 --double_pass
# version FashionMNIST (28x28, niveaux de gris, simple CNN) :
python -m src.predict --image_path /chemin/vers/image.jpg --checkpoint outputs/checkpoints/best_model.pth --dataset fashion_mnist --image_size 28 --double_pass
```
