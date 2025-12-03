# Image Classifier from Scratch

# Image Classifier from Scratch

Train a small CNN on FashionMNIST, save the best checkpoint, and produce loss/accuracy plots. CLI tools for training, evaluation, and custom image prediction.

## What’s inside (EN)
- Downloads FashionMNIST into `data/`.
- CNN: two conv+ReLU+maxpool blocks, then a small MLP.
- Train/val split (90/10), metrics: loss + accuracy.
- Saves best checkpoint: `outputs/checkpoints/best_model.pth` (highest val_acc).
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
Quick run:
```bash
python -m src.train --epochs 2 --batch_size 64
```
Longer/better:
```bash
python -m src.train --epochs 10 --batch_size 128 --lr 1e-3
```

## Evaluate (test split)
```bash
python -m src.evaluate --checkpoint outputs/checkpoints/best_model.pth
```

## Predict on a custom image
```bash
python -m src.predict --image_path /path/to/image.jpg --checkpoint outputs/checkpoints/best_model.pth
# if your object is dark on light background:
python -m src.predict --image_path /path/to/image.jpg --checkpoint outputs/checkpoints/best_model.pth --invert
# or let it try both and keep the most confident:
python -m src.predict --image_path /path/to/image.jpg --checkpoint outputs/checkpoints/best_model.pth --double_pass
```

---

## Ce que contient le projet (FR)
- Télécharge FashionMNIST dans `data/`.
- CNN : deux blocs conv+ReLU+maxpool, puis un petit MLP.
- Split train/val (90/10), métriques : loss + accuracy.
- Sauvegarde le meilleur modèle : `outputs/checkpoints/best_model.pth` (meilleure val_acc).
- Sauvegarde les courbes : `outputs/plots/loss_curve.png`, `outputs/plots/accuracy_curve.png`.
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
Rapide :
```bash
python -m src.train --epochs 2 --batch_size 64
```
Plus poussé :
```bash
python -m src.train --epochs 10 --batch_size 128 --lr 1e-3
```

## Évaluer (jeu de test)
```bash
python -m src.evaluate --checkpoint outputs/checkpoints/best_model.pth
```

## Prédire sur une image custom
```bash
python -m src.predict --image_path /chemin/vers/image.jpg --checkpoint outputs/checkpoints/best_model.pth
# si l’objet est sombre sur fond clair :
python -m src.predict --image_path /chemin/vers/image.jpg --checkpoint outputs/checkpoints/best_model.pth --invert
# ou tente normal + inversé et garde le plus confiant :
python -m src.predict --image_path /chemin/vers/image.jpg --checkpoint outputs/checkpoints/best_model.pth --double_pass
```
