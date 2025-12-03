# Image Classifier from Scratch

Train a tiny CNN on FashionMNIST with minimal glue code. Saves the best checkpoint and plots.

## What’s inside (EN)
- Downloads FashionMNIST (train split) into `data/`.
- Simple CNN: two conv+ReLU+maxpool blocks, then a small MLP.
- Train/val split (90/10 by default), metrics: loss + accuracy for both.
- Saves best checkpoint to `outputs/checkpoints/best_model.pth` (highest val_acc).
- Saves training curves to `outputs/plots/loss_curve.png` and `outputs/plots/accuracy_curve.png`.
- Optimizer: Adam + CrossEntropyLoss. Logs minibatch stats and epoch summaries.
- Runs on GPU if available, otherwise CPU.

## Code structure
- `src/train.py` — CLI entrypoint orchestrating the pipeline.
- `src/models/cnn.py` — model definition.
- `src/data/datamodule.py` — dataset + train/val dataloaders.
- `src/utils/training.py` — train/validate loops.
- `src/utils/plotting.py` — matplotlib curves.

## Setup (EN)
```bash
pip install -r requirements.txt
```

## Run a quick training (EN)
```bash
python -m src.train --epochs 2 --batch_size 64
```

---

## Ce que contient le projet (FR)
- Télécharge FashionMNIST (partie train) dans `data/`.
- CNN simple : deux blocs conv+ReLU+maxpool, suivi d’un petit MLP.
- Split train/val (90/10 par défaut), métriques : loss + accuracy pour les deux.
- Sauvegarde le meilleur modèle dans `outputs/checkpoints/best_model.pth` (meilleure val_acc).
- Sauvegarde les courbes dans `outputs/plots/loss_curve.png` et `outputs/plots/accuracy_curve.png`.
- Optimiseur : Adam + CrossEntropyLoss. Logs par minibatch et résumé par époque.
- S’exécute sur GPU si dispo, sinon CPU.

## Structure du code
- `src/train.py` — point d’entrée CLI qui orchestre le pipeline.
- `src/models/cnn.py` — définition du modèle.
- `src/data/datamodule.py` — dataset + dataloaders train/val.
- `src/utils/training.py` — boucles train/val.
- `src/utils/plotting.py` — courbes matplotlib.

## Installation (FR)
```bash
pip install -r requirements.txt
```

## Lancer un entraînement rapide (FR)
```bash
python -m src.train --epochs 2 --batch_size 64
```

La commande ci-dessus lance une époque complète. Ajuste `--epochs` et `--batch_size` selon ta machine.
