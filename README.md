# Image Classifier from Scratch

Train a tiny CNN on FashionMNIST with minimal glue code.

## What’s inside (EN)
- Downloads FashionMNIST (train split) into `data/`.
- Simple CNN: two conv+ReLU+maxpool blocks, then a small MLP.
- Optimizer: Adam + CrossEntropyLoss. Logs loss every 100 steps.
- Runs on GPU if available, otherwise CPU.

## Setup (EN)
```bash
pip install -r requirements.txt
```

## Run a quick training (EN)
```bash
python -m src.train --epochs 1 --batch_size 64
```

---

## Ce que contient le projet (FR)
- Télécharge FashionMNIST (partie train) dans `data/`.
- CNN simple : deux blocs conv+ReLU+maxpool, suivi d’un petit MLP.
- Optimiseur : Adam + CrossEntropyLoss. Logs de la perte toutes les 100 itérations.
- S’exécute sur GPU si dispo, sinon CPU.

## Installation (FR)
```bash
pip install -r requirements.txt
```

## Lancer un entraînement rapide (FR)
```bash
python -m src.train --epochs 1 --batch_size 64
```

La commande ci-dessus lance une époque complète. Ajuste `--epochs` et `--batch_size` selon ta machine.
