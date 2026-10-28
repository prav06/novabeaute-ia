"""
Entraînement d'un classifieur de lésions cutanées par transfer learning.

Approche : on part d'un ResNet50 pré-entraîné sur ImageNet (le modèle sait
déjà reconnaître des formes, textures, contours), on remplace sa dernière
couche par une couche à 7 sorties (nos classes), et on ré-entraîne
seulement cette partie + un fine-tuning léger du reste.

Pourquoi pas un modèle from scratch ? Avec ~7000 images d'entraînement
seulement, un CNN entraîné de zéro sur-apprendrait très vite (overfitting)
sans jamais bien généraliser.

Gestion du déséquilibre : les classes sont pondérées dans la loss
(CrossEntropyLoss avec poids inversement proportionnels à la fréquence)
pour éviter que le modèle ignore les classes minoritaires comme le
mélanome.

Usage : python src/train_classifier.py
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, f1_score, recall_score
from tqdm import tqdm

DATA_DIR = "data"
IMAGES_DIR = f"{DATA_DIR}/images"
SPLITS_DIR = f"{DATA_DIR}/splits"
OUTPUT_DIR = "outputs"
CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
BATCH_SIZE = 32
EPOCHS = 12
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
eval_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SkinLesionDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, f"{row['image_id']}.jpg")
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = CLASS_TO_IDX[row["dx"]]
        return image, label


def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # On gèle le corps du réseau, sauf les 2 derniers blocs (fine-tuning léger)
    for name, param in model.named_parameters():
        if not (name.startswith("layer4") or name.startswith("fc")):
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model.to(DEVICE)


def compute_class_weights(train_csv):
    df = pd.read_csv(train_csv)
    counts = df["dx"].value_counts()
    weights = [1.0 / counts[c] for c in CLASSES]
    weights = np.array(weights) / np.sum(weights) * len(CLASSES)
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_ds = SkinLesionDataset(f"{SPLITS_DIR}/train.csv", IMAGES_DIR, train_tfms)
    val_ds = SkinLesionDataset(f"{SPLITS_DIR}/val.csv", IMAGES_DIR, eval_tfms)
    test_ds = SkinLesionDataset(f"{SPLITS_DIR}/test.csv", IMAGES_DIR, eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=4)

    model = build_model()
    class_weights = compute_class_weights(f"{SPLITS_DIR}/train.csv")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    best_f1 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        y_true, y_pred = evaluate(model, val_loader)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        mel_recall = recall_score(y_true, y_pred, labels=[CLASS_TO_IDX["mel"]], average="macro")
        print(f"Epoch {epoch+1} — loss: {running_loss/len(train_loader):.4f} "
              f"— val macro-F1: {macro_f1:.4f} — rappel mélanome: {mel_recall:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/model.pt")
            print("  -> nouveau meilleur modèle sauvegardé")

    # Évaluation finale sur le test set avec le meilleur modèle
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/model.pt"))
    y_true, y_pred = evaluate(model, test_loader)
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    print("\n=== Rapport sur le test set ===")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nMétriques sauvegardées dans {OUTPUT_DIR}/metrics.json")


if __name__ == "__main__":
    main()
