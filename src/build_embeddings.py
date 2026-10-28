"""
Calcule un embedding (vecteur numérique) par image d'entraînement et
construit un index FAISS pour la recherche par similarité — c'est la
brique qui permet de dire "cette photo ressemble le plus à ces images
de la base de référence".

On réutilise le modèle entraîné (train_classifier.py), mais on lit la
sortie de l'avant-dernière couche (avant la classification finale) :
c'est un vecteur de 2048 valeurs qui capture les caractéristiques
visuelles de l'image, plus riche qu'une simple étiquette de classe.

Usage : python src/build_embeddings.py
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import faiss

DATA_DIR = "data"
IMAGES_DIR = f"{DATA_DIR}/images"
SPLITS_DIR = f"{DATA_DIR}/splits"
OUTPUT_DIR = "outputs"
CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_embedding_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/model.pt", map_location=DEVICE))
    model.fc = nn.Identity()  # on coupe la tête de classification -> sortie = embedding 2048d
    model.eval().to(DEVICE)
    return model


def embed_image(model, image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = tfms(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model(tensor).cpu().numpy().flatten()
    return embedding


def main():
    model = load_embedding_model()

    # On construit l'index à partir du train + val (le test set doit rester
    # "inconnu" pour simuler des photos jamais vues, comme en production)
    ref_df = pd.concat([
        pd.read_csv(f"{SPLITS_DIR}/train.csv"),
        pd.read_csv(f"{SPLITS_DIR}/val.csv"),
    ]).reset_index(drop=True)

    embeddings = []
    for _, row in tqdm(ref_df.iterrows(), total=len(ref_df), desc="Calcul des embeddings"):
        img_path = os.path.join(IMAGES_DIR, f"{row['image_id']}.jpg")
        embeddings.append(embed_image(model, img_path))

    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)  # normalisation -> similarité cosinus via produit scalaire

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, f"{OUTPUT_DIR}/faiss.index")
    ref_df.to_csv(f"{OUTPUT_DIR}/reference_index.csv", index=False)

    print(f"[OK] Index FAISS construit avec {index.ntotal} images de référence.")
    print(f"Fichiers sauvegardés : {OUTPUT_DIR}/faiss.index, {OUTPUT_DIR}/reference_index.csv")


if __name__ == "__main__":
    main()
