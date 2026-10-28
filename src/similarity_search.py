"""
Fonctions de recherche par similarité — trouve les k images de la base
de référence les plus proches d'une image donnée.

Utilisé par api.py. Peut aussi être testé seul en ligne de commande :
    python src/similarity_search.py chemin/vers/photo.jpg
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import faiss

OUTPUT_DIR = "outputs"
CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
DX_LABELS = {
    "nv": "Nævus mélanocytaire", "mel": "Mélanome", "bkl": "Kératose bénigne",
    "bcc": "Carcinome basocellulaire", "akiec": "Kératose actinique / Bowen",
    "vasc": "Lésion vasculaire", "df": "Dermatofibrome",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SimilarityEngine:
    """Charge le modèle + l'index une seule fois, réutilisable pour plusieurs requêtes."""

    def __init__(self):
        self.model = self._load_embedding_model()
        self.classifier = self._load_classifier()
        self.index = faiss.read_index(f"{OUTPUT_DIR}/faiss.index")
        self.ref_df = pd.read_csv(f"{OUTPUT_DIR}/reference_index.csv")

    def _load_embedding_model(self):
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
        model.load_state_dict(torch.load(f"{OUTPUT_DIR}/model.pt", map_location=DEVICE))
        model.fc = nn.Identity()
        model.eval().to(DEVICE)
        return model

    def _load_classifier(self):
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
        model.load_state_dict(torch.load(f"{OUTPUT_DIR}/model.pt", map_location=DEVICE))
        model.eval().to(DEVICE)
        return model

    def _preprocess(self, pil_image):
        return tfms(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)

    def classify(self, pil_image):
        tensor = self._preprocess(pil_image)
        with torch.no_grad():
            logits = self.classifier(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        ranked = sorted(zip(CLASSES, probs), key=lambda x: -x[1])
        return [{"code": c, "label": DX_LABELS[c], "confidence": round(float(p) * 100, 1)} for c, p in ranked]

    def find_similar(self, pil_image, k=5):
        tensor = self._preprocess(pil_image)
        with torch.no_grad():
            embedding = self.model(tensor).cpu().numpy().astype("float32")
        faiss.normalize_L2(embedding)
        scores, indices = self.index.search(embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            row = self.ref_df.iloc[idx]
            results.append({
                "image_id": row["image_id"],
                "dx": row["dx"],
                "label": DX_LABELS.get(row["dx"], row["dx"]),
                "similarity": round(float(score) * 100, 1),
            })
        return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python src/similarity_search.py chemin/vers/image.jpg")
        sys.exit(1)

    engine = SimilarityEngine()
    image = Image.open(sys.argv[1])

    print("\n=== Classification ===")
    for r in engine.classify(image):
        print(f"  {r['label']:35s} {r['confidence']}%")

    print("\n=== Images les plus similaires ===")
    for r in engine.find_similar(image, k=5):
        print(f"  {r['image_id']}  {r['label']:30s} similarité: {r['similarity']}%")
