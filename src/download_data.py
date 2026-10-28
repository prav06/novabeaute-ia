"""
Instructions de récupération du dataset HAM10000.

Le dataset n'est pas inclus dans ce dépôt (taille + licence CC-BY-NC).
Deux sources possibles :

1. Kaggle (le plus simple)
   -> https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
   - Créer un compte Kaggle si besoin, télécharger le zip, le décompresser
     dans data/

2. Harvard Dataverse (source officielle)
   -> https://doi.org/10.7910/DVN/DBW86T
   - Télécharger les fichiers HAM10000_images_part_1.zip,
     HAM10000_images_part_2.zip, et HAM10000_metadata.csv

Structure attendue une fois les fichiers en place :

    data/
    ├── HAM10000_metadata.csv
    └── images/
        ├── ISIC_0024306.jpg
        ├── ISIC_0024307.jpg
        └── ...

Toutes les images des deux parties doivent être fusionnées dans un seul
dossier data/images/ (peu importe qu'elles viennent de part_1 ou part_2).

Astuce : si vous utilisez Kaggle, vous pouvez aussi télécharger directement
depuis un notebook Colab avec l'API Kaggle (kaggle.json), ce qui évite de
télécharger en local puis re-uploader.
"""

import os

EXPECTED_METADATA = "data/HAM10000_metadata.csv"
EXPECTED_IMAGES_DIR = "data/images"

if __name__ == "__main__":
    print(__doc__)

    if os.path.exists(EXPECTED_METADATA) and os.path.isdir(EXPECTED_IMAGES_DIR):
        n_images = len([f for f in os.listdir(EXPECTED_IMAGES_DIR) if f.endswith(".jpg")])
        print(f"[OK] Metadata trouvée, {n_images} images détectées dans {EXPECTED_IMAGES_DIR}/")
    else:
        print("[!] Données non trouvées. Suivez les instructions ci-dessus avant de continuer.")
