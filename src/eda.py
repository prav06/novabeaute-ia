"""
Exploration des données HAM10000.

Objectif : comprendre la distribution des classes et repérer le piège de
data leakage (lesion_id) AVANT de faire quoi que ce soit d'autre.

Usage : python src/eda.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

METADATA_PATH = "data/HAM10000_metadata.csv"
OUTPUT_DIR = "outputs/eda"

DX_LABELS = {
    "nv": "Nævus mélanocytaire (bénin)",
    "mel": "Mélanome (malin)",
    "bkl": "Kératose bénigne",
    "bcc": "Carcinome basocellulaire",
    "akiec": "Kératose actinique / Bowen",
    "vasc": "Lésion vasculaire",
    "df": "Dermatofibrome",
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(METADATA_PATH)

    print(f"Nombre total d'images : {len(df)}")
    print(f"Nombre de lésions uniques (lesion_id) : {df['lesion_id'].nunique()}")
    print(f"Ratio images / lésions : {len(df) / df['lesion_id'].nunique():.2f}")
    print()

    # --- 1. Distribution des classes ---
    counts = df["dx"].value_counts()
    pct = (counts / counts.sum() * 100).round(1)
    print("Distribution des classes :")
    for code, n in counts.items():
        print(f"  {code:6s} ({DX_LABELS.get(code, code):30s}) : {n:5d}  ({pct[code]}%)")

    plt.figure(figsize=(9, 5))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title("Distribution des classes — HAM10000")
    plt.ylabel("Nombre d'images")
    plt.xlabel("Classe (dx)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/class_distribution.png", dpi=150)
    plt.close()

    # --- 2. Vérification du piège lesion_id ---
    dup = df.groupby("lesion_id").size()
    multi_image_lesions = (dup > 1).sum()
    print(f"\nLésions photographiées plusieurs fois : {multi_image_lesions} "
          f"({multi_image_lesions / df['lesion_id'].nunique() * 100:.1f}% des lésions)")
    print("=> Le split train/test DOIT être fait par lesion_id, pas par image.")

    plt.figure(figsize=(7, 4))
    sns.histplot(dup, bins=range(1, dup.max() + 2), discrete=True)
    plt.title("Nombre d'images par lésion")
    plt.xlabel("Images pour une même lésion")
    plt.ylabel("Nombre de lésions")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/images_per_lesion.png", dpi=150)
    plt.close()

    # --- 3. Répartition démographique (utile pour discuter des biais) ---
    if "age" in df.columns:
        plt.figure(figsize=(7, 4))
        sns.histplot(df["age"].dropna(), bins=20)
        plt.title("Distribution des âges")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/age_distribution.png", dpi=150)
        plt.close()

    print(f"\nGraphiques sauvegardés dans {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
