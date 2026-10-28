"""
Split train / val / test — GROUPÉ PAR lesion_id.

Pourquoi pas un simple train_test_split classique ?
Parce qu'une même lésion peut apparaître plusieurs fois dans le dataset
(photos sous plusieurs angles). Si on split au niveau image, une même
lésion peut se retrouver à la fois en train et en test : le modèle "a
déjà vu" cette lésion sous un autre angle, ses performances de test sont
alors artificiellement bonnes et ne reflètent pas sa vraie capacité de
généralisation.

On utilise donc GroupShuffleSplit (scikit-learn), qui garantit qu'un
même lesion_id ne se retrouve jamais dans deux splits différents.

Usage : python src/split_data.py
"""

import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

METADATA_PATH = "data/HAM10000_metadata.csv"
OUTPUT_DIR = "data/splits"
RANDOM_STATE = 42


def group_split(df, group_col, test_size, random_state):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx_a, idx_b = next(splitter.split(df, groups=df[group_col]))
    return df.iloc[idx_a].reset_index(drop=True), df.iloc[idx_b].reset_index(drop=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(METADATA_PATH)

    # 1) On sépare d'abord test (15%) du reste
    train_val, test = group_split(df, "lesion_id", test_size=0.15, random_state=RANDOM_STATE)

    # 2) Puis on sépare train (70% du total) et val (15% du total)
    #    -> 0.15 / 0.85 ≈ 0.176 pour obtenir 15% du total en val
    train, val = group_split(train_val, "lesion_id", test_size=0.176, random_state=RANDOM_STATE)

    # Vérification : aucun lesion_id ne doit apparaître dans plus d'un split
    sets = {"train": set(train.lesion_id), "val": set(val.lesion_id), "test": set(test.lesion_id)}
    overlap = (sets["train"] & sets["val"]) | (sets["train"] & sets["test"]) | (sets["val"] & sets["test"])
    assert len(overlap) == 0, f"FUITE DÉTECTÉE : {len(overlap)} lesion_id partagés entre splits !"
    print("[OK] Aucun chevauchement de lesion_id entre train/val/test.")

    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        path = f"{OUTPUT_DIR}/{name}.csv"
        split_df.to_csv(path, index=False)
        print(f"{name:5s} : {len(split_df):5d} images, {split_df['lesion_id'].nunique():5d} lésions -> {path}")
        print("        ", split_df["dx"].value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main()
