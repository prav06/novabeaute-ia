# NovaBeauté — Aide au repérage de grains de beauté par similarité d'image

**⚠️ Projet expérimental / portfolio.** Ce dépôt n'est **pas** un dispositif médical, n'a **pas** été validé cliniquement, et n'est **pas** destiné à un usage réel en cabinet médical. Il a été construit dans un but éducatif et de démonstration de compétences en data science / computer vision, à partir d'un jeu de données public.

## Le projet en une phrase

Un pipeline qui classe des photos de grains de beauté par type de lésion (bénin / à surveiller / suspect) et retrouve, dans une base de référence, les images les plus visuellement similaires à une photo donnée — avec une interface web de démonstration.

## Jeu de données

[HAM10000](https://doi.org/10.7910/DVN/DBW86T) ("Human Against Machine with 10000 training images"), Tschandl et al., Harvard Dataverse. 10 015 images dermoscopiques de lésions pigmentées, annotées par des dermatologues, réparties en 7 catégories :

| Code | Diagnostic | Nature |
|---|---|---|
| `nv` | Nævus mélanocytaire | Bénin |
| `mel` | Mélanome | Malin |
| `bkl` | Kératose bénigne (lentigo, kératose séborrhéique) | Bénin |
| `bcc` | Carcinome basocellulaire | Malin |
| `akiec` | Kératose actinique / maladie de Bowen | Pré-cancéreux |
| `vasc` | Lésion vasculaire | Bénin |
| `df` | Dermatofibrome | Bénin |

Licence : CC-BY-NC 4.0 — usage non commercial uniquement. Citez la source si vous réutilisez ce dépôt.

## Pourquoi ce dataset est plus piégeux qu'il n'y paraît

Deux problèmes structurels à connaître **avant** de toucher au code — je les documente ici parce qu'ils sont faciles à louper et qu'ils faussent complètement les résultats si on ne les traite pas :

1. **Déséquilibre des classes.** La classe `nv` représente à elle seule environ deux tiers des images. Un modèle entraîné naïvement obtient une bonne accuracy en prédisant quasiment toujours "nævus bénin" — ce qui le rend inutile, puisque c'est justement les cas minoritaires (mélanome) qui comptent cliniquement.
2. **Fuite de données (data leakage) via `lesion_id`.** Une même lésion physique est souvent photographiée plusieurs fois sous des angles différents. Si le split train/test se fait au niveau image et non au niveau lésion, des photos de la même lésion se retrouvent des deux côtés du split — le modèle "triche" en la reconnaissant, et les métriques de test sont artificiellement gonflées. **Le split doit être fait par `lesion_id`** (voir `src/split_data.py`).

## Structure du dépôt

```
novabeaute-ia/
├── data/                   # non versionné — voir instructions de téléchargement
├── src/
│   ├── download_data.py    # instructions de récupération du dataset
│   ├── eda.py               # exploration : distribution des classes, doublons de lésions
│   ├── split_data.py        # split train/val/test correct (par lesion_id)
│   ├── train_classifier.py  # entraînement d'un classifieur (transfer learning)
│   ├── build_embeddings.py  # calcul des embeddings + index FAISS
│   ├── similarity_search.py # recherche des k images les plus proches
│   └── api.py                # API FastAPI (classification + similarité)
├── frontend/
│   └── novabeaute.html      # interface web branchée sur l'API locale
├── requirements.txt
└── README.md
```

## Pipeline, étape par étape

1. `python src/download_data.py` — affiche les instructions de téléchargement (le dataset n'est pas inclus dans le dépôt pour des raisons de taille et de licence).
2. `python src/eda.py` — génère les graphiques d'exploration dans `outputs/eda/`.
3. `python src/split_data.py` — crée `data/splits/{train,val,test}.csv`, groupés par `lesion_id`.
4. `python src/train_classifier.py` — entraîne un ResNet50 pré-entraîné (fine-tuning), sauvegarde le meilleur modèle dans `outputs/model.pt` et les métriques dans `outputs/metrics.json`.
5. `python src/build_embeddings.py` — calcule un vecteur (embedding) par image d'entraînement et construit un index FAISS pour la recherche par similarité.
6. `python src/api.py` — lance une API locale (`http://localhost:8000`) exposant `/predict` (classification) et `/similar` (k plus proches voisins).
7. Ouvrir `frontend/novabeaute.html` dans un navigateur — l'interface envoie les photos à l'API locale et affiche les résultats.

## Résultats (à compléter après entraînement)

| Métrique | Valeur |
|---|---|
| Accuracy globale | — |
| F1-score classe `mel` (mélanome) | — |
| Rappel (sensibilité) classe `mel` | — |

> Le rappel sur la classe mélanome est la métrique la plus importante à surveiller ici : cliniquement, un faux négatif (mélanome classé bénin) est bien plus coûteux qu'un faux positif. L'accuracy globale seule masquerait ce problème vu le déséquilibre des classes.

## Limites connues

- Dataset majoritairement composé de peaux claires (populations européenne/australienne) — le modèle généraliserait mal sur d'autres teintes de peau. Un vrai déploiement nécessiterait un dataset plus représentatif.
- Pas de validation par un professionnel de santé sur les résultats produits par ce pipeline.
- Images dermoscopiques (prises avec un instrument grossissant/polarisant) vs. photos de smartphone : le modèle est entraîné sur les premières, ce qui peut réduire sa performance sur des photos classiques.

## Licence et usage

Projet éducatif. Dataset sous licence CC-BY-NC 4.0 (Tschandl et al., 2018). Code de ce dépôt libre de réutilisation à but non commercial, avec attribution.
