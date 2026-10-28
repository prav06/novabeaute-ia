"""
API locale exposant le modèle au site web (frontend/novabeaute.html).

Usage :
    pip install -r requirements.txt
    uvicorn src.api:app --reload --port 8000

Le site web appelle ensuite http://localhost:8000/analyze avec l'image
en multipart/form-data.
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from src.similarity_search import SimilarityEngine

app = FastAPI(title="NovaBeauté API — prototype expérimental")

# CORS ouvert pour le développement local uniquement.
# En production, restreindre à l'origine exacte du site.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None  # chargé au démarrage pour ne pas recharger le modèle à chaque requête


@app.on_event("startup")
def load_model():
    global engine
    engine = SimilarityEngine()
    print("[OK] Modèle et index chargés.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    classification = engine.classify(image)
    similar = engine.find_similar(image, k=3)

    return {
        "classification": classification,
        "similar_cases": similar,
        "disclaimer": (
            "Résultat généré par un prototype expérimental, non validé "
            "cliniquement. Ne remplace pas un avis médical."
        ),
    }
