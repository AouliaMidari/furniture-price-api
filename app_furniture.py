from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pickle

app = FastAPI(title="Furniture Price Prediction API")

# ---------- Chargement du modèle ----------
model = pickle.load(open('model.pkl', 'rb'))

# ---------- Schéma d'entrée (Pydantic) ----------
class FurnitureFeatures(BaseModel):
    category: int          # 0..16 (LabelEncoder)
    sellable_online: int   # 0 ou 1
    other_colors: int      # 0 ou 1
    depth: float
    height: float
    width: float

# ---------- Endpoints ----------
@app.get("/")
def home():
    return {"message": "ML model for Furniture price prediction"}

@app.post("/make_predictions")
async def make_predictions(features: FurnitureFeatures):
    data = [[
        features.category,
        features.sellable_online,
        features.other_colors,
        features.depth,
        features.height,
        features.width,
    ]]
    prediction = model.predict(data)[0]
    return {"prediction": float(prediction)}

if __name__ == "__main__":
    uvicorn.run("app_furniture:app", host="0.0.0.0", port=8080, reload=True)