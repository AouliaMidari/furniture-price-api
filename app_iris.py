from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pickle

app = FastAPI(title="Iris Prediction API")

# ---------- Schéma d'entrée (Pydantic) ----------
class Iris(BaseModel):
    a: float
    b: float
    c: float
    d: float

# ---------- Chargement du modèle ----------
model = pickle.load(open('model_iris', 'rb'))

# ---------- Endpoints ----------
@app.get("/")
def home():
    return {"message": "ML model for Iris prediction"}

@app.post("/make_predictions")
async def make_predictions(features: Iris):
    prediction = model.predict([[
        features.a,
        features.b,
        features.c,
        features.d,
    ]])[0]
    return {"prediction": str(prediction)}

if __name__ == "__main__":
    uvicorn.run("app_iris:app", host="0.0.0.0", port=8081, reload=True)