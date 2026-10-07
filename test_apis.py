from fastapi.testclient import TestClient

from app_furniture import app as app_furniture
from app_iris import app as app_iris

# ---------- Clients de test ----------
client_f = TestClient(app_furniture)
client_i = TestClient(app_iris)


# ============================================================
#  TESTS — API Furniture (port 8080)
# ============================================================

def test_furniture_home():
    """Le endpoint racine doit renvoyer un message d'accueil."""
    response = client_f.get("/")
    assert response.status_code == 200
    assert "Furniture" in response.json()["message"]


def test_furniture_prediction_valid():
    """Une prédiction valide doit renvoyer un nombre."""
    payload = {
        "category": 5,
        "sellable_online": 1,
        "other_colors": 0,
        "depth": 60.0,
        "height": 80.0,
        "width": 120.0,
    }
    response = client_f.post("/make_predictions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))
    assert data["prediction"] > 0


def test_furniture_prediction_validation_error():
    """Une valeur invalide doit être rejetée par Pydantic (422)."""
    payload = {
        "category": "abc",  # ← invalide : devrait être un int
        "sellable_online": 1,
        "other_colors": 0,
        "depth": 60.0,
        "height": 80.0,
        "width": 120.0,
    }
    response = client_f.post("/make_predictions", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("category" in str(err["loc"]) for err in detail)


def test_furniture_prediction_missing_field():
    """Un champ manquant doit être rejeté (422)."""
    payload = {
        "category": 5,
        "sellable_online": 1,
        # "other_colors" manquant
        "depth": 60.0,
        "height": 80.0,
        "width": 120.0,
    }
    response = client_f.post("/make_predictions", json=payload)
    assert response.status_code == 422


# ============================================================
#  TESTS — API Iris (port 8081)
# ============================================================

def test_iris_home():
    """Le endpoint racine doit renvoyer un message d'accueil."""
    response = client_i.get("/")
    assert response.status_code == 200
    assert "Iris" in response.json()["message"]


def test_iris_setosa():
    """Setosa → classe 0."""
    payload = {"a": 5.1, "b": 3.5, "c": 1.4, "d": 0.2}
    response = client_i.post("/make_predictions", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == "0"


def test_iris_versicolor():
    """Versicolor → classe 1."""
    payload = {"a": 6.0, "b": 2.9, "c": 4.5, "d": 1.5}
    response = client_i.post("/make_predictions", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == "1"


def test_iris_virginica():
    """Virginica → classe 2."""
    payload = {"a": 6.5, "b": 3.0, "c": 5.2, "d": 2.0}
    response = client_i.post("/make_predictions", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == "2"


def test_iris_validation_error():
    """Une valeur non numérique doit être rejetée par Pydantic (422)."""
    payload = {"a": "abc", "b": 3.5, "c": 1.4, "d": 0.2}
    response = client_i.post("/make_predictions", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("a" in str(err["loc"]) for err in detail)