from fastapi.testclient import TestClient
from api import app
import pandas as pd
import os

client = TestClient(app)

# Définir le répertoire de travail à la racine du projet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Chemin vers les données de test
data_path = os.path.join(BASE_DIR, "data", "test_data.csv")

def test_predict():
    # Chargement des données de test
    df = pd.read_csv(data_path, index_col=0)
    sample_data = df.iloc[0:1]  # Utilisation d'une seule ligne de données

    data_to_send = sample_data.to_dict(orient='records')[0]

    # Colonnes fournies pour vérification
    print("Provided features:", list(data_to_send.keys()))

    response = client.post("/predict", json=data_to_send)

    # Vérification du code de statut HTTP
    assert response.status_code == 200, f"Failed with response: {response.json()}"

    response_json = response.json()
    print("Response from API:", response_json)

    # Vérifications supplémentaires
    assert "client_id" in response_json, "Response missing client_id"
    assert "probability" in response_json, "Response missing probability"
    assert "class_prediction" in response_json, "Response missing class_prediction"
    assert "shap_values" in response_json, "Response missing shap_values"

    # Vérification des valeurs de prédiction
    assert isinstance(response_json["probability"], float), "Probability is not a float"
    assert response_json["probability"] >= 0, "Probability is less than 0"
    assert response_json["probability"] <= 100, "Probability is greater than 100"

    # Vérification de la classe prédite
    assert response_json["class_prediction"] in ["Refusé", "Accepté"], "Invalid class_prediction value"

    # Vérification des valeurs SHAP
    assert isinstance(response_json["shap_values"], dict), "SHAP values are not a dictionary"
    assert all(
        isinstance(value, float) for value in response_json["shap_values"].values()), "Not all SHAP values are floats"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

if __name__ == "__main__":
    import pytest
    pytest.main(["-v", "test_api.py"])