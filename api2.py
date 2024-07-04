# API

# Importation des librairies
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
import pandas as pd
import joblib
import os
import uvicorn

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Chemins relatifs pour les fichiers de modèle et de données
best_model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
explainer_path = os.path.join(BASE_DIR, "models", "explainer.pkl")
data_path = os.path.join(BASE_DIR, "data", "test_data.csv")

# Chargement du modèle et de l'explainer SHAP
model = joblib.load(best_model_path)
explainer = joblib.load(explainer_path)

# Chargement des données
df = pd.read_csv(data_path, index_col=0)


# Modèle de sortie de prédiction
class PredictionOutput(BaseModel):
    client_id: int
    probability: float
    class_prediction: str
    #shap_values: dict

# Endpoint de vérification de santé
# async défini une fonction asynchrone qui permet de gérer de nombreuses requêtes de manière efficace sans bloquer le thread principal
@app.get("/health")
async def health_check():
    return {"status": "ok"}


# Endpoint de prédiction
@app.get("/predict/{client_id}", response_model=PredictionOutput)
async def predict(client_id: int):
    try:
        # Recherche des données du client par ID
        if client_id not in df.index:
            raise HTTPException(status_code=404, detail="Client ID not found")

        # Extraction des données du client
        input_df = df.loc[[client_id]]

        # Prédiction
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_df)[:, 1][0] * 100
        else:
            proba = model.predict(input_df)[0] * 100

        pred = 0 if proba < 52 else 1

        # Calcul des valeurs SHAP pour l'entrée donnée
        #shap_values = explainer(input_df)
        #shap_values_dict = {f: v for f, v in zip(input_df.columns, shap_values.values[0])}

        # Réponse JSON
        response = {
            "client_id": client_id,
            "probability": round(proba, 2),
            "class_prediction": "Refusé" if pred == 1 else "Accepté"
            #, "shap_values": shap_values_dict
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
# L'API est démarrée uniquement si le script est exécuté directement
if __name__ == '__main__':
    #uvicorn.run("api2:app", reload=True, host="127.0.0.1", port=8000)
    uvicorn.run(app, host='127.0.0.1', port=8000)