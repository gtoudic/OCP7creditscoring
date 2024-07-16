# API

# Importation des librairies
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import joblib
import os
import uvicorn

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Chemins relatifs pour les fichiers de modèle et de données
prep_path = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
best_model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
explainer_path = os.path.join(BASE_DIR, "models", "explainer.pkl")
#data_path = os.path.join(BASE_DIR, "data", "test_brut_data.csv")
#data_path = os.path.join(BASE_DIR, "data", "test_data.csv")

# Chargement du modèle et de l'explainer SHAP
prep = joblib.load(prep_path)
model = joblib.load(best_model_path)
explainer = joblib.load(explainer_path)

# Définition manuelle du modèle Pydantic pour les données d'entrée
class DynamicClientInput(BaseModel):
    NAME_CONTRACT_TYPE: Optional[str] = None
    CODE_GENDER: Optional[str] = None
    FLAG_OWN_CAR: Optional[str] = None
    FLAG_OWN_REALTY: Optional[str] = None
    CNT_CHILDREN: Optional[int] = None
    NAME_INCOME_TYPE: Optional[str] = None
    NAME_EDUCATION_TYPE: Optional[str] = None
    NAME_FAMILY_STATUS: Optional[str] = None
    NAME_HOUSING_TYPE: Optional[str] = None
    REGION_POPULATION_RELATIVE: Optional[float] = None
    DAYS_BIRTH: Optional[int] = None
    DAYS_EMPLOYED: Optional[float] = None
    DAYS_REGISTRATION: Optional[float] = None
    DAYS_ID_PUBLISH: Optional[int] = None
    OWN_CAR_AGE: Optional[float] = None
    FLAG_MOBIL: Optional[int] = None
    FLAG_EMP_PHONE: Optional[int] = None
    FLAG_WORK_PHONE: Optional[int] = None
    FLAG_CONT_MOBILE: Optional[int] = None
    FLAG_PHONE: Optional[int] = None
    FLAG_EMAIL: Optional[int] = None
    OCCUPATION_TYPE: Optional[str] = None
    CNT_FAM_MEMBERS: Optional[float] = None
    REGION_RATING_CLIENT: Optional[int] = None
    ORGANIZATION_TYPE: Optional[str] = None
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None
    DAYS_LAST_PHONE_CHANGE: Optional[float] = None
    FLAG_DOCUMENT_2: Optional[int] = None
    FLAG_DOCUMENT_3: Optional[int] = None
    FLAG_DOCUMENT_4: Optional[int] = None
    FLAG_DOCUMENT_5: Optional[int] = None
    FLAG_DOCUMENT_6: Optional[int] = None
    FLAG_DOCUMENT_7: Optional[int] = None
    FLAG_DOCUMENT_8: Optional[int] = None
    FLAG_DOCUMENT_9: Optional[int] = None
    FLAG_DOCUMENT_10: Optional[int] = None
    FLAG_DOCUMENT_11: Optional[int] = None
    FLAG_DOCUMENT_12: Optional[int] = None
    FLAG_DOCUMENT_13: Optional[int] = None
    FLAG_DOCUMENT_14: Optional[int] = None
    FLAG_DOCUMENT_15: Optional[int] = None
    FLAG_DOCUMENT_16: Optional[int] = None
    FLAG_DOCUMENT_17: Optional[int] = None
    FLAG_DOCUMENT_18: Optional[int] = None
    FLAG_DOCUMENT_19: Optional[int] = None
    FLAG_DOCUMENT_20: Optional[int] = None
    FLAG_DOCUMENT_21: Optional[int] = None
    DAYS_EMPLOYED_ANOM: Optional[bool] = None
    #CREDIT_INCOME_PERCENT: Optional[float] = None
    #ANNUITY_INCOME_PERCENT: Optional[float] = None
    #CREDIT_TERM: Optional[float] = None
    #DAYS_EMPLOYED_PERCENT: Optional[float] = None
    ACTIVE_CREDIT_DAY_OVERDUE_MAX: Optional[float] = None
    CLOSED_CREDIT_DAY_OVERDUE_MAX: Optional[float] = None
    AMT_CREDIT: Optional[float] = None
    AMT_ANNUITY: Optional[float] = None
    AMT_INCOME_TOTAL: Optional[float] = None
    OBS_30_CNT_SOCIAL_CIRCLE: Optional[float] = None
    DEF_30_CNT_SOCIAL_CIRCLE: Optional[float] = None
    OBS_60_CNT_SOCIAL_CIRCLE: Optional[float] = None
    DEF_60_CNT_SOCIAL_CIRCLE: Optional[float] = None
    ACTIVE_AMT_CREDIT_SUM_DEBT_SUM: Optional[float] = None
    #AMT_CREDIT_log: Optional[float] = None
    #AMT_ANNUITY_log: Optional[float] = None
    #AMT_INCOME_TOTAL_log: Optional[float] = None
    #OBS_30_CNT_SOCIAL_CIRCLE_log: Optional[float] = None
    #DEF_30_CNT_SOCIAL_CIRCLE_log: Optional[float] = None
    #OBS_60_CNT_SOCIAL_CIRCLE_log: Optional[float] = None
    #DEF_60_CNT_SOCIAL_CIRCLE_log: Optional[float] = None
    #ACTIVE_AMT_CREDIT_SUM_DEBT_SUM_log: Optional[float] = None

# Chargement des données pour obtenir le format de données attendu
#df = pd.read_csv(data_path, index_col=0)
#fields = {
#    col: (Optional[int], None) if pd.api.types.is_integer_dtype(df[col]) else
#    (Optional[float], None) if pd.api.types.is_float_dtype(df[col]) else
#    (Optional[str], None) for col in df.columns
#}
# Création dynamique d'un modèle Pydantic basé sur les colonnes du DataFrame
#DynamicClientInput = create_model('DynamicClientInput', **fields)


# Modèle de sortie de prédiction
class PredictionOutput(BaseModel):
    probability: float
    class_prediction: str
    shap_values: dict

# Endpoint de vérification de santé
# async défini une fonction asynchrone qui permet de gérer de nombreuses requêtes de manière efficace sans bloquer le thread principal
@app.get("/health")
async def health_check():
    #return {"status": "ok"}
    return {"data": "Application ran successfully - FastAPI release v2.0"}

# Endpoint de prédiction
@app.post("/predict", response_model=PredictionOutput)
async def predict(data: DynamicClientInput):
    try:
        # Convertion des données d'entrée en df
        input_data = data.dict()
        input_df = pd.DataFrame([input_data])

        # Remplir les valeurs manquantes avec NaN si une variable est absente
        for col in input_df.columns:
            if pd.isnull(input_df[col].iloc[0]):
                input_df[col] = np.nan

        # Création des nouvelles variables
        input_df['CREDIT_INCOME_PERCENT'] = input_df['AMT_CREDIT'] / input_df['AMT_INCOME_TOTAL']
        input_df['ANNUITY_INCOME_PERCENT'] = input_df['AMT_ANNUITY'] / input_df['AMT_INCOME_TOTAL']
        input_df['CREDIT_TERM'] = input_df['AMT_ANNUITY'] / input_df['AMT_CREDIT']
        input_df['DAYS_EMPLOYED_PERCENT'] = input_df['DAYS_EMPLOYED'] / input_df['DAYS_BIRTH']
    
        # Variables pour la transformation logarithmique
        log_vars = [
        'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 
        'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 
        'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 
        'ACTIVE_AMT_CREDIT_SUM_DEBT_SUM'
        ]
        for var in log_vars:
            input_df[var + '_log'] = np.log1p(input_df[var])
        # Supprimer les variables transformées log
        input_df.drop(log_vars, axis=1, inplace=True)

        # Appliquer le préprocessing sur les données d'entrée
        input_preprocessed = prep.transform(input_df)

        # Prédiction
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_preprocessed)[:, 1][0] * 100
        else:
            proba = model.predict(input_preprocessed)[0] * 100

        pred = 0 if proba < 52 else 1

        # Calcul des valeurs SHAP
        input_preprocessed_df = pd.DataFrame(input_preprocessed, columns=prep.get_feature_names_out())
        shap_values = explainer.shap_values(input_preprocessed_df)
        shap_values_dict = {feature: shap_value for feature, shap_value in zip(input_preprocessed_df.columns, shap_values[1][0])}

        #shap_values = explainer(input_preprocessed_df)
        #shap_values_dict = {f: v for f, v in zip(input_preprocessed_df.columns, shap_values.values[0])}

        # Réponse JSON
        response = {
            "probability": round(proba, 2),
            "class_prediction": "Refusé" if pred == 1 else "Accepté"
            , "shap_values": shap_values_dict
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
# L'API est démarrée uniquement si le script est exécuté directement
if __name__ == '__main__':
    #uvicorn.run(app, host='127.0.0.1', port=8000)
    uvicorn.run(app, host='0.0.0.0', port=8000)