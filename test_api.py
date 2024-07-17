# test unitaires

import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

# vérifier la santé de mon API
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    #assert response.json() == {"status": "ok"}
    assert response.json() == {"data": "Application ran successfully - FastAPI release v2.0"}

# vérifier que les données du client n° 214010 de mon Test Set donnent lieu à une prédiction
def test_predict_id_214010():
    input_data = {
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "F",
        "FLAG_OWN_CAR": "Y",
        "FLAG_OWN_REALTY": "Y",
        "CNT_CHILDREN": 0,
        "NAME_INCOME_TYPE": "Commercial associate",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Single / not married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "REGION_POPULATION_RELATIVE": 0.006852,
        "DAYS_BIRTH": -14878,
        "DAYS_EMPLOYED": -1141.0,
        "DAYS_REGISTRATION": -1610.0,
        "DAYS_ID_PUBLISH": -4546,
        "OWN_CAR_AGE": 11.0,
        "FLAG_MOBIL": 1,
        "FLAG_EMP_PHONE": 1,
        "FLAG_WORK_PHONE": 0,
        "FLAG_CONT_MOBILE": 1,
        "FLAG_PHONE": 0,
        "FLAG_EMAIL": 1,
        "OCCUPATION_TYPE": "Managers",
        "CNT_FAM_MEMBERS": 1.0,
        "REGION_RATING_CLIENT": 3,
        "ORGANIZATION_TYPE": "Business Entity",
        "EXT_SOURCE_1": 0.430827,
        "EXT_SOURCE_2": 0.425351,
        "EXT_SOURCE_3": 0.712155,
        "DAYS_LAST_PHONE_CHANGE": -1071.0,
        "FLAG_DOCUMENT_2": 0,
        "FLAG_DOCUMENT_3": 1,
        "FLAG_DOCUMENT_4": 0,
        "FLAG_DOCUMENT_5": 0,
        "FLAG_DOCUMENT_6": 0,
        "FLAG_DOCUMENT_7": 0,
        "FLAG_DOCUMENT_8": 0,
        "FLAG_DOCUMENT_9": 0,
        "FLAG_DOCUMENT_10": 0,
        "FLAG_DOCUMENT_11": 0,
        "FLAG_DOCUMENT_12": 0,
        "FLAG_DOCUMENT_13": 0,
        "FLAG_DOCUMENT_14": 0,
        "FLAG_DOCUMENT_15": 0,
        "FLAG_DOCUMENT_16": 0,
        "FLAG_DOCUMENT_17": 0,
        "FLAG_DOCUMENT_18": 0,
        "FLAG_DOCUMENT_19": 0,
        "FLAG_DOCUMENT_20": 0,
        "FLAG_DOCUMENT_21": 0,
        "DAYS_EMPLOYED_ANOM": False,
        "ACTIVE_CREDIT_DAY_OVERDUE_MAX": 0.0,
        "CLOSED_CREDIT_DAY_OVERDUE_MAX": 0.0,
        "AMT_CREDIT": 1262538.59,
        "AMT_ANNUITY": 48558.15,
        "AMT_INCOME_TOTAL": 247142.13,
        "OBS_30_CNT_SOCIAL_CIRCLE": 2.0,
        "DEF_30_CNT_SOCIAL_CIRCLE": None,
        "OBS_60_CNT_SOCIAL_CIRCLE": 2.0,
        "DEF_60_CNT_SOCIAL_CIRCLE": None,
        "ACTIVE_AMT_CREDIT_SUM_DEBT_SUM": None
    }
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    assert "class_prediction" in response.json()
    assert "probability" in response.json()
    assert "shap_values" in response.json()