# Interface de test scoring client avec Streamlit

import streamlit as st
import requests
from datetime import datetime, timedelta
from PIL import Image

# Charger l'image
image = Image.open("E:/0-PROFESSIONNEL/formation data scientist OpenClassrooms/P7/pret_a_depenser.png")

# Afficher l'image en haut à droite
st.image(image, use_column_width=True, clamp=True, width=200)


# API url en local ou sur le cloud à paramétrer
API_URL = "http://127.0.0.1:8000/"
#API_URL = "http://0.0.0.0:8000/"
#API_URL = "https://fastapi-cd.azurewebsites.net/"
# si cloud alors ajouter pillow==10.0.1 a requirements_api.txt

# Définir les options pour les menus déroulants avec une option vide par défaut
contract_types = [""] + ["Cash loans", "Revolving loans"]
genders = [""] + ["F", "M"]
family_statuses = [""] + ["Single / not married", "Married", "Civil marriage", "Widow", "Separated", "Unknown"]
children_count = [""] + list(range(10))
date = None

# Interface utilisateur avec Streamlit
st.title("Démonstration de scoring crédit")

# Couleurs pour les prédictions
CLASSES_COLORS = {'Accepté': 'green', 'Refusé': 'red'}
DEFAULT_COLOR = "gray"

# Sélection des valeurs par l'utilisateur
name_contract_type = st.selectbox("Type de Contrat", contract_types)
amt_credit = st.number_input("Montant du crédit", value=0.00, step=1000.00)
duration_credit = st.number_input("Durée du crédit (en année)", value=1, min_value=1, max_value=25, step=1)
birthday = st.date_input("Date de naissance", date)
amt_income = st.number_input("Montant des revenus (annuels)", value=0.00, step=1000.00)
code_gender = st.selectbox("Genre", genders)
name_family_status = st.selectbox("Statut Familial", family_statuses)
cnt_children = st.selectbox("Nombre d'enfants", children_count)

# Bouton pour obtenir la prédiction
if st.button("Obtenir la prédiction"):
    # Vérifier que toutes les sélections ont été faites
    if name_contract_type and amt_credit != 0 and duration_credit != 0 and amt_income != 0 and code_gender and name_family_status and cnt_children != "":
        # Calculer amt_annuity 
        amt_annuity = amt_credit / duration_credit
        # Calculer DAYS_BIRTH
        current_date = datetime.now().date()
        age = current_date.year - birthday.year - ((current_date.month, current_date.day) < (birthday.month, birthday.day))
        # Vérifier si l'âge est compris entre 18 et 70 ans inclus
        if 18 <= age <= 70:
            days_birth = -age * 365
        else:
            st.error("L'âge doit être compris entre 18 et 70 ans inclus.")
        
        # Préparer les données pour l'API
        input_data = {
            "NAME_CONTRACT_TYPE": name_contract_type,
            "AMT_CREDIT": amt_credit,
            "AMT_ANNUITY": amt_annuity,
            "DAYS_BIRTH": days_birth,
            "AMT_INCOME_TOTAL": amt_income,
            "CODE_GENDER": code_gender,
            "NAME_FAMILY_STATUS": name_family_status,
            "CNT_CHILDREN": int(cnt_children),
        }

        # Appel à l'API de prédiction
        try:
            response = requests.post(API_URL + "predict/", json=input_data)
            response_data = response.json()
            
            if response.status_code == 200:
                prediction = response_data["class_prediction"]
                color = CLASSES_COLORS[prediction]
                # Afficher la prédiction avec la couleur appropriée
                st.markdown(f'<p style="color:{color}; font-size:24px;">{prediction}</p>', unsafe_allow_html=True)
            else:
                st.error(f"Erreur de l'API : {response_data['detail']}")
        except Exception as e:
            st.error(f"Erreur de connexion : {str(e)}")
    else:
        st.warning("Veuillez remplir toutes les options avant de soumettre.")