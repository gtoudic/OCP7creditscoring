# Interface de test scoring client avec Streamlit

import streamlit as st
import requests

# API url en local ou sur le cloud à tester/paramétrer
#API_URL = "http://0.0.0.0:8000/"
#API_URL = "https://scoringapp-api.azurewebsites.net/"


# Définir les options pour les menus déroulants avec une option vide
contract_types = [""] + ["Cash loans", "Revolving loans"]
genders = [""] + ["F", "M"]
family_statuses = [""] + ["Single / not married", "Married", "Civil marriage", "Widow", "Separated", "Unknown"]
children_count = [""] + list(range(10))

# Interface utilisateur avec Streamlit
st.title("Démonstration de Scoring")

# Couleurs pour les prédictions
CLASSES_COLORS = {'Accepté': 'green', 'Refusé': 'red'}

# Sélection des valeurs par l'utilisateur
name_contract_type = st.selectbox("Type de Contrat", contract_types)
code_gender = st.selectbox("Genre", genders)
cnt_children = st.selectbox("Nombre d'enfants", children_count)
name_family_status = st.selectbox("Statut Familial", family_statuses)

# Bouton pour obtenir la prédiction
if st.button("Obtenir la prédiction"):
    # Vérifier que toutes les sélections ont été faites
    if name_contract_type and code_gender and cnt_children != "" and name_family_status:
        # Préparer les données pour l'API
        input_data = {
            "NAME_CONTRACT_TYPE": name_contract_type,
            "CODE_GENDER": code_gender,
            "CNT_CHILDREN": int(cnt_children),
            "NAME_FAMILY_STATUS": name_family_status,
        }

        # Appel à l'API de prédiction
        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
            response_data = response.json()
            
            if response.status_code == 200:
                prediction = response_data["class_prediction"]
                color = CLASSES_COLORS[prediction]
                # Afficher la prédiction avec la couleur appropriée
                st.markdown(f'<p style="color:{color};">La prédiction est : {prediction}</p>', unsafe_allow_html=True)
            else:
                st.error(f"Erreur de l'API : {response_data['detail']}")
        except Exception as e:
            st.error(f"Erreur de connexion : {str(e)}")
    else:
        st.warning("Veuillez remplir toutes les options avant de soumettre.")