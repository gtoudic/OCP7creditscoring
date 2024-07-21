# P7 - Implémentez un modèle de scoring

Déploiement d'un modèle de prédiction de défaut de paiement des clients et classification automatique des demandes de prêt en Accepté ou Refusé (+ éléments explicatifs, feature importance)  
  via une API,   
  avec approche globale MLOps,  
  avec une interface de test    


## Structure du repository

- `README.md` : fichier introductif permettant de comprendre l'objectif du projet et le découpage des dossiers

- `.github/workflows/`: Contient le fichier de configuration pour l'intégration et le déploiement continu (CI/CD) avec GitHub Actions.
  - prod.workflow.yml : Script pour déployer l'API sur Azure
  
- `Data/`: Dossier pour les fichiers de données utilisés par l'API.
  - `test_brut_data.csv`: 1000 premiers clients de mon Test set.  
  - `test_data.csv`: 1000 premiers clients de mon Test set.  
  
- `Models/`: Contient les modèles de machine learning sérialisés et d'autres objets relatifs aux modèles.
  - `preprocessor.pkl`: Préprocessing 
  - `best_model.pkl`: Le meilleur modèle entraîné pour le scoring de crédit.
  - `explainer.pkl`: explainer SHAP 

- `api.py`: Le script principal qui définit l'API FastAPI.

- `test_api.py`: Le script des tests unitaires pour vérifier les fonctionnalités de l'API.

- `requirements.txt`: Liste des pacakages/dépendances Python nécessaires pour exécuter le projet.
- `requirements_API.txt`: Liste des packages/dépendances Python nécessaires pour exécuter l'API.

- `Dockerfile` : processus de construction de l'image conteneur de l'API

- `dashboard.py`: Le script de l'application Streamlit.


Lien vers l'API : https://fastapi-cd.azurewebsites.net/docs#


*Ce projet a été développé dans le cadre du parcours de formation de Data Scientist d'OpenClassRooms.*  
*Actuellement, ce projet est fourni à des fins éducatives et de démonstration.*