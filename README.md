# P7 - Implémentez un modèle de scoring

## Mission
L'organisme de financement " Prêt à dépenser" est spécialisé dans la distribution de crédit à la consommation. L'entreprise souhaite mettre en place un outil de scoring pour calculer la probabilité de défaut de paiement des clients et effectuer une classification automatique des demandes de prêt, les approuvant ou les rejetant selon les résultats obtenus afin de faciliter l’analyse des dossiers.
Pour assurer la compréhension des chargés d’études et la transparence des résultats, nous analyserons et expliquerons les features importances de manière globale et locale.  
Afin que notre modèle soit non seulement théorique mais également pratique, nous avons déployé ce modèle via une API. Nous avons également mis en place une interface pour tester cette API. 
Pour encadrer tout cela, nous avons adopté une approche globale MLOps. Cela commence par le suivi rigoureux de nos expérimentations jusqu'à l'analyse du data drift, en utilisant la librairie evidently. 

## Structure du répertoire

- `.github/workflows/`: Contient les fichiers de configuration pour l'intégration et le déploiement continu (CI/CD) avec GitHub Actions.
  
- `Data/`: Dossier pour les fichiers de données utilisés par l'API.
  - `test_data.csv`: Fichier de données.  
  
- `Models/`: Contient les modèles de machine learning sérialisés et d'autres objets relatifs aux modèles.
  - `best_model.pkl`: Le meilleur modèle entraîné pour le scoring de crédit.
  - `explainer.pkl`: explainer SHAP 
  - `shap_values.pkl`: Valeurs SHAP pré-calculées  
  
- `api.py`: Le script principal qui définit l'API FastAPI.

- `test_api.py`: Contient les tests unitaires pour vérifier les fonctionnalités de l'API.

- `requirements.txt`: Liste toutes les dépendances Python nécessaires pour exécuter le projet.


Les fichiers avec les packages utilisés pour l'ensemble du projet s'intitulent : requierements.txt.  

Lien vers l'API : 

Lien vers l'interface Streamlit : 

*Ce projet a été développé dans le cadre de la formation de Data Scientist chez OpenClassRooms. Actuellement, ce projet est fourni à des fins éducatives et de démonstration.*