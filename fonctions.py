import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, learning_curve
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, accuracy_score, precision_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer
import mlflow
import mlflow.sklearn
from mlflow import sklearn as mlflow_sklearn
from mlflow.models.signature import infer_signature



def outliers_z_score(data):
    """
    Recherche des outliers dans les variables numériques à partir du Z-Score.

    Paramètres :
    data (Series ou array-like) : Les données numériques à analyser pour les outliers.

    Retourne :
    Series : Les valeurs identifiées comme outliers.
    """
    data = pd.Series(data).dropna()
    mean = np.mean(data)
    std_dev = np.std(data)
    
    # Calcul des Z-Scores
    z_scores = (data - mean) / std_dev
    
    # Retourne les valeurs qui sont des outliers (Z-Score > 3 ou < -3)
    return data[np.abs(z_scores) > 3]


def apply_log_transform(df, columns_to_transform):
    """
    Applique une transformation logarithmique aux colonnes spécifiées d'un DataFrame
    et crée de nouvelles colonnes avec le suffixe '_log'.

    Parameters:
    df (pd.DataFrame): Le DataFrame sur lequel appliquer la transformation.
    columns_to_transform (list): Une liste des noms de colonnes à transformer.
    """
    for col in columns_to_transform:
        if col in df.columns:
            log_col_name = col + '_log'
            df[log_col_name] = df[col].apply(lambda x: np.log(x + 1) if x > 0 else np.nan)
        else:
            print(f"Warning: La colonne '{col}' n'existe pas dans le DataFrame.")

        
def missing_values(df):
        '''
        Fonction qui calcule les valeurs manquantes d'un dataset
        
        '''
        # Nombre total de valeurs manquantes
        mis_val = df.isna().sum()
        
        # Pourcentage de valeurs manquantes
        mis_val_percent =(df.isnull().sum() / len(df))*100
        
        # Construction d'un tableau avec les résultats
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Renommage des colonnes
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Valeurs manquantes', 1 : '% total des valeurs'})
        
        # On range de manière décroissante 
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% total des valeurs', ascending=False).round(1)
        
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns      


def description_dataset(data, name):
    '''
    Return the description of a dataset
    
        Parameters: 
            data: dataframe
            name: 
            
        Returns:
            the 3 first line of the dataframe
            the shape, the types, the number of null values,
            the number of unique values, the number of duplicated values
            
    '''
    print("On traite le dataset ", name)
    msno.bar(data)
    
    display(data.head(3))
    print(f'Taille :-------------------------------------------------------------- {data.shape}')
    
    print("--"*50)
    print("Valeurs manquantes par colonnes (%): ")
    table = missing_values(data)
    display(table.head(50))

    print("--"*50)
    print("Valeurs différentes par variables : ")
    for col in data:
        if data[col].nunique() < 30:
            print (f'{col :-<70} {data[col].unique()}')
        else : 
            print(f'{col :-<70} contient {data[col].nunique()} valeurs différentes')
    print("--"*50)
    print(f"Nombre de doublons : {data.duplicated().sum()}")


def business_cost(y_true, y_probs, threshold, cost_fn=10, cost_fp=1):
    """
    Calcule le coût d'affaires associé à la classification basée sur un seuil spécifique.

    Cette fonction renvoie le coût total associé à la classification effectuée selon le seuil donné.

    Paramètres:
        y_true (array-like): Les vraies étiquettes de classe.
        y_probs (array-like): Les probabilités prédites pour la classe positive.
        threshold (float): Le seuil de probabilité pour classifier les prédictions.
        cost_fn (int, optionnel): Coût d'un faux négatif.
        cost_fp (int, optionnel): Coût d'un faux positif.

    Retourne:
        float: Le coût total calculé.
    """
    
    # Calcul des prédictions binaires basées sur le seuil donné
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return fn * cost_fn + fp * cost_fp  


def business_cost_with_optimal_threshold(y_true, y_probs, cost_fn=10, cost_fp=1):
    """
    Détermine le seuil optimal pour la classification binaire basé sur le coût des faux négatifs et des faux positifs.

    Cette fonction calcule les coûts associés à différents seuils de décision et identifie le seuil qui minimise
    le coût total, combinant les coûts des faux négatifs et des faux positifs.

    Paramètres:
        y_true (array-like): Vecteur des vraies étiquettes de classe.
        y_probs (array-like): Vecteur des probabilités prédites pour la classe positive.
        cost_fn (int, optionnel): Coût d'un faux négatif. Valeur par défaut = 10.
        cost_fp (int, optionnel): Coût d'un faux positif. Valeur par défaut = 1.

    Retourne:
        tuple: Retourne un tuple contenant le seuil optimal et le coût associé à ce seuil.

    Calculs:
        1. Définit une série de seuils allant de 0 à 1.
        2. Calcule le coût pour chaque seuil en utilisant la fonction de coût définie précédemment.
        3. Identifie le seuil qui minimise le coût total.
    """
    thresholds = np.linspace(0, 1, 101)
    costs = [business_cost(y_true, y_probs, threshold, cost_fn, cost_fp) for threshold in thresholds]
    optimal_index = np.argmin(costs)
    optimal_threshold = thresholds[optimal_index]
    optimal_cost = costs[optimal_index]
    return optimal_threshold, optimal_cost


def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Trace la courbe d'apprentissage pour un estimateur donné afin d'évaluer son efficacité.

    Paramètres :
    estimator : objet estimateur utilisé pour les tests d'apprentissage et de validation.
    title : str, le titre du graphique.
    X : array-like, les données de caractéristiques.
    y : array-like, les données cible.
    cv : int, cross-validator ou un itérable, spécifie la stratégie de validation croisée.
    n_jobs : int ou None, le nombre de tâches parallèles à exécuter lors du calcul.
    train_sizes : array-like, les proportions de l'ensemble de données à générer des scores d'apprentissage.

    Cette fonction calcule les scores de l'estimateur pour différents ensembles de taille croissante,
    illustrant ainsi comment le modèle apprend progressivement à partir de l'ensemble de données.
    Les résultats sont représentés sous forme de graphique montrant les scores d'entraînement et de 
    validation croisée, avec leurs écarts-types, en fonction du nombre d'exemples d'entraînement.
    """
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    # Calcul des scores d'apprentissage et de validation croisée avec le scoring ROC AUC
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='roc_auc')
    # Calcul de la moyenne et de l'écart-type des scores d'entraînement et de test
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Configuration du graphique pour la visualisation des courbes
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.ylim(0, 1)
    plt.legend(loc="best")
    plt.show()
    plt.rcParams['figure.figsize'] = 10, 6


# Function to display confusion matrix
def display_confusion_matrix(y_test, y_pred):
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion matrix")
    plt.grid(visible=None)
    plt.show()


def compute_roc_curve(y_test, y_pred_proba, disp_best_th=False):
    # Compute metrics
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Create fig
    plt.figure(figsize=(6, 6))
    plt.title("ROC curve")

    # Display x=y
    sns.lineplot(x=[0, 1], y=[0, 1], linestyle='--')

    # Display ROC curve
    sns.lineplot(x=fpr, y=tpr, legend='brief', label="AUC = {:.3f}".format((auc)))

    if disp_best_th is True:
        # Compute best threshold
        o_tpr = tpr[np.argmin(np.abs(fpr+tpr-1))]
        o_fpr = fpr[np.argmin(np.abs(fpr+tpr-1))]
        o_threshold = thresholds[np.argmin(np.abs(fpr+tpr-1))]

        # Display best threshold
        sns.scatterplot(x=[o_fpr], y=[o_tpr], legend='brief', label="Best threshold = {:.3f}".format(o_threshold))

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()
    plt.rcParams['figure.figsize'] = 10, 6


def new_evaluate_model(model, model_name, X_train, y_train, X_test, y_test, scoring, stratified_kfold, param_grid=None, tag=None, model_alias=None):
    
    """
    Évalue et enregistre un modèle de machine learning en utilisant MLflow, en gérant les hyperparamètres,
    en effectuant une validation croisée et en enregistrant les résultats.

    Args:
        model (estimator): Modèle de machine learning à évaluer. Cela doit être un estimateur compatible scikit-learn.
        model_name (str): Nom sous lequel le modèle sera enregistré dans MLflow.
        X_train (DataFrame): Données d'entraînement, caractéristiques.
        y_train (Series): Données d'entraînement, cible.
        X_test (DataFrame): Données de test, caractéristiques.
        y_test (Series): Données de test, cible.
        scoring (dict or str): Métriques d'évaluation pour la validation croisée.
        stratified_kfold (StratifiedKFold): Méthode de division de validation croisée stratifiée.
        param_grid (dict, optional): Grille de paramètres pour la recherche d'hyperparamètres si spécifiée.
        tag (str, optional): Tag personnalisé à ajouter à la run de MLflow pour des informations supplémentaires.
        model_alias (str, optional): Alias à assigner à la version enregistrée du modèle pour une référence facile.

    Fonctionnalités:
        - Lance un run MLflow.
        - Entraîne le modèle avec des données d'entraînement, avec validation croisée stratifiée et recherche d'hyperparamètres. 
        - Calcule et enregistre des métriques comme le score F1 et l'AUCROC.
        - Trace les courbes d'apprentissage.
        - Retourne un DataFrame des résultats et un autre avec les prédictions des données de test.

    Returns:
        tuple: Contenant deux DataFrames, le premier avec les résultats des métriques d'évaluation et 
               le second avec les prédictions pour les données de test.
    """
    
    with mlflow.start_run(run_name=model_name) as run:
        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('classifier', model)
        ])
        
        # Utiliser HalvingGridSearchCV au lieu de GridSearchCV pour optimiser le processus de recherche d'hyperparamètres en réduisant le nombre d'évaluations nécessaires
        grid_search = HalvingGridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='roc_auc', cv=stratified_kfold, n_jobs=-1)
            
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time
        
        best_model = grid_search.best_estimator_
        cv_results = grid_search.cv_results_
        
        # Boucle pour obtenir les probabilités de prédiction pour chaque fold
        y_probs_list = []
        y_true_list = []
        for train_idx, valid_idx in stratified_kfold.split(X_train, y_train):
            pipeline.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            y_probs_list.append(pipeline.predict_proba(X_train.iloc[valid_idx])[:, 1])
            y_true_list.append(y_train.iloc[valid_idx])
        
        y_probs = np.concatenate(y_probs_list)
        y_true = np.concatenate(y_true_list)
        
        # Calcul du seuil optimal et du coût métier optimal
        optimal_threshold, optimal_cost = business_cost_with_optimal_threshold(y_true, y_probs)
        mlflow.log_metric('optimal_threshold', optimal_threshold)
        mlflow.log_metric('optimal_cost', optimal_cost)
        
        # Prédictions et calcul des métriques
        y_train_pred = best_model.predict(X_train)
        y_train_proba = best_model.predict_proba(X_train)[:, 1]
        y_test_pred = best_model.predict(X_test)
        y_test_proba = best_model.predict_proba(X_test)[:, 1]
        
        train_roc_auc = roc_auc_score(y_train, y_train_proba)
        test_roc_auc = roc_auc_score(y_test, y_test_proba)
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        # Display confusion matrix
        display_confusion_matrix(y_test, y_test_pred)

        # Draw roc curve
        compute_roc_curve(y_test, y_test_proba)
    
        # Logging des métriques
        mlflow.log_metric("train_roc_auc", train_roc_auc)
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("training_time", training_time)
        
        # Logging des hyperparamètres
        mlflow.log_params(grid_search.best_params_)
        
        if tag:
            mlflow.set_tag("custom_tag", tag)
        
        # Utilisation d'un sous-échantillon pour l'inférence de la signature car c'est très long
        X_sample = X_train.sample(n=1000, random_state=42) if len(X_train) > 1000 else X_train
        signature = mlflow.models.infer_signature(X_sample, best_model.predict(X_sample))
        mlflow.sklearn.log_model(best_model, "model", signature=signature)
   
        # Enregistrer le modèle (= sérialiser et stocker) dans le registre de modèles (MLflow Model Registry)
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, model_name)
    
        #Affichage des résultats
        print(f"{model_name} Results:")
        print(f"Train AUC: {train_roc_auc}")
        print(f"Test AUC: {test_roc_auc}")
        print(f"Optimal Threshold: {optimal_threshold}")
        print(f"Optimal Business Cost: {optimal_cost}")
        print(f"Training Time: {training_time}")
        
        # Courbes d'apprentissage
        plot_learning_curve(best_model, f"Learning Curve ({model_name})", X_train, y_train, cv=stratified_kfold, n_jobs=-1)
        
        # Enregistrement des résultats dans un DataFrame
        results = pd.DataFrame({
            'Model': [model_name],
            'Train AUC': [train_roc_auc],
            'Test AUC': [test_roc_auc],
            'Optimal Threshold': [optimal_threshold],
            'Optimal Business Cost': [optimal_cost],
            'Training Time': [training_time]
        })
    
        # DataFrame avec les prédictions et les ID des clients (index de X_test)
        test_results = pd.DataFrame({'prediction': y_test_proba}, index=X_test.index) 

        # Résultats de test
        print("Prédictions pour les données de test:")
        print(test_results.head())
        
        return results, test_results
    

