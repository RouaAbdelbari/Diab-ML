"""
src/training.py
---------------
Entraînement et comparaison équitable de 4 modèles avec :
  - même preprocessing pour tous
  - SMOTEENN intégré dans le pipeline imblearn (pas avant CV)
  - RandomizedSearchCV / GridSearchCV avec StratifiedKFold
  - prédictions OOF pour sélection du threshold
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    StratifiedKFold, RandomizedSearchCV, cross_val_predict
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN

from src.preprocessing import build_preprocessing_pipeline, FeatureEngineer, FEATURE_NAMES

# ─── Seed global ────────────────────────────────────────────────────────────
RANDOM_STATE = 42
# FIX 2: equal budget for all models, 10-fold as stated in the paper
CV_FOLDS = 10
N_ITER_ALL = 50  # identical for every model — auditable parity

# ─── Espaces de recherche d'hyperparamètres ──────────────────────────────────
# ─── Espaces de recherche d'hyperparamètres ──────────────────────────────────
# Préfixe "model__" car le nom du step dans ImbPipeline est "model"
PARAM_GRIDS = {
    'Logistic Regression': {
        'model__C':        [0.01, 0.1, 1, 10, 100],
        'model__penalty':  ['l1', 'l2'],
        'model__solver':   ['liblinear'],
        'model__max_iter': [2000],
    },
    'SVM (RBF)': {
        'model__C':      [0.1, 1, 10, 100],
        'model__gamma':  ['scale', 'auto', 0.001, 0.01],
        'model__kernel': ['rbf'],
    },
    'XGBoost': {
        'model__n_estimators':     [100, 200, 300],
        'model__max_depth':        [3, 5, 7],
        'model__learning_rate':    [0.01, 0.05, 0.1, 0.2],
        'model__subsample':        [0.7, 0.8, 1.0],
        'model__colsample_bytree': [0.7, 0.8, 1.0],
        'model__min_child_weight': [1, 3, 5],
    },
    'Random Forest': {
        'model__n_estimators':      [200, 300, 500],
        'model__max_depth':         [10, 15, 20, None],
        'model__min_samples_split': [2, 5, 10],
        'model__class_weight':      ['balanced', None],
        'model__max_features':      ['sqrt', 'log2'],
    },
}

N_ITER = {
    'Logistic Regression': N_ITER_ALL,
    'SVM (RBF)':           N_ITER_ALL,
    'XGBoost':             N_ITER_ALL,
    'Random Forest':       N_ITER_ALL,
}


def _build_model_instance(name: str):
    """Retourne une instance de modèle non-entraînée."""
    if name == 'Logistic Regression':
        return LogisticRegression(random_state=RANDOM_STATE)
    elif name == 'SVM (RBF)':
        return SVC(probability=True, random_state=RANDOM_STATE)
    elif name == 'XGBoost':
        return XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            verbosity=0,
            use_label_encoder=False,
        )
    elif name == 'Random Forest':
        return RandomForestClassifier(random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Modèle inconnu : {name}")


def build_full_pipeline(model_name: str):
    """
    Construit un pipeline imblearn complet APLATI :
        imputer → feature_eng → scaler → SMOTEENN → model

    SMOTEENN est à l'intérieur du pipeline, donc il est appliqué
    UNIQUEMENT sur les données d'entraînement de chaque fold.

    ATTENTION : imblearn.Pipeline ne supporte pas de sklearn.Pipeline
    imbriqué comme step intermédiaire. On aplatit donc toutes les étapes.
    """
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import RobustScaler

    smoteenn = SMOTEENN(random_state=RANDOM_STATE)
    model = _build_model_instance(model_name)

    return ImbPipeline([
        ('imputer',     KNNImputer(n_neighbors=5)),
        ('feature_eng', FeatureEngineer()),
        ('scaler',      RobustScaler()),
        ('smoteenn',    smoteenn),
        ('model',       model),
    ])


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    scoring: str = 'roc_auc',
    verbose: int = 1,
) -> dict:
    """
    Recherche d'hyperparamètres pour un modèle donné via RandomizedSearchCV.

    Returns
    -------
    dict avec :
        best_pipeline   : pipeline complet avec les meilleurs hyperparamètres
        best_params     : dict des meilleurs paramètres
        best_cv_score   : meilleur score ROC-AUC en CV
        search_object   : objet RandomizedSearchCV complet
    """
    pipeline = build_full_pipeline(model_name)
    param_grid = PARAM_GRIDS[model_name]
    n_iter = N_ITER[model_name]

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=verbose,
        refit=True,   # refit automatiquement sur tout X_train avec best params
    )
    search.fit(X_train, y_train)

    if verbose >= 1:
        print(f"[{model_name}] Best CV {scoring}: {search.best_score_:.4f}")
        print(f"[{model_name}] Best params: {search.best_params_}")

    return {
        'best_pipeline': search.best_estimator_,
        'best_params':   search.best_params_,
        'best_cv_score': search.best_score_,
        'search_object': search,
        'model_name':    model_name,
        # audit trail: parity is verifiable from these fields alone
        'n_iter':        n_iter,
        'cv_folds':      cv.n_splits,
        'scoring':       scoring,
    }


def get_oof_probabilities(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> np.ndarray:
    """
    Génère des probabilités OOF (Out-Of-Fold) via cross_val_predict.
    
    IMPORTANT : le pipeline complet (preprocessing + SMOTEENN + model)
    est ré-entraîné indépendamment sur chaque fold d'entraînement.
    Ces probabilités OOF servent UNIQUEMENT à sélectionner le seuil τ*.
    Le test set n'est jamais touché ici.
    """
    oof_probs = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method='predict_proba',
        n_jobs=-1,
    )
    return oof_probs[:, 1]


def train_final_model(
    best_result: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> object:
    """
    Reconstruit et ré-entraîne le pipeline final sur l'intégralité de X_train.
    Le modèle final sauvegardé sera EXACTEMENT celui évalué sur le test set.
    """
    model_name   = best_result['model_name']
    best_params  = best_result['best_params']

    pipeline = build_full_pipeline(model_name)
    # Ne pas passer les params smoteenn ni imputer qui n'ont pas changé
    model_params = {k: v for k, v in best_params.items() if k.startswith('model__')}
    if model_params:
        pipeline.set_params(**model_params)
    pipeline.fit(X_train, y_train)

    return pipeline
