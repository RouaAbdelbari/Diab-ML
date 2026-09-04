"""
src/preprocessing.py
--------------------
Preprocessing pipeline : KNNImputer + feature engineering + RobustScaler.
Tous les objets sont fitté UNIQUEMENT sur le train set.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

# ── Colonnes originales du dataset Pima
ORIGINAL_FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# ── Colonnes après feature engineering (ordre exact requis)
FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
    'glucose_bmi', 'bmi_category', 'glucose_category'
]

# ── Colonnes avec zéros biologiquement impossibles → NaN
ZERO_TO_NAN_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


def replace_zeros_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace les 0 biologiquement impossibles par NaN."""
    df = df.copy()
    for col in ZERO_TO_NAN_COLS:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformateur sklearn pour le feature engineering.

    Features créées :
    - glucose_bmi    : Glucose × BMI  — interaction capture le risque
                       métabolique combiné (hyperglycémie + obésité).
    - bmi_category   : 0=normal(<25), 1=surpoids(25-30), 2=obèse(≥30).
                       Catégorisation clinique standard de l'OMS.
    - glucose_category : 0=normal(<100), 1=prédiabète(100-126), 2=diabète(≥126).
                         Seuils ADA (American Diabetes Association).
    """

    def fit(self, X, y=None):
        # Aucun paramètre à apprendre dans ce transformateur
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=ORIGINAL_FEATURES[:X.shape[1]])

        # Feature 1 : interaction multiplicative glucose × BMI
        df['glucose_bmi'] = df['Glucose'] * df['BMI']

        # Feature 2 : catégorie BMI (OMS)
        df['bmi_category'] = np.where(
            df['BMI'] < 25, 0,
            np.where(df['BMI'] < 30, 1, 2)
        )

        # Feature 3 : catégorie glucose (ADA)
        df['glucose_category'] = np.where(
            df['Glucose'] < 100, 0,
            np.where(df['Glucose'] < 126, 1, 2)
        )

        return df[FEATURE_NAMES].values


def build_preprocessing_pipeline() -> Pipeline:
    """
    Retourne le pipeline de preprocessing complet :
      KNNImputer(n_neighbors=5) → FeatureEngineer → RobustScaler

    À fitter UNIQUEMENT sur X_train (via pipeline.fit(X_train, y_train)).
    """
    return Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('feature_eng', FeatureEngineer()),
        ('scaler', RobustScaler()),
    ])
