import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score, roc_auc_score, accuracy_score, roc_curve, confusion_matrix
from imblearn.combine import SMOTEENN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import shap
import mlflow
import mlflow.sklearn

# Créer le dossier outputs
os.makedirs("outputs", exist_ok=True)

# Charger les données (assumant target = Outcome)
df = pd.read_csv("data/diabetes.csv")

# Remplacer les 0 par NaN sur les colonnes spécifiques
cols_to_nan = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_to_nan] = df[cols_to_nan].replace(0, np.nan)

# Imputer avec KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Feature Engineering
df_imputed['glucose_bmi'] = df_imputed['Glucose'] * df_imputed['BMI']

def categorize_bmi(bmi):
    if bmi < 25: return 0
    elif bmi < 30: return 1
    else: return 2

def categorize_glucose(glu):
    if glu < 100: return 0
    elif glu < 126: return 1
    else: return 2

df_imputed['bmi_category'] = df_imputed['BMI'].apply(categorize_bmi)
df_imputed['glucose_category'] = df_imputed['Glucose'].apply(categorize_glucose)

# Séparation X et y
X = df_imputed.drop('Outcome', axis=1)
y = df_imputed['Outcome']

# Scaler
scaler = RobustScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# SMOTEENN pour rééquilibrer
smoteenn = SMOTEENN(random_state=42)
X_res, y_res = smoteenn.fit_resample(X_scaled, y)

# Split stratifié
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)
print("Dimensions : Entraînement", X_train.shape, "Test", X_test.shape)
