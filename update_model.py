import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN

# Charger les données
df = pd.read_csv("data/diabetes.csv")

# Remplacer les 0 par NaN sur les colonnes spécifiques
cols_to_nan = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_to_nan] = df[cols_to_nan].replace(0, np.nan)

# Séparation X et y
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split train/test
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Imputer avec KNNImputer (fit sur train uniquement)
imputer = KNNImputer(n_neighbors=5)
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X_train_raw.columns)

# Feature Engineering
def engineer_features(df_to_eng):
    df_new = df_to_eng.copy()
    df_new['glucose_bmi'] = df_new['Glucose'] * df_new['BMI']
    
    def categorize_bmi(bmi):
        if bmi < 25: return 0
        elif bmi < 30: return 1
        else: return 2
        
    def categorize_glucose(glu):
        if glu < 100: return 0
        elif glu < 126: return 1
        else: return 2
        
    df_new['bmi_category'] = df_new['BMI'].apply(categorize_bmi)
    df_new['glucose_category'] = df_new['Glucose'].apply(categorize_glucose)
    return df_new

X_train_eng = engineer_features(X_train_imputed)

# Scaler
scaler = RobustScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_eng), columns=X_train_eng.columns)

# SMOTEENN
smoteenn = SMOTEENN(random_state=42)
X_res, y_res = smoteenn.fit_resample(X_train_scaled, y_train)

# Grille de paramètres
param_dist = {
    'n_estimators': [200, 300, 500],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', None],
    'max_features': ['sqrt', 'log2']
}

# CV Aléatoire
rf = RandomForestClassifier(random_state=42)
rs = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, 
                        n_iter=50, cv=10, scoring='roc_auc', n_jobs=-1, random_state=42)
rs.fit(X_res, y_res)

best_model = rs.best_estimator_

print("Meilleurs paramètres :", rs.best_params_)

# Exporter best_model et scaler
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
    
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
print("Modèles exportés avec succès.")
