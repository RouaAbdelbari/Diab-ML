# Grille de paramètres
param_dist = {
    'n_estimators': [200, 300, 500],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', None],
    'max_features': ['sqrt', 'log2']
}

# CV Stratifiée
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Recherche Aléatoire
rf = RandomForestClassifier(random_state=42)
rs = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, 
                        n_iter=50, cv=cv, scoring='roc_auc', n_jobs=-1, random_state=42)
rs.fit(X_train, y_train)

best_model = rs.best_estimator_
print("Meilleurs paramètres :", rs.best_params_)
