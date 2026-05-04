cv_acc, cv_prec, cv_rec, cv_f1, cv_auc = [], [], [], [], []

# Evaluation sur l'ensemble entier X_res, y_res avec StratifiedKFold
for train_idx, val_idx in cv.split(X_res, y_res):
    X_f_train, X_f_val = X_res.iloc[train_idx], X_res.iloc[val_idx]
    y_f_train, y_f_val = y_res.iloc[train_idx], y_res.iloc[val_idx]
    
    model_cv = RandomForestClassifier(**rs.best_params_, random_state=42)
    model_cv.fit(X_f_train, y_f_train)
    
    probs = model_cv.predict_proba(X_f_val)[:, 1]
    preds = (probs >= optimal_threshold).astype(int)
    
    cv_acc.append(accuracy_score(y_f_val, preds))
    cv_prec.append(precision_score(y_f_val, preds))
    cv_rec.append(recall_score(y_f_val, preds))
    cv_f1.append(f1_score(y_f_val, preds))
    cv_auc.append(roc_auc_score(y_f_val, probs))

cv_metrics = pd.DataFrame({
    'Accuracy': cv_acc, 'Precision': cv_prec, 'Recall': cv_rec, 'F1': cv_f1, 'AUC': cv_auc
})

print("Résultats Cross-Validation (10-fold) avec seuil optimal :")
for col in cv_metrics.columns:
    print(f"{col}: {cv_metrics[col].mean():.3f} ± {cv_metrics[col].std():.3f}")
