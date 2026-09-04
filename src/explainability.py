"""
src/explainability.py
---------------------
SHAP calculé sur LE MÊME modèle final utilisé dans le dashboard.
Toutes les figures SHAP proviennent du même pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from src.preprocessing import FEATURE_NAMES


def compute_shap_values(final_pipeline, X_test_raw: pd.DataFrame):
    """
    Calcule les valeurs SHAP pour le modèle final.

    Le modèle final est un pipeline imblearn aplati.
    On applique les étapes preprocessing manuellement pour SHAP.

    Returns
    -------
    shap_values : np.ndarray (n_samples × n_features)
    X_test_transformed : pd.DataFrame (données transformées)
    explainer : objet SHAP
    """
    import shap

    # Transformer les données avec les étapes preprocessing du pipeline
    # Le pipeline aplati a : imputer → feature_eng → scaler → smoteenn → model
    X_imp = final_pipeline.named_steps['imputer'].transform(X_test_raw)
    X_eng = final_pipeline.named_steps['feature_eng'].transform(X_imp)
    X_scaled = final_pipeline.named_steps['scaler'].transform(X_eng)
    X_test_df = pd.DataFrame(X_scaled, columns=FEATURE_NAMES)

    # Extraire le modèle final
    model = final_pipeline.named_steps['model']
    model_type = type(model).__name__

    try:
        if model_type in ('RandomForestClassifier', 'XGBClassifier',
                          'GradientBoostingClassifier', 'ExtraTreesClassifier'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_df)
            # Pour RF, shap_values est une liste [class0, class1]
            if isinstance(shap_values, list):
                sv = shap_values[1]
            else:
                # XGBoost : shape (n, features) directement
                sv = shap_values
        else:
            # LogisticRegression, SVM : LinearExplainer ou KernelExplainer
            explainer = shap.LinearExplainer(model, X_test_df)
            sv = explainer.shap_values(X_test_df)
    except Exception:
        # Fallback : KernelExplainer (lent mais universel)
        predict_fn = lambda x: model.predict_proba(x)[:, 1]
        background = shap.sample(X_test_df, min(50, len(X_test_df)))
        explainer = shap.KernelExplainer(predict_fn, background)
        sv = explainer.shap_values(X_test_df)

    return sv, X_test_df, explainer


def plot_shap_summary(shap_values, X_test_df: pd.DataFrame,
                      model_name: str, save_dir: str):
    """Beeswarm summary plot global."""
    import shap

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_test_df,
        feature_names=FEATURE_NAMES,
        show=False, plot_size=None
    )
    plt.title(f'SHAP Summary — {model_name}', fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, 'shap_summary_beeswarm.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_shap_bar(shap_values, X_test_df: pd.DataFrame,
                  model_name: str, save_dir: str):
    """Bar plot importance globale SHAP."""
    import shap

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        shap_values, X_test_df,
        feature_names=FEATURE_NAMES,
        plot_type='bar', show=False
    )
    plt.title(f'SHAP Feature Importance — {model_name}', fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, 'shap_bar_global.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_shap_waterfall(shap_values, X_test_df: pd.DataFrame,
                        explainer, model_name: str,
                        patient_idx: int, save_dir: str):
    """Waterfall pour un patient exemple."""
    import shap

    try:
        # Gérer les deux formats de shap_values (list pour RF, array pour XGB)
        if isinstance(shap_values, list):
            # RF : liste [class0_array, class1_array], chaque array (n_samples, n_features)
            sv_all = shap_values[1]
        else:
            # XGB : array (n_samples, n_features) ou (n_samples, n_features, n_classes)
            if len(np.array(shap_values).shape) == 3:
                sv_all = np.array(shap_values)[:, :, 1]
            else:
                sv_all = np.array(shap_values)

        sv_patient = np.array(sv_all)[patient_idx]

        if hasattr(explainer, 'expected_value'):
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                base = float(ev[1]) if len(ev) > 1 else float(ev[0])
            else:
                base = float(ev)
        else:
            base = 0.0

        explanation = shap.Explanation(
            values=sv_patient,
            base_values=base,
            data=X_test_df.iloc[patient_idx].values,
            feature_names=FEATURE_NAMES,
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(explanation, show=False)
        plt.title(f'SHAP Waterfall — Patient {patient_idx} — {model_name}', fontsize=12)
        plt.tight_layout()
        path = os.path.join(save_dir, f'shap_waterfall_patient{patient_idx}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING: Waterfall non généré : {e}")
