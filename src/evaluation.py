"""
src/evaluation.py
-----------------
Calcul des métriques sur le test set (utilisation unique).
Génère les tableaux de comparaison et les figures.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import os


def compute_metrics(y_true, y_pred, y_prob, prefix: str = '') -> dict:
    """
    Calcule toutes les métriques pour un modèle donné.
    y_pred doit déjà être calculé avec le seuil optimal (OOF).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        f'{prefix}Accuracy':    round(accuracy_score(y_true, y_pred),        4),
        f'{prefix}Precision':   round(precision_score(y_true, y_pred,        zero_division=0), 4),
        f'{prefix}Recall':      round(recall_score(y_true, y_pred,           zero_division=0), 4),
        f'{prefix}F1':          round(f1_score(y_true, y_pred,               zero_division=0), 4),
        f'{prefix}ROC_AUC':     round(roc_auc_score(y_true, y_prob),          4),
        f'{prefix}PR_AUC':      round(average_precision_score(y_true, y_prob),4),
        f'{prefix}Specificity': round(tn / (tn + fp) if (tn + fp) > 0 else 0, 4),
        f'{prefix}Sensitivity': round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4),
        f'{prefix}TN': int(tn), f'{prefix}FP': int(fp),
        f'{prefix}FN': int(fn), f'{prefix}TP': int(tp),
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str):
    """Sauvegarde la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = np.array([
        [f"TN\n{cm[0,0]}\n{cm[0,0]/cm.sum():.1%}",
         f"FP\n{cm[0,1]}\n{cm[0,1]/cm.sum():.1%}"],
        [f"FN\n{cm[1,0]}\n{cm[1,0]/cm.sum():.1%}",
         f"TP\n{cm[1,1]}\n{cm[1,1]/cm.sum():.1%}"],
    ])
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues',
                annot_kws={"size": 12}, ax=ax)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=13)
    ax.set_ylabel('Vraie Classe')
    ax.set_xlabel('Classe Prédite')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curves(results: dict, y_test: np.ndarray, save_path: str):
    """
    Courbe ROC multi-modèles.
    results : {model_name: {'y_prob': ..., 'color': ...}}
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        auc = roc_auc_score(y_test, res['y_prob'])
        ax.plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
                label=f'{name} (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Aléatoire')
    ax.set_xlabel('Taux de Faux Positifs', fontsize=12)
    ax.set_ylabel('Taux de Vrais Positifs', fontsize=12)
    ax.set_title('Courbes ROC — Comparaison des modèles', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pr_curves(results: dict, y_test: np.ndarray, save_path: str):
    """Courbes Precision-Recall multi-modèles."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    for i, (name, res) in enumerate(results.items()):
        precision, recall, _ = precision_recall_curve(y_test, res['y_prob'])
        ap = average_precision_score(y_test, res['y_prob'])
        ax.plot(recall, precision, lw=2, color=colors[i % len(colors)],
                label=f'{name} (AP={ap:.3f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Courbes Precision-Recall — Comparaison', fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_threshold_curve(all_thresh_df: pd.DataFrame, optimal_threshold: float,
                         model_name: str, save_path: str):
    """Courbe F1/Recall/Precision vs seuil avec marquage du seuil optimal."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(all_thresh_df['threshold'], all_thresh_df['f1'],
            label='F1 Score', color='#2ca02c', lw=2)
    ax.plot(all_thresh_df['threshold'], all_thresh_df['recall'],
            label='Recall', color='#1f77b4', lw=2)
    ax.plot(all_thresh_df['threshold'], all_thresh_df['precision'],
            label='Precision', color='#ff7f0e', lw=2)
    ax.axvline(x=optimal_threshold, color='red', linestyle='--', lw=2,
               label=f'τ* = {optimal_threshold:.2f}')
    ax.axhline(y=0.75, color='gray', linestyle=':', lw=1.5,
               label='Recall min = 0.75')
    ax.set_xlabel('Seuil de décision τ', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Performance vs Seuil — {model_name} (OOF)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cv_boxplot(cv_results_df: pd.DataFrame, save_path: str):
    """Boxplots des métriques sur les folds CV."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    melt = cv_results_df.melt(var_name='Métrique', value_name='Score')
    sns.boxplot(x='Métrique', y='Score', data=melt, palette='coolwarm', ax=ax)
    ax.axhline(y=0.80, color='black', linestyle='--', alpha=0.7,
               label='Objectif 0.80')
    ax.set_title('Distribution des performances — CV 5-fold (OOF)', fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def build_comparison_table(all_results: list) -> pd.DataFrame:
    """
    Construit le tableau de comparaison final.

    RÈGLE DE TRI : OOF ROC-AUC uniquement — jamais par une colonne Test.
    Les colonnes Test sont présentes pour reporting seulement.

    OOF F1 et OOF ROC-AUC sont passés depuis oof_results['thresh_result']
    via les champs 'oof_f1' et 'cv_roc_auc' dans all_results.
    """
    rows = []
    for r in all_results:
        rows.append({
            'Model':          r['model_name'],
            'OOF ROC-AUC':    round(r['cv_roc_auc'],              4),  # CV best score
            'OOF F1':         round(r.get('oof_f1', 0.0),         4),  # F1 at τ* on OOF
            'OOF Recall':     round(r.get('oof_recall', 0.0),     4),  # Recall at τ* on OOF
            'OOF Threshold':  round(r['threshold'],               4),
            # Test columns — for reporting only, never used for selection
            'Test Accuracy':  round(r['test_metrics'].get('Accuracy', 0),  4),
            'Test Precision': round(r['test_metrics'].get('Precision', 0), 4),
            'Test Recall':    round(r['test_metrics'].get('Recall', 0),    4),
            'Test F1':        round(r['test_metrics'].get('F1', 0),        4),
            'Test ROC-AUC':   round(r['test_metrics'].get('ROC_AUC', 0),   4),
            'Test PR-AUC':    round(r['test_metrics'].get('PR_AUC', 0),    4),
        })
    df = pd.DataFrame(rows)
    # Sort by OOF ROC-AUC — the only information visible before test set
    df = df.sort_values('OOF ROC-AUC', ascending=False).reset_index(drop=True)
    return df
