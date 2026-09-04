"""
src/threshold.py
----------------
Sélection du seuil optimal τ* sur prédictions Out-Of-Fold (OOF).

Équation :
    τ* = argmax_{τ} F1(τ)   s.t.   Recall(τ) ≥ min_recall

Le test set n'est JAMAIS utilisé ici.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = 0.75,
    step: float = 0.01,
) -> Dict:
    """
    Trouve le seuil optimal τ* maximisant F1 sous contrainte Recall ≥ min_recall.

    Parameters
    ----------
    y_true     : labels réels (0/1)
    y_prob     : probabilités prédites pour la classe positive
    min_recall : contrainte minimale sur le recall (défaut 0.75)
    step       : pas de recherche sur [0.01, 0.99]

    Returns
    -------
    dict avec clés :
        threshold, f1, recall, precision,
        constraint_satisfied, all_thresholds_df
    """
    thresholds = np.arange(0.01, 1.00, step)
    records = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        accuracy  = (tp + tn) / len(y_true)

        records.append({
            'threshold': round(t, 4),
            'precision': round(precision, 6),
            'recall':    round(recall,    6),
            'f1':        round(f1,        6),
            'accuracy':  round(accuracy,  6),
        })

    df = pd.DataFrame(records)

    # Filtrer les seuils qui respectent la contrainte recall ≥ min_recall
    admissible = df[df['recall'] >= min_recall]

    if len(admissible) == 0:
        # Aucun seuil ne satisfait la contrainte : on signale et on prend le max F1
        best_row = df.loc[df['f1'].idxmax()]
        constraint_satisfied = False
    else:
        best_row = admissible.loc[admissible['f1'].idxmax()]
        constraint_satisfied = True

    return {
        'threshold':             float(best_row['threshold']),
        'f1':                    float(best_row['f1']),
        'recall':                float(best_row['recall']),
        'precision':             float(best_row['precision']),
        'accuracy':              float(best_row['accuracy']),
        'constraint_satisfied':  constraint_satisfied,
        'min_recall_required':   min_recall,
        'all_thresholds_df':     df,
    }
