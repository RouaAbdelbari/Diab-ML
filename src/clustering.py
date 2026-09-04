"""
src/clustering.py
-----------------
K-Means sur les données patients RÉELLES (pas SMOTEENN synthétiques).
Le clustering BI représente la population réelle.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.impute import KNNImputer

from src.preprocessing import ZERO_TO_NAN_COLS, FEATURE_NAMES


def prepare_data_for_clustering(df_original: pd.DataFrame) -> tuple:
    """
    Prépare les données originales (non-synthétiques) pour le clustering.
    Applique imputation + feature engineering + scaling indépendamment.

    Returns
    -------
    X_scaled : np.ndarray prêt pour KMeans
    X_eng    : pd.DataFrame avec les features engineered (avant scaling)
    imputer  : fitted KNNImputer
    scaler   : fitted RobustScaler
    """
    df = df_original.copy()

    # Remplacement des 0 impossibles
    for col in ZERO_TO_NAN_COLS:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    # Features originales seulement (sans Outcome)
    X_raw = df.drop(columns=['Outcome'], errors='ignore')
    original_cols = X_raw.columns.tolist()

    # Imputation KNN
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_raw),
        columns=original_cols
    )

    # Feature engineering
    X_imputed['glucose_bmi'] = X_imputed['Glucose'] * X_imputed['BMI']
    X_imputed['bmi_category'] = np.where(
        X_imputed['BMI'] < 25, 0,
        np.where(X_imputed['BMI'] < 30, 1, 2)
    )
    X_imputed['glucose_category'] = np.where(
        X_imputed['Glucose'] < 100, 0,
        np.where(X_imputed['Glucose'] < 126, 1, 2)
    )

    X_eng = X_imputed[FEATURE_NAMES].copy()

    # Scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_eng)

    return X_scaled, X_eng, imputer, scaler


def select_k_kmeans(X_scaled: np.ndarray, k_range=range(2, 8),
                    save_path: str = None) -> pd.DataFrame:
    """
    Compare plusieurs valeurs de K avec Silhouette, Davies-Bouldin,
    Calinski-Harabasz pour justifier le choix de K.
    """
    scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        scores.append({
            'K': k,
            'Silhouette':          round(silhouette_score(X_scaled, labels),         4),
            'Davies-Bouldin':      round(davies_bouldin_score(X_scaled, labels),     4),
            'Calinski-Harabasz':   round(calinski_harabasz_score(X_scaled, labels),  4),
        })

    df_scores = pd.DataFrame(scores)
    print("\n=== Scores de sélection de K (K-Means) ===")
    print(df_scores.to_string(index=False))

    if save_path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(df_scores['K'], df_scores['Silhouette'], 'bo-')
        axes[0].set_title('Silhouette Score (↑ mieux)')
        axes[0].set_xlabel('K')
        axes[0].grid(alpha=0.3)

        axes[1].plot(df_scores['K'], df_scores['Davies-Bouldin'], 'ro-')
        axes[1].set_title('Davies-Bouldin (↓ mieux)')
        axes[1].set_xlabel('K')
        axes[1].grid(alpha=0.3)

        axes[2].plot(df_scores['K'], df_scores['Calinski-Harabasz'], 'go-')
        axes[2].set_title('Calinski-Harabasz (↑ mieux)')
        axes[2].set_xlabel('K')
        axes[2].grid(alpha=0.3)

        plt.suptitle('Sélection du nombre de clusters K', fontsize=13)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return df_scores


def train_kmeans(X_scaled: np.ndarray, n_clusters: int = 4) -> KMeans:
    """
    Entraîne le K-Means final sur données réelles.
    n_init=10 pour reproductibilité robuste.
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    return kmeans


def assign_cluster_labels(X_eng: pd.DataFrame, clusters: np.ndarray,
                          outcome: pd.Series = None) -> dict:
    """
    Attribue les labels de risque selon les caractéristiques RÉELLES des clusters.
    Les labels sont déterminés par les statistiques (glucose moyen, BMI, taux diabète),
    non par l'index du cluster.

    Returns
    -------
    dict : {cluster_id: label_string}
    """
    df_analysis = X_eng.copy()
    df_analysis['cluster'] = clusters
    if outcome is not None:
        df_analysis['outcome'] = outcome.values

    # Calcul des moyennes par cluster pour classer le risque
    agg = {}
    for k in sorted(df_analysis['cluster'].unique()):
        mask = df_analysis['cluster'] == k
        subset = df_analysis[mask]
        agg[k] = {
            'glucose_mean': subset['Glucose'].mean(),
            'bmi_mean':     subset['BMI'].mean(),
            'age_mean':     subset['Age'].mean(),
            'n':            mask.sum(),
            'outcome_rate': subset['outcome'].mean() if outcome is not None else np.nan,
        }

    # Trier les clusters par taux diabète (ou glucose si pas de outcome)
    sort_key = 'outcome_rate' if outcome is not None else 'glucose_mean'
    sorted_clusters = sorted(agg.keys(), key=lambda k: agg[k][sort_key])

    # Labels selon rang croissant de risque
    risk_labels = ['Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk']
    cluster_label_map = {}
    for i, ck in enumerate(sorted_clusters):
        label = risk_labels[min(i, len(risk_labels) - 1)]
        cluster_label_map[ck] = label

    print("\n=== Profils des clusters ===")
    for ck in sorted(agg.keys()):
        print(f"  Cluster {ck} → {cluster_label_map[ck]} | "
              f"Glucose={agg[ck]['glucose_mean']:.1f} | "
              f"BMI={agg[ck]['bmi_mean']:.1f} | "
              f"Age={agg[ck]['age_mean']:.1f} | "
              f"N={agg[ck]['n']} | "
              f"Diabète={agg[ck]['outcome_rate']:.2%}")

    return cluster_label_map, agg


def plot_kmeans_pca(X_scaled: np.ndarray, clusters: np.ndarray,
                   kmeans: KMeans, cluster_label_map: dict,
                   save_path: str):
    """Visualisation PCA 2D des clusters."""
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)
    centroids_pca = pca.transform(kmeans.cluster_centers_)

    n_clusters = len(set(clusters))
    colors = plt.colormaps['tab10'].resampled(n_clusters)

    fig, ax = plt.subplots(figsize=(9, 7))
    for k in sorted(set(clusters)):
        mask = clusters == k
        label = cluster_label_map.get(k, f'Cluster {k}')
        ax.scatter(components[mask, 0], components[mask, 1],
                   c=[colors(k)], alpha=0.6, s=20, label=label)

    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
               marker='X', s=200, c='black', zorder=5, label='Centroïdes')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.set_title('K-Means Clustering (PCA 2D) — Données patients réelles', fontsize=13)
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cluster_profiles(agg: dict, cluster_label_map: dict, save_path: str):
    """Barplot des profils des clusters."""
    rows = []
    for ck, stats in agg.items():
        rows.append({
            'Cluster': cluster_label_map.get(ck, f'C{ck}'),
            'Glucose Moyen': stats['glucose_mean'],
            'BMI Moyen': stats['bmi_mean'],
            'Taux Diabète (%)': stats['outcome_rate'] * 100,
        })
    df_plot = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    sns.barplot(data=df_plot, x='Cluster', y='Glucose Moyen',
                palette='YlOrRd', ax=axes[0])
    axes[0].set_title('Glucose moyen par cluster')

    sns.barplot(data=df_plot, x='Cluster', y='BMI Moyen',
                palette='YlOrRd', ax=axes[1])
    axes[1].set_title('BMI moyen par cluster')

    sns.barplot(data=df_plot, x='Cluster', y='Taux Diabète (%)',
                palette='YlOrRd', ax=axes[2])
    axes[2].set_title('Taux de diabète (%) par cluster')

    for ax in axes:
        ax.tick_params(axis='x', rotation=20)

    plt.suptitle('Profils des segments patients (K-Means)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
