import os
import mlflow
import pathlib

# Résolution des erreurs d'artefacts avec la base SQLite sous Windows
tracking_uri = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(tracking_uri)

# Créer une expérience dédiée avec un chemin explicite et local vers les artefacts via pathlib (file:///)
exp_name = "Diabetes_Optimization"
artifact_path = pathlib.Path(os.path.abspath("mlruns")).as_uri()

exp = mlflow.get_experiment_by_name(exp_name)
if exp is None:
    mlflow.create_experiment(exp_name, artifact_location=artifact_path)
mlflow.set_experiment(exp_name)

# Terminer tout run existant pour le relancer proprement
if mlflow.active_run():
    mlflow.end_run()

with mlflow.start_run(run_name="OptimizedPipeline_Final"):
    # Paramètres
    mlflow.log_params(rs.best_params_)
    mlflow.log_param("knn_neighbors", 5)
    mlflow.log_param("smoteenn_random_state", 42)
    
    # Métriques
    mlflow.log_metric("threshold_optimal", optimal_threshold)
    mlflow.log_metric("f1_optimized", optimal_f1)
    mlflow.log_metric("recall_optimized", optimal_recall)
    mlflow.log_metric("precision_optimized", optimal_precision)
    mlflow.log_metric("auc_test", auc_val)
    mlflow.log_metric("silhouette_score_k4", sil_score)
    
    # Métriques issues de la cross-validation
    for col in cv_metrics.columns:
        mlflow.log_metric(f"CV_mean_{col}", cv_metrics[col].mean())
        mlflow.log_metric(f"CV_std_{col}", cv_metrics[col].std())
        
    # Modèle complet
    mlflow.sklearn.log_model(best_model, "RandomForest_Optimized")
    
    # Artefacts (Figures issues du dossier outputs)
    figs_to_log = [
        "shap_summary_beeswarm.png", "shap_bar_global.png", "shap_waterfall_patient0.png",
        "clustering_kmeans.png", "fig1_roc_comparaison.png", "fig2_confusion_matrix.png",
        "fig3_threshold_curve.png", "fig4_cv_boxplots.png", "fig5_shap_importance.png",
        "fig6_clusters.png"
    ]
    for fig in figs_to_log:
        f_path = os.path.join("outputs", fig)
        if os.path.exists(f_path):
            mlflow.log_artifact(f_path, artifact_path="figures")

print("Pipeline optimisé finalisé. MLflow log effectué avec succès !")
