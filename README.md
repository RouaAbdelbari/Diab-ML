# Integrated Business Intelligence and Machine Learning Framework for Diabetes Prediction and Clinical Decision Support

**AICCSA 2026 — Submitted Paper**

---

## Overview

This repository contains the full implementation of the framework presented in the paper:

> *Integrated Business Intelligence and Machine Learning Framework for Diabetes Prediction and Clinical Decision Support*

The system combines supervised machine learning (classification), unsupervised learning (patient segmentation), explainability (SHAP), and an interactive clinical dashboard (Streamlit).

---

## Dataset

**Pima Indians Diabetes Database** — 768 female patients, 8 clinical features, binary outcome (diabetic / non-diabetic).

Source: [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

Place the file at: `data/diabetes.csv`

---

## Project Structure

```
├── app.py                    # Streamlit dashboard
├── run_all.py                # Full reproducible pipeline (one command)
├── src/
│   ├── preprocessing.py      # KNNImputer + feature engineering + RobustScaler
│   ├── training.py           # 4-model training with equal search budget
│   ├── evaluation.py         # Metrics, figures, comparison table
│   ├── threshold.py          # τ* selection on OOF (Recall ≥ 0.75)
│   ├── clustering.py         # K-Means on real patient data
│   ├── explainability.py     # SHAP (TreeExplainer)
│   └── statistics.py         # Bootstrap CIs + DeLong AUC test
├── data/
│   └── diabetes.csv
├── models/                   # Saved after run_all.py
│   ├── final_model.pkl
│   ├── model_metadata.json
│   └── kmeans_artifacts.pkl
├── results/
│   ├── figures/              # All generated figures
│   ├── tables/               # CSV + LaTeX comparison tables
│   ├── metrics/              # Per-model metrics CSV
│   ├── statistics.json       # Bootstrap CIs + DeLong results
│   └── audit_report.md       # Scientific integrity checklist
├── requirements.txt
└── ml_pipeline.ipynb / ml_optimization.ipynb
```

---

## Methodology

### Models Compared (equal conditions)
| Model | Search Budget | CV |
|---|---|---|
| Logistic Regression | 50 iterations | 10-fold |
| SVM (RBF) | 50 iterations | 10-fold |
| XGBoost | 50 iterations | 10-fold |
| **Random Forest** | 50 iterations | 10-fold |

All models share the same preprocessing pipeline, same CV object, same scoring (`roc_auc`).

### Pipeline Architecture
```
Dataset
  └── Stratified 80/20 split
        └── [Train set only]
              ├── KNNImputer (n=5)
              ├── Feature Engineering (glucose_bmi, bmi_category, glucose_category)
              ├── RobustScaler
              ├── SMOTEENN (inside each CV fold only)
              └── RandomizedSearchCV (n_iter=50, 10-fold, roc_auc)
                    └── OOF probabilities → τ* selection
                          └── Final model trained on full train set
                                └── Test set evaluation (once)
```

### Threshold Selection
τ* is selected on **out-of-fold (OOF) predictions** only:

```
τ* = argmax F1(τ)   s.t.   Recall(τ) ≥ 0.75
     τ ∈ [0.01, 0.99]
```

The test set is never used for threshold selection.

### Primary Model Selection
Random Forest is declared as the primary model **a priori**, before any data is seen.  
Rationale: TreeSHAP (exact, O(TLD)) is required for real-time local explanations in the dashboard.

### Statistical Validation
- **Bootstrap 95% CIs**: 2000 stratified resamples
- **DeLong AUC test**: pairwise, primary model vs each other
- Finding: all models are statistically indistinguishable at n=154 test samples (all p > 0.05)

---

## Results

| Model | OOF ROC-AUC | τ* | Test Recall | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8449 | 0.57 | 0.7407 | 0.6154 | 0.7996 |
| SVM (RBF) | 0.8448 | 0.61 | 0.7778 | 0.6269 | 0.7987 |
| **Random Forest** | 0.8412 | **0.48** | **0.8333** | **0.6667** | 0.8182 |
| XGBoost | 0.8403 | 0.62 | 0.7963 | 0.6667 | 0.8269 |

*Primary model selected on deployment grounds, OOF ranking shown above.*

### Patient Segmentation (K-Means, K=4)
| Cluster | Label | N | Glucose | BMI | Diabetes Rate |
|---|---|---|---|---|---|
| 0 | Very High Risk | 138 | 160.0 | 39.4 | 74.6% |
| 1 | Moderate Risk | 230 | 105.5 | 35.1 | 24.4% |
| 2 | High Risk | 183 | 134.1 | 32.2 | 50.3% |
| 3 | Low Risk | 217 | 103.7 | 25.3 | 7.8% |

---

## Reproducing the Results

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python run_all.py
```

This single command:
- Loads and preprocesses the data
- Trains and tunes all 4 models (equal budget)
- Generates OOF predictions and selects τ*
- Evaluates on the test set (once)
- Runs statistical analysis (bootstrap + DeLong)
- Trains K-Means on real patient data
- Generates all SHAP figures
- Saves all artifacts and the audit report

### 3. Launch the dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Reproducibility

All random seeds are fixed at `42`. Results are fully reproducible with the specified package versions in `requirements.txt`.

Key packages:
- Python 3.12.2
- scikit-learn 1.4.2
- imbalanced-learn 0.14.2
- xgboost 3.4.1
- shap 0.49.1
- streamlit 1.56.0

---

## Scientific Integrity

The `results/audit_report.md` file documents:
- No data leakage (SMOTEENN inside CV folds only)
- Test set used exactly once
- Threshold selected on OOF, not test set
- Equal search budget across all models
- Model selected a priori, not on test performance
- K-Means trained on real data (not synthetic SMOTEENN samples)
- SHAP and dashboard use the same final model

---

## Dashboard Features

**Patient Space**
- Risk prediction with local SHAP explanation
- Personalized recommendations
- What-If interactive simulator
- Consultation history tracking

**Physician Space**
- Population BI dashboard (KPIs, distributions)
- Patient segmentation (K-Means clusters)
- Risk factor prevalence
- ML figures (ROC, PR, confusion matrix)
- High-risk patient table
- Rapid diagnostic tool

---

## Authors

University Research Project — AICCSA 2026
