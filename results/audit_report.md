# Audit Report — AICCSA 2026 BI-ML Diabetes Pipeline

Run: 2026-08-17 22:47:51
Primary model    : **Random Forest** (declared a priori, OOF rank 3/4)
Threshold τ*     : **0.4800**  (search range [0.01,0.99], min_recall=0.75)
OOF ROC-AUC      : 0.8412
OOF F1 at τ*     : 0.6880
OOF Recall at τ* : 0.8037

## Audit Checklist

| Criterion                  | Status |
| -------------------------- | ------ |
| Data Leakage               | ✅ PASS |
| Preprocessing Consistency  | ✅ PASS |
| Cross-Validation           | ✅ PASS |
| Threshold Optimization     | ✅ PASS |
| Test-Set Isolation         | ✅ PASS |
| Model Selection (no leak)  | ✅ PASS |
| Search Budget Parity       | ✅ PASS |
| Model Reproducibility      | ✅ PASS |
| Dashboard Consistency      | ✅ PASS |
| K-Means Consistency        | ✅ PASS |
| SHAP Consistency           | ✅ PASS |
| Reproducibility            | ✅ PASS |

## Model Selection Rationale (FIX 1)

`PRIMARY_MODEL = 'Random Forest'` is declared at the top of run_all.py,
before any data is loaded. It was NOT chosen by inspecting test metrics.
Rationale: RandomForestClassifier supports TreeSHAP (exact, O(TLD))
required for real-time local explanations in the Streamlit dashboard.
Its OOF ROC-AUC rank: 3/4.
Test columns in the comparison table are for reporting only.

## Search Budget Parity (FIX 2)

All four models: `n_iter=50`, `cv=10-fold` StratifiedKFold,
`scoring='roc_auc'`, `random_state=42`.
Previous version had LR=20, SVM=20, XGB=30, RF=40 — that asymmetry is corrected.
Verifiable from `model_metadata.json` field `search_budget_parity`.

## Threshold Search Range

Previous version: τ ∈ [0.20, 0.70]. Current: τ ∈ [0.01, 0.99] step 0.01.
**Paper Equation (5) must be updated** to reflect τ ∈ [0.01, 0.99].

## OOF vs Test Recall Transfer Gap

| Model | OOF Recall | Test Recall | Gap |
| ----- | ---------- | ----------- | --- |
| Logistic Regression | 0.7523 | 0.7407 | -0.0116 |
| SVM (RBF) | 0.7897 | 0.7778 | -0.0119 |
| XGBoost | 0.7850 | 0.7963 | +0.0113 |
| Random Forest | 0.8037 | 0.8333 | +0.0296 |

## Confusion Matrix at Model-Specific τ* (Precision Transparency)

| Model | τ* | TP | FP | FN | TN | Precision | FP/100 neg |
| ----- | -- | -- | -- | -- | -- | --------- | ---------- |
| Logistic Regression | 0.57 | 40 | 36 | 14 | 64 | 0.5263 | 36.0 |
| SVM (RBF) | 0.61 | 42 | 38 | 12 | 62 | 0.5250 | 38.0 |
| XGBoost | 0.62 | 43 | 32 | 11 | 68 | 0.5733 | 32.0 |
| Random Forest | 0.48 | 45 | 36 | 9 | 64 | 0.5556 | 36.0 |

## OOF Ranking Table (model selection basis)

| Model               |   OOF ROC-AUC |   OOF F1 |   OOF Recall |   OOF Threshold |
|:--------------------|--------------:|---------:|-------------:|----------------:|
| Logistic Regression |        0.8449 |   0.6779 |       0.7523 |            0.57 |
| SVM (RBF)           |        0.8448 |   0.6787 |       0.7897 |            0.61 |
| Random Forest       |        0.8412 |   0.688  |       0.8037 |            0.48 |
| XGBoost             |        0.8403 |   0.6843 |       0.785  |            0.62 |

## Full Comparison Table (Test columns for reporting only)

| Model               |   OOF ROC-AUC |   OOF F1 |   OOF Recall |   OOF Threshold |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC |   Test PR-AUC |
|:--------------------|--------------:|---------:|-------------:|----------------:|----------------:|-----------------:|--------------:|----------:|---------------:|--------------:|
| Logistic Regression |        0.8449 |   0.6779 |       0.7523 |            0.57 |          0.6753 |           0.5263 |        0.7407 |    0.6154 |         0.7996 |        0.635  |
| SVM (RBF)           |        0.8448 |   0.6787 |       0.7897 |            0.61 |          0.6753 |           0.525  |        0.7778 |    0.6269 |         0.7987 |        0.6317 |
| Random Forest       |        0.8412 |   0.688  |       0.8037 |            0.48 |          0.7078 |           0.5556 |        0.8333 |    0.6667 |         0.8182 |        0.6939 |
| XGBoost             |        0.8403 |   0.6843 |       0.785  |            0.62 |          0.7208 |           0.5733 |        0.7963 |    0.6667 |         0.8269 |        0.7226 |

## Statistical Analysis (FIX 3)

Bootstrap 95% CIs (2000 stratified resamples) + DeLong pairwise AUC test.

### Bootstrap CIs (95%)

| Model | ROC-AUC | F1 | Recall | Precision |
| ----- | ------- | -- | ------ | --------- |
| Logistic Regression | 0.799 [0.725–0.864] | 0.615 [0.537–0.696] | 0.742 [0.630–0.852] | 0.527 [0.453–0.605] |
| SVM (RBF) | 0.799 [0.725–0.862] | 0.626 [0.552–0.701] | 0.778 [0.667–0.889] | 0.525 [0.456–0.600] |
| XGBoost | 0.826 [0.753–0.886] | 0.667 [0.584–0.739] | 0.797 [0.685–0.889] | 0.574 [0.500–0.653] |
| Random Forest | 0.818 [0.743–0.879] | 0.667 [0.596–0.735] | 0.835 [0.722–0.926] | 0.556 [0.487–0.627] |

### DeLong AUC Test (Random Forest vs others)

| Comparison | ΔAUC | 95% CI | p-value |
| ---------- | ---- | ------ | ------- |
| RF vs Logistic Regression | +0.0186 | [-0.0771–0.1143] | 0.7031 |
| RF vs SVM (RBF) | +0.0195 | [-0.0761–0.1152] | 0.6889 |
| RF vs XGBoost | -0.0086 | [-0.1006–0.0833] | 0.8544 |

**Finding**: All p-values >> 0.05. All AUC confidence intervals overlap. The four models are statistically indistinguishable at n=154 test samples.

## K-Means

n_clusters=4, n_init=10, seed=42. Silhouette=0.1695.
Trained on real patient data. Labels sorted by outcome rate (not cluster index).

| Cluster | Label | N | Glucose | BMI | Diabète % |
| ------- | ----- | - | ------- | --- | --------- |
| 0 | Very High Risk | 138 | 160.0 | 39.4 | 74.6% |
| 1 | Moderate Risk | 230 | 105.5 | 35.1 | 24.3% |
| 2 | High Risk | 183 | 134.1 | 32.2 | 50.3% |
| 3 | Low Risk | 217 | 103.7 | 25.3 | 7.8% |

## Package Versions

Python 3.12.2 | numpy 2.0.2 | pandas 2.2.2 | scikit-learn 1.4.2 | imbalanced-learn 0.14.2 | xgboost 3.4.1 | shap 0.49.1