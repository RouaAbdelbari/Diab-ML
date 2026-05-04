y_probs = best_model.predict_proba(X_test)[:, 1]
thresholds = np.arange(0.20, 0.71, 0.01)

f1_scores, recalls, precisions = [], [], []

for t in thresholds:
    y_pred_t = (y_probs >= t).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_t))
    recalls.append(recall_score(y_test, y_pred_t))
    precisions.append(precision_score(y_test, y_pred_t))

# Trouver le seuil optimal pour F1
opt_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[opt_idx]
optimal_f1 = f1_scores[opt_idx]
optimal_recall = recalls[opt_idx]
optimal_precision = precisions[opt_idx]

print(f"Seuil optimal : {optimal_threshold:.2f} (F1: {optimal_f1:.3f}, Recall: {optimal_recall:.3f}, Precision: {optimal_precision:.3f})")

# Préparation Fig 3
plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_scores, label='F1 Score', color='#2ca02c')
plt.plot(thresholds, recalls, label='Recall', color='#1f77b4')
plt.plot(thresholds, precisions, label='Precision', color='#ff7f0e')
plt.axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Optimal ({optimal_threshold:.2f})')
plt.title('Performance vs Seuil de décision', fontsize=14)
plt.xlabel('Seuil de décision', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('outputs/fig3_threshold_curve.png', dpi=300, bbox_inches='tight')
plt.show()
