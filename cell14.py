sns.set_theme(style="whitegrid")

# Fig 1 : ROC Comparaison
plt.figure(figsize=(8,6))
fpr, tpr, _ = roc_curve(y_test, y_probs)
auc_val = roc_auc_score(y_test, y_probs)
plt.plot(fpr, tpr, color='#d62728', lw=2, label=f'Random Forest Opt. (AUC = {auc_val:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Diagonale')
plt.xlabel('Taux Faux Positifs', fontsize=12)
plt.ylabel('Taux Vrais Positifs', fontsize=12)
plt.title('Courbe ROC', fontsize=14)
plt.legend(loc="lower right")
plt.savefig('outputs/fig1_roc_comparaison.png', dpi=300, bbox_inches='tight')
plt.close()

# Fig 2 : Confusion Matrix heatmap (valeurs + % )
plt.figure(figsize=(6,5))
y_pred_opt = (y_probs >= optimal_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_opt)
cat_labels = ['Vrai Neg', 'Faux Pos', 'Faux Neg', 'Vrai Pos']
counts = [f"{value}" for value in cm.flatten()]
percentages = [f"{value:.1%}" for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(cat_labels, counts, percentages)]
labels = np.asarray(labels).reshape(2,2)

sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', annot_kws={"size": 12})
plt.title(f'Matrice de Confusion', fontsize=14)
plt.ylabel('Vraie Classe')
plt.xlabel('Classe Prédite')
plt.savefig('outputs/fig2_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Fig 4 : CV Boxplots
plt.figure(figsize=(10,6))
cv_melt = cv_metrics.melt(var_name='Métrique', value_name='Score')
sns.boxplot(x='Métrique', y='Score', data=cv_melt, palette="coolwarm")
plt.axhline(y=0.80, color='black', linestyle='--', alpha=0.7, label='Objectif 0.80')
plt.title('Distribution des performances sur 10 folds', fontsize=14)
plt.legend()
plt.savefig('outputs/fig4_cv_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()
