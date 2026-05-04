kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_res)

sil_score = silhouette_score(X_res, clusters)
print(f"Silhouette Score : {sil_score:.3f}")

# PCA pour visualisation
pca = PCA(n_components=2)
components = pca.fit_transform(X_res)

plt.figure(figsize=(8,6))
scatter = plt.scatter(components[:, 0], components[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.title('PCA - Clustering K-Means (4 Clusters)', fontsize=14)
plt.xlabel('Composant Principal 1')
plt.ylabel('Composant Principal 2')
plt.colorbar(scatter, label='Cluster')

# Marquer les centroïdes estimatifs
centroids = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='red', label='Centroïdes')
plt.legend()
plt.savefig('outputs/clustering_kmeans.png', dpi=300, bbox_inches='tight')
plt.savefig('outputs/fig6_clusters.png', dpi=300, bbox_inches='tight')
plt.show()

# Profils par cluster
X_orig = pd.DataFrame(scaler.inverse_transform(X_res.copy()), columns=X_res.columns)
X_orig['Cluster'] = clusters
profils = X_orig.groupby('Cluster').mean()
print("Moyennes des features par cluster :\n", profils.round(2))
