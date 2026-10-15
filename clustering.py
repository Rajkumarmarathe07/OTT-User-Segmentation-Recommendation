
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1. Load preprocessed dataset
df = pd.read_csv("data/ott_users_preprocessed.csv")

print("Preprocessed dataset loaded:", df.shape)

# 2. Select features for clustering
X = df.drop(columns=["customer_id"])

# 3. Elbow Method
inertias = []

for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(X)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), inertias, marker="o")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.xticks(range(2, 11))
plt.tight_layout()
plt.savefig("graphs/elbow_method.png")
plt.show()

# 4. Train K-Means with 4 clusters initially
k = 4

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X)

# 5. Evaluate the model
score = silhouette_score(X, df["cluster"])

print("\nK-Means clustering completed!")
print("Number of clusters:", k)
print("Silhouette Score:", round(score, 4))

# 6. Count users in each cluster
print("\nUsers in each cluster:")
print(df["cluster"].value_counts().sort_index())

# 7. Visualize clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plot_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Cluster": df["cluster"].astype(str)
})

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=plot_df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="tab10",
    alpha=0.7
)

plt.title("OTT User Segmentation Using K-Means")
plt.tight_layout()
plt.savefig("graphs/kmeans_clusters.png")
plt.show()

# 8. Save cluster assignments
df.to_csv("data/ott_users_clustered.csv", index=False)

print("\nClustered dataset saved successfully!")
print("Graphs saved inside the graphs folder.")
