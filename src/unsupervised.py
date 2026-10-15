import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage
from src.config import SEED, FIGURES_DIR

def run_unsupervised_analysis(X_train_transformed, y_train):
    inertias = []
    silhouettes = []
    k_range = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        km.fit(X_train_transformed)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_train_transformed, km.labels_))
    plot_elbow_and_silhouette(k_range, inertias, silhouettes)
    best_k = int(list(k_range)[int(np.argmax(silhouettes))])
    best_km = KMeans(n_clusters=best_k, random_state=SEED, n_init=10)
    km_labels = best_km.fit_predict(X_train_transformed)
    plot_dendrogram(X_train_transformed[:100])
    pca = PCA(random_state=SEED)
    pca.fit(X_train_transformed)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_comp_90 = int(np.argmax(cum_var >= 0.90) + 1)
    plot_pca_variance(cum_var, n_comp_90)
    pca_2d = PCA(n_components=2, random_state=SEED)
    X_pca_2d = pca_2d.fit_transform(X_train_transformed)
    plot_pca_projection(X_pca_2d, y_train, km_labels)
    ari = round(float(adjusted_rand_score(y_train, km_labels)), 4)
    nmi = round(float(normalized_mutual_info_score(y_train, km_labels)), 4)
    return {
        "best_k": best_k,
        "n_components_90_var": n_comp_90,
        "ari": ari,
        "nmi": nmi
    }

def plot_elbow_and_silhouette(k_range, inertias, silhouettes):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(k_range, inertias, "b-o")
    ax1.set_xlabel("Number of clusters k")
    ax1.set_ylabel("Inertia", color="b")
    ax2 = ax1.twinx()
    ax2.plot(k_range, silhouettes, "r--s")
    ax2.set_ylabel("Silhouette coefficient", color="r")
    plt.title("Choosing k: elbow method and silhouette coefficient")
    plt.savefig(FIGURES_DIR / "elbow_silhouette.png", bbox_inches="tight")
    plt.close()

def plot_dendrogram(sample_data):
    linked = linkage(sample_data, method="ward")
    plt.figure(figsize=(10, 5))
    dendrogram(linked, truncate_mode="lastp", p=20)
    plt.title("Hierarchical clustering dendrogram (Ward)")
    plt.xlabel("Sample indices / cluster sizes")
    plt.ylabel("Distance")
    plt.savefig(FIGURES_DIR / "dendrogram.png", bbox_inches="tight")
    plt.close()

def plot_pca_variance(cum_var, n_comp_90):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cum_var) + 1), cum_var, marker=".")
    plt.axhline(y=0.90, color="r", linestyle="--")
    plt.axvline(x=n_comp_90, color="r", linestyle="--")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title(f"PCA: {n_comp_90} components for 90% of variance")
    plt.savefig(FIGURES_DIR / "pca_variance.png", bbox_inches="tight")
    plt.close()

def plot_pca_projection(X_pca, y_true, km_labels):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap="coolwarm", alpha=0.6)
    ax1.set_title("PCA 2D - True classes")
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=km_labels, cmap="viridis", alpha=0.6)
    ax2.set_title("PCA 2D - K-means assignments")
    plt.savefig(FIGURES_DIR / "pca_clusters.png", bbox_inches="tight")
    plt.close()