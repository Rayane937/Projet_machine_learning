from sklearn.metrics import adjusted_rand_score, silhouette_score
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN as SklearnDBSCAN

# 🟢 KMeans personnalisé
class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X):
        np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.cluster_centers_ = X[indices]

        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            if np.all(new_centers == self.cluster_centers_):
                break
            self.cluster_centers_ = new_centers

        self.labels_ = labels

# 🟢 DBSCAN personnalisé
class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        dbscan = SklearnDBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels_ = dbscan.fit_predict(X)
        print(f"Labels DBSCAN générés : {set(self.labels_)}")

# 🟢 Fonction pour les métriques avec gestion des erreurs
def show_metric(labels_true, labels_pred, descriptors, name_descriptor="", name_model="kmeans"):
    if labels_pred is None or len(labels_pred) == 0 or len(set(labels_pred)) <= 1:
        print(f"❌ Aucune cluster trouvé par {name_model} avec {name_descriptor}")
        return {"ami": 0, "ari": 0, "silhouette": 0, "homogeneity": 0, "completeness": 0, "v_measure": 0, "descriptor": name_descriptor, "name_model": name_model}

    homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
    ami = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    silhouette = silhouette_score(descriptors, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)

    return {"ami": ami, "ari": ari, "silhouette": silhouette, "homogeneity": homogeneity, "completeness": completeness, "v_measure": v_measure, "descriptor": name_descriptor, "name_model": name_model}
