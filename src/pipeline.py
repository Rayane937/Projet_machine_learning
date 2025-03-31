from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import pandas as pd
import numpy as np
from skimage import io
from skimage.transform import resize
from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN
from features import compute_hog_descriptors, compute_gray_histograms, compute_color_histograms, compute_cnn_descriptors, compute_lbp_descriptors
from clustering import KMeans, show_metric  
from utils import conversion_3d, create_df_to_export
from constant import PATH_OUTPUT, PATH_DATA
import hdbscan
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score

# Masquer les FutureWarnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def load_images_and_labels():
    images = []
    labels = []
    label_map = {}
    label_counter = 0

    test_path = os.path.join(PATH_DATA, 'test')
    for category in os.listdir(test_path):
        category_path = os.path.join(test_path, category)
        if os.path.isdir(category_path):
            if category not in label_map:
                label_map[category] = label_counter
                label_counter += 1
            for filename in os.listdir(category_path):
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    img = io.imread(os.path.join(category_path, filename))
                    img_resized = resize(img, (224, 224))
                    images.append(img_resized)
                    labels.append(label_map[category])

    return np.array(images), np.array(labels), label_map


#chercher les meilleurs parametres pour DBSCAN

def find_best_dbscan_params(X, labels_true, eps_list=None, min_samples_list=None, verbose=True):
    if eps_list is None:
        eps_list = [2.25,2.55,2.3,2.35]
    if min_samples_list is None:
        min_samples_list = list(range(1, 11))  

    best_score = -1
    best_params = {}
    results = []

    for eps in eps_list:
        for min_samples in min_samples_list:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)

            if len(set(labels)) <= 1 or len(set(labels)) == len(X):
                continue  # Pas de vrai clustering

            # Calcul des métriques
            ami = adjusted_mutual_info_score(labels_true, labels)
            ari = adjusted_rand_score(labels_true, labels)

            try:
                silhouette = silhouette_score(X, labels)
            except:
                silhouette = -1  # Silhouette échoue parfois (ex : un seul cluster sans bruit)

            # Score moyen pondéré
            avg_score = 0.6 * ami + 0.2 * ari + 0.2 * silhouette

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'ami': ami,
                'ari': ari,
                'silhouette': silhouette,
                'avg_score': avg_score,
                'clusters': n_clusters,
                'noise': n_noise
            })

            if avg_score > best_score:
                best_score = avg_score
                best_params = {
                    'eps': eps,
                    'min_samples': min_samples,
                    'ami': ami,
                    'ari': ari,
                    'silhouette': silhouette,
                    'avg_score': avg_score,
                    'labels': labels,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise
                }

    if verbose and best_params:
        print("\n✅ Meilleurs paramètres DBSCAN :")
        print(f"eps={best_params['eps']}, min_samples={best_params['min_samples']} → "
              f"Score={best_params['avg_score']:.3f} "
              f"(AMI={best_params['ami']:.3f}, ARI={best_params['ari']:.3f}, "
              f"Silhouette={best_params['silhouette']:.3f}), "
              f"clusters={best_params['n_clusters']}, noise={best_params['n_noise']}")

    return best_params, results
#chercher les meilleurs parametres pour HDBSCAN


def find_best_hdbscan_params(X, labels_true, min_cluster_size_list=None, min_samples_list=None, verbose=True):
    if min_cluster_size_list is None:
        min_cluster_size_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 70, 90, 100]
    if min_samples_list is None:
        min_samples_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]

    best_score = -1
    best_params = {}
    results = []

    for min_cluster_size in min_cluster_size_list:
        for min_samples in min_samples_list:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
            labels = clusterer.fit_predict(X)

            if len(set(labels)) <= 1 or len(set(labels)) == len(X):
                continue  # Pas de vrai clustering

            ami = adjusted_mutual_info_score(labels_true, labels)
            ari = adjusted_rand_score(labels_true, labels)
            try:
                silhouette = silhouette_score(X, labels)
            except:
                silhouette = 0  # Silhouette échoue si un seul cluster ou si cluster déséquilibré

            score_total = 0.4 * ami + 0.3 * ari + 0.3 * silhouette

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            results.append({
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'ami': ami,
                'ari': ari,
                'silhouette': silhouette,
                'score_total': score_total,
                'clusters': n_clusters,
                'noise': n_noise
            })

            if score_total > best_score:
                best_score = score_total
                best_params = {
                    'min_cluster_size': min_cluster_size,
                    'min_samples': min_samples,
                    'ami': ami,
                    'ari': ari,
                    'silhouette': silhouette,
                    'score_total': score_total,
                    'labels': labels,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise
                }

            if verbose:
                print(f"Test min_cluster_size={min_cluster_size}, min_samples={min_samples} → "
                      f"Score={score_total:.3f} (AMI={ami:.3f}, ARI={ari:.3f}, Silhouette={silhouette:.3f}), "
                      f"clusters={n_clusters}, noise={n_noise}")

    return best_params, results

def pipeline():
    print("Chargement des images...")
    images, labels_true, label_map = load_images_and_labels()

    print("\n- Extraction des descripteurs HOG...")
    descriptors_hog = compute_hog_descriptors(images)
    print("- Extraction des descripteurs Histogram...")
    descriptors_hist = compute_gray_histograms(images)
    print("- Extraction des descripteurs de couleur...")
    descriptors_color = compute_color_histograms(images, color_space='HSV')
    print("- Extraction des descripteurs CNN...")
    descriptors_cnn = compute_cnn_descriptors(images)
    print("- Extraction des descripteurs LBP...")
    descriptors_lbp = compute_lbp_descriptors(images)
    

    descriptors_hog = np.array(descriptors_hog)
    descriptors_hist = np.array(descriptors_hist)
    descriptors_color = np.array(descriptors_color)
    descriptors_cnn = np.array(descriptors_cnn)
    descriptors_lbp = np.array(descriptors_lbp)


    scaler = StandardScaler()
    descriptors_hog_norm = scaler.fit_transform(descriptors_hog)
    descriptors_hist_norm = scaler.fit_transform(descriptors_hist)
    descriptors_color_norm = scaler.fit_transform(descriptors_color)
    descriptors_cnn_norm = scaler.fit_transform(descriptors_cnn)
    descriptors_lbp_norm = scaler.fit_transform(descriptors_lbp)


    pca_hog = PCA(n_components=3)
    descriptors_hog_pca = pca_hog.fit_transform(descriptors_hog_norm)

    pca_hist = PCA(n_components=3)
    descriptors_hist_pca = pca_hist.fit_transform(descriptors_hist_norm)

    pca_color = PCA(n_components=3)
    descriptors_color_pca = pca_color.fit_transform(descriptors_color_norm)

    pca_cnn = PCA(n_components=3)
    descriptors_cnn_pca = pca_cnn.fit_transform(descriptors_cnn_norm)

    pca_lbp = PCA(n_components=3)
    descriptors_lbp_pca = pca_lbp.fit_transform(descriptors_lbp_norm)


    print("--------------------------------------------------------------------------")

    print("Shape avant export (HISTOGRAM):", descriptors_hist_pca.shape)  
    print("Shape avant export (HOG):", descriptors_hog_pca.shape)  
    print("Shape avant export (COLOR_HIST):", descriptors_color_pca.shape)  
    print("Shape avant export (CNN):", descriptors_cnn_pca.shape)
    print("Shape avant export (LBP):", descriptors_lbp_pca.shape)

    print("--------------------------------------------------------------------------")

    print("\n- KMeans avec HOG...")
    kmeans_hog = KMeans(n_clusters=20)
    kmeans_hog.fit(descriptors_hog_pca)

    print("- KMeans avec Histogram...")
    kmeans_hist = KMeans(n_clusters=20)
    kmeans_hist.fit(descriptors_hist_pca)

    print("- KMeans avec Couleur...")
    kmeans_color = KMeans(n_clusters=20)
    kmeans_color.fit(descriptors_color_pca)

    print("- KMeans avec CNN...")
    kmeans_cnn = KMeans(n_clusters=20)
    kmeans_cnn.fit(descriptors_cnn_pca)

    print("- KMeans avec LBP...")
    kmeans_lbp = KMeans(n_clusters=20)
    kmeans_lbp.fit(descriptors_lbp_pca)


    print("--------------------------------------------------------------------------")
    #pour determiner les meilleurs parametres DBSCAN, nous l'avons deja fait, on ne le refait pas a chaque fois pour que le programme se lance plus vite
    best_dbscan, all_results = find_best_dbscan_params( descriptors_cnn_pca, labels_true)
    #best_params, results = find_best_hdbscan_params(descriptors_lbp_pca, labels_true)


    print("\n- DBSCAN avec CNN...")
    dbscan_cnn = DBSCAN(eps=5, min_samples=11)
    dbscan_cnn.fit(descriptors_cnn_pca)

    print("\n- DBSCAN avec HOG...")
    dbscan_hog = DBSCAN(eps=1.2, min_samples=12)
    dbscan_hog.fit(descriptors_hog_pca)

    print("\n- DBSCAN avec Histogram...")
    dbscan_hist = DBSCAN(eps=3.2, min_samples=13)
    dbscan_hist.fit(descriptors_hist_pca)

    print("\n- DBSCAN avec couleur...")
    dbscan_color = DBSCAN(eps=0.5, min_samples=15)
    dbscan_color.fit(descriptors_color_pca)

    print("\n- DBSCAN avec LBP...")
    dbscan_lbp = DBSCAN(eps=0.7, min_samples=10)
    dbscan_lbp.fit(descriptors_lbp_pca)


    print("--------------------------------------------------------------------------")

    
    print("\n- HDBSCAN avec CNN...")
    hdbscan_cnn = HDBSCAN(min_cluster_size=9, min_samples=1)
    hdbscan_cnn.fit(descriptors_cnn_pca)

    print("\n- HDBSCAN avec HOG...")
    hdbscan_hog = HDBSCAN(min_cluster_size=2, min_samples=7)
    hdbscan_hog.fit(descriptors_hog_pca)

    print("\n- HDBSCAN avec Histogram...")
    hdbscan_hist = HDBSCAN(min_cluster_size=20, min_samples=1)
    hdbscan_hist.fit(descriptors_hist_pca)

    print("\n- HDBSCAN avec couleur...")
    hdbscan_color = HDBSCAN(min_cluster_size=30, min_samples=1)
    hdbscan_color.fit(descriptors_color_pca)

    print("\n- HDBSCAN avec LBP...")
    hdbscan_lbp = HDBSCAN(min_cluster_size=2, min_samples=1)
    hdbscan_lbp.fit(descriptors_lbp_pca)

    
    print("--------------------------------------------------------------------------")

    
    list_dict = [
        
        show_metric(labels_true, kmeans_hog.labels_, descriptors_hog_pca, name_descriptor="HOG", name_model="kmeans"),
        show_metric(labels_true, kmeans_hist.labels_, descriptors_hist_pca, name_descriptor="HISTOGRAM", name_model="kmeans"),
        show_metric(labels_true, kmeans_color.labels_, descriptors_color_pca, name_descriptor="COLOR_HIST", name_model="kmeans"),
        show_metric(labels_true, kmeans_cnn.labels_, descriptors_cnn_pca, name_descriptor="CNN", name_model="kmeans"),
        show_metric(labels_true, kmeans_lbp.labels_, descriptors_lbp_pca, name_descriptor="LBP", name_model="kmeans"),


        show_metric(labels_true, dbscan_hog.labels_, descriptors_hog_pca, name_descriptor="HOG_DBSCAN", name_model="dbscan"),
        show_metric(labels_true, dbscan_hist.labels_, descriptors_hist_pca, name_descriptor="HISTOGRAM_DBSCAN", name_model="dbscan"),
        show_metric(labels_true, dbscan_color.labels_, descriptors_color_pca, name_descriptor="COLOR_HIST_DBSCAN", name_model="dbscan"),
        show_metric(labels_true, dbscan_cnn.labels_, descriptors_cnn_pca, name_descriptor="CNN", name_model="dbscan"),
        show_metric(labels_true, dbscan_lbp.labels_, descriptors_lbp_pca, name_descriptor="LBP_DBSCAN", name_model="dbscan"),

        
        show_metric(labels_true, hdbscan_hog.labels_, descriptors_hog_pca, name_descriptor="HOG_HDBSCAN", name_model="hdbscan"),
        show_metric(labels_true, hdbscan_hist.labels_, descriptors_hist_pca, name_descriptor="HISTOGRAM_HDBSCAN", name_model="hdbscan"),
        show_metric(labels_true, hdbscan_color.labels_, descriptors_color_pca, name_descriptor="COLOR_HIST_HDBSCAN", name_model="hdbscan"),
        show_metric(labels_true, hdbscan_cnn.labels_, descriptors_cnn_pca, name_descriptor="CNN_HDBSCAN", name_model="hdbscan"),
        show_metric(labels_true, hdbscan_lbp.labels_, descriptors_lbp_pca, name_descriptor="LBP_HDBSCAN", name_model="hdbscan"),

    ]

    df_metric = pd.DataFrame(list_dict)

    df_hist = create_df_to_export(descriptors_hist_pca, labels_true, kmeans_hist.labels_)
    df_hog = create_df_to_export(descriptors_hog_pca, labels_true, kmeans_hog.labels_)
    df_color = create_df_to_export(descriptors_color_pca, labels_true, kmeans_color.labels_)
    df_cnn = create_df_to_export(descriptors_cnn_pca, labels_true, kmeans_cnn.labels_)
    df_lbp = create_df_to_export(descriptors_lbp_pca, labels_true, kmeans_lbp.labels_)


    df_hist_dbscan = create_df_to_export(descriptors_hist_pca, labels_true, dbscan_hist.labels_)
    df_hog_dbscan = create_df_to_export(descriptors_hog_pca, labels_true, dbscan_hog.labels_)
    df_color_dbscan = create_df_to_export(descriptors_color_pca, labels_true, dbscan_color.labels_)
    df_cnn_dbscan = create_df_to_export(descriptors_cnn_pca, labels_true, dbscan_cnn.labels_)
    df_lbp_dbscan = create_df_to_export(descriptors_lbp_pca, labels_true, dbscan_lbp.labels_)


    df_hist_hdbscan = create_df_to_export(descriptors_hist_pca, labels_true, hdbscan_hist.labels_)
    df_hog_hdbscan = create_df_to_export(descriptors_hog_pca, labels_true, hdbscan_hog.labels_)
    df_color_hdbscan = create_df_to_export(descriptors_color_pca, labels_true, hdbscan_color.labels_)
    df_cnn_hdbscan = create_df_to_export(descriptors_cnn_pca, labels_true, hdbscan_cnn.labels_)
    df_lbp_hdbscan = create_df_to_export(descriptors_lbp_pca, labels_true, hdbscan_lbp.labels_)


    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    print("- Sauvegarde des fichiers Excel...")
    df_hist.to_excel(f"{PATH_OUTPUT}/save_clustering_hist_kmeans.xlsx", index=False)
    df_hog.to_excel(f"{PATH_OUTPUT}/save_clustering_hog_kmeans.xlsx", index=False)
    df_color.to_excel(f"{PATH_OUTPUT}/save_clustering_color_kmeans.xlsx", index=False)
    df_cnn.to_excel(f"{PATH_OUTPUT}/save_clustering_cnn_kmeans.xlsx", index=False)
    df_lbp.to_excel(f"{PATH_OUTPUT}/save_clustering_lbp_kmeans.xlsx", index=False)

    df_hist_dbscan.to_excel(f"{PATH_OUTPUT}/save_clustering_hist_dbscan.xlsx", index=False)
    df_hog_dbscan.to_excel(f"{PATH_OUTPUT}/save_clustering_hog_dbscan.xlsx", index=False)
    df_color_dbscan.to_excel(f"{PATH_OUTPUT}/save_clustering_color_dbscan.xlsx", index=False)
    df_cnn_dbscan.to_excel(f"{PATH_OUTPUT}/save_clustering_cnn_dbscan.xlsx", index=False)
    df_lbp_dbscan.to_excel(f"{PATH_OUTPUT}/save_clustering_lbp_dbscan.xlsx", index=False)


    df_hist_hdbscan.to_excel(f"{PATH_OUTPUT}/save_clustering_hist_hdbscan.xlsx", index=False)
    df_hog_hdbscan.to_excel(f"{PATH_OUTPUT}/save_clustering_hog_hdbscan.xlsx", index=False)
    df_color_hdbscan.to_excel(f"{PATH_OUTPUT}/save_clustering_color_hdbscan.xlsx", index=False)
    df_cnn_hdbscan.to_excel(f"{PATH_OUTPUT}/save_clustering_cnn_hdbscan.xlsx", index=False)
    df_lbp_hdbscan.to_excel(f"{PATH_OUTPUT}/save_clustering_lbp_hdbscan.xlsx", index=False)


    df_metric.to_excel(f"{PATH_OUTPUT}/save_metric.xlsx", index=False)

    print("Export terminé.")

if __name__ == "__main__":
    pipeline()
