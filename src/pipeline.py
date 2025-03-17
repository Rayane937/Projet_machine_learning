from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import pandas as pd
import numpy as np
from skimage import io
from skimage.transform import resize

from features import compute_hog_descriptors, compute_gray_histograms, compute_color_histograms
from clustering import KMeans, DBSCAN, show_metric  
from utils import conversion_3d, create_df_to_export
from constant import PATH_OUTPUT, PATH_DATA  

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
                    img = io.imread(os.path.join(category_path, filename), as_gray=True)
                    img_resized = resize(img, (64, 64))
                    images.append(img_resized)
                    labels.append(label_map[category])

    return np.array(images), np.array(labels), label_map

def pipeline():
    print("Chargement des images...")
    images, labels_true, label_map = load_images_and_labels()

    print("\n- Extraction des descripteurs HOG...")
    descriptors_hog = compute_hog_descriptors(images)
    print("- Extraction des descripteurs Histogram...")
    descriptors_hist = compute_gray_histograms(images)
    print("- Extraction des descripteurs de couleur...")
    descriptors_color = compute_color_histograms(images, color_space='HSV')

    # Convertir en numpy array
    descriptors_hog = np.array(descriptors_hog)
    descriptors_hist = np.array(descriptors_hist)
    descriptors_color = np.array(descriptors_color)

    # Normalisation
    scaler = StandardScaler()
    descriptors_hog_norm = scaler.fit_transform(descriptors_hog)
    descriptors_hist_norm = scaler.fit_transform(descriptors_hist)
    descriptors_color_norm = scaler.fit_transform(descriptors_color)

    # PCA pour réduire les dimensions
    pca_hog = PCA(n_components=3)
    descriptors_hog_pca = pca_hog.fit_transform(descriptors_hog_norm)

    pca_hist = PCA(n_components=3)
    descriptors_hist_pca = pca_hist.fit_transform(descriptors_hist_norm)

    pca_color = PCA(n_components=3)
    descriptors_color_pca = pca_color.fit_transform(descriptors_color_norm)

    # Vérification des dimensions après PCA
    print("Shape avant export (HISTOGRAM):", descriptors_hist_pca.shape)  
    print("Shape avant export (HOG):", descriptors_hog_pca.shape)  
    print("Shape avant export (COLOR_HIST):", descriptors_color_pca.shape)  

    # KMeans
    print("\n- KMeans avec HOG...")
    kmeans_hog = KMeans(n_clusters=20)
    kmeans_hog.fit(descriptors_hog_pca)

    print("- KMeans avec Histogram...")
    kmeans_hist = KMeans(n_clusters=20)
    kmeans_hist.fit(descriptors_hist_pca)

    print("- KMeans avec Couleur...")
    kmeans_color = KMeans(n_clusters=20)
    kmeans_color.fit(descriptors_color_pca)

    # DBSCAN
    print("\n- DBSCAN avec HOG...")
    dbscan_hog = DBSCAN(eps=0.5, min_samples=5)
    dbscan_hog.fit(descriptors_hog_pca)

    print("- DBSCAN avec Histogram...")
    dbscan_hist = DBSCAN(eps=1, min_samples=5)
    dbscan_hist.fit(descriptors_hist_pca)

    print("- DBSCAN avec Couleur...")
    dbscan_color = DBSCAN(eps=0.2, min_samples=5)
    dbscan_color.fit(descriptors_color_pca)

    # Calcul des métriques
    list_dict = [
        show_metric(labels_true, kmeans_hog.labels_, descriptors_hog_pca, name_descriptor="HOG", name_model="kmeans"),
        show_metric(labels_true, kmeans_hist.labels_, descriptors_hist_pca, name_descriptor="HISTOGRAM", name_model="kmeans"),
        show_metric(labels_true, kmeans_color.labels_, descriptors_color_pca, name_descriptor="COLOR_HIST", name_model="kmeans"),
        show_metric(labels_true, dbscan_hog.labels_, descriptors_hog_pca, name_descriptor="HOG_DBSCAN", name_model="dbscan"),
        show_metric(labels_true, dbscan_hist.labels_, descriptors_hist_pca, name_descriptor="HISTOGRAM_DBSCAN", name_model="dbscan"),
        show_metric(labels_true, dbscan_color.labels_, descriptors_color_pca, name_descriptor="COLOR_HIST_DBSCAN", name_model="dbscan")
    ]

    df_metric = pd.DataFrame(list_dict)

    # Création des DataFrames pour sauvegarde
    df_hist = create_df_to_export(descriptors_hist_pca, labels_true, kmeans_hist.labels_)
    df_hog = create_df_to_export(descriptors_hog_pca, labels_true, kmeans_hog.labels_)
    df_hist_dbscan = create_df_to_export(descriptors_hist_pca, labels_true, dbscan_hist.labels_)
    df_hog_dbscan = create_df_to_export(descriptors_hog_pca, labels_true, dbscan_hog.labels_)
    df_color = create_df_to_export(descriptors_color_pca, labels_true, kmeans_color.labels_)
    df_color_dbscan = create_df_to_export(descriptors_color_pca, labels_true, dbscan_color.labels_)

    # Création du dossier output s'il n'existe pas
    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    # Sauvegarde des fichiers Excel
    print("- Sauvegarde des fichiers Excel...")
    df_hist.to_excel(f"{PATH_OUTPUT}/save_clustering_hist_kmeans.xlsx", index=False)
    df_hog.to_excel(f"{PATH_OUTPUT}/save_clustering_hog_kmeans.xlsx", index=False)
    df_hist_dbscan.to_excel(f"{PATH_OUTPUT}/save_clustering_hist_dbscan.xlsx", index=False)
    df_hog_dbscan.to_excel(f"{PATH_OUTPUT}/save_clustering_hog_dbscan.xlsx", index=False)
    df_color.to_excel(f"{PATH_OUTPUT}/save_clustering_color_kmeans.xlsx", index=False)
    df_color_dbscan.to_excel(f"{PATH_OUTPUT}/save_clustering_color_dbscan.xlsx", index=False)
    df_metric.to_excel(f"{PATH_OUTPUT}/save_metric.xlsx", index=False)

    print("Export terminé.")

if __name__ == "__main__":
    pipeline()
