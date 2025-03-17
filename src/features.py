import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import transform
import itertools


def compute_gray_histograms(images):
    """
    Calcule les histogrammes de niveau de gris pour les images MNIST.
    Input : images (list) : liste des images en niveaux de gris
    Output : descriptors (list) : liste des descripteurs d'histogrammes de niveau de gris
    """
    descriptors = []
    # Conversion des images en uint8
    images = (images / images.max() * 255).astype('uint8')  # Normalisation et conversion en uint8
    for img in images:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])  # Calcul de l'histogramme
        hist = cv2.normalize(hist, hist).flatten()  # Normalisation
        descriptors.append(hist)  # Ajout aux descripteurs
    return descriptors


def compute_hog_descriptors(images):
    """
    Calcule les descripteurs HOG pour les images en niveaux de gris.
    Input : images (array) : tableau numpy des images
    Output : descriptors (list) : liste des descripteurs HOG
    """
    descriptors = []
    # Conversion des images en uint8
    images = (images / images.max() * 255).astype('uint8')  # Normalisation et conversion en uint8
    for img in images:
        # Redimensionnement en 16x16
        img_resized = cv2.resize(img, (16, 16))
        # Calcul des descripteurs HOG
        hog_descriptor = hog(img_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        descriptors.append(hog_descriptor)  # Ajout aux descripteurs
    return descriptors




def compute_color_histograms(images, color_space='HSV', bins=(8, 8, 8)):
    """
    Calcule les histogrammes de couleur pour chaque image.
    
    :param images: liste d'images en niveaux de gris ou en RGB
    :param color_space: 'RGB' ou 'HSV'
    :param bins: nombre de bins par canal
    :return: liste des descripteurs d'histogrammes de couleur
    """
    descriptors = []
    
    for img in images:
        # Convertir en RGB si l'image est en niveaux de gris
        if len(img.shape) == 2:
            img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        # Convertir en espace de couleur spécifié
        if color_space == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Calculer l'histogramme de couleur
        hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()  # Normalisation

        descriptors.append(hist)
    
    return np.array(descriptors)
