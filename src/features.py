import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import transform
import itertools
from skimage.feature import local_binary_pattern
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from skimage.color import rgb2gray

def compute_gray_histograms(images):
    descriptors = []
    images = (images / images.max() * 255).astype('uint8')
    for img in images:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        descriptors.append(hist)
    return descriptors

def compute_hog_descriptors(images):
    descriptors = []
    images = (images / images.max() * 255).astype('uint8')
    for img in images:
        gray_img = rgb2gray(img)  # conversion en niveaux de gris
        img_resized = cv2.resize(gray_img, (16, 16))
        hog_descriptor = hog(
            img_resized,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False
        )
        descriptors.append(hog_descriptor)
    return descriptors

def compute_color_histograms(images, color_space='RGB'):
    descriptors = []
    for img in images:
        img = (img * 255).astype('uint8')  # ← conversion ici

        if color_space == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LAB':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        hist = []
        for channel in range(img.shape[2]):
            h = cv2.calcHist([img], [channel], None, [256], [0, 256])
            h = cv2.normalize(h, h).flatten()
            hist.extend(h)
        descriptors.append(hist)
    return descriptors

# Chargement du modèle CNN (ResNet50 sans la dernière couche)
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Transformation des images pour ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def compute_cnn_descriptors(images):
    descriptors = []
    for img in images:
        # Si img est en float, on le convertit correctement
        img_rgb = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

        # Vérification de la forme de l'image
        if img_rgb.shape[-1] != 3:
            continue  # on saute si ce n'est pas une image RGB

        img_pil = Image.fromarray(img_rgb)  # plus besoin de np.stack
        img_tensor = transform(img_pil).unsqueeze(0)

        with torch.no_grad():
            feat = model(img_tensor).squeeze().numpy()
        descriptors.append(feat)
    return descriptors



def compute_lbp_descriptors(images, radius=1, n_points=8):
    descriptors = []
    images = (images / images.max() * 255).astype('uint8')

    for img in images:
        gray_img = rgb2gray(img)  # ← conversion en niveaux de gris
        lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= hist.sum() + 1e-6  # ← petite sécurité pour éviter division par zéro
        descriptors.append(hist)

    return np.array(descriptors)