import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import transform
import itertools

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

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
        img_resized = cv2.resize(img, (16, 16))
        hog_descriptor = hog(img_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        descriptors.append(hog_descriptor)
    return descriptors

def compute_color_histograms(images, color_space='HSV', bins=(8, 8, 8)):
    descriptors = []
    for img in images:
        if len(img.shape) == 2:
            img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if color_space == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        descriptors.append(hist)
    return np.array(descriptors)

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
        img_rgb = np.stack((img,) * 3, axis=-1)
        img_pil = Image.fromarray((img_rgb * 255).astype(np.uint8))
        img_tensor = transform(img_pil).unsqueeze(0)
        with torch.no_grad():
            feat = model(img_tensor).squeeze().numpy()
        descriptors.append(feat)
    return descriptors