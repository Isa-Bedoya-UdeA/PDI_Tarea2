"""
preprocessing.py
----------------
Módulo para preprocesamiento de imágenes.
Incluye redimensionamiento, aplicación de filtros y normalización.
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path


# ============================================================================
# CONFIGURACIÓN DE PARÁMETROS DE PREPROCESAMIENTO
# ============================================================================

DEFAULT_SIZE = (128, 128)  # Tamaño estándar para todas las imágenes


# ============================================================================
# REDIMENSIONAMIENTO
# ============================================================================

def resize_image(image, size=DEFAULT_SIZE, interpolation=cv2.INTER_AREA):
    return cv2.resize(image, size, interpolation=interpolation)


# ============================================================================
# FILTROS DE SUAVIZADO (REDUCCIÓN DE RUIDO)
# ============================================================================

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    return cv2.GaussianBlur(image, kernel_size, sigma)

def apply_median_blur(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


# ============================================================================
# NORMALIZACIÓN
# ============================================================================

def normalize_image(image, method='minmax'):
    if method == 'minmax':
        img_min = image.min()
        img_max = image.max()
        if img_max - img_min == 0:
            return image
        return (image - img_min) / (img_max - img_min)
    
    elif method == 'standard':
        mean = image.mean()
        std = image.std()
        if std == 0:
            return image - mean
        return (image - mean) / std
    
    elif method == 'uint8':
        return np.clip(image, 0, 255).astype(np.uint8)
    
    else:
        raise ValueError(f"Método '{method}' no reconocido. Usa 'minmax', 'standard' o 'uint8'")


# ============================================================================
# CONVERSIÓN DE ESPACIO DE COLOR
# ============================================================================

def convert_to_grayscale(image):
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convert_to_rgb(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def convert_to_lab(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# ============================================================================
# ECUALIZACIÓN DE HISTOGRAMA
# ============================================================================

def equalize_histogram(image):
    if len(image.shape) == 3:
        image = convert_to_grayscale(image)
    return cv2.equalizeHist(image)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    if len(image.shape) == 3:
        image = convert_to_grayscale(image)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


# ============================================================================
# PIPELINE COMPLETO DE PREPROCESAMIENTO
# ============================================================================

def preprocess_image(image, 
                     resize=True, 
                     target_size=DEFAULT_SIZE,
                     grayscale=False,
                     apply_filter=None,
                     normalize=True,
                     equalize=False):
    processed = image.copy()
    
    if resize:
        processed = resize_image(processed, target_size)
    if grayscale:
        processed = convert_to_grayscale(processed)
        
    if apply_filter == 'gaussian':
        processed = apply_gaussian_blur(processed)
    elif apply_filter == 'median':
        processed = apply_median_blur(processed)
    elif apply_filter == 'bilateral':
        processed = apply_bilateral_filter(processed)
        
    if equalize:
        processed = apply_clahe(processed)
    if normalize:
        processed = normalize_image(processed, method='minmax')
        
    return processed

def preprocess_for_hog(image, target_size=DEFAULT_SIZE):
    return preprocess_image(
        image,
        resize=True,
        target_size=target_size,
        grayscale=True,
        apply_filter='gaussian',
        normalize=False,
        equalize=True
    )

def preprocess_for_lbp(image, target_size=DEFAULT_SIZE):
    return preprocess_image(
        image,
        resize=True,
        target_size=target_size,
        grayscale=True,
        apply_filter='median',
        normalize=False,
        equalize=True
    )

# ============================================================================
# COMPARACIÓN ANTES/DESPUÉS DE PREPROCESAMIENTO
# ============================================================================

def show_preprocessing_comparison(image, steps=['original', 'resized', 'filtered', 'normalized']):
    import matplotlib.pyplot as plt
    
    images = {}
    
    if 'original' in steps:
        images['Original'] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if 'resized' in steps:
        resized = resize_image(image)
        images['Resized'] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    if 'filtered' in steps:
        resized = resize_image(image)
        filtered = apply_gaussian_blur(resized)
        images['Gaussian Filter'] = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
    
    if 'normalized' in steps:
        resized = resize_image(image)
        filtered = apply_gaussian_blur(resized)
        normalized = normalize_image(filtered, method='minmax')
        images['Normalized'] = normalized
    
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]
    
    for ax, (title, img) in zip(axes, images.items()):
        ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        ax.set_title(title, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# BATCH PREPROCESSING (para todo el dataset)
# ============================================================================

def preprocess_dataset_split(split='train', 
                             dataset_path='data',
                             output_path='data/processed',
                             target_size=DEFAULT_SIZE,
                             descriptor='hog'):
    from tqdm import tqdm
    
    split_output = os.path.join(output_path, split)
    os.makedirs(split_output, exist_ok=True)
    
    split_input = os.path.join(dataset_path, split)
    images = [f for f in os.listdir(split_input) if f.endswith('.jpg')]
    
    print(f"Preprocesando {len(images)} imágenes de {split}...")
    
    preprocess_fn = preprocess_for_hog if descriptor == 'hog' else preprocess_for_lbp
    
    for img_name in tqdm(images, desc=f"Procesando {split}"):
        img_path = os.path.join(split_input, img_name)
        image = cv2.imread(img_path)
        
        if image is None:
            continue
        
        processed = preprocess_fn(image, target_size)
        output_img_path = os.path.join(split_output, img_name)
        cv2.imwrite(output_img_path, processed)
    
    print(f"✓ {len(images)} imágenes preprocesadas guardadas en {split_output}")
    return len(images)

if __name__ == "__main__":
    test_images = glob.glob("data/train/*.jpg")
    
    if test_images:
        print(f"Probando preprocesamiento con: {test_images[0]}")
        test_image = cv2.imread(test_images[0])
        
        if test_image is not None:
            show_preprocessing_comparison(test_image)
            
            hog_ready = preprocess_for_hog(test_image)
            print(f"Imagen lista para HOG: {hog_ready.shape}")
            
            lbp_ready = preprocess_for_lbp(test_image)
            print(f"Imagen lista para LBP: {lbp_ready.shape}")
    else:
        print("No se encontraron imágenes en data/train/ para probar el preprocesamiento.")