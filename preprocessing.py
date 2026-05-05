"""
preprocessing.py
----------------
Módulo para preprocesamiento de imágenes.
Incluye redimensionamiento, aplicación de filtros y normalización.
"""

import cv2
import numpy as np
from pathlib import Path


# ============================================================================
# CONFIGURACIÓN DE PARÁMETROS DE PREPROCESAMIENTO
# ============================================================================

DEFAULT_SIZE = (128, 128)  # Tamaño estándar para todas las imágenes


# ============================================================================
# REDIMENSIONAMIENTO
# ============================================================================

def resize_image(image, size=DEFAULT_SIZE, interpolation=cv2.INTER_AREA):
    """
    Redimensiona una imagen al tamaño especificado.
    
    Args:
        image (np.ndarray): Imagen a redimensionar
        size (tuple): Tamaño objetivo (width, height)
        interpolation: Método de interpolación de OpenCV
    
    Returns:
        np.ndarray: Imagen redimensionada
    """
    return cv2.resize(image, size, interpolation=interpolation)


# ============================================================================
# FILTROS DE SUAVIZADO (REDUCCIÓN DE RUIDO)
# ============================================================================

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    """
    Aplica filtro Gaussiano para reducir ruido.
    
    Args:
        image (np.ndarray): Imagen de entrada
        kernel_size (tuple): Tamaño del kernel (debe ser impar)
        sigma (float): Desviación estándar (0 = auto)
    
    Returns:
        np.ndarray: Imagen filtrada
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)


def apply_median_blur(image, kernel_size=5):
    """
    Aplica filtro de Mediana para reducir ruido.
    Especialmente efectivo contra ruido sal y pimienta.
    
    Args:
        image (np.ndarray): Imagen de entrada
        kernel_size (int): Tamaño del kernel (debe ser impar)
    
    Returns:
        np.ndarray: Imagen filtrada
    """
    return cv2.medianBlur(image, kernel_size)


def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    Aplica filtro Bilateral que preserva bordes mientras suaviza.
    
    Args:
        image (np.ndarray): Imagen de entrada
        d (int): Diámetro del vecindario
        sigma_color (float): Filtro de rango de color
        sigma_space (float): Filtro de rango espacial
    
    Returns:
        np.ndarray: Imagen filtrada
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


# ============================================================================
# NORMALIZACIÓN
# ============================================================================

def normalize_image(image, method='minmax'):
    """
    Normaliza los valores de píxeles de una imagen.
    
    Args:
        image (np.ndarray): Imagen de entrada
        method (str): Método de normalización
            - 'minmax': Escala a rango [0, 1]
            - 'standard': Estandarización (media=0, std=1)
            - 'uint8': Convierte a rango [0, 255]
    
    Returns:
        np.ndarray: Imagen normalizada
    """
    if method == 'minmax':
        # Normalización Min-Max a [0, 1]
        img_min = image.min()
        img_max = image.max()
        if img_max - img_min == 0:
            return image
        return (image - img_min) / (img_max - img_min)
    
    elif method == 'standard':
        # Estandarización (z-score)
        mean = image.mean()
        std = image.std()
        if std == 0:
            return image - mean
        return (image - mean) / std
    
    elif method == 'uint8':
        # Asegurar rango [0, 255] en uint8
        return np.clip(image, 0, 255).astype(np.uint8)
    
    else:
        raise ValueError(f"Método '{method}' no reconocido. Usa 'minmax', 'standard' o 'uint8'")


# ============================================================================
# CONVERSIÓN DE ESPACIO DE COLOR
# ============================================================================

def convert_to_grayscale(image):
    """
    Convierte una imagen BGR a escala de grises.
    
    Args:
        image (np.ndarray): Imagen BGR
    
    Returns:
        np.ndarray: Imagen en escala de grises
    """
    if len(image.shape) == 2:
        # Ya está en escala de grises
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def convert_to_rgb(image):
    """
    Convierte una imagen BGR a RGB.
    
    Args:
        image (np.ndarray): Imagen BGR
    
    Returns:
        np.ndarray: Imagen RGB
    """
    if len(image.shape) == 2:
        # Es escala de grises, convertir a RGB
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_to_lab(image):
    """
    Convierte una imagen BGR al espacio de color LAB.
    
    Args:
        image (np.ndarray): Imagen BGR
    
    Returns:
        np.ndarray: Imagen en espacio LAB
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


def convert_to_hsv(image):
    """
    Convierte una imagen BGR al espacio de color HSV.
    
    Args:
        image (np.ndarray): Imagen BGR
    
    Returns:
        np.ndarray: Imagen en espacio HSV
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# ============================================================================
# ECUALIZACIÓN DE HISTOGRAMA
# ============================================================================

def equalize_histogram(image):
    """
    Aplica ecualización de histograma para mejorar contraste.
    
    Args:
        image (np.ndarray): Imagen en escala de grises
    
    Returns:
        np.ndarray: Imagen ecualizada
    """
    if len(image.shape) == 3:
        # Convertir a escala de grises primero
        image = convert_to_grayscale(image)
    
    return cv2.equalizeHist(image)


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image (np.ndarray): Imagen en escala de grises
        clip_limit (float): Límite de contraste
        tile_grid_size (tuple): Tamaño de la cuadrícula para regiones
    
    Returns:
        np.ndarray: Imagen con CLAHE aplicado
    """
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
    """
    Pipeline completo de preprocesamiento de imagen.
    
    Args:
        image (np.ndarray): Imagen de entrada (BGR)
        resize (bool): Si redimensionar
        target_size (tuple): Tamaño objetivo
        grayscale (bool): Si convertir a escala de grises
        apply_filter (str): Filtro a aplicar ('gaussian', 'median', 'bilateral', None)
        normalize (bool): Si normalizar a [0, 1]
        equalize (bool): Si aplicar CLAHE
    
    Returns:
        np.ndarray: Imagen preprocesada
    """
    processed = image.copy()
    
    # 1. Redimensionar
    if resize:
        processed = resize_image(processed, target_size)
    
    # 2. Convertir a escala de grises si se requiere
    if grayscale:
        processed = convert_to_grayscale(processed)
    
    # 3. Aplicar filtro de suavizado
    if apply_filter == 'gaussian':
        processed = apply_gaussian_blur(processed)
    elif apply_filter == 'median':
        processed = apply_median_blur(processed)
    elif apply_filter == 'bilateral':
        processed = apply_bilateral_filter(processed)
    
    # 4. Ecualización de histograma
    if equalize:
        processed = apply_clahe(processed)
    
    # 5. Normalización
    if normalize:
        processed = normalize_image(processed, method='minmax')
    
    return processed


def preprocess_for_hog(image, target_size=DEFAULT_SIZE):
    """
    Preprocesamiento específico para extracción HOG.
    HOG funciona mejor con imágenes en escala de grises.
    
    Args:
        image (np.ndarray): Imagen BGR
        target_size (tuple): Tamaño objetivo
    
    Returns:
        np.ndarray: Imagen preprocesada para HOG
    """
    return preprocess_image(
        image,
        resize=True,
        target_size=target_size,
        grayscale=True,
        apply_filter='gaussian',
        normalize=False,  # HOG no requiere normalización previa
        equalize=True     # CLAHE mejora detección de gradientes
    )


def preprocess_for_lbp(image, target_size=DEFAULT_SIZE):
    """
    Preprocesamiento específico para extracción LBP.
    LBP requiere escala de grises.
    
    Args:
        image (np.ndarray): Imagen BGR
        target_size (tuple): Tamaño objetivo
    
    Returns:
        np.ndarray: Imagen preprocesada para LBP
    """
    return preprocess_image(
        image,
        resize=True,
        target_size=target_size,
        grayscale=True,
        apply_filter='median',  # LBP es sensible a ruido
        normalize=False,
        equalize=True
    )


# ============================================================================
# COMPARACIÓN ANTES/DESPUÉS DE PREPROCESAMIENTO
# ============================================================================

def show_preprocessing_comparison(image, steps=['original', 'resized', 'filtered', 'normalized']):
    """
    Muestra comparación visual de las etapas de preprocesamiento.
    
    Args:
        image (np.ndarray): Imagen original
        steps (list): Etapas a mostrar
    """
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
    
    # Crear subplot
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
    """
    Preprocesa todas las imágenes de un split del dataset.
    
    Args:
        split (str): 'train', 'valid' o 'test'
        dataset_path (str): Ruta del dataset original
        output_path (str): Ruta donde guardar imágenes preprocesadas
        target_size (tuple): Tamaño objetivo
        descriptor (str): Tipo de descriptor ('hog' o 'lbp')
    
    Returns:
        int: Número de imágenes procesadas
    """
    import os
    from tqdm import tqdm
    
    # Crear directorio de salida
    split_output = os.path.join(output_path, split)
    os.makedirs(split_output, exist_ok=True)
    
    # Obtener lista de imágenes
    split_input = os.path.join(dataset_path, split)
    images = [f for f in os.listdir(split_input) if f.endswith('.jpg')]
    
    print(f"Preprocesando {len(images)} imágenes de {split}...")
    
    # Seleccionar función de preprocesamiento
    preprocess_fn = preprocess_for_hog if descriptor == 'hog' else preprocess_for_lbp
    
    for img_name in tqdm(images, desc=f"Procesando {split}"):
        # Leer imagen
        img_path = os.path.join(split_input, img_name)
        image = cv2.imread(img_path)
        
        if image is None:
            continue
        
        # Preprocesar
        processed = preprocess_fn(image, target_size)
        
        # Guardar
        output_img_path = os.path.join(split_output, img_name)
        cv2.imwrite(output_img_path, processed)
    
    print(f"✓ {len(images)} imágenes preprocesadas guardadas en {split_output}")
    
    return len(images)

if __name__ == "__main__":
    # Cargar imagen de ejemplo
    test_image = cv2.imread("data/train/-003_png.rf.f3ba0c67474f7343f5348d72b97d99ea.jpg")
    
    if test_image is not None:
        # Mostrar comparación
        show_preprocessing_comparison(test_image)
        
        # Preprocesar para HOG
        hog_ready = preprocess_for_hog(test_image)
        print(f"Imagen lista para HOG: {hog_ready.shape}")
        
        # Preprocesar para LBP
        lbp_ready = preprocess_for_lbp(test_image)
        print(f"Imagen lista para LBP: {lbp_ready.shape}")