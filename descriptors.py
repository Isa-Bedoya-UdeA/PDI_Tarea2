"""
descriptors.py
--------------
Módulo para extracción de características usando descriptores HOG y LBP.
Incluye visualización de características extraídas.
"""

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURACIÓN DE PARÁMETROS DE DESCRIPTORES
# ============================================================================

# Parámetros HOG
HOG_ORIENTATIONS = 9          # Número de orientaciones del histograma
HOG_PIXELS_PER_CELL = (8, 8)  # Tamaño de celda en píxeles
HOG_CELLS_PER_BLOCK = (2, 2)  # Número de celdas por bloque
HOG_VISUALIZE = True           # Si se quiere visualización
HOG_BLOCK_NORM = 'L2-Hys'      # Tipo de normalización

# Parámetros LBP
LBP_RADIUS = 3                 # Radio del patrón circular
LBP_N_POINTS = 8 * LBP_RADIUS  # Número de puntos en el círculo
LBP_METHOD = 'uniform'         # Método: 'uniform', 'default', 'ror', 'var'
LBP_N_BINS = 256               # Número de bins del histograma


# ============================================================================
# EXTRACCIÓN HOG (Histogram of Oriented Gradients)
# ============================================================================

def extract_hog_features(image, visualize=False):
    """
    Extrae características HOG de una imagen.
    
    Args:
        image (np.ndarray): Imagen en escala de grises (2D)
        visualize (bool): Si retornar también la visualización
    
    Returns:
        np.ndarray: Vector de características HOG (1D)
        np.ndarray (opcional): Imagen de visualización HOG
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    if visualize:
        features, hog_image = hog(
            image,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            block_norm=HOG_BLOCK_NORM,
            visualize=True,
            feature_vector=True
        )
        return features, hog_image
    else:
        features = hog(
            image,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            block_norm=HOG_BLOCK_NORM,
            visualize=False,
            feature_vector=True
        )
        return features


def visualize_hog(image):
    """Visualiza las características HOG de una imagen."""
    features, hog_image = extract_hog_features(image, visualize=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Imagen Original', fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(hog_image, cmap='gray')
    axes[1].set_title(f'HOG Features (dim={len(features)})', fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Dimensionalidad del vector HOG: {len(features)}")
    print(f"Rango de valores: [{features.min():.4f}, {features.max():.4f}]")


# ============================================================================
# EXTRACCIÓN LBP (Local Binary Patterns)
# ============================================================================

def extract_lbp_features(image, n_bins=None):
    """
    Extrae características LBP de una imagen.
    
    Args:
        image (np.ndarray): Imagen en escala de grises (2D)
        n_bins (int): Número de bins del histograma
    
    Returns:
        np.ndarray: Histograma LBP normalizado (vector de características)
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    lbp = local_binary_pattern(
        image,
        P=LBP_N_POINTS,
        R=LBP_RADIUS,
        method=LBP_METHOD
    )
    
    if n_bins is None:
        if LBP_METHOD == 'uniform':
            n_bins = LBP_N_POINTS + 2
        else:
            n_bins = LBP_N_BINS
    
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=True
    )
    
    return hist


def visualize_lbp(image):
    """Visualiza el patrón LBP y su histograma."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    lbp = local_binary_pattern(image, LBP_N_POINTS, LBP_RADIUS, LBP_METHOD)
    hist = extract_lbp_features(image)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Imagen Original', fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(lbp, cmap='gray')
    axes[1].set_title(f'LBP Pattern (R={LBP_RADIUS}, P={LBP_N_POINTS})', fontweight='bold')
    axes[1].axis('off')
    
    axes[2].bar(range(len(hist)), hist, color='#19CAE1', edgecolor='black')
    axes[2].set_title(f'LBP Histogram (dim={len(hist)})', fontweight='bold')
    axes[2].set_xlabel('Bin')
    axes[2].set_ylabel('Frecuencia Normalizada')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Dimensionalidad del vector LBP: {len(hist)}")
    print(f"Rango de valores: [{hist.min():.4f}, {hist.max():.4f}]")


# ============================================================================
# MAPEO DE CLASES STRING A NUMÉRICO
# ============================================================================

def map_class_to_numeric(class_name):
    """
    Convierte nombres de clase a valores numéricos.
    
    Args:
        class_name (str): 'good bean' o 'bad bean'
    
    Returns:
        int: 0 para 'bad bean', 1 para 'good bean'
    """
    class_mapping = {
        'bad bean': 0,
        'good bean': 1
    }
    return class_mapping.get(class_name, -1)


# ============================================================================
# EXTRACCIÓN BATCH PARA TODO EL DATASET
# ============================================================================

def extract_features_from_dataset(split='train', 
                                  dataset_path='data',
                                  descriptor='hog',
                                  preprocessed=False):
    """
    Extrae características de todas las imágenes de un split.
    
    Args:
        split (str): 'train', 'valid' o 'test'
        dataset_path (str): Ruta del dataset
        descriptor (str): 'hog' o 'lbp'
        preprocessed (bool): Si usar imágenes ya preprocesadas
    
    Returns:
        tuple: (X, y, filenames)
    """
    import os
    from tqdm import tqdm
    from preprocessing import preprocess_for_hog, preprocess_for_lbp
    from data_loader import load_classes_csv
    
    if preprocessed:
        split_path = os.path.join(dataset_path, 'processed', split)
    else:
        split_path = os.path.join(dataset_path, split)
    
    # Leer archivo de clases
    df = load_classes_csv(split, dataset_path)
    
    if df is None or len(df) == 0:
        print(f"ERROR: No se pudo cargar el CSV de {split}")
        return None, None, None
    
    features_list = []
    labels_list = []
    filenames_list = []
    
    if descriptor == 'hog':
        extract_fn = extract_hog_features
        preprocess_fn = preprocess_for_hog
    else:
        extract_fn = extract_lbp_features
        preprocess_fn = preprocess_for_lbp
    
    print(f"\nExtrayendo características {descriptor.upper()} de {split}...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Procesando {split}"):
        img_path = os.path.join(dataset_path, split, row['filename'])
        
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        if not preprocessed:
            image = preprocess_fn(image)
        
        features = extract_fn(image)
        
        # Convertir clase de string a numérico
        numeric_label = map_class_to_numeric(row['class'])
        
        if numeric_label == -1:
            print(f"Advertencia: clase desconocida '{row['class']}' en {row['filename']}")
            continue
        
        features_list.append(features)
        labels_list.append(numeric_label)
        filenames_list.append(row['filename'])
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"✓ Características extraídas:")
    print(f"  Shape: {X.shape}")
    print(f"  Clases únicas: {np.unique(y)} (0=bad bean, 1=good bean)")
    print(f"  Distribución: bad bean={np.sum(y==0)}, good bean={np.sum(y==1)}")
    
    return X, y, filenames_list


def extract_all_features(dataset_path='data', descriptor='hog'):
    """
    Extrae características de train, valid y test.
    
    Returns:
        dict: Diccionario con X, y para cada split
    """
    data = {}
    
    for split in ['train', 'valid', 'test']:
        X, y, filenames = extract_features_from_dataset(
            split=split,
            dataset_path=dataset_path,
            descriptor=descriptor,
            preprocessed=False
        )
        data[split] = {'X': X, 'y': y, 'filenames': filenames}
    
    return data


# ============================================================================
# COMPARACIÓN DE DESCRIPTORES
# ============================================================================

def compare_descriptors(image):
    """Compara visualmente HOG vs LBP en la misma imagen."""
    from preprocessing import preprocess_for_hog, preprocess_for_lbp
    
    hog_img = preprocess_for_hog(image)
    lbp_img = preprocess_for_lbp(image)
    
    hog_features, hog_vis = extract_hog_features(hog_img, visualize=True)
    lbp_features = extract_lbp_features(lbp_img)
    lbp_pattern = local_binary_pattern(lbp_img, LBP_N_POINTS, LBP_RADIUS, LBP_METHOD)
    
    fig = plt.figure(figsize=(15, 8))
    
    # HOG
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(hog_img, cmap='gray')
    ax1.set_title('Imagen (HOG preprocessing)', fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(hog_vis, cmap='gray')
    ax2.set_title(f'HOG Visualization', fontweight='bold')
    ax2.axis('off')
    
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(hog_features[:100], color='#19CAE1', linewidth=2)
    ax3.set_title(f'HOG Features (dim={len(hog_features)})', fontweight='bold')
    ax3.set_xlabel('Feature index (primeras 100)')
    ax3.grid(alpha=0.3)
    
    # LBP
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(lbp_img, cmap='gray')
    ax4.set_title('Imagen (LBP preprocessing)', fontweight='bold')
    ax4.axis('off')
    
    ax5 = plt.subplot(2, 3, 5)
    ax5.imshow(lbp_pattern, cmap='gray')
    ax5.set_title(f'LBP Pattern', fontweight='bold')
    ax5.axis('off')
    
    ax6 = plt.subplot(2, 3, 6)
    ax6.bar(range(len(lbp_features)), lbp_features, color='#19CAE1', edgecolor='black')
    ax6.set_title(f'LBP Histogram (dim={len(lbp_features)})', fontweight='bold')
    ax6.set_xlabel('Bin')
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("COMPARACIÓN DE DESCRIPTORES")
    print("="*60)
    print(f"HOG: {len(hog_features)} características")
    print(f"LBP: {len(lbp_features)} características")
    print("="*60)


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    import os
    import glob
    
    test_images = glob.glob("data/train/*.jpg")
    
    if test_images:
        print(f"Probando con: {test_images[0]}")
        image = cv2.imread(test_images[0])
        
        compare_descriptors(image)
        
        # Extraer características de todo el dataset
        # hog_data = extract_all_features(descriptor='hog')
        # lbp_data = extract_all_features(descriptor='lbp')
    else:
        print("No se encontraron imágenes en data/train/")