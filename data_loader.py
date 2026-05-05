"""
data_loader.py
--------------
Módulo para descarga, carga y exploración del dataset de granos de café.
Maneja la obtención desde Roboflow y análisis estadístico del dataset.
"""

import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

DATASET_PATH = "data"
ROBOFLOW_API_KEY = "xNGFWu6CtqfW7sGfMab0"
ROBOFLOW_WORKSPACE = "yzu"  # ACTUALIZADO
ROBOFLOW_PROJECT = "good-bad-bean"  # ACTUALIZADO
ROBOFLOW_VERSION = 1


# ============================================================================
# DESCARGA DEL DATASET DESDE ROBOFLOW
# ============================================================================

def download_dataset_from_roboflow(destination="data"):
    """
    Descarga el dataset desde Roboflow usando la API.
    """
    try:
        from roboflow import Roboflow
        
        print("=" * 60)
        print("DESCARGANDO DATASET DESDE ROBOFLOW")
        print("=" * 60)
        
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
        version = project.version(ROBOFLOW_VERSION)
        
        print(f"Proyecto: {ROBOFLOW_PROJECT}")
        print(f"Versión: {ROBOFLOW_VERSION}")
        print(f"Descargando a: {destination}/\n")
        
        dataset = version.download("multiclass", location=destination)
        
        print("\n✓ Dataset descargado exitosamente")
        print(f"Ubicación: {dataset.location}")
        print("=" * 60)
        
        return dataset.location
        
    except ImportError:
        print("ERROR: La librería 'roboflow' no está instalada.")
        print("Instala con: pip install roboflow")
        return None
    except Exception as e:
        print(f"ERROR al descargar dataset: {e}")
        return None


# ============================================================================
# VERIFICACIÓN DE ESTRUCTURA DEL DATASET
# ============================================================================

def verify_dataset_structure(dataset_path=DATASET_PATH):
    """
    Verifica que la estructura del dataset sea correcta.
    """
    print("\n" + "=" * 60)
    print("VERIFICANDO ESTRUCTURA DEL DATASET")
    print("=" * 60)
    
    splits = ['train', 'valid', 'test']
    status = {}
    
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        csv_path = os.path.join(split_path, '_classes.csv')
        
        exists = os.path.exists(split_path)
        has_csv = os.path.exists(csv_path)
        
        if exists:
            images = [f for f in os.listdir(split_path) if f.endswith('.jpg')]
            num_images = len(images)
        else:
            num_images = 0
        
        status[split] = {
            'exists': exists,
            'has_csv': has_csv,
            'num_images': num_images
        }
        
        status_icon = "✓" if exists and has_csv else "✗"
        print(f"{status_icon} {split:10s}: {num_images:4d} imágenes | CSV: {'Sí' if has_csv else 'No'}")
    
    print("=" * 60)
    
    return status


# ============================================================================
# LECTURA DE ARCHIVOS _classes.csv (MODIFICADO)
# ============================================================================

def load_classes_csv(split='train', dataset_path=DATASET_PATH):
    """
    Lee el archivo _classes.csv adaptándose al formato one-hot (multiclass).
    Convierte las columnas binarias en una única columna 'class'.
    """
    csv_path = os.path.join(dataset_path, split, '_classes.csv')
    
    if not os.path.exists(csv_path):
        print(f"ERROR: No se encontró {csv_path}")
        return None
    
    try:
        # Cargar el CSV
        df = pd.read_csv(csv_path)
        
        # Eliminar espacios extraños en los nombres de las columnas
        df.columns = [col.strip() for col in df.columns]
        
        # Detectar columnas de clases (todas excepto 'filename')
        class_columns = [col for col in df.columns if col != 'filename']
        
        if len(class_columns) > 0:
            # Encontrar el nombre de la columna donde el valor es máximo (el '1')
            df['class'] = df[class_columns].idxmax(axis=1)
            # Mantener solo las columnas necesarias para el código original
            df = df[['filename', 'class']]
        else:
            # Backup por si el dataset viene en el formato clásico antiguo
            df.columns = ['filename', 'class']
            
    except Exception as e:
        print(f"Error procesando el CSV de {split}: {e}")
        return None
        
    return df


# ============================================================================
# EXPLORACIÓN DEL DATASET
# ============================================================================

def get_dataset_stats(dataset_path=DATASET_PATH):
    """
    Obtiene estadísticas completas del dataset.
    """
    print("\n" + "=" * 60)
    print("ESTADÍSTICAS DEL DATASET")
    print("=" * 60)
    
    splits = ['train', 'valid', 'test']
    stats = {}
    total_images = 0
    all_classes = set()
    
    for split in splits:
        df = load_classes_csv(split, dataset_path)
        
        if df is None:
            continue
        
        num_images = len(df)
        classes = df['class'].unique()
        class_counts = df['class'].value_counts().to_dict()
        
        stats[split] = {
            'num_images': num_images,
            'num_classes': len(classes),
            'classes': sorted(classes),
            'class_distribution': class_counts
        }
        
        total_images += num_images
        all_classes.update(classes)
        
        print(f"\n{split.upper()}:")
        print(f"  Total imágenes: {num_images}")
        print(f"  Clases: {sorted(classes)}")
        print(f"  Distribución:")
        for cls, count in sorted(class_counts.items()):
            print(f"    Clase {cls}: {count} imágenes ({count/num_images*100:.1f}%)")
    
    print(f"\n{'TOTAL':10s}: {total_images} imágenes")
    print(f"{'CLASES':10s}: {sorted(all_classes)}")
    print("=" * 60)
    
    stats['total'] = {
        'num_images': total_images,
        'all_classes': sorted(all_classes),
        'num_classes': len(all_classes)
    }
    
    return stats


# ============================================================================
# ANÁLISIS DE DIMENSIONES DE IMÁGENES
# ============================================================================

def analyze_image_dimensions(split='train', dataset_path=DATASET_PATH, max_samples=100):
    """
    Analiza las dimensiones de las imágenes en un split.
    """
    print(f"\nAnalizando dimensiones de imágenes en {split}...")
    
    split_path = os.path.join(dataset_path, split)
    images = [f for f in os.listdir(split_path) if f.endswith('.jpg')][:max_samples]
    
    widths = []
    heights = []
    
    for img_name in images:
        img_path = os.path.join(split_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
    
    if not widths:
        return None
        
    dimensions = {
        'min_width': min(widths),
        'max_width': max(widths),
        'avg_width': np.mean(widths),
        'min_height': min(heights),
        'max_height': max(heights),
        'avg_height': np.mean(heights),
        'most_common': Counter(zip(widths, heights)).most_common(1)[0]
    }
    
    print(f"  Ancho:  min={dimensions['min_width']}, max={dimensions['max_width']}, avg={dimensions['avg_width']:.0f}")
    print(f"  Alto:   min={dimensions['min_height']}, max={dimensions['max_height']}, avg={dimensions['avg_height']:.0f}")
    print(f"  Dimensión más común: {dimensions['most_common'][0]} ({dimensions['most_common'][1]} imágenes)")
    
    return dimensions


# ============================================================================
# VISUALIZACIÓN DE MUESTRAS DEL DATASET
# ============================================================================

def visualize_samples(split='train', dataset_path=DATASET_PATH, num_samples=9, seed=42):
    """
    Muestra un grid de imágenes aleatorias del dataset.
    """
    np.random.seed(seed)
    
    df = load_classes_csv(split, dataset_path)
    if df is None or len(df) == 0:
        return
    
    samples = df.sample(n=min(num_samples, len(df)))
    
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    
    # Manejar caso de grid 1x1
    if grid_size == 1:
        axes = np.array([[axes]])
        
    fig.suptitle(f'Muestras del Dataset - {split.upper()}', fontsize=16, fontweight='bold')
    
    for idx, (_, row) in enumerate(samples.iterrows()):
        if idx >= num_samples:
            break
        
        img_path = os.path.join(dataset_path, split, row['filename'])
        img = cv2.imread(img_path)
        
        ax = axes[idx // grid_size, idx % grid_size]
        
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(f"Clase: {row['class']}", fontsize=10)
        ax.axis('off')
    
    for idx in range(len(samples), grid_size * grid_size):
        axes[idx // grid_size, idx % grid_size].axis('off')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# GRÁFICO DE DISTRIBUCIÓN DE CLASES
# ============================================================================

def plot_class_distribution(dataset_path=DATASET_PATH):
    """
    Genera gráficos de barras mostrando la distribución de clases.
    """
    splits = ['train', 'valid', 'test']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Distribución de Clases por Split', fontsize=16, fontweight='bold')
    
    for idx, split in enumerate(splits):
        df = load_classes_csv(split, dataset_path)
        if df is None or len(df) == 0:
            continue
        
        class_counts = df['class'].value_counts().sort_index()
        
        ax = axes[idx]
        ax.bar(class_counts.index.astype(str), class_counts.values, color='#19CAE1', edgecolor='black')
        ax.set_title(split.upper(), fontweight='bold')
        ax.set_xlabel('Clase')
        ax.set_ylabel('Cantidad de Imágenes')
        ax.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(class_counts.values):
            ax.text(i, v + (v*0.02), str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# FUNCIÓN PRINCIPAL DE EXPLORACIÓN COMPLETA
# ============================================================================

def explore_full_dataset(dataset_path=DATASET_PATH, visualize=True):
    structure = verify_dataset_structure(dataset_path)
    stats = get_dataset_stats(dataset_path)
    
    dims = analyze_image_dimensions('train', dataset_path)
    stats['dimensions'] = dims
    
    if visualize:
        visualize_samples('train', dataset_path, num_samples=9)
        plot_class_distribution(dataset_path)
    
    return stats

if __name__ == "__main__":
    download_dataset_from_roboflow()

    if os.path.exists(DATASET_PATH):
        stats = explore_full_dataset(DATASET_PATH, visualize=True)