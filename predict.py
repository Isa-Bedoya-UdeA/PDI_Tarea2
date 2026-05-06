"""
predict.py
----------
Módulo para predicción de nuevas imágenes usando modelos entrenados.
Incluye funciones para cargar modelos y predecir sobre imágenes individuales.
"""

import cv2
import numpy as np
import pickle
import os
from preprocessing import preprocess_for_hog, preprocess_for_lbp
from descriptors import extract_hog_features, extract_lbp_features


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

MODELS_PATH = 'models'
CLASS_NAMES = {0: 'Bad Bean', 1: 'Good Bean'}


# ============================================================================
# CARGA DE MODELOS
# ============================================================================

def load_model(descriptor='hog', model_type='nn'):
    """
    Carga un modelo entrenado y su scaler.
    
    Args:
        descriptor (str): 'hog' o 'lbp'
        model_type (str): 'nn' o 'svm'
    
    Returns:
        tuple: (model, scaler) o (None, None) si hay error
    """
    model_filename = f'{descriptor}_{model_type}.pkl'
    scaler_filename = f'{descriptor}_scaler.pkl'
    
    model_path = os.path.join(MODELS_PATH, model_filename)
    scaler_path = os.path.join(MODELS_PATH, scaler_filename)
    
    if not os.path.exists(model_path):
        print(f"ERROR: No se encontró el modelo en {model_path}")
        return None, None
    
    if not os.path.exists(scaler_path):
        print(f"ERROR: No se encontró el scaler en {scaler_path}")
        return None, None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        print(f"✓ Modelo cargado: {model_filename}")
        print(f"✓ Scaler cargado: {scaler_filename}")
        
        return model, scaler
        
    except Exception as e:
        print(f"ERROR al cargar modelo: {e}")
        return None, None


# ============================================================================
# PREDICCIÓN SOBRE IMAGEN
# ============================================================================

def predict_image(image_path, descriptor='hog', model_type='nn'):
    """
    Predice la clase de una imagen.
    
    Args:
        image_path (str): Ruta de la imagen
        descriptor (str): 'hog' o 'lbp'
        model_type (str): 'nn' o 'svm'
    
    Returns:
        dict: {
            'class': int,
            'class_name': str,
            'confidence': float,
            'probabilities': dict
        }
    """
    # Cargar modelo
    model, scaler = load_model(descriptor, model_type)
    if model is None or scaler is None:
        return None
    
    # Cargar imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: No se pudo cargar la imagen {image_path}")
        return None
    
    # Preprocesar
    if descriptor == 'hog':
        processed = preprocess_for_hog(image)
        features = extract_hog_features(processed)
    else:  # lbp
        processed = preprocess_for_lbp(image)
        features = extract_lbp_features(processed)
    
    # Escalar características
    features_scaled = scaler.transform([features])
    
    # Predecir
    prediction = model.predict(features_scaled)[0]
    
    # Obtener probabilidades
    try:
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = probabilities[prediction]
        proba_dict = {
            'Bad Bean': float(probabilities[0]),
            'Good Bean': float(probabilities[1])
        }
    except:
        confidence = 1.0
        proba_dict = {'Bad Bean': 0.0, 'Good Bean': 0.0}
        proba_dict[CLASS_NAMES[prediction]] = 1.0
    
    result = {
        'class': int(prediction),
        'class_name': CLASS_NAMES[prediction],
        'confidence': float(confidence),
        'probabilities': proba_dict
    }
    
    return result


def predict_image_with_preloaded_model(image, model, scaler, descriptor='hog'):
    """
    Predice usando un modelo ya cargado en memoria (más rápido para GUI).
    
    Args:
        image (np.ndarray): Imagen BGR de OpenCV
        model: Modelo entrenado
        scaler: Scaler entrenado
        descriptor (str): 'hog' o 'lbp'
    
    Returns:
        dict: Resultado de predicción
    """
    # Preprocesar
    if descriptor == 'hog':
        processed = preprocess_for_hog(image)
        features = extract_hog_features(processed)
    else:
        processed = preprocess_for_lbp(image)
        features = extract_lbp_features(processed)
    
    # Escalar
    features_scaled = scaler.transform([features])
    
    # Predecir
    prediction = model.predict(features_scaled)[0]
    
    # Probabilidades
    try:
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = probabilities[prediction]
        proba_dict = {
            'Bad Bean': float(probabilities[0]),
            'Good Bean': float(probabilities[1])
        }
    except:
        confidence = 1.0
        proba_dict = {'Bad Bean': 0.0, 'Good Bean': 0.0}
        proba_dict[CLASS_NAMES[prediction]] = 1.0
    
    return {
        'class': int(prediction),
        'class_name': CLASS_NAMES[prediction],
        'confidence': float(confidence),
        'probabilities': proba_dict
    }


# ============================================================================
# VISUALIZACIÓN DE PREDICCIÓN
# ============================================================================

def visualize_prediction(image_path, result, descriptor='hog', model_type='nn'):
    """
    Muestra la imagen con el resultado de predicción.
    
    Args:
        image_path (str): Ruta de la imagen
        result (dict): Resultado de predict_image()
        descriptor (str): Descriptor usado
        model_type (str): Tipo de modelo usado
    """
    import matplotlib.pyplot as plt
    
    # Cargar imagen
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Crear figura
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Mostrar imagen
    ax.imshow(image_rgb)
    
    # Color según predicción
    color = 'green' if result['class'] == 1 else 'red'
    
    # Título con resultado
    title = f"Predicción: {result['class_name']}\n"
    title += f"Confianza: {result['confidence']*100:.1f}%\n"
    title += f"Modelo: {descriptor.upper()}-{model_type.upper()}"
    
    ax.set_title(title, fontsize=14, fontweight='bold', color=color)
    ax.axis('off')
    
    # Mostrar probabilidades
    prob_text = f"Bad Bean: {result['probabilities']['Bad Bean']*100:.1f}%\n"
    prob_text += f"Good Bean: {result['probabilities']['Good Bean']*100:.1f}%"
    
    ax.text(0.02, 0.98, prob_text, 
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# BATCH PREDICTION
# ============================================================================

def predict_batch(image_paths, descriptor='hog', model_type='nn'):
    """
    Predice sobre un batch de imágenes.
    
    Args:
        image_paths (list): Lista de rutas de imágenes
        descriptor (str): 'hog' o 'lbp'
        model_type (str): 'nn' o 'svm'
    
    Returns:
        list: Lista de resultados de predicción
    """
    from tqdm import tqdm
    
    # Cargar modelo una sola vez
    model, scaler = load_model(descriptor, model_type)
    if model is None or scaler is None:
        return []
    
    results = []
    
    print(f"\nProcesando {len(image_paths)} imágenes...")
    
    for img_path in tqdm(image_paths, desc="Prediciendo"):
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        result = predict_image_with_preloaded_model(image, model, scaler, descriptor)
        result['image_path'] = img_path
        results.append(result)
    
    return results


def save_predictions_to_csv(results, output_path='predictions.csv'):
    """
    Guarda resultados de predicción en un archivo CSV.
    
    Args:
        results (list): Lista de resultados de predict_batch()
        output_path (str): Ruta del archivo CSV de salida
    """
    import pandas as pd
    
    data = []
    for r in results:
        data.append({
            'image_path': r['image_path'],
            'predicted_class': r['class'],
            'predicted_class_name': r['class_name'],
            'confidence': r['confidence'],
            'prob_bad_bean': r['probabilities']['Bad Bean'],
            'prob_good_bean': r['probabilities']['Good Bean']
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Predicciones guardadas en: {output_path}")


# ============================================================================
# UTILIDADES
# ============================================================================

def get_available_models():
    """
    Lista todos los modelos disponibles en la carpeta models/.
    
    Returns:
        dict: {descriptor: [model_types]}
    """
    if not os.path.exists(MODELS_PATH):
        return {}
    
    files = os.listdir(MODELS_PATH)
    models = {}
    
    for f in files:
        if f.endswith('.pkl') and not f.endswith('scaler.pkl'):
            parts = f.replace('.pkl', '').split('_')
            if len(parts) == 2:
                descriptor, model_type = parts
                if descriptor not in models:
                    models[descriptor] = []
                models[descriptor].append(model_type)
    
    return models


def print_available_models():
    """Imprime los modelos disponibles."""
    models = get_available_models()
    
    if not models:
        print("No hay modelos entrenados disponibles.")
        print("Ejecuta train.py primero.")
        return
    
    print("\n" + "="*60)
    print("MODELOS DISPONIBLES")
    print("="*60)
    
    for descriptor, model_types in models.items():
        print(f"\n{descriptor.upper()}:")
        for mt in model_types:
            print(f"  - {mt.upper()}")
    
    print("="*60)


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    import glob
    
    # Listar modelos disponibles
    print_available_models()
    
    # Obtener una imagen de prueba
    test_images = glob.glob("data/test/*.jpg")
    
    if test_images:
        print(f"\nProbando predicción con: {test_images[0]}")
        
        # Predicción con HOG-NN
        result = predict_image(test_images[0], descriptor='hog', model_type='nn')
        
        if result:
            print("\n" + "="*60)
            print("RESULTADO DE PREDICCIÓN")
            print("="*60)
            print(f"Clase predicha: {result['class_name']} ({result['class']})")
            print(f"Confianza: {result['confidence']*100:.2f}%")
            print(f"\nProbabilidades:")
            for clase, prob in result['probabilities'].items():
                print(f"  {clase}: {prob*100:.2f}%")
            print("="*60)
            
            # Visualizar
            visualize_prediction(test_images[0], result, 'hog', 'nn')
    else:
        print("\nNo se encontraron imágenes en data/test/")