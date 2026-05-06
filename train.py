"""
train.py
--------
Módulo para entrenamiento y evaluación de modelos de clasificación.
Incluye Red Neuronal y SVM, con manejo de clases desbalanceadas.
"""

import numpy as np
import pickle
import os
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# CONFIGURACIÓN DE HIPERPARÁMETROS
# ============================================================================

# Red Neuronal (MLP)
NN_HIDDEN_LAYERS = (256, 128, 64)  # Arquitectura de capas ocultas
NN_ACTIVATION = 'relu'              # Función de activación
NN_SOLVER = 'adam'                  # Optimizador
NN_ALPHA = 0.0001                   # Regularización L2
NN_LEARNING_RATE = 'adaptive'       # Tasa de aprendizaje
NN_MAX_ITER = 500                   # Máximo de iteraciones
NN_EARLY_STOPPING = True            # Parada temprana
NN_VALIDATION_FRACTION = 0.1        # Fracción para validación
NN_RANDOM_STATE = 42

# SVM
SVM_KERNEL = 'rbf'                  # Tipo de kernel: 'linear', 'rbf', 'poly'
SVM_C = 10.0                        # Parámetro de regularización
SVM_GAMMA = 'scale'                 # Coeficiente del kernel
SVM_MAX_ITER = 1000                 # Máximo de iteraciones
SVM_RANDOM_STATE = 42

# Manejo de desbalance de clases
USE_CLASS_WEIGHT = True             # Usar pesos de clase automáticos
USE_SMOTE = False                   # Usar SMOTE para balanceo

# Rutas
MODELS_PATH = 'models'
os.makedirs(MODELS_PATH, exist_ok=True)


# ============================================================================
# PREPROCESAMIENTO DE DATOS
# ============================================================================

def scale_features(X_train, X_valid=None, X_test=None):
    """Estandariza las características usando StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    X_valid_scaled = None
    X_test_scaled = None
    
    if X_valid is not None:
        X_valid_scaled = scaler.transform(X_valid)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_valid_scaled, X_test_scaled, scaler


def balance_dataset_smote(X_train, y_train):
    """
    Balancea el dataset usando SMOTE.
    Requiere: pip install imbalanced-learn
    """
    try:
        from imblearn.over_sampling import SMOTE
        
        print("\nAplicando SMOTE...")
        print(f"Distribución original: bad bean={np.sum(y_train==0)}, good bean={np.sum(y_train==1)}")
        
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"Distribución después de SMOTE: bad bean={np.sum(y_resampled==0)}, good bean={np.sum(y_resampled==1)}")
        
        return X_resampled, y_resampled
        
    except ImportError:
        print("ERROR: imbalanced-learn no está instalado.")
        print("Instala con: pip install imbalanced-learn")
        return X_train, y_train


def compute_class_weights(y_train):
    """Calcula pesos de clase para manejar desbalance."""
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    
    print(f"\nPesos de clase calculados:")
    print(f"  Clase 0 (bad bean): {class_weight_dict[0]:.4f}")
    print(f"  Clase 1 (good bean): {class_weight_dict[1]:.4f}")
    
    return class_weight_dict


# ============================================================================
# ENTRENAMIENTO DE RED NEURONAL
# ============================================================================

def train_neural_network(X_train, y_train, X_valid=None, y_valid=None, 
                            class_weight=None, save_path=None):
    """Entrena una Red Neuronal (MLP) para clasificación."""
    print("\n" + "="*60)
    print("ENTRENANDO RED NEURONAL (MLP)")
    print("="*60)
    print(f"Arquitectura: {NN_HIDDEN_LAYERS}")
    print(f"Activación: {NN_ACTIVATION}")
    print(f"Optimizador: {NN_SOLVER}")
    print(f"Regularización (alpha): {NN_ALPHA}")
    
    model = MLPClassifier(
        hidden_layer_sizes=NN_HIDDEN_LAYERS,
        activation=NN_ACTIVATION,
        solver=NN_SOLVER,
        alpha=NN_ALPHA,
        learning_rate=NN_LEARNING_RATE,
        max_iter=NN_MAX_ITER,
        early_stopping=NN_EARLY_STOPPING,
        validation_fraction=NN_VALIDATION_FRACTION,
        random_state=NN_RANDOM_STATE,
        verbose=True
    )
    
    print("\nEntrenando...")
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"\n✓ Accuracy en TRAIN: {train_acc:.4f}")
    
    if X_valid is not None and y_valid is not None:
        y_valid_pred = model.predict(X_valid)
        valid_acc = accuracy_score(y_valid, y_valid_pred)
        print(f"✓ Accuracy en VALID: {valid_acc:.4f}")
    
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"\n✓ Modelo guardado en: {save_path}")
    
    print("="*60)
    
    return model


# ============================================================================
# ENTRENAMIENTO DE SVM
# ============================================================================

def train_svm(X_train, y_train, class_weight=None, save_path=None):
    """Entrena una SVM para clasificación."""
    print("\n" + "="*60)
    print("ENTRENANDO SVM")
    print("="*60)
    print(f"Kernel: {SVM_KERNEL}")
    print(f"C (regularización): {SVM_C}")
    print(f"Gamma: {SVM_GAMMA}")
    
    model = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma=SVM_GAMMA,
        class_weight=class_weight,
        max_iter=SVM_MAX_ITER,
        random_state=SVM_RANDOM_STATE,
        probability=True,
        verbose=True
    )
    
    print("\nEntrenando...")
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"\n✓ Accuracy en TRAIN: {train_acc:.4f}")
    
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"\n✓ Modelo guardado en: {save_path}")
    
    print("="*60)
    
    return model


# ============================================================================
# MÉTRICAS DE EVALUACIÓN
# ============================================================================

def calculate_metrics(y_true, y_pred, y_proba=None):
    """Calcula todas las métricas de evaluación."""
    metrics = {}
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
    
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
        except:
            metrics['roc_auc'] = None
    
    return metrics


def print_metrics(metrics, split_name="TEST"):
    """Imprime las métricas de forma legible."""
    print("\n" + "="*60)
    print(f"MÉTRICAS EN {split_name}")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    if 'true_positives' in metrics:
        print(f"\nVerdaderos Positivos (TP):  {metrics['true_positives']}")
        print(f"Verdaderos Negativos (TN):  {metrics['true_negatives']}")
        print(f"Falsos Positivos (FP):      {metrics['false_positives']}")
        print(f"Falsos Negativos (FN):      {metrics['false_negatives']}")
    
    print("="*60)


def plot_confusion_matrix(cm, class_names=['Bad Bean (0)', 'Good Bean (1)'], title='Matriz de Confusión'):
    """Visualiza la matriz de confusión."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Cantidad'})
    plt.title(title, fontweight='bold', fontsize=14)
    plt.ylabel('Etiqueta Verdadera', fontweight='bold')
    plt.xlabel('Etiqueta Predicha', fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_proba, title='Curva ROC'):
    """Visualiza la curva ROC."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1])
    auc = roc_auc_score(y_true, y_proba[:, 1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#19CAE1', linewidth=2, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate (FPR)', fontweight='bold')
    plt.ylabel('True Positive Rate (TPR)', fontweight='bold')
    plt.title(title, fontweight='bold', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# EVALUACIÓN COMPLETA DE MODELO
# ============================================================================

def evaluate_model(model, X_test, y_test, split_name="TEST", show_plots=True):
    """Evaluación completa de un modelo."""
    y_pred = model.predict(X_test)
    
    try:
        y_proba = model.predict_proba(X_test)
    except:
        y_proba = None
    
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    
    print_metrics(metrics, split_name)
    
    print("\nREPORTE DE CLASIFICACIÓN:")
    print(classification_report(y_test, y_pred, target_names=['Bad Bean (0)', 'Good Bean (1)']))
    
    if show_plots:
        plot_confusion_matrix(metrics['confusion_matrix'], 
                            title=f'Matriz de Confusión - {split_name}')
        
        if y_proba is not None:
            plot_roc_curve(y_test, y_proba, title=f'Curva ROC - {split_name}')
    
    return metrics


# ============================================================================
# PIPELINE COMPLETO DE ENTRENAMIENTO
# ============================================================================

def train_complete_pipeline(descriptor='hog', use_smote=USE_SMOTE, use_class_weight=USE_CLASS_WEIGHT):
    """
    Pipeline completo: extracción, entrenamiento y evaluación.
    
    Args:
        descriptor (str): 'hog' o 'lbp'
        use_smote (bool): Si aplicar SMOTE
        use_class_weight (bool): Si usar pesos de clase
    
    Returns:
        dict: Modelos y métricas
    """
    from descriptors import extract_all_features
    
    print("\n" + "="*80)
    print(f"PIPELINE COMPLETO - DESCRIPTOR: {descriptor.upper()}")
    print("="*80)
    
    # 1. Extraer características
    print("\n[1/5] Extrayendo características...")
    data = extract_all_features(descriptor=descriptor)
    
    X_train, y_train = data['train']['X'], data['train']['y']
    X_valid, y_valid = data['valid']['X'], data['valid']['y']
    X_test, y_test = data['test']['X'], data['test']['y']
    
    # 2. Escalar características
    print("\n[2/5] Escalando características...")
    X_train_scaled, X_valid_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_valid, X_test
    )
    
    # 3. Balancear dataset (opcional)
    if use_smote:
        print("\n[3/5] Balanceando dataset con SMOTE...")
        X_train_scaled, y_train = balance_dataset_smote(X_train_scaled, y_train)
    else:
        print("\n[3/5] Sin SMOTE. Usando pesos de clase.")
    
    # Calcular pesos de clase
    class_weights = compute_class_weights(y_train) if use_class_weight else None
    
    # 4. Entrenar modelos
    print("\n[4/5] Entrenando modelos...")
    
    # Red Neuronal
    nn_model = train_neural_network(
        X_train_scaled, y_train,
        X_valid_scaled, y_valid,
        save_path=os.path.join(MODELS_PATH, f'{descriptor}_nn.pkl')
    )
    
    # SVM
    svm_model = train_svm(
        X_train_scaled, y_train,
        class_weight=class_weights,
        save_path=os.path.join(MODELS_PATH, f'{descriptor}_svm.pkl')
    )
    
    # Guardar scaler
    with open(os.path.join(MODELS_PATH, f'{descriptor}_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # 5. Evaluar modelos
    print("\n[5/5] Evaluando modelos...")
    
    print("\n" + "🔹"*40)
    print("RED NEURONAL - EVALUACIÓN")
    print("🔹"*40)
    nn_metrics = evaluate_model(nn_model, X_test_scaled, y_test, "TEST", show_plots=True)
    
    print("\n" + "🔸"*40)
    print("SVM - EVALUACIÓN")
    print("🔸"*40)
    svm_metrics = evaluate_model(svm_model, X_test_scaled, y_test, "TEST", show_plots=True)
    
    # Comparación
    compare_models_results({
        f'{descriptor.upper()}-NN': nn_metrics,
        f'{descriptor.upper()}-SVM': svm_metrics
    })
    
    return {
        'models': {'nn': nn_model, 'svm': svm_model, 'scaler': scaler},
        'metrics': {'nn': nn_metrics, 'svm': svm_metrics}
    }


# ============================================================================
# COMPARACIÓN DE MODELOS
# ============================================================================

def compare_models_results(results_dict):
    """
    Compara métricas de varios modelos en una tabla.
    
    Args:
        results_dict (dict): {nombre_modelo: metrics_dict}
    """
    print("\n" + "="*80)
    print("COMPARACIÓN DE MODELOS")
    print("="*80)
    
    # Crear tabla
    print(f"{'Modelo':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
    print("-" * 80)
    
    for model_name, metrics in results_dict.items():
        roc_auc = metrics.get('roc_auc', 0.0) or 0.0
        print(f"{model_name:<20} "
                f"{metrics['accuracy']:<12.4f} "
                f"{metrics['precision']:<12.4f} "
                f"{metrics['recall']:<12.4f} "
                f"{metrics['f1_score']:<12.4f} "
                f"{roc_auc:<12.4f}")
    
    print("="*80)
    
    # Gráfico comparativo
    plot_models_comparison(results_dict)


def plot_models_comparison(results_dict):
    """Gráfico de barras comparativo de métricas."""
    models = list(results_dict.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    for i, (model_name, metrics) in enumerate(results_dict.items()):
        values = [metrics[m] for m in metrics_names]
        ax.bar(x + i*width, values, width, label=model_name)
    
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Comparación de Modelos', fontweight='bold', fontsize=14)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Entrenar con HOG
    results_hog = train_complete_pipeline(descriptor='hog')
    
    # Entrenar con LBP
    # results_lbp = train_complete_pipeline(descriptor='lbp')