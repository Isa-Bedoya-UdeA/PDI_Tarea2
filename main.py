"""
main.py
-------
Script principal para ejecutar el pipeline completo del proyecto.
Permite entrenar modelos, evaluar y hacer predicciones desde la línea de comandos.
"""

import argparse
import sys
import os


# ============================================================================
# FUNCIONES PRINCIPALES
# ============================================================================

def train_models(descriptor='hog', use_smote=False):
    """
    Entrena modelos con el descriptor especificado.
    
    Args:
        descriptor (str): 'hog' o 'lbp'
        use_smote (bool): Si aplicar SMOTE para balanceo
    """
    from train import train_complete_pipeline
    
    print("\n" + "="*80)
    print(f"ENTRENAMIENTO DE MODELOS - DESCRIPTOR: {descriptor.upper()}")
    print("="*80)
    
    results = train_complete_pipeline(
        descriptor=descriptor,
        use_smote=use_smote,
        use_class_weight=True
    )
    
    print("\n✓ Entrenamiento completado exitosamente")
    return results


def predict_single(image_path, descriptor='hog', model_type='nn', show_viz=True):
    """
    Predice sobre una imagen individual.
    
    Args:
        image_path (str): Ruta de la imagen
        descriptor (str): 'hog' o 'lbp'
        model_type (str): 'nn' o 'svm'
        show_viz (bool): Si mostrar visualización
    """
    from predict import predict_image, visualize_prediction
    
    print(f"\nPrediciendo imagen: {image_path}")
    print(f"Modelo: {descriptor.upper()}-{model_type.upper()}")
    
    result = predict_image(image_path, descriptor, model_type)
    
    if result:
        print("\n" + "="*60)
        print("RESULTADO")
        print("="*60)
        print(f"Clase: {result['class_name']} ({result['class']})")
        print(f"Confianza: {result['confidence']*100:.2f}%")
        print(f"\nProbabilidades:")
        for clase, prob in result['probabilities'].items():
            print(f"  {clase}: {prob*100:.2f}%")
        print("="*60)
        
        if show_viz:
            visualize_prediction(image_path, result, descriptor, model_type)
    else:
        print("ERROR: No se pudo realizar la predicción")


def predict_folder(folder_path, descriptor='hog', model_type='nn', save_csv=False):
    """
    Predice sobre todas las imágenes de una carpeta.
    
    Args:
        folder_path (str): Ruta de la carpeta con imágenes
        descriptor (str): 'hog' o 'lbp'
        model_type (str): 'nn' o 'svm'
        save_csv (bool): Si guardar resultados en CSV
    """
    import glob
    from predict import predict_batch, save_predictions_to_csv
    
    # Buscar imágenes
    image_paths = glob.glob(os.path.join(folder_path, "*.jpg"))
    image_paths += glob.glob(os.path.join(folder_path, "*.png"))
    
    if not image_paths:
        print(f"ERROR: No se encontraron imágenes en {folder_path}")
        return
    
    print(f"\nEncontradas {len(image_paths)} imágenes en {folder_path}")
    
    # Predecir
    results = predict_batch(image_paths, descriptor, model_type)
    
    # Resumen
    if results:
        bad_count = sum(1 for r in results if r['class'] == 0)
        good_count = sum(1 for r in results if r['class'] == 1)
        
        print("\n" + "="*60)
        print("RESUMEN DE PREDICCIONES")
        print("="*60)
        print(f"Total procesadas: {len(results)}")
        print(f"Bad Bean:  {bad_count} ({bad_count/len(results)*100:.1f}%)")
        print(f"Good Bean: {good_count} ({good_count/len(results)*100:.1f}%)")
        print("="*60)
        
        if save_csv:
            save_predictions_to_csv(results, 'predictions.csv')


def explore_dataset():
    """Explora el dataset y muestra estadísticas."""
    from data_loader import explore_full_dataset
    
    print("\nExplorando dataset...")
    stats = explore_full_dataset('data', visualize=True)


def list_models():
    """Lista los modelos disponibles."""
    from predict import print_available_models
    print_available_models()


def run_gui():
    """Lanza la interfaz gráfica."""
    print("\nLanzando interfaz gráfica...")
    import subprocess
    subprocess.run([sys.executable, "app.py"])


# ============================================================================
# CLI (Command Line Interface)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Sistema de Clasificación de Granos de Café',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Ejemplos de uso:
        # Entrenar modelos con HOG
        python main.py train --descriptor hog
        
        # Entrenar con LBP y SMOTE
        python main.py train --descriptor lbp --smote
        
        # Predecir una imagen
        python main.py predict --image data/test/imagen.jpg --descriptor hog --model nn
        
        # Predecir carpeta completa
        python main.py predict-folder --folder data/test --descriptor hog --model svm --csv
        
        # Explorar dataset
        python main.py explore
        
        # Listar modelos disponibles
        python main.py list-models
        
        # Lanzar GUI
        python main.py gui
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # Comando: train
    train_parser = subparsers.add_parser('train', help='Entrenar modelos')
    train_parser.add_argument('--descriptor', type=str, default='hog', 
                            choices=['hog', 'lbp'], help='Descriptor a usar')
    train_parser.add_argument('--smote', action='store_true', 
                            help='Aplicar SMOTE para balanceo')
    
    # Comando: predict
    predict_parser = subparsers.add_parser('predict', help='Predecir imagen individual')
    predict_parser.add_argument('--image', type=str, required=True, 
                                help='Ruta de la imagen')
    predict_parser.add_argument('--descriptor', type=str, default='hog',
                                choices=['hog', 'lbp'], help='Descriptor')
    predict_parser.add_argument('--model', type=str, default='nn',
                                choices=['nn', 'svm'], help='Tipo de modelo')
    predict_parser.add_argument('--no-viz', action='store_true',
                                help='No mostrar visualización')
    
    # Comando: predict-folder
    folder_parser = subparsers.add_parser('predict-folder', 
                                        help='Predecir carpeta de imágenes')
    folder_parser.add_argument('--folder', type=str, required=True,
                                help='Ruta de la carpeta')
    folder_parser.add_argument('--descriptor', type=str, default='hog',
                                choices=['hog', 'lbp'], help='Descriptor')
    folder_parser.add_argument('--model', type=str, default='nn',
                                choices=['nn', 'svm'], help='Tipo de modelo')
    folder_parser.add_argument('--csv', action='store_true',
                                help='Guardar resultados en CSV')
    
    # Comando: explore
    subparsers.add_parser('explore', help='Explorar dataset')
    
    # Comando: list-models
    subparsers.add_parser('list-models', help='Listar modelos disponibles')
    
    # Comando: gui
    subparsers.add_parser('gui', help='Lanzar interfaz gráfica')
    
    # Parsear argumentos
    args = parser.parse_args()
    
    # Ejecutar comando
    if args.command == 'train':
        train_models(args.descriptor, args.smote)
    
    elif args.command == 'predict':
        predict_single(args.image, args.descriptor, args.model, 
                        show_viz=not args.no_viz)
    
    elif args.command == 'predict-folder':
        predict_folder(args.folder, args.descriptor, args.model, args.csv)
    
    elif args.command == 'explore':
        explore_dataset()
    
    elif args.command == 'list-models':
        list_models()
    
    elif args.command == 'gui':
        run_gui()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()