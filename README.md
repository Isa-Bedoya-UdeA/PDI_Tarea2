# Tarea 2

## Equipo

* Rafael Alemán
* Esteban Correa
* Isabela Bedoya Gaviria

## Requerimientos

* CustomTkinter: `pip3 install customtkinter`
* Cv2: `pip install opencv-python`
* Pandas: `pip install pandas`
* Numpy: `pip install numpy`
* Matplotlib: `pip install matplotlib`
* Roboflow: `pip install roboflow`
* Scikit: `pip install scikit-image`
* Imblearn: `pip install imbalanced-learn`
* Seaborn: `pip install seaborn`

## Flujo de trabajo

```plain text
1. Entrenar modelos:
   $ python main.py --train

2. Ejecutar interfaz gráfica:
   $ python gui/app.py

3. Predicción por línea de comandos (opcional):
   $ python -m src.predict --image ruta/imagen.jpg --model hog_svm
```

## Pasos

Estos pasos son solo de lo implementado hasta el momento.

1. Ejecuta `data_loader.py` para ver la exploración del dataset y descargarlo.
2. Ejecuta `preprocessing.py` para ver los pasos de preprocesado.
