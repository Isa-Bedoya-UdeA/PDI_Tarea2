# Tarea 2

## Equipo

* Rafael Alemán
* Esteban Correa
* Isabela Bedoya Gaviria

## Requerimientos

* CustomTkinter: `pip3 install customtkinter`

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

1. Ve a `data_loader.py` y ejecuta por primera vez para descargar el dataset. Debes estar en la ruta raiz de este proyecto.
2. En `if __name__ == "__main__":` del `data_loader.py` comenta la línea `download_dataset_from_roboflow()`, y descomenta `stats = explore_full_dataset(DATASET_PATH, visualize=True)`
3. Ejecuta nuevamente `data_loader.py` para ver la exploración del dataset.
