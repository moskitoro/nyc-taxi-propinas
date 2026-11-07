"""Lógica compartida por train.py y predict.py: propina generosa en taxis de Nueva York.

    datos.py         lectura del CSV, esquema de entrada y validación
    variables.py     variable objetivo, limpieza e ingeniería de variables
    modelo.py        preprocesamiento + modelo (Pipeline), modelo base y métricas
    versiones.py     guardar y cargar versiones (modelo, metrics.json, metadata.json)
    entrenamiento.py proceso completo de entrenamiento
    prediccion.py    predicción y escritura segura del CSV de salida

Importar el paquete no lee archivos ni entrena nada.
"""
