"""Predicción con una versión guardada. No entrena ni reajusta nada.

predict.py solo lee los argumentos y llama a estas funciones; la API REST de la
Fase 3 (/predict) podrá reutilizarlas.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .datos import ETIQUETAS
from .modelo import UMBRAL_DECISION
from .variables import preparar_para_predecir

COLUMNAS_SALIDA = ["fila", "prediccion", "significado", "probabilidad_propina_generosa"]


def predecir(pipe, viajes: pd.DataFrame, umbral: float = UMBRAL_DECISION) -> pd.DataFrame:
    """Una predicción por viaje, en el mismo orden de entrada.

    fila: posición del viaje en el CSV (1 = primer viaje, debajo del encabezado).
    prediccion: 1 = propina generosa (>= 20 % de la tarifa), 0 = no generosa.
    """
    X = preparar_para_predecir(viajes)
    X = X[list(pipe.feature_names_in_)]          # mismo orden de variables del entrenamiento
    prob = pipe.predict_proba(X)[:, 1]
    if len(prob) != len(viajes) or not np.isfinite(prob).all():
        raise RuntimeError("El modelo no devolvió una probabilidad válida para cada viaje.")
    pred = (prob >= umbral).astype(int)
    return pd.DataFrame({
        "fila": np.arange(1, len(viajes) + 1),
        "prediccion": pred,
        "significado": [ETIQUETAS[p] for p in pred],
        "probabilidad_propina_generosa": prob.round(4),
    })


def guardar_predicciones(resultado: pd.DataFrame, ruta: str | Path) -> Path:
    """Escribe primero un archivo temporal y lo renombra al final: si algo falla,
    no queda un CSV a medias que parezca válido."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fd, temporal = tempfile.mkstemp(prefix=".predicciones_", suffix=".tmp", dir=ruta.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            resultado.to_csv(f, index=False)
        os.chmod(temporal, 0o644)                   # mkstemp lo crea solo legible por el dueño
        os.replace(temporal, ruta)
    except BaseException:
        Path(temporal).unlink(missing_ok=True)
        raise
    return ruta
