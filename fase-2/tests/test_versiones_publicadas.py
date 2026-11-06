"""Pruebas sobre las versiones del modelo que están en el repositorio (modelos/v1 y modelos/v2).

v1 es el modelo del notebook de la Fase 1 y v2 se obtuvo con train.py. Estas pruebas
convierten en verificación automática la evidencia del README: train.py reproduce la Fase 1.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predict import predecir
from propinas.versiones import cargar_version, leer_registro

RAIZ = Path(__file__).resolve().parents[1]
MODELOS = RAIZ / "modelos"
EJEMPLO = RAIZ / "ejemplos" / "viajes_ejemplo.csv"

# Probabilidades que dio el modelo de la Fase 1 sobre ejemplos/viajes_ejemplo.csv
ESPERADAS = [0.5275, 0.5238, 0.5190, 0.6425, 0.2777]


def test_registro_tiene_v1_y_v2_y_la_actual_existe():
    registro = leer_registro(MODELOS)
    versiones = [v["version"] for v in registro["versiones"]]
    assert versiones[:2] == ["v1", "v2"]
    assert registro["actual"] in versiones
    assert (MODELOS / registro["actual"] / "modelo.joblib").exists()


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_version_publicada_carga_y_su_sha256_coincide(version):
    pipe, meta = cargar_version(MODELOS, version)   # falla si el archivo fue alterado
    assert meta["version"] == version
    assert hasattr(pipe, "predict_proba")


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_version_publicada_da_las_probabilidades_documentadas(version):
    pipe, _ = cargar_version(MODELOS, version)
    prob = predecir(pipe, pd.read_csv(EJEMPLO))["prob_propina_generosa"]
    np.testing.assert_allclose(prob, ESPERADAS, atol=1e-4)


def test_train_py_reproduce_las_metricas_de_la_fase_1():
    meta = {v: json.loads((MODELOS / v / "metadatos.json").read_text(encoding="utf-8")) for v in ["v1", "v2"]}
    for metrica in ["roc_auc", "exactitud_balanceada", "f1_macro"]:
        assert meta["v1"]["metricas_prueba"][metrica] == meta["v2"]["metricas_prueba"][metrica]
    assert meta["v1"]["datos"]["filas_entrenamiento"] == meta["v2"]["datos"]["filas_entrenamiento"]
