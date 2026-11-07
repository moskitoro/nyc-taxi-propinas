"""Pruebas sobre las versiones entregadas en models/ (generadas con train.py).

Convierten en verificación automática la evidencia del README: cada versión carga,
su sha256 coincide, trae métricas y metadatos, y la que usa la configuración de la
Fase 1 (semilla 42, muestra del 5 %, 300 iteraciones) reproduce ese modelo.
"""
import json

import numpy as np
import pytest

from conftest import EJEMPLO, RAIZ
from propinas.datos import leer_csv_prediccion
from propinas.prediccion import predecir
from propinas.versiones import cargar_modelo, listar_versiones

MODELS = RAIZ / "models"
FASE1 = RAIZ.parent / "fase-1" / "modelo"
VERSIONES = listar_versiones(MODELS)

# Probabilidades del modelo de la Fase 1 sobre data/ejemplo_prediccion.csv
ESPERADAS_FASE1 = [0.5275, 0.5238, 0.5190, 0.6425, 0.2777]

necesita_versiones = pytest.mark.skipif(not VERSIONES, reason="models/ todavía no tiene versiones: genérelas con train.py")


def config_fase1(meta):
    return (meta["semilla"] == 42 and meta["datos"]["fraccion_muestra"] == 0.05
            and meta["parametros_modelo"]["max_iter"] == 300 and meta["separacion"]["proporcion_prueba"] == 0.2)


@necesita_versiones
def test_hay_al_menos_dos_versiones():
    assert len(VERSIONES) >= 2, VERSIONES


@necesita_versiones
@pytest.mark.parametrize("version", VERSIONES)
def test_version_completa_carga_y_predice(version):
    carpeta = MODELS / version
    pipe, meta = cargar_modelo(carpeta)                 # falla si el sha256 no coincide
    metricas = json.loads((carpeta / "metrics.json").read_text(encoding="utf-8"))
    assert meta["version"] == metricas["version"] == version
    assert metricas["modelo"]["prueba"]["roc_auc"] > metricas["modelo_base"]["prueba"]["roc_auc"]
    assert meta["codigo"]["huella_sha256_codigo"]
    res = predecir(pipe, leer_csv_prediccion(EJEMPLO)[0])
    assert len(res) == 5 and res["probabilidad_propina_generosa"].between(0, 1).all()


@necesita_versiones
@pytest.mark.parametrize("version", VERSIONES)
def test_configuracion_de_la_fase_1_reproduce_su_modelo(version):
    pipe, meta = cargar_modelo(MODELS / version)
    if not config_fase1(meta):
        pytest.skip(f"{version} usa otra configuración")
    prob = predecir(pipe, leer_csv_prediccion(EJEMPLO)[0])["probabilidad_propina_generosa"]
    np.testing.assert_allclose(prob, ESPERADAS_FASE1, atol=1e-4)
    metricas = json.loads((MODELS / version / "metrics.json").read_text(encoding="utf-8"))
    assert metricas["modelo"]["prueba"]["roc_auc"] == 0.6661
    assert meta["separacion"]["filas_entrenamiento"] == 313292


@pytest.mark.skipif(not (FASE1 / "modelo_propina_generosa.joblib").is_file(),
                    reason="la carpeta fase-1 no está disponible (por ejemplo, dentro de Docker)")
def test_modelo_de_la_fase_1_da_las_probabilidades_documentadas():
    pipe, _ = cargar_modelo(FASE1 / "modelo_propina_generosa.joblib")
    prob = predecir(pipe, leer_csv_prediccion(EJEMPLO)[0])["probabilidad_propina_generosa"]
    np.testing.assert_allclose(prob, ESPERADAS_FASE1, atol=1e-4)
