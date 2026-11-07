"""Las cinco pruebas mínimas de la especificación de la Fase 2 (sección 12).

1. Entrenamiento y almacenamiento   2. Carga y predicción   3. Valores faltantes
4. Entrada inválida                 5. Versionamiento
"""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import EJEMPLO, RAIZ
from propinas.datos import COLUMNAS_ENTRADA, leer_csv_prediccion
from propinas.entrenamiento import entrenar
from propinas.prediccion import predecir
from propinas.variables import preparar_para_predecir
from propinas.versiones import cargar_modelo, sha256_archivo


def correr(*args):
    r = subprocess.run([sys.executable, *args], cwd=RAIZ, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def contenido(carpeta):
    """Bytes de cada archivo de una carpeta, para comprobar que nada cambió."""
    return {p.name: p.read_bytes() for p in sorted(carpeta.iterdir())}


# 1. Entrenamiento y almacenamiento ------------------------------------------------
def test_1_entrenamiento_genera_artefacto_cargable_metricas_y_metadatos(entrenado):
    carpeta = entrenado.rutas["modelo"].parent
    assert sorted(p.name for p in carpeta.iterdir()) == ["metadata.json", "metrics.json", "modelo.joblib"]

    pipe, meta = cargar_modelo(carpeta)                     # se puede cargar (y su sha256 coincide)
    assert hasattr(pipe, "predict_proba")

    metricas = json.loads((carpeta / "metrics.json").read_text(encoding="utf-8"))
    modelo, base = metricas["modelo"]["prueba"], metricas["modelo_base"]["prueba"]
    assert metricas["metrica_principal"] == "roc_auc"
    assert base["roc_auc"] == 0.5 and base["exactitud_balanceada"] == 0.5    # siempre la clase mayoritaria
    assert modelo["roc_auc"] > base["roc_auc"]              # los datos sintéticos tienen señal

    assert meta["version"] == "v1"
    assert meta["tipo_problema"] == "clasificación binaria"
    assert meta["variable_objetivo"]["nombre"] == "propina_generosa"
    assert "HistGradientBoostingClassifier" in meta["algoritmo"]
    assert list(meta["variables_entrada"]) == COLUMNAS_ENTRADA and all(meta["variables_entrada"].values())
    assert meta["semilla"] == 42 and meta["parametros_modelo"]["max_iter"] == 30
    assert meta["datos"]["archivo_usado"] == "viajes_sinteticos.csv" and meta["datos"]["filas_archivo"] == 6000
    sep = meta["separacion"]
    assert sep["filas_entrenamiento"] + sep["filas_prueba"] == meta["datos"]["filas_validas"]
    assert meta["entorno"]["python"] and meta["dependencias"]["archivo"].endswith("requirements.txt")
    assert len(meta["codigo"]["huella_sha256_codigo"]) == 64
    assert meta["sha256_modelo"] == sha256_archivo(carpeta / "modelo.joblib")


# 2. Carga y predicción -------------------------------------------------------------
def test_2_modelo_cargado_predice_igual_que_antes_de_guardarlo(entrenado):
    viajes, _ = leer_csv_prediccion(EJEMPLO)
    X = preparar_para_predecir(viajes)
    antes = entrenado.pipeline.predict_proba(X)[:, 1]          # modelo en memoria, recién entrenado

    for ruta in [entrenado.rutas["modelo"], entrenado.rutas["modelo"].parent]:   # archivo o carpeta
        cargado, _ = cargar_modelo(ruta)
        np.testing.assert_allclose(cargado.predict_proba(X)[:, 1], antes, rtol=0, atol=1e-12)

    resultado = predecir(cargado, viajes)
    assert len(resultado) == len(pd.read_csv(EJEMPLO))         # una predicción por fila
    assert resultado["fila"].tolist() == list(range(1, len(viajes) + 1))   # en el mismo orden
    assert set(resultado["prediccion"]) <= {0, 1}
    np.testing.assert_allclose(resultado["probabilidad_propina_generosa"], antes, atol=1e-4)


def test_2b_el_orden_de_las_columnas_no_cambia_la_prediccion(entrenado, tmp_path):
    df = pd.read_csv(EJEMPLO)
    desordenado = tmp_path / "desordenado.csv"
    df[df.columns[::-1]].to_csv(desordenado, index=False)
    a = predecir(entrenado.pipeline, leer_csv_prediccion(EJEMPLO)[0])
    b = predecir(entrenado.pipeline, leer_csv_prediccion(desordenado)[0])
    pd.testing.assert_frame_equal(a, b)


# 3. Valores faltantes --------------------------------------------------------------
def test_3_celdas_vacias_se_imputan_con_la_mediana_del_entrenamiento(entrenado, viaje_base):
    pipe = entrenado.pipeline
    imputador = pipe.named_steps["prepro"].named_transformers_["num"]
    mediana = dict(zip(imputador.feature_names_in_, imputador.statistics_))["passenger_count"]

    con_nulo = predecir(pipe, viaje_base.assign(passenger_count=np.nan))
    con_mediana = predecir(pipe, viaje_base.assign(passenger_count=mediana))
    assert con_nulo["probabilidad_propina_generosa"][0] == con_mediana["probabilidad_propina_generosa"][0]


def test_3b_csv_con_celdas_vacias_da_una_prediccion_por_fila(entrenado, tmp_path):
    df = pd.read_csv(EJEMPLO, dtype=str)
    df.loc[0, "VendorID"] = None                     # categórica vacía
    df.loc[1, "passenger_count"] = None              # numérica vacía
    df.loc[2, "tpep_dropoff_datetime"] = None        # fecha vacía
    df.loc[3, ["pickup_latitude", "pickup_longitude"]] = None   # coordenadas vacías
    entrada, salida = tmp_path / "con_nulos.csv", tmp_path / "pred.csv"
    df.to_csv(entrada, index=False)

    codigo, texto = correr("predict.py", "--model", str(entrenado.rutas["modelo"]),
                           "--input", str(entrada), "--output", str(salida))
    assert codigo == 0, texto
    res = pd.read_csv(salida)
    assert len(res) == 5 and res["probabilidad_propina_generosa"].between(0, 1).all()


# 4. Entrada inválida ---------------------------------------------------------------
def test_4_falta_una_columna_requerida_error_claro_y_codigo_distinto_de_cero(entrenado, tmp_path):
    entrada, salida = tmp_path / "sin_tarifa.csv", tmp_path / "pred.csv"
    pd.read_csv(EJEMPLO).drop(columns="fare_amount").to_csv(entrada, index=False)

    with pytest.raises(ValueError, match="faltan estas columnas.*fare_amount"):
        leer_csv_prediccion(entrada)

    codigo, texto = correr("predict.py", "--model", str(entrenado.rutas["modelo"]),
                           "--input", str(entrada), "--output", str(salida))
    assert codigo != 0
    assert "fare_amount" in texto and "Traceback" not in texto
    assert not salida.exists()                       # no queda un archivo que parezca válido


# 5. Versionamiento -----------------------------------------------------------------
def test_5_segunda_version_conserva_la_primera_y_no_se_sobrescribe(csv_sintetico, tmp_path):
    silencio = dict(fraccion=0.5, max_iter=10, informar=lambda _: None)
    v1 = entrenar(csv_sintetico, tmp_path, version="v1", **silencio)
    antes = contenido(tmp_path / "v1")
    v2 = entrenar(csv_sintetico, tmp_path, semilla=7, **silencio)          # sin --version: la siguiente

    assert v2.version == "v2"
    assert contenido(tmp_path / "v1") == antes                            # v1 intacta
    for version in ["v1", "v2"]:
        pipe, meta = cargar_modelo(tmp_path / version)                    # ambas cargan
        assert meta["version"] == version
        assert len(predecir(pipe, leer_csv_prediccion(EJEMPLO)[0])) == 5
    assert v1.metadatos["semilla"] == 42 and v2.metadatos["semilla"] == 7

    with pytest.raises(ValueError, match="ya existe"):
        entrenar(csv_sintetico, tmp_path, version="v1", **silencio)
    assert contenido(tmp_path / "v1") == antes                            # identificador repetido: nada cambió
    assert sorted(p.name for p in tmp_path.iterdir()) == ["v1", "v2"]

    codigo, texto = correr("train.py", "--data", str(csv_sintetico), "--output-dir", str(tmp_path),
                           "--version", "v2", "--max-iter", "5")
    assert codigo != 0 and "ya existe" in texto and "Traceback" not in texto
