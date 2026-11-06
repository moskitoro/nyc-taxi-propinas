"""Pruebas de entrenamiento, versionado y predicción de extremo a extremo."""
import json
from pathlib import Path

import pandas as pd
import pytest

import predict
import train
from propinas.versiones import cargar_version, leer_registro

RAIZ = Path(__file__).resolve().parents[1]
EJEMPLO = RAIZ / "ejemplos" / "viajes_ejemplo.csv"


def test_entrenar_crea_version_con_metadatos(carpeta_modelos):
    registro = leer_registro(carpeta_modelos)
    assert registro["actual"] == "v1"
    meta = json.loads((carpeta_modelos / "v1" / "metadatos.json").read_text(encoding="utf-8"))
    for clave in ["version", "fecha", "hiperparametros", "metricas_prueba", "librerias", "sha256_modelo"]:
        assert clave in meta
    assert 0 <= meta["metricas_prueba"]["roc_auc"] <= 1


def test_el_modelo_aprende_algo_mejor_que_el_azar(carpeta_modelos):
    _, meta = cargar_version(carpeta_modelos)
    assert meta["metricas_prueba"]["roc_auc"] > 0.55


def test_reentrenar_crea_nueva_version_sin_borrar_la_anterior(csv_sintetico, tmp_path):
    for _ in range(2):
        train.main(["--datos", str(csv_sintetico), "--fraccion", "0.5", "--max-iter", "10",
                    "--modelos", str(tmp_path)])
    assert (tmp_path / "v1" / "modelo.joblib").exists()
    assert (tmp_path / "v2" / "modelo.joblib").exists()
    assert leer_registro(tmp_path)["actual"] == "v2"


def test_modelo_alterado_se_detecta(csv_sintetico, tmp_path):
    train.main(["--datos", str(csv_sintetico), "--fraccion", "0.3", "--max-iter", "5",
                "--modelos", str(tmp_path)])
    with open(tmp_path / "v1" / "modelo.joblib", "ab") as f:
        f.write(b"cambio")
    with pytest.raises(ValueError, match="sha256"):
        cargar_version(tmp_path)


def test_predict_sobre_el_ejemplo(carpeta_modelos, tmp_path):
    salida = tmp_path / "pred.csv"
    predict.main(["--entrada", str(EJEMPLO), "--modelos", str(carpeta_modelos), "--salida", str(salida)])
    res = pd.read_csv(salida)
    assert len(res) == len(pd.read_csv(EJEMPLO))
    assert res["prob_propina_generosa"].between(0, 1).all()
    assert set(res["propina_generosa_predicha"]) <= {0, 1}


def test_umbral_de_decision(carpeta_modelos):
    todos = predict.main(["--entrada", str(EJEMPLO), "--modelos", str(carpeta_modelos), "--umbral", "0"])
    ninguno = predict.main(["--entrada", str(EJEMPLO), "--modelos", str(carpeta_modelos), "--umbral", "1"])
    assert todos["propina_generosa_predicha"].eq(1).all()
    assert ninguno["propina_generosa_predicha"].eq(0).all()


def test_version_inexistente_da_error(carpeta_modelos):
    with pytest.raises(FileNotFoundError):
        predict.main(["--entrada", str(EJEMPLO), "--modelos", str(carpeta_modelos), "--version", "v99"])
