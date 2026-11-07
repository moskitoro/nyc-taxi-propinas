"""Los comandos tal como los usa una persona en la terminal.

Ejecutan train.py y predict.py como procesos aparte y revisan el código de salida
y que los errores se expliquen con un mensaje claro, sin volcados de Python (traceback).
"""
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from conftest import EJEMPLO, RAIZ


def correr(*args):
    r = subprocess.run([sys.executable, *args], cwd=RAIZ, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def ejemplo_modificado(tmp_path, columna, valor):
    df = pd.read_csv(EJEMPLO, dtype=str)
    df.loc[1, columna] = valor
    ruta = tmp_path / "entrada.csv"
    df.to_csv(ruta, index=False)
    return ruta


@pytest.mark.parametrize("script, opciones", [
    ("train.py", ["--data", "--target", "--output-dir", "--version", "--seed"]),
    ("predict.py", ["--model", "--input", "--output"]),
])
def test_ayuda(script, opciones):
    codigo, texto = correr(script, "--help")
    assert codigo == 0 and all(o in texto for o in opciones)


def test_train_desde_la_terminal_muestra_resumen_y_rutas(csv_sintetico, tmp_path):
    codigo, texto = correr("train.py", "--data", str(csv_sintetico), "--target", "propina_generosa",
                           "--output-dir", str(tmp_path), "--version", "v-prueba", "--seed", "42",
                           "--sample-fraction", "0.5", "--max-iter", "10")
    assert codigo == 0, texto
    assert "ROC-AUC prueba" in texto and "modelo base" in texto
    for archivo in ["modelo.joblib", "metrics.json", "metadata.json"]:
        assert (tmp_path / "v-prueba" / archivo).is_file() and archivo in texto


def test_predict_desde_la_terminal(entrenado, tmp_path):
    salida = tmp_path / "carpeta" / "nueva" / "pred.csv"          # la carpeta no existe todavía
    codigo, texto = correr("predict.py", "--model", str(entrenado.rutas["modelo"].parent),
                           "--input", str(EJEMPLO), "--output", str(salida))
    assert codigo == 0, texto
    res = pd.read_csv(salida)
    assert list(res.columns) == ["fila", "prediccion", "significado", "probabilidad_propina_generosa"]
    assert len(res) == 5 and "store_and_fwd_flag" in texto        # informa la columna que ignora


@pytest.mark.parametrize("columna, valor, mensaje", [
    ("fare_amount", "diez", "no son números"),
    ("tpep_pickup_datetime", "ayer", "fechas no válidas"),
])
def test_tipo_incompatible_da_mensaje_claro_y_no_crea_salida(entrenado, tmp_path, columna, valor, mensaje):
    salida = tmp_path / "pred.csv"
    codigo, texto = correr("predict.py", "--model", str(entrenado.rutas["modelo"]),
                           "--input", str(ejemplo_modificado(tmp_path, columna, valor)), "--output", str(salida))
    assert codigo != 0
    assert mensaje in texto and "filas: 2" in texto and "Traceback" not in texto
    assert not salida.exists()


@pytest.mark.parametrize("caso", ["sin_viajes", "entrada_inexistente", "modelo_inexistente", "modelo_alterado"])
def test_errores_previsibles(entrenado, tmp_path, caso):
    entrada, modelo, salida = EJEMPLO, entrenado.rutas["modelo"], tmp_path / "pred.csv"
    if caso == "sin_viajes":
        entrada = tmp_path / "vacio.csv"
        entrada.write_text(EJEMPLO.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    elif caso == "entrada_inexistente":
        entrada = tmp_path / "no_existe.csv"
    elif caso == "modelo_inexistente":
        modelo = tmp_path / "v99"
    else:
        copia = tmp_path / "copia"
        shutil.copytree(modelo.parent, copia)
        with open(copia / "modelo.joblib", "ab") as f:
            f.write(b"cambio")
        modelo = copia
    codigo, texto = correr("predict.py", "--model", str(modelo), "--input", str(entrada), "--output", str(salida))
    esperado = {"sin_viajes": "no tiene viajes", "entrada_inexistente": "No existe el archivo",
                "modelo_inexistente": "No existe el modelo", "modelo_alterado": "sha256"}[caso]
    assert codigo != 0 and esperado in texto and "Traceback" not in texto
    assert not salida.exists()


def test_entrenar_sin_la_propina_da_error_claro(tmp_path):
    datos = tmp_path / "sin_propina.csv"
    pd.read_csv(EJEMPLO).to_csv(datos, index=False)               # el ejemplo no trae tip_amount
    codigo, texto = correr("train.py", "--data", str(datos), "--output-dir", str(tmp_path / "m"))
    assert codigo != 0 and "tip_amount" in texto and "Traceback" not in texto
    assert not (tmp_path / "m" / "v1").exists()


def test_identificador_de_version_no_valido(csv_sintetico, tmp_path):
    codigo, texto = correr("train.py", "--data", str(csv_sintetico), "--output-dir", str(tmp_path),
                           "--version", "../afuera")
    assert codigo != 0 and "no válido" in texto


@pytest.mark.parametrize("args", [
    ["predict.py", "--model", "models/v1", "--input", "x.csv", "--output", "y.csv", "--threshold", "5"],
    ["predict.py", "--model", "models/v1", "--input", "x.csv"],                    # falta --output
    ["train.py", "--data", "x.csv", "--target", "otra_columna"],
    ["train.py", "--data", "x.csv", "--sample-fraction", "0"],
    ["train.py", "--data", "x.csv", "--sample-fraction", "2"],
    ["train.py", "--data", "x.csv", "--test-size", "1"],
    ["train.py", "--data", "x.csv", "--max-iter", "0"],
    ["train.py"],                                                                   # falta --data
])
def test_argumentos_invalidos_se_rechazan(args):
    codigo, texto = correr(*args)
    assert codigo == 2 and "error:" in texto      # argparse explica el problema y sale con código 2
