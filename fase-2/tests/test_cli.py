"""Pruebas de los comandos tal como los usa una persona en la terminal.

Ejecutan train.py y predict.py como procesos aparte y revisan que los errores
se expliquen con un mensaje claro, sin volcados de Python (traceback).
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
EJEMPLO = RAIZ / "ejemplos" / "viajes_ejemplo.csv"


def correr(*args):
    r = subprocess.run([sys.executable, *args], cwd=RAIZ, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def ejemplo_modificado(tmp_path, columna, valor):
    df = pd.read_csv(EJEMPLO)
    df[columna] = df[columna].astype(object)
    df.loc[1, columna] = valor
    ruta = tmp_path / "entrada.csv"
    df.to_csv(ruta, index=False)
    return ruta


def test_predict_desde_la_terminal(tmp_path):
    salida = tmp_path / "carpeta" / "nueva" / "pred.csv"     # la carpeta no existe todavía
    codigo, texto = correr("predict.py", "--entrada", str(EJEMPLO), "--salida", str(salida))
    assert codigo == 0, texto
    assert len(pd.read_csv(salida)) == 5


def test_train_desde_la_terminal(csv_sintetico, tmp_path):
    codigo, texto = correr("train.py", "--datos", str(csv_sintetico), "--fraccion", "0.5",
                           "--max-iter", "10", "--modelos", str(tmp_path))
    assert codigo == 0, texto
    assert "Versión guardada" in texto and (tmp_path / "v1" / "modelo.joblib").exists()


@pytest.mark.parametrize("columna, valor, mensaje", [
    ("fare_amount", "diez", "no son números"),
    ("tpep_pickup_datetime", "ayer", "fechas no válidas"),
])
def test_valores_invalidos_dan_mensaje_claro(tmp_path, columna, valor, mensaje):
    codigo, texto = correr("predict.py", "--entrada", str(ejemplo_modificado(tmp_path, columna, valor)))
    assert codigo != 0
    assert mensaje in texto and "fila" in texto and "Traceback" not in texto


def test_archivo_sin_viajes(tmp_path):
    vacio = tmp_path / "vacio.csv"
    vacio.write_text(EJEMPLO.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    codigo, texto = correr("predict.py", "--entrada", str(vacio))
    assert codigo != 0 and "no tiene viajes" in texto and "Traceback" not in texto


def test_archivo_inexistente():
    codigo, texto = correr("predict.py", "--entrada", "no_existe.csv")
    assert codigo != 0 and "Traceback" not in texto


@pytest.mark.parametrize("args", [
    ["predict.py", "--entrada", str(EJEMPLO), "--umbral", "5"],
    ["train.py", "--datos", "x.csv", "--fraccion", "0"],
    ["train.py", "--datos", "x.csv", "--fraccion", "2"],
    ["train.py", "--datos", "x.csv", "--prueba", "1"],
    ["train.py", "--datos", "x.csv", "--max-iter", "0"],
    ["train.py"],                                      # falta --datos o --descargar
])
def test_argumentos_fuera_de_rango_se_rechazan(args):
    codigo, texto = correr(*args)
    assert codigo == 2 and "error:" in texto      # argparse explica el problema y sale con código 2
