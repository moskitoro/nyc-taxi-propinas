"""Predice si los viajes de un CSV recibirán propina generosa (>= 20 % de la tarifa).

Ejemplos:
    python predict.py --entrada ejemplos/viajes_ejemplo.csv
    python predict.py --entrada viajes.csv --salida predicciones.csv
    python predict.py --entrada viajes.csv --version v1
    python predict.py --entrada viajes.csv --modelo ../fase-1/modelo/modelo_propina_generosa.joblib

El CSV de entrada usa las columnas de la NYC TLC (ver COLUMNAS_ENTRADA en propinas/datos.py).
La propina NO se necesita: si viene, se ignora.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from propinas.datos import preparar_para_predecir
from propinas.versiones import cargar_version


def probabilidad(texto: str) -> float:
    valor = float(texto)
    if not 0 <= valor <= 1:
        raise argparse.ArgumentTypeError("el umbral debe estar entre 0 y 1")
    return valor


def leer_argumentos(argv=None):
    p = argparse.ArgumentParser(description="Predice propina generosa para viajes de taxi de NYC.")
    p.add_argument("--entrada", required=True, help="CSV con los viajes a predecir.")
    p.add_argument("--salida", help="CSV donde guardar las predicciones. Si se omite, se muestran en pantalla.")
    p.add_argument("--modelos", default="modelos", help="Carpeta de versiones (por defecto 'modelos').")
    p.add_argument("--version", help="Versión a usar, por ejemplo v1. Por defecto, la actual del registro.")
    p.add_argument("--modelo", help="Ruta directa a un .joblib (por ejemplo, el de la Fase 1).")
    p.add_argument("--umbral", type=probabilidad, default=0.5,
                   help="Probabilidad mínima para predecir 1, entre 0 y 1 (por defecto 0.5).")
    return p.parse_args(argv)


def predecir(pipe, viajes: pd.DataFrame, umbral: float = 0.5) -> pd.DataFrame:
    X = preparar_para_predecir(viajes)
    salida = viajes.copy()
    salida["prob_propina_generosa"] = pipe.predict_proba(X)[:, 1].round(4)
    salida["propina_generosa_predicha"] = (salida["prob_propina_generosa"] >= umbral).astype(int)
    return salida


def main(argv=None) -> pd.DataFrame:
    args = leer_argumentos(argv)
    if args.modelo:
        pipe, origen = joblib.load(args.modelo), args.modelo
    else:
        pipe, meta = cargar_version(args.modelos, args.version)
        origen = f"{args.modelos}/{meta['version']} (entrenado {meta['fecha']})"
    print(f"Modelo: {origen}")

    viajes = pd.read_csv(args.entrada)
    resultado = predecir(pipe, viajes, args.umbral)

    if args.salida:
        Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
        resultado.to_csv(args.salida, index=False)
        print(f"{len(resultado)} predicciones guardadas en {args.salida}")
    else:
        cols = ["fare_amount", "trip_distance", "prob_propina_generosa", "propina_generosa_predicha"]
        cols = [c for c in cols if c in resultado.columns]
        print(resultado[cols].to_string(index=False))
    generosas = resultado["propina_generosa_predicha"].mean()
    print(f"Viajes con propina generosa predicha: {generosas:.0%}")
    return resultado


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error: {e}")
