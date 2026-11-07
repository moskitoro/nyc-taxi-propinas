"""Variable objetivo, limpieza e ingeniería de variables.

Reproduce el procesamiento del notebook de la Fase 1
(fase-1/fase1_modelo_predictivo.ipynb, secciones 3, 4, 6, 7 y 8). Las mismas
funciones se usan al entrenar y al predecir, así que las transformaciones no
están duplicadas.

Nada de lo que hay aquí aprende de los datos: las reglas son fijas. Lo que sí
aprende (imputación con la mediana y one-hot) va dentro del Pipeline de
modelo.py y se ajusta solo con el conjunto de entrenamiento.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .datos import OBJETIVO, UMBRAL_PROPINA, convertir_tipos, normalizar_columnas

# Caja geográfica amplia alrededor de Nueva York (incluye JFK, LaGuardia y Newark)
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 40.45, 41.00, -74.30, -73.65
MIDTOWN = (40.7580, -73.9855)
AEROPUERTOS = {"JFK": (40.6413, -73.7781), "LGA": (40.7769, -73.8740), "EWR": (40.6895, -74.1745)}

# Variables que recibe el modelo, en este orden
VARIABLES_NUMERICAS = [
    "fare_amount", "trip_distance", "duracion_min", "velocidad_mph", "tarifa_por_milla",
    "extra", "tolls_amount", "passenger_count",
    "hora_sin", "hora_cos", "dia_semana", "fin_de_semana",
    "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude",
    "dist_centro_recogida", "dist_centro_destino", "aeropuerto", "coords_faltantes",
]
VARIABLES_CATEGORICAS = ["vendorid", "ratecodeid"]
VARIABLES = VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS

# Variables con fuga de información: nunca pueden ser predictoras
PROHIBIDAS = ["tip_amount", "total_amount", "tasa_propina", "payment_type", OBJETIVO]


def crear_objetivo(df: pd.DataFrame) -> pd.Series:
    """1 si la propina es >= 20 % de la tarifa, 0 en caso contrario."""
    return ((df["tip_amount"] / df["fare_amount"]) >= UMBRAL_PROPINA).astype(int).rename(OBJETIVO)


def calcular_basicas(df: pd.DataFrame) -> pd.DataFrame:
    """Tipos, duración, velocidad y coordenadas fuera de Nueva York -> NaN.
    No elimina filas, por eso sirve tanto para entrenar como para predecir."""
    d = convertir_tipos(normalizar_columnas(df))
    d["duracion_min"] = (d["tpep_dropoff_datetime"] - d["tpep_pickup_datetime"]).dt.total_seconds() / 60
    d["velocidad_mph"] = d["trip_distance"] / (d["duracion_min"] / 60)
    d["velocidad_mph"] = d["velocidad_mph"].replace([np.inf, -np.inf], np.nan)
    for p in ["pickup", "dropoff"]:
        fuera = ~(d[f"{p}_latitude"].between(LAT_MIN, LAT_MAX) & d[f"{p}_longitude"].between(LON_MIN, LON_MAX))
        d.loc[fuera, [f"{p}_latitude", f"{p}_longitude"]] = np.nan
    return d


def filtrar_invalidos(d: pd.DataFrame) -> pd.DataFrame:
    """Elimina registros imposibles. Solo se usa al ENTRENAR: al predecir no se borra ninguna fila."""
    mascara = (
        (d["trip_distance"] > 0) & (d["trip_distance"] < 100)
        & d["duracion_min"].between(1, 180)
        & d["passenger_count"].between(1, 6)
        & (d["fare_amount"] < 500)
        & d["ratecodeid"].between(1, 6)
        & (d["velocidad_mph"] < 80)
    )
    if "tip_amount" in d.columns:
        mascara &= d["tip_amount"] >= 0
    return d[mascara].drop_duplicates().copy()


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def crear_variables(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    recogida = d["tpep_pickup_datetime"]
    hora = recogida.dt.hour
    d["hora_sin"] = np.sin(2 * np.pi * hora / 24)
    d["hora_cos"] = np.cos(2 * np.pi * hora / 24)
    d["dia_semana"] = recogida.dt.dayofweek
    # Si la fecha está vacía, fin_de_semana queda vacío (y se imputa) en vez de suponer "entre semana"
    d["fin_de_semana"] = (d["dia_semana"] >= 5).astype(float).where(d["dia_semana"].notna())
    d["tarifa_por_milla"] = (d["fare_amount"] / d["trip_distance"]).replace([np.inf, -np.inf], np.nan)
    d["dist_centro_recogida"] = haversine_km(d["pickup_latitude"], d["pickup_longitude"], *MIDTOWN)
    d["dist_centro_destino"] = haversine_km(d["dropoff_latitude"], d["dropoff_longitude"], *MIDTOWN)
    cerca = pd.Series(False, index=d.index)
    for lat, lon in AEROPUERTOS.values():
        cerca |= haversine_km(d["pickup_latitude"], d["pickup_longitude"], lat, lon) < 2
        cerca |= haversine_km(d["dropoff_latitude"], d["dropoff_longitude"], lat, lon) < 2
    d["aeropuerto"] = (cerca | d["ratecodeid"].isin([2, 3])).astype(int)
    d["coords_faltantes"] = d[["pickup_latitude", "pickup_longitude",
                               "dropoff_latitude", "dropoff_longitude"]].isna().any(axis=1).astype(int)
    return d


def preparar_X(d: pd.DataFrame) -> pd.DataFrame:
    """Deja solo las variables predictoras, en el orden que espera el modelo.
    Las categóricas vacías quedan como "-1": una categoría que el modelo no vio
    y que el one-hot ignora (todas sus columnas en 0)."""
    if set(VARIABLES) & set(PROHIBIDAS):
        raise RuntimeError("Hay una variable con fuga de información entre las predictoras")
    X = d[VARIABLES].copy()
    for col in VARIABLES_CATEGORICAS:
        X[col] = X[col].fillna(-1).astype(int).astype(str)
    return X


def preparar_para_entrenar(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    d = filtrar_invalidos(calcular_basicas(df_raw))
    d = crear_variables(d)
    return preparar_X(d), crear_objetivo(d)


def preparar_para_predecir(viajes: pd.DataFrame) -> pd.DataFrame:
    """Una fila de X por cada viaje, en el mismo orden. No elimina nada."""
    return preparar_X(crear_variables(calcular_basicas(viajes)))
