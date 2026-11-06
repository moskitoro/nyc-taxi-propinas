"""Carga, limpieza e ingeniería de variables.

Reproduce exactamente el procesamiento del notebook de la Fase 1
(fase-1/fase1_modelo_predictivo.ipynb), para que los modelos entrenados
aquí y el de la Fase 1 reciban las mismas variables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes del problema
# ---------------------------------------------------------------------------
UMBRAL_PROPINA = 0.20          # propina >= 20 % de la tarifa -> "generosa"
OBJETIVO = "propina_generosa"
SEMILLA = 42

# Caja geográfica amplia alrededor de Nueva York (incluye JFK, LaGuardia y Newark)
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 40.45, 41.00, -74.30, -73.65
MIDTOWN = (40.7580, -73.9855)
AEROPUERTOS = {"JFK": (40.6413, -73.7781), "LGA": (40.7769, -73.8740), "EWR": (40.6895, -74.1745)}

# Columnas que debe traer un viaje para poder predecir (sin la propina)
COLUMNAS_ENTRADA = [
    "vendorid", "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
    "trip_distance", "pickup_longitude", "pickup_latitude", "ratecodeid",
    "dropoff_longitude", "dropoff_latitude", "fare_amount", "extra", "tolls_amount",
]

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


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """En 2015 la columna es RateCodeID y en 2016 RatecodeID: todo a minúsculas."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df


def cargar_csv(ruta: str, fraccion: float = 0.05, semilla: int = SEMILLA,
               tamano_bloque: int = 1_000_000) -> tuple[pd.DataFrame, dict]:
    """Lee el CSV de la TLC por bloques, deja solo pagos con tarjeta y tarifa > 0
    y toma una muestra aleatoria reproducible de cada bloque."""
    total, filtradas, partes = 0, 0, []
    for bloque in pd.read_csv(ruta, chunksize=tamano_bloque, low_memory=False):
        bloque = normalizar_columnas(bloque)
        total += len(bloque)
        bloque = bloque[(bloque["payment_type"] == 1) & (bloque["fare_amount"] > 0)]
        filtradas += len(bloque)
        partes.append(bloque.sample(frac=fraccion, random_state=semilla))
    df = pd.concat(partes, ignore_index=True)
    resumen = {"filas_archivo": total, "filas_tarjeta": filtradas, "filas_muestra": len(df)}
    return df, resumen


# ---------------------------------------------------------------------------
# Variable objetivo y limpieza
# ---------------------------------------------------------------------------
def crear_objetivo(df: pd.DataFrame) -> pd.Series:
    """1 si la propina es >= 20 % de la tarifa, 0 en caso contrario."""
    return ((df["tip_amount"] / df["fare_amount"]) >= UMBRAL_PROPINA).astype(int).rename(OBJETIVO)


def calcular_basicas(df: pd.DataFrame) -> pd.DataFrame:
    """Fechas, duración, velocidad y coordenadas inválidas -> NaN.
    No elimina filas, por eso sirve tanto para entrenar como para predecir."""
    d = normalizar_columnas(df)
    d["tpep_pickup_datetime"] = pd.to_datetime(d["tpep_pickup_datetime"])
    d["tpep_dropoff_datetime"] = pd.to_datetime(d["tpep_dropoff_datetime"])
    d["duracion_min"] = (d["tpep_dropoff_datetime"] - d["tpep_pickup_datetime"]).dt.total_seconds() / 60
    d["velocidad_mph"] = d["trip_distance"] / (d["duracion_min"] / 60)
    d["velocidad_mph"] = d["velocidad_mph"].replace([np.inf, -np.inf], np.nan)
    for p in ["pickup", "dropoff"]:
        fuera = ~(d[f"{p}_latitude"].between(LAT_MIN, LAT_MAX) & d[f"{p}_longitude"].between(LON_MIN, LON_MAX))
        d.loc[fuera, [f"{p}_latitude", f"{p}_longitude"]] = np.nan
    return d


def filtrar_invalidos(d: pd.DataFrame) -> pd.DataFrame:
    """Elimina registros imposibles (solo se usa al entrenar)."""
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


# ---------------------------------------------------------------------------
# Ingeniería de variables
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def crear_variables(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    hora = d["tpep_pickup_datetime"].dt.hour
    d["hora_sin"] = np.sin(2 * np.pi * hora / 24)
    d["hora_cos"] = np.cos(2 * np.pi * hora / 24)
    d["dia_semana"] = d["tpep_pickup_datetime"].dt.dayofweek
    d["fin_de_semana"] = (d["dia_semana"] >= 5).astype(int)
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
    """Deja solo las variables predictoras, en el orden que espera el modelo."""
    assert not set(VARIABLES) & set(PROHIBIDAS), "Hay una variable con fuga de información"
    X = d[VARIABLES].copy()
    for col in VARIABLES_CATEGORICAS:
        X[col] = X[col].fillna(-1).astype(int).astype(str)
    return X


COLUMNAS_FECHA = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]


def _filas(mascara: pd.Series) -> str:
    """Números de fila como los ve una persona en el CSV (la 1 es el primer viaje)."""
    filas = [str(i + 1) for i in np.flatnonzero(mascara.to_numpy())]
    return ", ".join(filas[:10]) + (" ..." if len(filas) > 10 else "")


def validar_entrada(df: pd.DataFrame) -> None:
    """Revisa el archivo antes de predecir y explica el problema en lenguaje claro."""
    d = normalizar_columnas(df)
    faltan = [c for c in COLUMNAS_ENTRADA if c not in d.columns]
    if faltan:
        raise ValueError(f"Al archivo de entrada le faltan columnas: {', '.join(faltan)}")
    if d.empty:
        raise ValueError("El archivo de entrada no tiene viajes (solo encabezados).")
    for col in COLUMNAS_ENTRADA:
        if col in COLUMNAS_FECHA:
            convertida = pd.to_datetime(d[col], errors="coerce")
            problema = "fechas no válidas (formato esperado: AAAA-MM-DD HH:MM:SS)"
        else:
            convertida = pd.to_numeric(d[col], errors="coerce")
            problema = "valores que no son números"
        malas = convertida.isna() & d[col].notna()
        if malas.any():
            raise ValueError(f"La columna '{col}' tiene {problema} en las filas: {_filas(malas)}")
        if col in COLUMNAS_FECHA and convertida.isna().any():
            raise ValueError(f"La columna '{col}' está vacía en las filas: {_filas(convertida.isna())}")


def preparar_para_entrenar(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    d = filtrar_invalidos(calcular_basicas(df_raw))
    d = crear_variables(d)
    return preparar_X(d), crear_objetivo(d)


def preparar_para_predecir(df_raw: pd.DataFrame) -> pd.DataFrame:
    validar_entrada(df_raw)
    return preparar_X(crear_variables(calcular_basicas(df_raw)))
