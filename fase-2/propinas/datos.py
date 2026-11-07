"""Lectura y validación de los datos de viajes (CSV de la NYC TLC).

Aquí está el esquema de entrada: qué columnas se necesitan, de qué tipo
y qué hacer cuando un archivo no cumple. No crea variables ni entrena.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Problema y fuente de los datos
# ---------------------------------------------------------------------------
OBJETIVO = "propina_generosa"
UMBRAL_PROPINA = 0.20          # propina >= 20 % de la tarifa -> "generosa"
SEMILLA = 42
ETIQUETAS = {
    0: "no generosa: propina menor al 20 % de la tarifa",
    1: "generosa: propina igual o mayor al 20 % de la tarifa",
}
FUENTE_DATOS = "Kaggle elemento/nyc-yellow-taxi-trip-data (NYC TLC, taxis amarillos)"
ARCHIVO_DATOS = "yellow_tripdata_2015-01.csv"

# ---------------------------------------------------------------------------
# Esquema de entrada
# ---------------------------------------------------------------------------
# Columnas que necesita un viaje para predecir (sin la propina). Los nombres se
# comparan sin importar mayúsculas: VendorID y vendorid son la misma columna.
TIPOS_ENTRADA = {
    "vendorid": "entero (categórica: 1 o 2)",
    "tpep_pickup_datetime": "fecha y hora AAAA-MM-DD HH:MM:SS",
    "tpep_dropoff_datetime": "fecha y hora AAAA-MM-DD HH:MM:SS",
    "passenger_count": "entero",
    "trip_distance": "número decimal (millas)",
    "pickup_longitude": "número decimal (grados)",
    "pickup_latitude": "número decimal (grados)",
    "ratecodeid": "entero (categórica: 1 a 6)",
    "dropoff_longitude": "número decimal (grados)",
    "dropoff_latitude": "número decimal (grados)",
    "fare_amount": "número decimal (USD)",
    "extra": "número decimal (USD)",
    "tolls_amount": "número decimal (USD)",
}
COLUMNAS_ENTRADA = list(TIPOS_ENTRADA)
COLUMNAS_FECHA = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
COLUMNAS_NUMERICAS_ENTRADA = [c for c in COLUMNAS_ENTRADA if c not in COLUMNAS_FECHA]

# Para entrenar además se necesitan la propina (para construir el objetivo) y el medio de pago
COLUMNAS_ENTRENAMIENTO = COLUMNAS_ENTRADA + ["payment_type", "tip_amount"]


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """En 2015 la columna es RateCodeID y en 2016 RatecodeID: todo a minúsculas y sin espacios."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower()
    return df


def _filas(mascara: pd.Series) -> str:
    """Números de fila como los ve una persona (la fila 1 es el primer viaje, debajo del encabezado)."""
    filas = [str(i + 1) for i in np.flatnonzero(mascara.to_numpy())]
    return ", ".join(filas[:10]) + (" ..." if len(filas) > 10 else "")


def validar_columnas(columnas, requeridas: list[str], que: str = "predecir") -> None:
    """Falla con un mensaje claro si falta alguna columna o hay columnas repetidas."""
    normalizadas = pd.Index(columnas).astype(str).str.strip().str.lower()
    repetidas = sorted(set(normalizadas[normalizadas.duplicated()]))
    if repetidas:
        raise ValueError(f"El archivo tiene columnas repetidas: {', '.join(repetidas)}")
    faltan = [c for c in requeridas if c not in normalizadas]
    if faltan:
        raise ValueError(f"Para {que} faltan estas columnas en el archivo: {', '.join(faltan)}")


def validar_tipos(df: pd.DataFrame) -> None:
    """Revisa que las columnas numéricas tengan números y las de fecha, fechas.

    Una celda vacía NO es un error (el modelo la imputa); un texto como "diez"
    en una columna numérica sí lo es, y se informan las filas.
    """
    if df.empty:
        raise ValueError("El archivo de entrada no tiene viajes (solo encabezados).")
    for col in COLUMNAS_ENTRADA:
        if col in COLUMNAS_FECHA:
            convertida = pd.to_datetime(df[col], errors="coerce", format="ISO8601")
            problema = "fechas no válidas (formato esperado: AAAA-MM-DD HH:MM:SS)"
        else:
            convertida = pd.to_numeric(df[col], errors="coerce")
            problema = "valores que no son números"
        vacias = df[col].isna() | df[col].astype(str).str.strip().eq("")
        malas = convertida.isna() & ~vacias
        if malas.any():
            raise ValueError(f"La columna '{col}' tiene {problema} en las filas: {_filas(malas)}")


def convertir_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte las columnas del esquema a su tipo (después de validar_tipos).
    Las celdas vacías quedan como NaN / NaT."""
    d = df.copy()
    for col in COLUMNAS_FECHA:
        d[col] = pd.to_datetime(d[col], errors="coerce", format="ISO8601")
    for col in COLUMNAS_NUMERICAS_ENTRADA:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    return d


def leer_csv_prediccion(ruta: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Lee el CSV de viajes a predecir, lo valida y devuelve (viajes, columnas ignoradas).

    - El orden de las columnas no importa: se usan por nombre.
    - La variable objetivo y la propina NO son necesarias; si vienen, se ignoran.
    - Las columnas que el modelo no usa se informan y se ignoran.
    """
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el archivo de entrada: {ruta}")
    try:
        crudo = pd.read_csv(ruta, dtype=str, keep_default_na=True)
    except pd.errors.EmptyDataError:
        raise ValueError(f"El archivo de entrada está vacío: {ruta}") from None
    except (UnicodeDecodeError, pd.errors.ParserError) as e:
        raise ValueError(f"No se pudo leer {ruta} como CSV: {e}") from None
    validar_columnas(crudo.columns, COLUMNAS_ENTRADA, "predecir")
    viajes = normalizar_columnas(crudo)
    validar_tipos(viajes)
    ignoradas = [c for c in viajes.columns if c not in COLUMNAS_ENTRADA]
    return convertir_tipos(viajes), ignoradas


def cargar_csv_entrenamiento(ruta: str | Path, fraccion: float = 0.05, semilla: int = SEMILLA,
                             tamano_bloque: int = 1_000_000) -> tuple[pd.DataFrame, dict]:
    """Lee el CSV de la TLC por bloques (pesa 2 GB), deja solo pagos con tarjeta y
    tarifa > 0 y toma una muestra aleatoria reproducible de cada bloque.

    Igual que en la Fase 1: las propinas en efectivo no se registran en la TLC
    (aparecen en 0), por eso solo se usan los pagos con tarjeta (payment_type = 1).
    """
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el archivo de datos: {ruta}. "
                                "Vea en el README cómo descargarlo y dónde ubicarlo.")
    try:
        encabezado = pd.read_csv(ruta, nrows=0).columns
    except pd.errors.EmptyDataError:
        raise ValueError(f"El archivo de datos está vacío: {ruta}") from None
    validar_columnas(encabezado, COLUMNAS_ENTRENAMIENTO, "entrenar")

    total, filtradas, partes = 0, 0, []
    for bloque in pd.read_csv(ruta, chunksize=tamano_bloque, low_memory=False):
        bloque = normalizar_columnas(bloque)
        total += len(bloque)
        bloque = bloque[(bloque["payment_type"] == 1) & (bloque["fare_amount"] > 0)]
        filtradas += len(bloque)
        partes.append(bloque.sample(frac=fraccion, random_state=semilla))
    df = pd.concat(partes, ignore_index=True)
    if df.empty:
        raise ValueError("El archivo no tiene viajes pagados con tarjeta y tarifa mayor a 0: no hay con qué entrenar.")
    validar_tipos(df)
    for col in ["tip_amount"] + COLUMNAS_NUMERICAS_ENTRADA:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"La columna '{col}' debería ser numérica y no lo es.")
    resumen = {"filas_archivo": total, "filas_tarjeta": filtradas, "filas_muestra": len(df)}
    return df, resumen
