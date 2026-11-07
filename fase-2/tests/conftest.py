"""Datos sintéticos con la misma estructura del CSV de la NYC TLC.

Las pruebas no descargan nada: generan unos miles de viajes falsos (pero con
señal que aprender) y entrenan en carpetas temporales (tmp_path), sin tocar
las versiones de models/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

EJEMPLO = RAIZ / "data" / "ejemplo_prediccion.csv"


def generar_viajes(n: int = 6000, semilla: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    recogida = pd.Timestamp("2015-01-01") + pd.to_timedelta(rng.integers(0, 31 * 24 * 3600, n), unit="s")
    distancia = np.round(np.abs(rng.gamma(1.5, 2, n)) + 0.1, 2)
    duracion = distancia * 4 + rng.gamma(2, 3, n) + 1
    tarifa = np.round(2.5 + distancia * 2.5 + rng.normal(0, 1, n).clip(-1, 1), 2)
    vendor = rng.integers(1, 3, n)
    pago = rng.choice([1, 2], n, p=[0.65, 0.35])
    # Propina generosa más probable en el proveedor 2 y de noche: hay señal que aprender
    p_generosa = 0.45 + 0.25 * (vendor == 2) + 0.1 * (recogida.hour >= 18)
    generosa = rng.random(n) < p_generosa
    propina = np.where(generosa, tarifa * rng.choice([0.2, 0.25, 0.3], n), tarifa * rng.uniform(0, 0.15, n))
    propina = np.round(np.where(pago == 1, propina, 0), 2)
    lat = 40.75 + rng.normal(0, 0.03, n)
    lon = -73.98 + rng.normal(0, 0.03, n)
    ceros = rng.random(n) < 0.02
    lat[ceros], lon[ceros] = 0, 0
    df = pd.DataFrame({
        "VendorID": vendor,
        "tpep_pickup_datetime": recogida.strftime("%Y-%m-%d %H:%M:%S"),
        "tpep_dropoff_datetime": (recogida + pd.to_timedelta(duracion, unit="m")).strftime("%Y-%m-%d %H:%M:%S"),
        "passenger_count": rng.choice([0, 1, 2, 3, 5, 6], n, p=[0.01, 0.7, 0.15, 0.06, 0.05, 0.03]),
        "trip_distance": distancia,
        "pickup_longitude": lon, "pickup_latitude": lat,
        "RateCodeID": rng.choice([1, 2, 5, 99], n, p=[0.95, 0.03, 0.019, 0.001]),
        "store_and_fwd_flag": "N",
        "dropoff_longitude": lon + rng.normal(0, 0.02, n), "dropoff_latitude": lat + rng.normal(0, 0.02, n),
        "payment_type": pago, "fare_amount": tarifa, "extra": rng.choice([0, 0.5, 1], n),
        "mta_tax": 0.5, "tip_amount": propina, "tolls_amount": rng.choice([0, 5.33], n, p=[0.95, 0.05]),
        "improvement_surcharge": 0.3,
    })
    df["total_amount"] = df[["fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount",
                             "improvement_surcharge"]].sum(axis=1)
    return df


@pytest.fixture(scope="session")
def csv_sintetico(tmp_path_factory) -> Path:
    ruta = tmp_path_factory.mktemp("datos") / "viajes_sinteticos.csv"
    generar_viajes().to_csv(ruta, index=False)
    return ruta


@pytest.fixture(scope="session")
def entrenado(tmp_path_factory, csv_sintetico):
    """Entrena una vez (unos segundos) la versión v1 en una carpeta temporal."""
    from propinas.entrenamiento import entrenar
    carpeta = tmp_path_factory.mktemp("models")
    return entrenar(csv_sintetico, carpeta, version="v1", fraccion=1.0, max_iter=30, informar=lambda _: None)


@pytest.fixture
def viaje_base() -> pd.DataFrame:
    """Un viaje válido en Manhattan, un jueves a las 18:00."""
    return pd.DataFrame([{
        "VendorID": 2, "tpep_pickup_datetime": "2015-01-15 18:00:00",
        "tpep_dropoff_datetime": "2015-01-15 18:15:00", "passenger_count": 1,
        "trip_distance": 2.0, "pickup_longitude": -73.9855, "pickup_latitude": 40.7580,
        "RateCodeID": 1, "dropoff_longitude": -73.9712, "dropoff_latitude": 40.7644,
        "fare_amount": 10.0, "extra": 1.0, "tolls_amount": 0.0,
    }])
