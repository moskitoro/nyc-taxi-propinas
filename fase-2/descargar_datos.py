"""Descarga el CSV de enero de 2015 desde Kaggle y lo deja en la carpeta data/.

Uso (desde la carpeta fase-2):
    python descargar_datos.py --output-dir data

Es opcional: también se puede descargar a mano desde la página de Kaggle (ver README).
No necesita credenciales para este conjunto de datos público; si Kaggle las pidiera,
kagglehub las lee de las variables KAGGLE_USERNAME y KAGGLE_KEY (nunca van en el repositorio).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DATASET_KAGGLE = "elemento/nyc-yellow-taxi-trip-data"
ARCHIVO = "yellow_tripdata_2015-01.csv"


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(description=f"Descarga {ARCHIVO} (unos 2 GB) desde Kaggle con kagglehub.")
    p.add_argument("--output-dir", default="data", help="Carpeta destino (por defecto data).")
    args = p.parse_args(argv)

    destino = Path(args.output_dir) / ARCHIVO
    if destino.is_file():
        print(f"Ya existe {destino.as_posix()}: no se descarga de nuevo.")
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)

    import kagglehub
    print(f"Descargando {ARCHIVO} de https://www.kaggle.com/datasets/{DATASET_KAGGLE} (puede tardar varios minutos)...")
    ruta = Path(kagglehub.dataset_download(DATASET_KAGGLE, path=ARCHIVO))
    temporal = destino.with_suffix(".csv.parcial")
    shutil.copyfile(ruta, temporal)               # copia completa primero; luego se renombra
    temporal.replace(destino)
    print(f"Listo: {destino.as_posix()} ({destino.stat().st_size / 1e9:.2f} GB)")
    return destino


if __name__ == "__main__":
    try:
        main()
    except Exception as e:                        # red, permisos o Kaggle no disponible
        sys.exit(f"Error al descargar: {type(e).__name__}: {e}\n"
                 "Alternativa: descárguelo a mano desde Kaggle y póngalo en la carpeta data/ (ver README).")
