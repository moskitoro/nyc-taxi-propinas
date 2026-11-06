"""Versionado de modelos.

Cada entrenamiento crea una carpeta nueva y nunca sobrescribe las anteriores:

    modelos/
    ├── registro.json          # lista de versiones y cuál es la actual
    ├── v1/
    │   ├── modelo.joblib
    │   └── metadatos.json     # fecha, datos, hiperparámetros, métricas, librerías, sha256
    └── v2/ ...
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

ARCHIVO_REGISTRO = "registro.json"


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def leer_registro(carpeta: str | Path) -> dict:
    ruta = Path(carpeta) / ARCHIVO_REGISTRO
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    return {"actual": None, "versiones": []}


def siguiente_version(carpeta: str | Path) -> str:
    carpeta = Path(carpeta)
    numeros = [int(m.group(1)) for p in carpeta.glob("v*") if (m := re.fullmatch(r"v(\d+)", p.name)) and p.is_dir()]
    return f"v{max(numeros, default=0) + 1}"


def versiones_librerias() -> dict:
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__}


def guardar_version(pipe, metadatos: dict, carpeta: str | Path = "modelos") -> Path:
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    version = siguiente_version(carpeta)
    destino = carpeta / version
    destino.mkdir()

    ruta_modelo = destino / "modelo.joblib"
    joblib.dump(pipe, ruta_modelo)

    meta = {
        "version": version,
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **metadatos,
        "librerias": metadatos.get("librerias") or versiones_librerias(),
        "sha256_modelo": _sha256(ruta_modelo),
    }
    (destino / "metadatos.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    registro = leer_registro(carpeta)
    registro["versiones"].append({
        "version": version,
        "fecha": meta["fecha"],
        "roc_auc_prueba": meta.get("metricas_prueba", {}).get("roc_auc"),
    })
    registro["actual"] = version
    (carpeta / ARCHIVO_REGISTRO).write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    return destino


def cargar_version(carpeta: str | Path = "modelos", version: str | None = None):
    """Carga la versión pedida (o la actual) y verifica que el archivo no cambió."""
    carpeta = Path(carpeta)
    registro = leer_registro(carpeta)
    version = version or registro["actual"]
    if version is None:
        raise FileNotFoundError(f"No hay modelos en '{carpeta}'. Primero ejecute train.py.")
    destino = carpeta / version
    ruta_modelo = destino / "modelo.joblib"
    if not ruta_modelo.exists():
        raise FileNotFoundError(f"No existe la versión '{version}' en '{carpeta}'.")
    meta = json.loads((destino / "metadatos.json").read_text(encoding="utf-8"))
    if meta.get("sha256_modelo") and meta["sha256_modelo"] != _sha256(ruta_modelo):
        raise ValueError(f"El archivo de la versión {version} no coincide con su sha256: fue modificado.")
    return joblib.load(ruta_modelo), meta
