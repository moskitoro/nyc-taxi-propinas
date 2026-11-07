"""Almacenamiento y carga de versiones del modelo.

Cada versión es una carpeta con todo lo necesario para predecir y para saber
cómo se obtuvo:

    models/
    ├── v1/
    │   ├── modelo.joblib    # Pipeline completo: imputación + one-hot + Gradient Boosting
    │   ├── metrics.json     # métricas del modelo y del modelo base (misma partición)
    │   └── metadata.json    # datos, variables y tipos, parámetros, entorno, commit, sha256
    └── v2/ ...

Una versión existente nunca se sobrescribe: si el identificador ya existe, se rechaza.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

import joblib

ARCHIVO_MODELO = "modelo.joblib"
ARCHIVO_METRICAS = "metrics.json"
ARCHIVO_METADATOS = "metadata.json"
PATRON_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
RAIZ_CODIGO = Path(__file__).resolve().parents[1]          # carpeta fase-2


def sha256_archivo(ruta: str | Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def validar_identificador(version: str) -> str:
    if not PATRON_VERSION.fullmatch(version or ""):
        raise ValueError(f"Identificador de versión no válido: '{version}'. Use letras, números, "
                         "punto, guion o guion bajo (por ejemplo v1, v2 o v-docker).")
    return version


def siguiente_version(carpeta: str | Path) -> str:
    """v1, v2, ... según las carpetas vN que ya existan."""
    carpeta = Path(carpeta)
    numeros = [int(m.group(1)) for p in carpeta.glob("v*")
               if p.is_dir() and (m := re.fullmatch(r"v(\d+)", p.name))]
    return f"v{max(numeros, default=0) + 1}"


def comprobar_disponible(carpeta: str | Path, version: str) -> Path:
    """Se llama ANTES de entrenar, para no esperar minutos y luego fallar."""
    destino = Path(carpeta) / validar_identificador(version)
    if destino.exists():
        raise ValueError(f"La versión '{version}' ya existe en {destino}. No se sobrescribe: "
                         "use otro --version u omítalo para crear la siguiente automáticamente.")
    return destino


def listar_versiones(carpeta: str | Path) -> list[str]:
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        return []
    return sorted(p.name for p in carpeta.iterdir() if (p / ARCHIVO_MODELO).is_file())


def referencia_codigo() -> dict:
    """Identifica el código que entrenó la versión.

    - commit: hash de Git (o la variable de entorno GIT_COMMIT, que se pasa al construir
      la imagen Docker, porque dentro del contenedor no está la carpeta .git).
    - huella_sha256_codigo: sha256 de train.py, predict.py y propinas/*.py. Identifica el
      código aunque no haya Git o haya cambios sin commit.
    """
    commit, cambios, origen = None, None, "no disponible"
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ_CODIGO, capture_output=True,
                                text=True, timeout=10, check=True).stdout.strip() or None
        estado = subprocess.run(["git", "status", "--porcelain", "--", "train.py", "predict.py", "propinas"],
                                cwd=RAIZ_CODIGO, capture_output=True, text=True, timeout=10, check=True).stdout
        cambios, origen = bool(estado.strip()), "git"
    except (OSError, subprocess.SubprocessError):
        if os.environ.get("GIT_COMMIT"):
            commit, origen = os.environ["GIT_COMMIT"], "variable de entorno GIT_COMMIT (imagen Docker)"

    h = hashlib.sha256()
    archivos = [RAIZ_CODIGO / "train.py", RAIZ_CODIGO / "predict.py", *sorted((RAIZ_CODIGO / "propinas").glob("*.py"))]
    for ruta in archivos:
        if ruta.is_file():
            h.update(ruta.relative_to(RAIZ_CODIGO).as_posix().encode())
            h.update(ruta.read_bytes().replace(b"\r\n", b"\n"))   # igual en Windows y Linux
    return {"commit_git": commit, "origen_commit": origen, "cambios_sin_commit": cambios,
            "huella_sha256_codigo": h.hexdigest()}


def _escribir_json(ruta: Path, contenido: dict) -> None:
    ruta.write_text(json.dumps(contenido, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def guardar_version(pipe, metricas: dict, metadatos: dict, carpeta: str | Path, version: str) -> dict[str, Path]:
    """Crea la carpeta de la versión con el modelo, metrics.json y metadata.json.
    Si algo falla a mitad de camino, borra la carpeta para no dejar versiones incompletas."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    destino = comprobar_disponible(carpeta, version)
    destino.mkdir()                     # falla si otra ejecución la creó mientras tanto
    rutas = {"modelo": destino / ARCHIVO_MODELO, "metricas": destino / ARCHIVO_METRICAS,
             "metadatos": destino / ARCHIVO_METADATOS}
    try:
        joblib.dump(pipe, rutas["modelo"])
        _escribir_json(rutas["metricas"], {"version": version, **metricas})
        _escribir_json(rutas["metadatos"], {"version": version, **metadatos,
                                            "sha256_modelo": sha256_archivo(rutas["modelo"])})
    except BaseException:
        shutil.rmtree(destino, ignore_errors=True)
        raise
    return rutas


def resolver_ruta_modelo(ruta: str | Path) -> Path:
    """Acepta el archivo .joblib o la carpeta de la versión (models/v1)."""
    ruta = Path(ruta)
    if ruta.is_dir():
        ruta = ruta / ARCHIVO_MODELO
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el modelo: {ruta}. Revise la ruta de --model "
                                "(por ejemplo models/v1 o models/v1/modelo.joblib).")
    return ruta


def cargar_modelo(ruta: str | Path):
    """Carga una versión guardada. Devuelve (pipeline, metadatos o None).

    Si junto al modelo está metadata.json, verifica el sha256: un archivo alterado se rechaza.
    """
    ruta = resolver_ruta_modelo(ruta)
    ruta_meta = ruta.parent / ARCHIVO_METADATOS
    meta = None
    if ruta_meta.is_file():
        meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
        esperado = meta.get("sha256_modelo")
        if esperado and esperado != sha256_archivo(ruta):
            raise ValueError(f"El archivo {ruta} no coincide con el sha256 de {ruta_meta}: fue modificado.")
    try:
        pipe = joblib.load(ruta)
    except Exception as e:                          # archivo dañado o que no es un modelo
        raise ValueError(f"No se pudo leer el modelo {ruta}: {type(e).__name__}: {e}") from None
    if not hasattr(pipe, "predict_proba") or not hasattr(pipe, "feature_names_in_"):
        raise ValueError(f"{ruta} no es un modelo entrenado de este proyecto.")
    return pipe, meta


def entorno() -> dict:
    import numpy, pandas, scipy, sklearn
    return {"python": platform.python_version(), "sistema": platform.platform(terse=True),
            "librerias": {"scikit-learn": sklearn.__version__, "pandas": pandas.__version__,
                          "numpy": numpy.__version__, "scipy": scipy.__version__,
                          "joblib": joblib.__version__}}


def huella_dependencias() -> dict:
    ruta = RAIZ_CODIGO / "requirements.txt"
    return {"archivo": "fase-2/requirements.txt",
            "sha256": hashlib.sha256(ruta.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            if ruta.is_file() else None}

