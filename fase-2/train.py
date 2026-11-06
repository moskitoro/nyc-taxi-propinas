"""Entrena el modelo de propina generosa y guarda una nueva versión.

Ejemplos:
    python train.py --datos datos/yellow_tripdata_2015-01.csv
    python train.py --descargar                     # baja el CSV de Kaggle
    python train.py --datos datos/archivo.csv --fraccion 0.02 --modelos modelos
"""
from __future__ import annotations

import argparse
import sys
import time

from sklearn.model_selection import train_test_split

from propinas.datos import OBJETIVO, SEMILLA, UMBRAL_PROPINA, VARIABLES, cargar_csv, preparar_para_entrenar
from propinas.modelo import HIPERPARAMETROS_FASE1, construir_pipeline, evaluar, metricas_base
from propinas.versiones import guardar_version

DATASET_KAGGLE = "elemento/nyc-yellow-taxi-trip-data"
ARCHIVO_KAGGLE = "yellow_tripdata_2015-01.csv"


def fraccion(texto: str) -> float:
    valor = float(texto)
    if not 0 < valor <= 1:
        raise argparse.ArgumentTypeError("debe ser mayor que 0 y menor o igual a 1")
    return valor


def proporcion_prueba(texto: str) -> float:
    valor = float(texto)
    if not 0 < valor < 1:
        raise argparse.ArgumentTypeError("debe estar entre 0 y 1 (sin incluirlos)")
    return valor


def entero_positivo(texto: str) -> int:
    valor = int(texto)
    if valor < 1:
        raise argparse.ArgumentTypeError("debe ser un entero mayor o igual a 1")
    return valor


def leer_argumentos(argv=None):
    p = argparse.ArgumentParser(description="Entrena el modelo de propina generosa y lo guarda versionado.")
    origen = p.add_mutually_exclusive_group(required=True)
    origen.add_argument("--datos", help="Ruta al CSV de viajes de la NYC TLC.")
    origen.add_argument("--descargar", action="store_true",
                        help=f"Descarga {ARCHIVO_KAGGLE} desde Kaggle con kagglehub.")
    p.add_argument("--fraccion", type=fraccion, default=0.05,
                   help="Fracción de los viajes con tarjeta que se usa para entrenar (por defecto 0.05).")
    p.add_argument("--prueba", type=proporcion_prueba, default=0.20, help="Proporción del conjunto de prueba (por defecto 0.20).")
    p.add_argument("--semilla", type=int, default=SEMILLA)
    p.add_argument("--modelos", default="modelos", help="Carpeta donde se guardan las versiones (por defecto 'modelos').")
    p.add_argument("--max-iter", type=entero_positivo, default=HIPERPARAMETROS_FASE1["max_iter"],
                   help="Iteraciones del Gradient Boosting (por defecto, las de la Fase 1).")
    return p.parse_args(argv)


def main(argv=None) -> str:
    args = leer_argumentos(argv)
    inicio = time.time()

    ruta = args.datos
    if args.descargar:
        import kagglehub
        print(f"Descargando {ARCHIVO_KAGGLE} desde Kaggle...")
        ruta = kagglehub.dataset_download(DATASET_KAGGLE, path=ARCHIVO_KAGGLE)

    print(f"[1/4] Cargando {ruta} (muestra del {args.fraccion:.0%} de los viajes con tarjeta)")
    df_raw, resumen_carga = cargar_csv(ruta, fraccion=args.fraccion, semilla=args.semilla)
    print(f"      {resumen_carga['filas_archivo']:,} filas en el archivo -> {resumen_carga['filas_muestra']:,} en la muestra")

    print("[2/4] Limpiando y creando variables")
    X, y = preparar_para_entrenar(df_raw)
    if len(X) < 100 or y.nunique() < 2:
        raise ValueError(f"Después de limpiar quedaron {len(X)} viajes válidos: no alcanza para entrenar. "
                         "Revise el archivo o aumente --fraccion.")
    print(f"      {len(X):,} viajes válidos | {y.mean():.1%} con propina generosa")

    # La prueba se separa ANTES de entrenar: el preprocesamiento va dentro del Pipeline
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.prueba, stratify=y, random_state=args.semilla)

    print(f"[3/4] Entrenando Gradient Boosting con {len(X_train):,} viajes")
    hiper = {**HIPERPARAMETROS_FASE1, "max_iter": args.max_iter}
    pipe = construir_pipeline(hiper)
    pipe.fit(X_train, y_train)

    met_train = evaluar(pipe, X_train, y_train)
    met_test = evaluar(pipe, X_test, y_test)
    base = metricas_base(y_train, y_test)

    print("[4/4] Guardando versión")
    destino = guardar_version(pipe, {
        "modelo": "HistGradientBoostingClassifier (class_weight='balanced')",
        "hiperparametros": hiper,
        "umbral_propina": UMBRAL_PROPINA,
        "umbral_decision": 0.5,
        "objetivo": OBJETIVO,
        "variables": VARIABLES,
        "datos": {"origen": str(ruta), "fraccion": args.fraccion, "semilla": args.semilla, **resumen_carga,
                  "filas_entrenamiento": len(X_train), "filas_prueba": len(X_test)},
        "metricas_entrenamiento": met_train,
        "metricas_prueba": met_test,
        "modelo_base_prueba": base,
    }, carpeta=args.modelos)

    print("\n=========== RESULTADO ===========")
    print(f"Versión guardada:     {destino}")
    print(f"ROC-AUC prueba:       {met_test['roc_auc']:.4f}  (base 0.5000, entrenamiento {met_train['roc_auc']:.4f})")
    print(f"Exactitud balanceada: {met_test['exactitud_balanceada']:.4f}  (base {base['exactitud_balanceada']:.4f})")
    print(f"F1 macro:             {met_test['f1_macro']:.4f}")
    print(f"Tiempo total:         {time.time() - inicio:,.0f} s")
    return destino.name


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error: {e}")
