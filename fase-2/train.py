"""Entrena el modelo de propina generosa y guarda una versión nueva.

Uso (desde la carpeta fase-2):
    python train.py --data data/yellow_tripdata_2015-01.csv --target propina_generosa --output-dir models --version v1 --seed 42
"""
from __future__ import annotations

import argparse
import shlex
import sys

from propinas.datos import ARCHIVO_DATOS, OBJETIVO, SEMILLA
from propinas.modelo import HIPERPARAMETROS_FASE1


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


def leer_argumentos(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entrena el modelo de propina generosa (taxis amarillos de NYC) y guarda una "
                    "versión con modelo.joblib, metrics.json y metadata.json.",
        epilog=f"Ejemplo: python train.py --data data/{ARCHIVO_DATOS} --target {OBJETIVO} "
               "--output-dir models --version v1 --seed 42")
    p.add_argument("--data", required=True,
                   help=f"CSV de viajes de la NYC TLC con la propina (por ejemplo data/{ARCHIVO_DATOS}).")
    p.add_argument("--target", default=OBJETIVO, choices=[OBJETIVO],
                   help=f"Variable objetivo. Se construye como tip_amount / fare_amount >= 0.20 "
                        f"(por defecto {OBJETIVO}).")
    p.add_argument("--output-dir", default="models",
                   help="Carpeta donde se guardan las versiones (por defecto models).")
    p.add_argument("--version",
                   help="Identificador de la versión, por ejemplo v1. Si ya existe, se rechaza. "
                        "Si se omite, se usa el siguiente libre (v1, v2, ...).")
    p.add_argument("--seed", type=int, default=SEMILLA,
                   help=f"Semilla de la muestra, la separación y el modelo (por defecto {SEMILLA}).")
    p.add_argument("--test-size", type=proporcion_prueba, default=0.20,
                   help="Proporción del conjunto de prueba, entre 0 y 1 (por defecto 0.20).")
    p.add_argument("--sample-fraction", type=fraccion, default=0.05,
                   help="Fracción de los viajes con tarjeta que se usa, mayor que 0 y hasta 1 "
                        "(por defecto 0.05, como en la Fase 1).")
    p.add_argument("--max-iter", type=entero_positivo, default=HIPERPARAMETROS_FASE1["max_iter"],
                   help=f"Iteraciones del Gradient Boosting (por defecto {HIPERPARAMETROS_FASE1['max_iter']}, "
                        "como en la Fase 1).")
    return p.parse_args(argv)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = leer_argumentos(argv)
    from propinas.entrenamiento import entrenar          # import pesado solo después de validar argumentos

    r = entrenar(args.data, carpeta_salida=args.output_dir, version=args.version, objetivo=args.target,
                 semilla=args.seed, proporcion_prueba=args.test_size, fraccion=args.sample_fraction,
                 max_iter=args.max_iter, comando=shlex.join(["python", "train.py", *argv]))

    prueba, base = r.metricas["modelo"]["prueba"], r.metricas["modelo_base"]["prueba"]
    print("\n================ RESULTADO ================")
    print(f"Versión:               {r.version}")
    print(f"ROC-AUC prueba:        {prueba['roc_auc']:.4f}   (modelo base {base['roc_auc']:.4f})")
    print(f"Exactitud balanceada:  {prueba['exactitud_balanceada']:.4f}   (modelo base {base['exactitud_balanceada']:.4f})")
    print(f"F1 macro:              {prueba['f1_macro']:.4f}   (modelo base {base['f1_macro']:.4f})")
    print(f"Tiempo:                {r.metadatos['duracion_entrenamiento_s']:,.0f} s")
    print("Archivos generados:")
    for ruta in r.rutas.values():
        print(f"  {ruta.as_posix()}")
    return r


if __name__ == "__main__":
    try:
        main()
    except PermissionError as e:
        sys.exit(f"Error: no hay permiso para leer o escribir {e.filename}. Si usa Docker en Linux, "
                 'agregue --user "$(id -u):$(id -g)" al comando docker run (ver README).')
    except (OSError, ValueError) as e:            # OSError incluye archivo inexistente y permisos
        sys.exit(f"Error: {e}")
