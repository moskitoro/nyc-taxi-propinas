"""Predice si los viajes de un CSV recibirán propina generosa (>= 20 % de la tarifa).

Uso (desde la carpeta fase-2):
    python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
"""
from __future__ import annotations

import argparse
import sys

from propinas.datos import COLUMNAS_ENTRADA


def probabilidad(texto: str) -> float:
    valor = float(texto)
    if not 0 <= valor <= 1:
        raise argparse.ArgumentTypeError("debe estar entre 0 y 1")
    return valor


def leer_argumentos(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Predice propina generosa para viajes de taxi de NYC con una versión guardada del modelo. "
                    "No entrena ni reajusta nada.",
        epilog="Ejemplo: python predict.py --model models/v1/modelo.joblib "
               "--input data/ejemplo_prediccion.csv --output outputs/predicciones.csv. "
               f"Columnas requeridas del CSV: {', '.join(COLUMNAS_ENTRADA)}.")
    p.add_argument("--model", required=True,
                   help="Versión a usar: el archivo (models/v1/modelo.joblib) o su carpeta (models/v1).")
    p.add_argument("--input", required=True,
                   help="CSV con los viajes. No necesita la propina ni la variable objetivo.")
    p.add_argument("--output", required=True,
                   help="CSV donde se guardan las predicciones (si la carpeta no existe, se crea).")
    p.add_argument("--threshold", type=probabilidad, default=0.5,
                   help="Probabilidad mínima para predecir 1 (propina generosa), entre 0 y 1 (por defecto 0.5).")
    return p.parse_args(argv)


def main(argv=None):
    args = leer_argumentos(argv)
    from propinas.datos import leer_csv_prediccion
    from propinas.prediccion import guardar_predicciones, predecir
    from propinas.versiones import cargar_modelo

    pipe, meta = cargar_modelo(args.model)
    if meta:
        print(f"Modelo: {args.model} (versión {meta['version']}, creada {meta['fecha_creacion']})")
    else:
        print(f"Modelo: {args.model} (sin metadata.json: no se pudo verificar su sha256)")

    viajes, ignoradas = leer_csv_prediccion(args.input)
    if ignoradas:
        print(f"Columnas ignoradas (el modelo no las usa): {', '.join(ignoradas)}")
    resultado = predecir(pipe, viajes, args.threshold)
    ruta = guardar_predicciones(resultado, args.output)

    print(resultado[["fila", "prediccion", "probabilidad_propina_generosa"]].head(20).to_string(index=False))
    if len(resultado) > 20:
        print(f"... (en pantalla solo las primeras 20 de {len(resultado)} filas)")
    print(f"{len(resultado)} predicciones guardadas en {ruta.as_posix()}")
    print("prediccion: 1 = propina generosa (>= 20 % de la tarifa), 0 = no generosa")
    return resultado


if __name__ == "__main__":
    try:
        main()
    except PermissionError as e:
        sys.exit(f"Error: no hay permiso para leer o escribir {e.filename}. Si usa Docker en Linux, "
                 'agregue --user "$(id -u):$(id -g)" al comando docker run (ver README).')
    except (OSError, ValueError) as e:            # OSError incluye archivo inexistente y permisos
        sys.exit(f"Error: {e}")
