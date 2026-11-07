"""Verifica con validación cruzada (solo entrenamiento) qué modelo se debe elegir.

Corrige la observación del profesor sobre la Fase 1: el notebook escogía el modelo
final con el mayor ROC-AUC del conjunto de PRUEBA. Aquí la elección se hace solo con
validación cruzada de 5 particiones sobre el ENTRENAMIENTO; la prueba no se toca.

Usa la misma muestra, limpieza y partición que train.py (semilla 42, 5 %, 80/20), así
que el entrenamiento es exactamente el de la versión v1.

Uso (desde la carpeta fase-2):
    python seleccion_modelo.py --data data/yellow_tripdata_2015-01.csv
"""
from __future__ import annotations

import argparse
import sys
import time

from propinas.datos import SEMILLA


def leer_argumentos(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compara los modelos candidatos de la Fase 1 con validación cruzada sobre el "
                    "entrenamiento, sin usar el conjunto de prueba.")
    p.add_argument("--data", required=True, help="CSV de viajes de la NYC TLC con la propina.")
    p.add_argument("--seed", type=int, default=SEMILLA, help=f"Semilla (por defecto {SEMILLA}).")
    p.add_argument("--sample-fraction", type=float, default=0.05,
                   help="Fracción de los viajes con tarjeta (por defecto 0.05, como v1).")
    p.add_argument("--test-size", type=float, default=0.20,
                   help="Proporción de prueba que se aparta y NO se usa (por defecto 0.20).")
    p.add_argument("--folds", type=int, default=5, help="Particiones de la validación cruzada (por defecto 5).")
    return p.parse_args(argv)


def candidatos(semilla: int) -> dict:
    """Los mismos candidatos del notebook de la Fase 1 (secciones 10 y 11)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from propinas.modelo import construir_pipeline, construir_preprocesamiento
    from propinas.variables import VARIABLES_CATEGORICAS, VARIABLES_NUMERICAS

    prepro_lineal = ColumnTransformer([
        ("num", Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())]),
         VARIABLES_NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), VARIABLES_CATEGORICAS),
    ])
    return {
        "Modelo base (clase mayoritaria)": Pipeline([("prepro", construir_preprocesamiento()),
                                                     ("modelo", DummyClassifier(strategy="most_frequent"))]),
        "Regresión logística": Pipeline([("prepro", prepro_lineal),
                                         ("modelo", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "Gradient Boosting por defecto": Pipeline([
            ("prepro", construir_preprocesamiento()),
            ("modelo", HistGradientBoostingClassifier(class_weight="balanced", random_state=semilla))]),
        "Gradient Boosting ajustado (v1)": construir_pipeline(semilla=semilla),
    }


def main(argv=None):
    args = leer_argumentos(argv)
    from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

    from propinas.datos import cargar_csv_entrenamiento
    from propinas.variables import preparar_para_entrenar

    print(f"Leyendo {args.data} (muestra del {args.sample_fraction:.0%}, semilla {args.seed})")
    df_raw, _ = cargar_csv_entrenamiento(args.data, fraccion=args.sample_fraction, semilla=args.seed)
    X, y = preparar_para_entrenar(df_raw)
    # Misma partición que train.py; X_test y y_test se descartan a propósito
    X_train, _, y_train, _ = train_test_split(X, y, test_size=args.test_size, stratify=y,
                                              random_state=args.seed)
    print(f"Validación cruzada de {args.folds} particiones sobre {len(X_train):,} viajes de entrenamiento "
          "(la prueba no se usa)\n")

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    metricas = {"roc_auc": "roc_auc", "exactitud_balanceada": "balanced_accuracy", "f1_macro": "f1_macro"}
    resultados = {}
    for nombre, pipe in candidatos(args.seed).items():
        inicio = time.time()
        r = cross_validate(pipe, X_train, y_train, cv=cv, scoring=metricas)
        resultados[nombre] = r
        print(f"{nombre:<33} ROC-AUC {r['test_roc_auc'].mean():.4f} ± {r['test_roc_auc'].std():.4f} | "
              f"exactitud bal. {r['test_exactitud_balanceada'].mean():.4f} | "
              f"F1 macro {r['test_f1_macro'].mean():.4f}   ({time.time() - inicio:,.0f} s)")

    elegido = max(resultados, key=lambda n: resultados[n]["test_roc_auc"].mean())
    print(f"\nModelo elegido por validación cruzada (mayor ROC-AUC medio): {elegido}")
    return resultados, elegido


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as e:
        sys.exit(f"Error: {e}")
