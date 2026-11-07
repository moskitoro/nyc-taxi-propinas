"""Proceso completo de entrenamiento: datos -> versión guardada.

train.py solo lee los argumentos y llama a entrenar(). Así, la API REST de la
Fase 3 (/train) podrá reutilizar esta función sin copiar código.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sklearn.model_selection import train_test_split

from .datos import (ARCHIVO_DATOS, ETIQUETAS, FUENTE_DATOS, OBJETIVO, SEMILLA, TIPOS_ENTRADA,
                    UMBRAL_PROPINA, cargar_csv_entrenamiento)
from .modelo import (ALGORITMO, HIPERPARAMETROS_FASE1, METRICA_PRINCIPAL, UMBRAL_DECISION,
                     construir_modelo_base, construir_pipeline, evaluar)
from .variables import VARIABLES, VARIABLES_CATEGORICAS, VARIABLES_NUMERICAS, preparar_para_entrenar
from .versiones import (comprobar_disponible, entorno, guardar_version, huella_dependencias,
                        referencia_codigo, sha256_archivo, siguiente_version)

MINIMO_VIAJES = 100


@dataclass
class Resultado:
    version: str
    rutas: dict
    metricas: dict
    metadatos: dict
    pipeline: object


def entrenar(datos: str | Path, carpeta_salida: str | Path = "models", version: str | None = None,
             objetivo: str = OBJETIVO, semilla: int = SEMILLA, proporcion_prueba: float = 0.20,
             fraccion: float = 0.05, max_iter: int | None = None, comando: str | None = None,
             informar=print) -> Resultado:
    inicio = time.time()
    if objetivo != OBJETIVO:
        raise ValueError(f"La variable objetivo de este proyecto es '{OBJETIVO}' (se construye con "
                         f"tip_amount / fare_amount >= {UMBRAL_PROPINA}); no se reconoce '{objetivo}'.")
    datos = Path(datos)
    version = version or siguiente_version(carpeta_salida)
    comprobar_disponible(carpeta_salida, version)          # falla rápido si la versión ya existe

    informar(f"[1/5] Leyendo {datos} (muestra del {fraccion:.0%} de los viajes con tarjeta)")
    df_raw, resumen = cargar_csv_entrenamiento(datos, fraccion=fraccion, semilla=semilla)
    informar(f"      {resumen['filas_archivo']:,} filas en el archivo -> {resumen['filas_muestra']:,} en la muestra")

    informar("[2/5] Limpiando, creando la variable objetivo y las variables de entrada")
    X, y = preparar_para_entrenar(df_raw)
    if len(X) < MINIMO_VIAJES or y.nunique() < 2:
        raise ValueError(f"Después de limpiar quedaron {len(X)} viajes válidos y {y.nunique()} clase(s): "
                         "no alcanza para entrenar. Revise el archivo o aumente --sample-fraction.")
    informar(f"      {len(X):,} viajes válidos | {y.mean():.1%} con propina generosa")

    # La prueba se separa ANTES de ajustar cualquier transformación: imputación y one-hot
    # están dentro del Pipeline y aprenden solo del entrenamiento.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=proporcion_prueba, stratify=y, random_state=semilla)

    hiper = {**HIPERPARAMETROS_FASE1, **({"max_iter": max_iter} if max_iter else {})}
    informar(f"[3/5] Entrenando con {len(X_train):,} viajes (prueba: {len(X_test):,})")
    pipe = construir_pipeline(hiper, semilla=semilla).fit(X_train, y_train)
    base = construir_modelo_base().fit(X_train, y_train)

    informar("[4/5] Evaluando el modelo y el modelo base con el mismo conjunto de prueba")
    met_prueba, met_base = evaluar(pipe, X_test, y_test), evaluar(base, X_test, y_test)
    metricas = {
        "metrica_principal": METRICA_PRINCIPAL,
        "umbral_decision": UMBRAL_DECISION,
        "conjunto_evaluacion": f"prueba: {len(X_test)} viajes ({proporcion_prueba:.0%}) no usados para ajustar",
        "modelo": {"prueba": met_prueba, "entrenamiento": evaluar(pipe, X_train, y_train)},
        "modelo_base": {
            "descripcion": "DummyClassifier(strategy='most_frequent'): siempre predice la clase más "
                           "frecuente del entrenamiento (el mismo modelo base de la Fase 1)",
            "clase_predicha": int(base.named_steps["modelo"].classes_[base.named_steps["modelo"].class_prior_.argmax()]),
            "prueba": met_base,
        },
        "mejora_roc_auc_sobre_base": round(met_prueba["roc_auc"] - met_base["roc_auc"], 4),
    }

    informar(f"[5/5] Guardando la versión {version}")
    onehot = pipe.named_steps["prepro"].named_transformers_["cat"]
    metadatos = {
        "fecha_creacion": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "tipo_problema": "clasificación binaria",
        "variable_objetivo": {
            "nombre": OBJETIVO,
            "definicion": f"1 si tip_amount / fare_amount >= {UMBRAL_PROPINA}; 0 en otro caso",
            "etiquetas": {str(k): v for k, v in ETIQUETAS.items()},
            "proporcion_clase_1": round(float(y.mean()), 4),
        },
        "algoritmo": ALGORITMO,
        "parametros_modelo": hiper,
        "semilla": semilla,
        "variables_entrada": TIPOS_ENTRADA,
        "variables_modelo": {
            "orden": VARIABLES,
            "numericas": VARIABLES_NUMERICAS,
            "categoricas": VARIABLES_CATEGORICAS,
            "categorias_aprendidas": {c: [str(v) for v in cats]
                                      for c, cats in zip(VARIABLES_CATEGORICAS, onehot.categories_)},
        },
        "tratamiento_faltantes": {
            "numericas": "mediana del entrenamiento (SimpleImputer dentro del Pipeline)",
            "categoricas": "celda vacía o categoría no vista -> todas las columnas one-hot en 0",
            "coordenadas": "fuera de Nueva York o en 0 -> faltantes; indicador coords_faltantes = 1",
        },
        "datos": {
            "fuente": FUENTE_DATOS,
            "archivo_esperado": ARCHIVO_DATOS,
            "archivo_usado": datos.name,
            "tamano_bytes": datos.stat().st_size,
            "sha256_archivo": sha256_archivo(datos),
            "filtro": "payment_type = 1 (tarjeta) y fare_amount > 0",
            "fraccion_muestra": fraccion,
            **resumen,
            "filas_validas": int(len(X)),
        },
        "separacion": {
            "estrategia": "train_test_split aleatorio, estratificado por la variable objetivo, "
                          "hecho antes de ajustar cualquier transformación",
            "proporcion_prueba": proporcion_prueba,
            "filas_entrenamiento": int(len(X_train)),
            "filas_prueba": int(len(X_test)),
            "semilla": semilla,
        },
        "evaluacion": {"metrica_principal": METRICA_PRINCIPAL, "umbral_decision": UMBRAL_DECISION,
                       "roc_auc_prueba": met_prueba["roc_auc"], "roc_auc_base": met_base["roc_auc"],
                       "archivo_metricas": "metrics.json"},
        "entorno": entorno(),
        "dependencias": huella_dependencias(),
        "codigo": referencia_codigo(),
        "comando": comando,
        "duracion_entrenamiento_s": round(time.time() - inicio, 1),
    }
    rutas = guardar_version(pipe, metricas, metadatos, carpeta_salida, version)
    return Resultado(version, rutas, metricas, metadatos, pipe)
