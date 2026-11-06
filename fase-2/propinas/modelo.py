"""Definición del modelo (el mismo elegido en la Fase 1) y sus métricas."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .datos import SEMILLA, VARIABLES_CATEGORICAS, VARIABLES_NUMERICAS

# Mejores hiperparámetros encontrados en la Fase 1 (fase-1/modelo/metricas.json)
HIPERPARAMETROS_FASE1 = {
    "learning_rate": 0.1,
    "max_iter": 300,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 100,
    "l2_regularization": 0.0,
}


def construir_pipeline(hiperparametros: dict | None = None) -> Pipeline:
    """Preprocesamiento + Gradient Boosting dentro de un único Pipeline,
    para que imputación y codificación solo aprendan del entrenamiento."""
    params = {**HIPERPARAMETROS_FASE1, **(hiperparametros or {})}
    prepro = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), VARIABLES_NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), VARIABLES_CATEGORICAS),
    ])
    modelo = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEMILLA, **params)
    return Pipeline([("prepro", prepro), ("modelo", modelo)])


def evaluar(pipe, X, y, umbral: float = 0.5) -> dict:
    prob = pipe.predict_proba(X)[:, 1]
    pred = (prob >= umbral).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y, prob)), 4),
        "exactitud": round(float(accuracy_score(y, pred)), 4),
        "exactitud_balanceada": round(float(balanced_accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
    }


def metricas_base(y_train, y_test) -> dict:
    """Modelo de referencia: predecir siempre la clase mayoritaria del entrenamiento."""
    clase = int(np.mean(y_train) >= 0.5)
    pred = np.full(len(y_test), clase)
    return {
        "clase_predicha": clase,
        "exactitud": round(float(accuracy_score(y_test, pred)), 4),
        "exactitud_balanceada": round(float(balanced_accuracy_score(y_test, pred)), 4),
        "roc_auc": 0.5,
    }
