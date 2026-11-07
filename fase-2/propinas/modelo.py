"""Preprocesamiento + modelo (el elegido en la Fase 1), modelo base y métricas."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .datos import SEMILLA
from .variables import VARIABLES_CATEGORICAS, VARIABLES_NUMERICAS

ALGORITMO = "HistGradientBoostingClassifier (scikit-learn) dentro de un Pipeline con ColumnTransformer"
METRICA_PRINCIPAL = "roc_auc"
UMBRAL_DECISION = 0.5

# Mejores hiperparámetros encontrados en la Fase 1 con búsqueda aleatoria y validación
# cruzada sobre el entrenamiento (fase-1/modelo/metricas.json). No se vuelven a buscar.
HIPERPARAMETROS_FASE1 = {
    "learning_rate": 0.1,
    "max_iter": 300,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 100,
    "l2_regularization": 0.0,
    "class_weight": "balanced",
}


def construir_preprocesamiento() -> ColumnTransformer:
    """Mediana para numéricas y one-hot para categóricas. Se ajusta dentro del Pipeline,
    o sea, solo con los datos de entrenamiento."""
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), VARIABLES_NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), VARIABLES_CATEGORICAS),
    ])


def construir_pipeline(hiperparametros: dict | None = None, semilla: int = SEMILLA) -> Pipeline:
    params = {**HIPERPARAMETROS_FASE1, **(hiperparametros or {})}
    modelo = HistGradientBoostingClassifier(random_state=semilla, **params)
    return Pipeline([("prepro", construir_preprocesamiento()), ("modelo", modelo)])


def construir_modelo_base() -> Pipeline:
    """Modelo de referencia de la Fase 1: siempre predice la clase más frecuente del entrenamiento."""
    return Pipeline([("prepro", construir_preprocesamiento()),
                     ("modelo", DummyClassifier(strategy="most_frequent"))])


def evaluar(pipe, X, y, umbral: float = UMBRAL_DECISION) -> dict:
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
