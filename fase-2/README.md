# Fase 2 – Scripts y Docker

[![Pruebas](https://github.com/moskitoro/nyc-taxi-propinas/actions/workflows/pruebas.yml/badge.svg)](https://github.com/moskitoro/nyc-taxi-propinas/actions/workflows/pruebas.yml)

Convierte el modelo de la Fase 1 en una **aplicación de línea de comandos** que puede entrenarse y generar predicciones sin abrir el notebook. No es un modelo nuevo: `train.py` reproduce exactamente el procesamiento y el modelo elegido en la Fase 1 (Gradient Boosting con `class_weight="balanced"` y los mismos hiperparámetros).

## Estructura

```bash
fase-2/
├── train.py                  # entrena y guarda una nueva versión del modelo
├── predict.py                # predice a partir de un CSV de viajes
├── Dockerfile                # imagen para entrenar, predecir y probar
├── .dockerignore             # deja por fuera de la imagen los datos y archivos temporales
├── requirements.txt          # versiones fijadas (las mismas de la Fase 1)
├── propinas/                 # código compartido por train.py y predict.py
│   ├── datos.py              #   carga, limpieza, variables y control de fuga
│   ├── modelo.py             #   pipeline y métricas
│   └── versiones.py          #   versionado de modelos
├── modelos/                  # versiones del modelo
│   ├── registro.json         #   historial de versiones y versión actual
│   └── v1/                   #   modelo de la Fase 1
│       ├── modelo.joblib
│       └── metadatos.json
├── ejemplos/
│   └── viajes_ejemplo.csv    # 5 viajes para probar predict.py
└── tests/                    # pruebas con pytest
    ├── conftest.py           #   datos sintéticos (no necesita descargar Kaggle)
    ├── test_datos.py         #   limpieza, variables y control de fuga
    ├── test_modelo.py        #   entrenamiento, versionado y predicción
    ├── test_versiones_publicadas.py  # v1 y v2 del repo dan las predicciones documentadas
    └── test_cli.py           #   comandos en la terminal y mensajes de error
```

## Instalación (sin Docker)

Requiere Python 3.13 (es la versión con la que se entrenó el modelo de la Fase 1).

```bash
cd fase-2
pip install -r requirements.txt
```

## Uso

### Predecir

```bash
python predict.py --entrada ejemplos/viajes_ejemplo.csv
python predict.py --entrada ejemplos/viajes_ejemplo.csv --salida predicciones.csv
```

| Opción | Descripción |
|---|---|
| `--entrada` | CSV con los viajes. Usa las columnas de la NYC TLC; la propina **no** se necesita (si viene, se ignora). |
| `--salida` | CSV donde se guardan las predicciones. Si se omite, se muestran en pantalla. |
| `--version` | Versión del modelo, por ejemplo `v1`. Por defecto, la actual de `modelos/registro.json`. |
| `--modelo` | Ruta directa a un `.joblib`, por ejemplo `../fase-1/modelo/modelo_propina_generosa.joblib`. |
| `--umbral` | Probabilidad mínima para predecir "generosa" (por defecto 0,5). |

Columnas obligatorias de entrada: `VendorID`, `tpep_pickup_datetime`, `tpep_dropoff_datetime`, `passenger_count`, `trip_distance`, `pickup_longitude`, `pickup_latitude`, `RateCodeID`, `dropoff_longitude`, `dropoff_latitude`, `fare_amount`, `extra`, `tolls_amount`.

La salida agrega `prob_propina_generosa` (probabilidad entre 0 y 1) y `propina_generosa_predicha` (0 o 1). A diferencia del entrenamiento, `predict.py` **no elimina filas**: los valores imposibles (coordenadas en 0, distancia 0) se tratan como faltantes y el pipeline los imputa.

Antes de predecir, `predict.py` revisa el archivo y, si hay un problema, lo explica sin volcados de Python: columnas faltantes, archivo sin viajes, valores que no son números o fechas no válidas (indicando las filas). También rechaza un `--umbral` fuera de [0, 1]. Si la carpeta de `--salida` no existe, la crea.

### Entrenar

Necesita el archivo `yellow_tripdata_2015-01.csv` (2 GB) de [Kaggle](https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data).

```bash
python train.py --datos datos/yellow_tripdata_2015-01.csv
python train.py --descargar           # lo descarga con kagglehub
```

| Opción | Descripción |
|---|---|
| `--datos` / `--descargar` | Ruta al CSV, o descarga automática desde Kaggle. Una de las dos es obligatoria. |
| `--fraccion` | Fracción de los viajes con tarjeta que se usa, mayor que 0 y hasta 1 (por defecto 0,05, como en la Fase 1). |
| `--prueba` | Proporción del conjunto de prueba, entre 0 y 1 (por defecto 0,20). |
| `--semilla` | Semilla aleatoria (por defecto 42). |
| `--max-iter` | Iteraciones del Gradient Boosting (por defecto 300, como en la Fase 1). |
| `--modelos` | Carpeta de versiones (por defecto `modelos`). |

Pasos que ejecuta: carga por bloques y filtra pagos con tarjeta → limpia registros imposibles → crea las variables → separa entrenamiento y prueba (80/20 estratificado) **antes** de entrenar → entrena el pipeline → evalúa contra el modelo base → guarda una nueva versión.

## Versionado del modelo

Cada entrenamiento crea una carpeta nueva (`v1`, `v2`, …) y **nunca sobrescribe** las anteriores. `registro.json` guarda el historial y cuál es la versión actual, que es la que usa `predict.py` por defecto.

Cada `metadatos.json` registra:

- fecha, origen de los datos, fracción, semilla y número de filas;
- hiperparámetros y variables usadas;
- métricas en entrenamiento y prueba, y las del modelo base;
- versiones de Python, scikit-learn, pandas y NumPy;
- el **sha256** del archivo del modelo. `predict.py` lo verifica al cargar y se niega a usar un modelo alterado.

Versiones actuales del repositorio:

| Versión | Origen | Fecha | ROC-AUC prueba |
|---|---|---|---|
| `v1` | Modelo entrenado en la Fase 1 con el notebook (Colab) | 2026-09-30 04:36 UTC | 0,6661 |
| `v2` (actual) | `python train.py --descargar` en Colab | 2026-11-06 21:16 UTC | 0,6661 |

Las fechas se guardan en UTC, sin depender de la zona horaria del equipo donde se entrena.

Para usar una versión anterior: `python predict.py --entrada ... --version v1`. Cada nuevo entrenamiento crea `v3`, `v4`, …

## Docker

```bash
cd fase-2
docker build -t nyc-taxi-propinas:fase2 .
```

La imagen ya incluye las versiones de `modelos/`, así que puede predecir sin montar nada:

```bash
# Predecir con el ejemplo incluido en la imagen (versión actual, o una específica)
docker run --rm nyc-taxi-propinas:fase2 predict.py --entrada ejemplos/viajes_ejemplo.csv
docker run --rm nyc-taxi-propinas:fase2 predict.py --entrada ejemplos/viajes_ejemplo.csv --version v1

# Entrenar con un CSV propio (va en fase-2/datos/; la nueva versión queda en fase-2/modelos/)
docker run --rm -v "$(pwd)/datos:/app/datos" -v "$(pwd)/modelos:/app/modelos" \
  nyc-taxi-propinas:fase2 train.py --datos datos/yellow_tripdata_2015-01.csv

# Entrenar descargando desde Kaggle (el CSV queda guardado en fase-2/datos/kagglehub/ para la próxima vez)
docker run --rm -v "$(pwd)/datos:/app/datos" -v "$(pwd)/modelos:/app/modelos" \
  nyc-taxi-propinas:fase2 train.py --descargar

# Pruebas
docker run --rm nyc-taxi-propinas:fase2 -m pytest -v
```

En Windows (PowerShell), cambie `$(pwd)` por `${PWD}` y la `\` del final de línea por una comilla invertida `` ` ``. Las carpetas `datos` y `modelos` se montan como volúmenes: así el CSV no se copia a la imagen, la descarga de Kaggle no se pierde al cerrar el contenedor y los modelos entrenados quedan en el computador.

El contenedor corre con un usuario sin privilegios (`app`, uid 1000), no como root. En Linux, si su usuario no es el 1000 y Docker no puede escribir en las carpetas montadas, agregue `--user "$(id -u):$(id -g)"` al `docker run`.

## Pruebas

```bash
python -m pytest -v
```

Las pruebas usan datos sintéticos con la misma estructura del CSV de la TLC, así que no necesitan descargar nada. Cubren:

| Archivo | Qué verifica |
|---|---|
| `test_datos.py` | Umbral del 20 % (inclusivo); cálculo de duración, velocidad, hora cíclica, día y distancia a Midtown en un viaje conocido; detección de viajes al aeropuerto; coordenadas en (0, 0) convertidas en faltantes; eliminación de registros imposibles; **ninguna variable con fuga** entre las predictoras; la propina no altera las variables de entrada; predicción sin perder filas; error claro si falta una columna. |
| `test_modelo.py` | `train.py` crea una versión con metadatos completos; el modelo supera al azar; reentrenar crea `v2` sin borrar `v1`; un modelo modificado se detecta por su sha256; `predict.py` devuelve una fila por viaje con probabilidades en [0, 1]; el umbral de decisión funciona; una versión inexistente da error. |
| `test_versiones_publicadas.py` | `v1` y `v2` del repositorio cargan y su sha256 coincide; ambas dan exactamente las probabilidades documentadas; `v2` (train.py) tiene las mismas métricas que `v1` (Fase 1). |
| `test_cli.py` | `train.py` y `predict.py` funcionan como comandos de terminal; valores de texto, fechas no válidas, archivo vacío o inexistente dan un mensaje claro sin traceback; argumentos fuera de rango se rechazan. |

En total son **34 pruebas**. Además, GitHub Actions (`.github/workflows/pruebas.yml`) las corre en cada cambio del repositorio, construye la imagen Docker y predice con `v1` y `v2`; el resultado se ve en la insignia del inicio y en la pestaña **Actions**.

## Resultados y verificación

### `train.py` reproduce exactamente la Fase 1

Al ejecutar `python train.py --descargar` en Colab (misma muestra del 5 %, semilla 42 e hiperparámetros) se obtuvo `v2`, con las **mismas métricas** que el modelo del notebook:

| Métrica | Fase 1 – notebook (`v1`) | `train.py` (`v2`) |
|---|---|---|
| Viajes de entrenamiento / prueba | 313.292 / 78.323 | 313.292 / 78.323 |
| ROC-AUC prueba | 0,6661 | 0,6661 |
| ROC-AUC entrenamiento | 0,6831 | 0,6831 |
| Exactitud balanceada | 0,6106 | 0,6106 |
| F1 macro | 0,5878 | 0,5878 |
| Tiempo de entrenamiento | — | 91 s |

Además, `v1` y `v2` producen **las mismas probabilidades** sobre `ejemplos/viajes_ejemplo.csv`:

| Tarifa (USD) | Distancia (millas) | Probabilidad `v1` | Probabilidad `v2` | Predicción |
|---|---|---|---|---|
| 10,0 | 1,8 | 0,5275 | 0,5275 | 1 |
| 52,0 | 17,3 | 0,5238 | 0,5238 | 1 |
| 11,5 | 2,6 | 0,5190 | 0,5190 | 1 |
| 5,5 | 0,7 | 0,6425 | 0,6425 | 1 |
| 29,5 | 9,1 | 0,2777 | 0,2777 | 0 |

El último viaje tiene coordenadas en (0, 0): `predict.py` no lo descarta, las trata como faltantes y el pipeline las imputa.

### Ejecución verificada

| Entorno | Construcción | Pruebas | `predict.py` |
|---|---|---|---|
| Google Colab (Python 3.13) | — | 34 passed | `v1` y `v2` |
| Docker Desktop en Windows (imagen `python:3.13-slim`) | `docker build` correcto | 34 passed | `v1` y `v2` |
| GitHub Actions (Ubuntu) | `docker build` en cada cambio | ver insignia | `v1` y `v2` |
