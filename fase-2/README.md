# Fase 2 – Scripts y Docker

Convierte el modelo de la Fase 1 en una **aplicación de línea de comandos** que puede entrenarse y generar predicciones sin abrir el notebook. No es un modelo nuevo: `train.py` reproduce exactamente el procesamiento y el modelo elegido en la Fase 1 (Gradient Boosting con `class_weight="balanced"` y los mismos hiperparámetros).

## Estructura

```bash
fase-2/
├── train.py                  # entrena y guarda una nueva versión del modelo
├── predict.py                # predice a partir de un CSV de viajes
├── Dockerfile                # imagen para entrenar, predecir y probar
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
    ├── test_datos.py
    └── test_modelo.py
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

### Entrenar

Necesita el archivo `yellow_tripdata_2015-01.csv` (2 GB) de [Kaggle](https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data).

```bash
python train.py --datos datos/yellow_tripdata_2015-01.csv
python train.py --descargar           # lo descarga con kagglehub
```

| Opción | Descripción |
|---|---|
| `--datos` / `--descargar` | Ruta al CSV, o descarga automática desde Kaggle. Una de las dos es obligatoria. |
| `--fraccion` | Fracción de los viajes con tarjeta que se usa (por defecto 0,05, como en la Fase 1). |
| `--prueba` | Proporción del conjunto de prueba (por defecto 0,20). |
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

**v1** es el modelo entrenado en la Fase 1 (ROC-AUC 0,666 en prueba). Al ejecutar `train.py` se crea `v2`, y así sucesivamente. Para volver a una versión anterior: `python predict.py --entrada ... --version v1`.

## Docker

```bash
cd fase-2
docker build -t nyc-taxi-propinas:fase2 .
```

```bash
# Predecir con el ejemplo incluido en la imagen
docker run --rm -v "$(pwd)/modelos:/app/modelos" nyc-taxi-propinas:fase2 predict.py --entrada ejemplos/viajes_ejemplo.csv

# Entrenar (el CSV va en fase-2/datos/; la nueva versión queda en fase-2/modelos/)
docker run --rm -v "$(pwd)/datos:/app/datos" -v "$(pwd)/modelos:/app/modelos" \
  nyc-taxi-propinas:fase2 train.py --datos datos/yellow_tripdata_2015-01.csv

# Pruebas
docker run --rm nyc-taxi-propinas:fase2 -m pytest -v
```

En Windows (PowerShell), cambie `$(pwd)` por `${PWD}`. Las carpetas `datos` y `modelos` se montan como volúmenes: así el CSV no se copia a la imagen y los modelos entrenados quedan en el computador.

## Pruebas

```bash
python -m pytest -v
```

Las pruebas usan datos sintéticos con la misma estructura del CSV de la TLC, así que no necesitan descargar nada. Cubren:

| Archivo | Qué verifica |
|---|---|
| `test_datos.py` | Umbral del 20 % (inclusivo); cálculo de duración, velocidad, hora cíclica, día y distancia a Midtown en un viaje conocido; detección de viajes al aeropuerto; coordenadas en (0, 0) convertidas en faltantes; eliminación de registros imposibles; **ninguna variable con fuga** entre las predictoras; la propina no altera las variables de entrada; predicción sin perder filas; error claro si falta una columna. |
| `test_modelo.py` | `train.py` crea una versión con metadatos completos; el modelo supera al azar; reentrenar crea `v2` sin borrar `v1`; un modelo modificado se detecta por su sha256; `predict.py` devuelve una fila por viaje con probabilidades en [0, 1]; el umbral de decisión funciona; una versión inexistente da error. |
