# Fase 2 – Scripts y Docker: propina generosa en taxis de Nueva York

[![Pruebas](https://github.com/moskitoro/nyc-taxi-propinas/actions/workflows/pruebas.yml/badge.svg)](https://github.com/moskitoro/nyc-taxi-propinas/actions/workflows/pruebas.yml)

**Proyecto Integrado – Modelos y Simulación de Sistemas I – Universidad de Antioquia (2026-2)** · Docente: Andrés Felipe Parra Barragán

| Integrante | Correo |
|---|---|
| Oscar Julian Toro Arroyave | <o.toro@udea.edu.co> |
| Esteban Barrera Sanabria | <esteban.barreras@udea.edu.co> |

**Versión entregada (Entrega 2):** etiqueta de Git [`entrega-2`](https://github.com/moskitoro/nyc-taxi-propinas/tree/entrega-2) en la rama `main`. Este identificador del código es distinto de las versiones del modelo guardadas en `models/`.

## 1. Problema y relación con la Fase 1

Clasificación binaria: a partir de las características de un viaje en taxi amarillo de Nueva York pagado con tarjeta, predecir si la propina será **generosa**.

| Etiqueta (`prediccion`) | Significado |
|---|---|
| `1` | Propina generosa: `tip_amount / fare_amount >= 0.20` |
| `0` | Propina no generosa: menor al 20 % de la tarifa |

La Fase 1 ([`fase-1/fase1_modelo_predictivo.ipynb`](../fase-1/fase1_modelo_predictivo.ipynb)) exploró los datos, comparó modelos y eligió un **Gradient Boosting** (`HistGradientBoostingClassifier` con `class_weight="balanced"`). La Fase 2 **no cambia el problema, los datos ni el modelo**: convierte el notebook en una aplicación que entrena y predice con comandos, sin abrir el notebook. Con la misma configuración (muestra del 5 %, semilla 42), `train.py` reproduce exactamente el modelo de la Fase 1 (sección 9).

## 2. Correcciones y cambios respecto a la Fase 1

| Tema | Fase 1 | Fase 2 |
|---|---|---|
| `fare_amount` | El informe inicial la marcaba como fuga de información. | Se conserva como predictora (ya corregido en la sección 8 del notebook): la tarifa se conoce antes de que el pasajero decida la propina; el modelo nunca recibe la propina. |
| Hiperparámetros | Búsqueda aleatoria con validación cruzada dentro del entrenamiento. | `train.py` usa los hiperparámetros que eligió esa búsqueda (no la repite); quedan en `metadata.json`. |
| Modelo base | Se evaluó una vez en el notebook. | Se evalúa en **cada** entrenamiento, con la misma partición y las mismas métricas, y se guarda en `metrics.json`. |
| Filas con datos imposibles | Se eliminan (solo tiene sentido al entrenar). | `train.py` las elimina igual que la Fase 1; `predict.py` **no elimina ninguna fila**: devuelve una predicción por viaje. |
| Celdas vacías al predecir | No se contemplaba (el notebook solo trabajaba con datos ya filtrados). | Numéricas: mediana del entrenamiento. Categóricas: se ignoran (one-hot en 0). Fecha vacía: las variables de tiempo quedan vacías y se imputan. |
| Validación de la entrada | No había. | Columnas obligatorias, tipos (números y fechas), archivo vacío, modelo inexistente o alterado: mensaje claro y código de salida distinto de 0. |
| Versionamiento | Un solo archivo `modelo_propina_generosa.joblib`. | Una carpeta por versión con modelo, métricas y metadatos; una versión existente nunca se sobrescribe. |

## 3. Estructura

```text
fase-2/
├── train.py                 # entrena y guarda una versión nueva
├── predict.py               # predice con una versión guardada
├── descargar_datos.py       # (opcional) descarga el CSV de Kaggle a data/
├── requirements.txt         # dependencias con versiones fijas
├── Dockerfile
├── .dockerignore
├── propinas/                # lógica compartida (train.py y predict.py solo leen argumentos)
│   ├── datos.py             #   lectura, esquema de entrada y validación
│   ├── variables.py         #   variable objetivo, limpieza y variables derivadas
│   ├── modelo.py            #   Pipeline (imputación + one-hot + Gradient Boosting), modelo base, métricas
│   ├── versiones.py         #   guardar y cargar versiones, sha256, commit de Git
│   ├── entrenamiento.py     #   proceso completo de entrenamiento
│   └── prediccion.py        #   predicción y escritura segura del CSV de salida
├── data/
│   └── ejemplo_prediccion.csv   # 5 viajes sin la propina ni la variable objetivo
├── models/                  # versiones del modelo (v1, v2), cada una con
│   └── v1/                  #   modelo.joblib, metrics.json y metadata.json
├── outputs/                 # predicciones generadas (no se suben a Git)
└── tests/                   # pruebas con pytest
```

## 4. Entorno local

Versión de Python: **3.13** (la misma de la imagen Docker y con la que se entrenó la Fase 1).

```powershell
# Windows – PowerShell, desde la carpeta fase-2
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# Linux / macOS – Bash, desde la carpeta fase-2
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` fija todas las versiones: numpy 2.1.3, pandas 2.2.3, scipy 1.15.3, scikit-learn 1.6.1, joblib 1.5.1, kagglehub 0.3.13 y pytest 8.4.2. Un modelo `.joblib` se debe cargar con la misma versión de scikit-learn con la que se entrenó.

## 5. Datos

**Fuente:** [NYC Yellow Taxi Trip Data](https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data) en Kaggle (registros de la NYC Taxi and Limousine Commission). **Archivo requerido:** `yellow_tripdata_2015-01.csv` (enero de 2015, unos 2 GB, 12.748.986 viajes). Por su tamaño no está en GitHub.

**Ubicación esperada:** `fase-2/data/yellow_tripdata_2015-01.csv`. Dos formas de obtenerlo:

1. **A mano:** entrar a la página de Kaggle (requiere una cuenta gratuita), descargar `yellow_tripdata_2015-01.csv` (si llega en `.zip`, descomprimirlo) y copiarlo a `fase-2/data/`.
2. **Con el script** (no necesita credenciales para este conjunto público):

   ```bash
   python descargar_datos.py --output-dir data
   ```

El entrenamiento solo necesita el archivo: no depende del notebook ni de una sesión de Colab o Kaggle. Las credenciales, si se usaran, van en variables de entorno y nunca en el repositorio ni en la imagen.

### Esquema del CSV de predicción

Una fila por viaje, con estas 13 columnas (el orden no importa y las mayúsculas tampoco: `VendorID` = `vendorid`). La propina y la variable objetivo **no** se necesitan. Ejemplo: [`data/ejemplo_prediccion.csv`](data/ejemplo_prediccion.csv).

| Columna | Tipo esperado |
|---|---|
| `VendorID` | entero, categórica (1 o 2) |
| `tpep_pickup_datetime`, `tpep_dropoff_datetime` | fecha y hora `AAAA-MM-DD HH:MM:SS` |
| `passenger_count` | entero |
| `trip_distance` | número (millas) |
| `pickup_longitude`, `pickup_latitude`, `dropoff_longitude`, `dropoff_latitude` | número (grados) |
| `RateCodeID` | entero, categórica (1 a 6) |
| `fare_amount`, `extra`, `tolls_amount` | número (USD) |

| Situación | Qué hace `predict.py` |
|---|---|
| Falta una columna obligatoria | Error claro con el nombre de la columna; código de salida 1; no crea archivo de salida. |
| Texto en una columna numérica o fecha no válida | Error claro indicando columna y filas; código de salida 1; no crea archivo de salida. |
| Celda vacía (columna presente) | No es error: numéricas → mediana del entrenamiento; categóricas → se ignoran; fecha vacía → variables de tiempo imputadas. |
| Coordenadas en 0 o fuera de Nueva York | Se tratan como faltantes (igual que en la Fase 1) y `coords_faltantes = 1`. |
| Columnas adicionales (por ejemplo `store_and_fwd_flag`, `tip_amount`) | Se ignoran y se informan en pantalla. |
| Categoría no vista al entrenar (por ejemplo `RateCodeID = 99`) | El one-hot la ignora (`handle_unknown="ignore"`): todas sus columnas quedan en 0. |
| Viajes con valores imposibles (distancia 0, etc.) | Se predicen igual; no se elimina ninguna fila. |

## 6. Uso

Todos los comandos se ejecutan desde la carpeta `fase-2`. Para ver las opciones:

```bash
python train.py --help
python predict.py --help
```

### Entrenar

```bash
python train.py --data data/yellow_tripdata_2015-01.csv --target propina_generosa --output-dir models --version v1 --seed 42
```

| Opción | Descripción | Por defecto |
|---|---|---|
| `--data` | CSV de la TLC con la propina (obligatorio). | — |
| `--target` | Variable objetivo; se construye con `tip_amount / fare_amount >= 0.20`. | `propina_generosa` |
| `--output-dir` | Carpeta de versiones. | `models` |
| `--version` | Identificador de la versión. **Si ya existe, se rechaza** (no se sobrescribe). Si se omite, se usa el siguiente libre (`v1`, `v2`, …). | siguiente libre |
| `--seed` | Semilla de la muestra, la separación y el modelo. | `42` |
| `--test-size` | Proporción de prueba. | `0.20` |
| `--sample-fraction` | Fracción de los viajes con tarjeta que se usa. | `0.05` (como la Fase 1) |
| `--max-iter` | Iteraciones del Gradient Boosting. | `300` (como la Fase 1) |

Pasos: verifica columnas y tipos → lee por bloques y deja solo pagos con tarjeta → toma la muestra → elimina registros imposibles → crea la variable objetivo y las variables derivadas → separa entrenamiento y prueba (estratificado) **antes** de ajustar cualquier transformación (no hay grupos naturales que respetar: cada fila es un viaje y el archivo no identifica taxis ni pasajeros) → entrena el Pipeline → evalúa el modelo y el modelo base con la misma prueba → guarda la versión y muestra las rutas de los archivos.

### Predecir

```bash
python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
```

| Opción | Descripción |
|---|---|
| `--model` | Versión a usar: el archivo (`models/v1/modelo.joblib`) o la carpeta (`models/v1`). Así se elige explícitamente la versión. |
| `--input` | CSV de viajes (esquema de la sección 5). |
| `--output` | CSV de salida; si la carpeta no existe, se crea. |
| `--threshold` | Probabilidad mínima para predecir `1` (por defecto 0,5, el mismo umbral de la Fase 1). |

`predict.py` no entrena ni reajusta nada: carga el Pipeline guardado (que incluye la imputación y el one-hot aprendidos), verifica su sha256 contra `metadata.json` y ordena las variables según el esquema del entrenamiento. El CSV de salida conserva el orden de entrada:

| Columna | Contenido |
|---|---|
| `fila` | Posición del viaje en el CSV de entrada (1 = primer viaje). |
| `prediccion` | `1` = propina generosa, `0` = no generosa. |
| `significado` | La etiqueta en palabras. |
| `probabilidad_propina_generosa` | Puntaje entre 0 y 1 (ver limitaciones). |

El archivo se escribe primero como temporal y se renombra al final, así que si el proceso falla no queda un CSV que parezca válido.

## 7. Versiones, métricas, metadatos y resultados

| Qué | Dónde |
|---|---|
| Versiones del modelo | `models/<versión>/modelo.joblib` (Pipeline completo: preprocesamiento + modelo) |
| Métricas del modelo y del modelo base | `models/<versión>/metrics.json` |
| Metadatos | `models/<versión>/metadata.json` |
| Predicciones | la ruta de `--output` (por ejemplo `outputs/predicciones.csv`) |

`metadata.json` registra: versión y fecha (UTC); tipo de problema, variable objetivo con sus etiquetas y algoritmo; variables de entrada con su tipo esperado, variables del modelo en orden y categorías aprendidas; tratamiento de faltantes; hiperparámetros y semilla; datos usados (fuente, archivo, tamaño, sha256 del archivo, filas del archivo, con tarjeta, en la muestra y válidas); estrategia de separación y tamaños de entrenamiento y prueba; métrica principal; versión de Python y de las librerías; referencia a `requirements.txt` (con su sha256); **commit de Git** del código y una huella sha256 de `train.py`, `predict.py` y `propinas/`; el comando ejecutado; y el sha256 del modelo.

Versiones entregadas (ambas generadas con `train.py` en Docker):

| Versión | Configuración | Propósito |
|---|---|---|
| `v1` | `--sample-fraction 0.05 --seed 42` | Misma configuración de la Fase 1: debe reproducir su modelo. |
| `v2` | `--sample-fraction 0.10 --seed 42` | El doble de datos, para ver si más datos mejoran el modelo. |

## 8. Docker

Imagen base `python:3.13-slim` (versión explícita), directorio de trabajo `/app`, usuario sin privilegios y **sin `ENTRYPOINT`**: el comando (`python train.py …` o `python predict.py …`) se escribe al ejecutar. El modelo **no** se entrena durante `docker build`. Los datos se montan desde el computador (solo lectura) y los modelos y predicciones se escriben en carpetas montadas, así que **persisten aunque el contenedor se elimine** (`--rm`).

### Windows – PowerShell (terminal usada por el equipo)

Desde `fase-2`, cada comando en una sola línea:

```powershell
docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t nyc-taxi-propinas:fase2 .

# (opcional) descargar los datos a data/
docker run --rm -v "${PWD}/data:/app/data" nyc-taxi-propinas:fase2 python descargar_datos.py --output-dir data

# entrenar: los datos entran de solo lectura y la versión queda en models/ del computador
docker run --rm -v "${PWD}/data:/app/data:ro" -v "${PWD}/models:/app/models" nyc-taxi-propinas:fase2 python train.py --data data/yellow_tripdata_2015-01.csv --target propina_generosa --output-dir models --version v-docker --seed 42

# predecir: el resultado queda en outputs/ del computador
docker run --rm -v "${PWD}/data:/app/data:ro" -v "${PWD}/models:/app/models:ro" -v "${PWD}/outputs:/app/outputs" nyc-taxi-propinas:fase2 python predict.py --model models/v-docker/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv

# pruebas dentro del contenedor
docker run --rm nyc-taxi-propinas:fase2 python -m pytest -v
```

### Linux / macOS – Bash

```bash
docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t nyc-taxi-propinas:fase2 .
mkdir -p models outputs
docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/models:/app/models" \
  nyc-taxi-propinas:fase2 python train.py \
  --data data/yellow_tripdata_2015-01.csv --target propina_generosa \
  --output-dir models --version v-docker --seed 42
docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/outputs:/app/outputs" \
  nyc-taxi-propinas:fase2 python predict.py \
  --model models/v-docker/modelo.joblib \
  --input data/ejemplo_prediccion.csv \
  --output outputs/predicciones.csv
```

El contenedor corre con el usuario `app` (uid 1000). En Linux, si su usuario no es el 1000 y aparece un error de permisos al escribir en `models/` u `outputs/`, agregue `--user "$(id -u):$(id -g)"` después de `docker run --rm`. La imagen también trae `models/` y el CSV de ejemplo, así que `docker run --rm nyc-taxi-propinas:fase2 python predict.py --model models/v1 --input data/ejemplo_prediccion.csv --output outputs/p.csv` funciona sin montajes (pero ese resultado se pierde al cerrar el contenedor).

`.dockerignore` deja por fuera de la imagen: `.git`, entornos virtuales, cachés, credenciales (`.env`, `kaggle.json`), los datos completos, `outputs/` y archivos del editor.

## 9. Pruebas

```bash
python -m pytest -v
```

Usan datos sintéticos con la estructura del CSV de la TLC (sin descargas), entrenan en carpetas temporales (`tmp_path`) con pocos datos y no modifican `models/`. Tardan alrededor de medio minuto.

| Archivo | Qué comprueba |
|---|---|
| `test_aplicacion.py` | **Las cinco pruebas mínimas:** (1) entrenar genera un modelo cargable, `metrics.json` y `metadata.json` completos; (2) el modelo cargado da las mismas probabilidades que antes de guardarlo (tolerancia 1e-12), una por fila y en orden, y el orden de las columnas no cambia el resultado; (3) una celda vacía se imputa exactamente con la mediana aprendida y un CSV con vacíos da una predicción por fila; (4) si falta `fare_amount` hay error claro, código distinto de 0 y no se crea el archivo de salida; (5) una segunda versión conserva la primera byte a byte, ambas cargan y un identificador repetido se rechaza sin tocar nada. |
| `test_variables.py` | Umbral del 20 % inclusivo; duración, velocidad, hora cíclica, día y distancia a Midtown en un viaje conocido; aeropuerto; coordenadas en 0 como faltantes; limpieza de registros imposibles; ninguna variable con fuga; la propina no altera las variables de entrada; columnas sin importar mayúsculas. |
| `test_cli.py` | `--help` de ambos scripts; entrenar y predecir desde la terminal; tipos incompatibles, archivo vacío o inexistente, modelo inexistente o alterado (sha256), entrenar sin `tip_amount`: mensaje claro, sin *traceback*, código distinto de 0 y sin archivo de salida; argumentos fuera de rango rechazados (código 2). |
| `test_versiones_publicadas.py` | Las versiones de `models/` cargan, su sha256 coincide y predicen el ejemplo; la que usa la configuración de la Fase 1 da exactamente sus probabilidades y métricas. |

Además, GitHub Actions (`.github/workflows/pruebas.yml`) corre las pruebas en cada cambio, construye la imagen y repite dentro del contenedor el flujo de la entrega: entrenar con una carpeta montada, predecir, verificar que los archivos persisten afuera y que una entrada inválida falla sin crear salida.

## 10. Resultados y limitaciones

Métrica principal: **ROC-AUC** en el conjunto de prueba (no depende del umbral ni del desbalance: 69 % de los viajes tienen propina generosa).

| Modelo (prueba) | ROC-AUC | Exactitud balanceada | F1 macro | Exactitud |
|---|---|---|---|---|
| Modelo base (siempre "generosa") | 0,5000 | 0,5000 | 0,4082 | 0,6899 |
| `v1` (5 %, igual a la Fase 1) | ⏳ | ⏳ | ⏳ | ⏳ |
| `v2` (10 %) | ⏳ | ⏳ | ⏳ | ⏳ |

El modelo supera al modelo base en las métricas que no se dejan engañar por la clase mayoritaria. Su exactitud es menor que la del modelo base porque este acierta el 69 % solo por predecir siempre "generosa" sin detectar nunca una propina baja.

**Reproducibilidad.** Con los mismos datos, semilla y versiones de librerías, el resultado es el mismo: la prueba `test_versiones_publicadas.py` comprueba que `v1` da exactamente las probabilidades del modelo de la Fase 1. La semilla controla la muestra, la separación y el Gradient Boosting (incluida su parada temprana interna). Entre computadores pueden aparecer diferencias numéricas mínimas porque el Gradient Boosting usa varios hilos (OpenMP) y operaciones de punto flotante.

**Limitaciones conocidas:**

1. ROC-AUC alrededor de 0,67: las características del viaje explican solo una parte de la propina; el resto depende de factores personales que no están en los datos.
2. Solo enero de 2015 y solo pagos con tarjeta (las propinas en efectivo no se registran). La separación es aleatoria dentro del mismo mes, no temporal.
3. Por el `class_weight="balanced"`, `probabilidad_propina_generosa` es un puntaje para ordenar viajes, no una probabilidad calibrada.
4. Se entrena con una muestra (5 % o 10 %) para que el proceso corra en un computador personal.
5. La validación revisa columnas y tipos, pero no rangos: un viaje de otra ciudad o de otro año se predice igual, sin aviso.

## 11. Evidencia de ejecución

⏳ Se completa con las salidas reales al entrenar `v1` y `v2` en Docker.

## 12. Trabajo colaborativo (Git Flow)

| Rama | Uso |
|---|---|
| `main` | Versión estable de cada entrega (etiqueta `entrega-2`). |
| `develop` | Integración del trabajo de la fase. |
| `feature/*` | Una tarea por rama, creada desde `develop` e integrada con un Pull Request revisado por el otro integrante. |

## 13. Fuentes consultadas y código externo

- Documentación de scikit-learn: `Pipeline`, `ColumnTransformer`, `SimpleImputer`, `OneHotEncoder`, `HistGradientBoostingClassifier`, `DummyClassifier`, `train_test_split`.
- Documentación de pandas (`read_csv` por bloques con `chunksize`, `to_datetime`), de `argparse` y de pytest (`tmp_path`, `pytest.raises`).
- Documentación de Docker: referencia de `Dockerfile`, `.dockerignore` y *bind mounts*.
- [Diccionario de datos de la NYC TLC](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf) y la página del conjunto de datos en Kaggle.
- Fórmula de *haversine* para la distancia entre coordenadas (fórmula estándar, implementada con NumPy).
- Herramientas de IA: se usó Claude (Anthropic) como asistente para organizar el código, las pruebas y la documentación; el equipo revisó, ejecutó y verificó los resultados.
