# Predicción de propina generosa en taxis de Nueva York

**Proyecto Integrado – Modelos y Simulación de Sistemas I – Universidad de Antioquia (2026)**
Docente: Andrés Felipe Parra Barragán

| Integrante | Correo |
|---|---|
| Oscar Julian Toro Arroyave | o.toro@udea.edu.co |
| Esteban Barrera Sanabria | esteban.barreras@udea.edu.co |

---

## 1. Problema

Predecir si un pasajero de taxi amarillo en Nueva York dejará una **propina generosa** a partir de las características de su viaje (distancia, duración, hora, zona, número de pasajeros, tarifa, entre otras). Es útil para que conductores y plataformas de transporte anticipen el comportamiento de propina de sus usuarios.

- **Tipo de problema:** clasificación binaria (aprendizaje supervisado).
- **Variable objetivo:** `propina_generosa`
  - `1` si la propina es **igual o mayor al 20 %** de la tarifa (`tip_amount / fare_amount >= 0.20`).
  - `0` si es menor al 20 %.
- **No es un problema de series temporales:** cada viaje se predice por sí mismo a partir de sus características.

## 2. Datos

- **Fuente:** [NYC Yellow Taxi Trip Data](https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data) (Kaggle), construido a partir de los *Trip Record Data* de la NYC Taxi and Limousine Commission (TLC).
- **Archivo usado:** `yellow_tripdata_2015-01.csv` (enero de 2015), con **12.748.986 viajes** y **19 columnas**.
- **Filtro:** solo viajes **pagados con tarjeta** y con tarifa mayor a cero (7.880.666 registros). En los registros de la TLC las propinas en efectivo no quedan registradas (aparecen como cero); incluir esos viajes haría que el modelo aprendiera el método de pago en lugar del comportamiento de propina.
- **Muestra de trabajo:** muestra aleatoria reproducible del 5 % de los viajes filtrados (semilla 42), leída por bloques para no cargar los 2 GB en memoria.
- **Balance de clases:** aproximadamente dos tercios de los viajes tienen propina generosa y un tercio no.

Los datos **no se suben al repositorio** por su tamaño: el notebook los descarga automáticamente desde Kaggle.

## 3. Estructura del repositorio

```
nyc-taxi-propinas/
├── README.md
├── requirements.txt
└── fase-1/
    ├── fase1_modelo_predictivo.ipynb   # notebook ejecutable
    └── modelo/
        ├── modelo_propina_generosa.joblib   # pipeline entrenado (preprocesamiento + modelo)
        └── metricas.json                    # métricas, variables y versiones
```

## 4. Fase 1 – Modelo predictivo

### Cómo ejecutar

**Opción A – Google Colab (recomendada):**

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moskitoro/nyc-taxi-propinas/blob/main/fase-1/fase1_modelo_predictivo.ipynb)

Abrir el enlace y usar *Entorno de ejecución → Ejecutar todas*. Tarda entre 10 y 15 minutos.

**Opción B – Local:**

```bash
git clone https://github.com/moskitoro/nyc-taxi-propinas.git
cd nyc-taxi-propinas
pip install -r requirements.txt
jupyter notebook fase-1/fase1_modelo_predictivo.ipynb
```

### Qué hace el notebook

1. **Carga** por bloques, filtro a pagos con tarjeta y muestreo reproducible.
2. **Exploración:** distribución de la variable objetivo y de la tasa de propina, tasa de propina generosa por hora, día, proveedor, código de tarifa, pasajeros, distancia y tarifa.
3. **Calidad de datos:**
   - Coordenadas en (0, 0) o fuera de Nueva York (≈ 2 % de los viajes): se convierten en **valores faltantes** y se imputan con la mediana dentro del pipeline; se agrega el indicador `coords_faltantes`.
   - Se eliminan registros imposibles: distancia ≤ 0 o ≥ 100 millas, duración < 1 min o > 3 h, pasajeros fuera de 1 a 6, propina negativa, tarifa ≥ 500 USD, códigos de tarifa inexistentes y velocidades ≥ 80 mph.
4. **Ingeniería de variables:** duración, velocidad, tarifa por milla, hora cíclica (seno/coseno), día de la semana, fin de semana, distancia a Midtown Manhattan, indicador de aeropuerto (JFK, LaGuardia, Newark).
5. **Modelos:** modelo base (clase mayoritaria), regresión logística y Gradient Boosting (`HistGradientBoostingClassifier`), con validación cruzada estratificada de 5 particiones y ajuste de hiperparámetros por búsqueda aleatoria.
6. **Evaluación final** en un conjunto de prueba reservado (20 %), usado una sola vez.
7. **Importancia de variables** por permutación.
8. **Guardado** del pipeline completo con `joblib`.

### Variables

**Predictoras:** `fare_amount`, `trip_distance`, `duracion_min`, `velocidad_mph`, `tarifa_por_milla`, `extra`, `tolls_amount`, `passenger_count`, `hora_sin`, `hora_cos`, `dia_semana`, `fin_de_semana`, coordenadas de recogida y destino, `dist_centro_recogida`, `dist_centro_destino`, `aeropuerto`, `coords_faltantes`, `vendorid`, `ratecodeid`.

**Excluidas:**

| Variable | Motivo |
|---|---|
| `tip_amount` | Es la propina; con ella se construye la variable objetivo (fuga directa). |
| `total_amount` | Incluye la propina (fuga directa). |
| `payment_type` | Después del filtro solo vale 1 (tarjeta); no aporta información. |
| `mta_tax`, `improvement_surcharge` | Prácticamente constantes. |
| `store_and_fwd_flag` | Prácticamente constante; describe la transmisión del registro, no el viaje. |
| Fechas originales | Se reemplazan por variables derivadas (hora, día, duración). |

> **Corrección respecto al informe inicial:** `fare_amount` **sí se usa** como predictora. La tarifa se conoce antes de que el pasajero decida la propina (aparece en la pantalla del taxi), así que no contiene la respuesta. El modelo recibe la tarifa, pero nunca la propina.

### Control de fuga de información

1. Se excluyen todas las variables que contienen la propina (`tip_amount`, `total_amount`) y se verifica con un `assert` en el código.
2. El conjunto de prueba se separa **antes** de cualquier ajuste y se usa una sola vez al final.
3. Imputación, escalado y codificación van **dentro de un `Pipeline`**, así que solo aprenden de los datos de entrenamiento en cada partición de la validación cruzada.
4. Se trabaja solo con pagos con tarjeta para que el modelo no aprenda el método de pago.

### Resultados (conjunto de prueba)

| Modelo | ROC-AUC | Exactitud balanceada | F1 macro |
|---|---|---|---|
| Base (clase mayoritaria) | 0,500 | 0,500 | — |
| Regresión logística | — | — | — |
| Gradient Boosting | — | — | — |
| Gradient Boosting ajustado | — | — | — |

*Los valores se completan con `fase-1/modelo/metricas.json`, que genera el notebook al ejecutarse.*

### Limitaciones

1. Solo aplica a viajes pagados con tarjeta; no se extiende a pagos en efectivo.
2. Solo enero de 2015: el comportamiento de propina puede cambiar según la temporada.
3. La propina depende de factores personales que no están en los datos (costumbres del pasajero, satisfacción con el servicio), lo que pone un límite a lo que cualquier modelo puede predecir con las características del viaje.
4. La ubicación viene como coordenadas y no como zonas; se aproxima con distancias a puntos de referencia.
