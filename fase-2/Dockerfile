# Imagen para entrenar y predecir el modelo de propina generosa (Fase 2)
# Versión explícita de Python (la misma del entorno local y de la Fase 1), nunca "latest"
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Primero las dependencias, para aprovechar la caché de Docker
COPY requirements.txt .
RUN pip install -r requirements.txt

# Usuario sin privilegios: el contenedor no corre como root
RUN useradd --create-home --uid 1000 app

# Código, pruebas, CSV de ejemplo y versiones del modelo.
# Los datos completos NO se copian: se montan desde el computador al ejecutar.
COPY --chown=app:app propinas/ propinas/
COPY --chown=app:app train.py predict.py descargar_datos.py seleccion_modelo.py ./
COPY --chown=app:app tests/ tests/
COPY --chown=app:app data/ejemplo_prediccion.csv data/
COPY --chown=app:app models/ models/
RUN mkdir -p data models outputs && chown -R app:app /app

# Commit de Git del código (dentro de la imagen no está la carpeta .git). Se pasa al construir:
#   docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t nyc-taxi-propinas:fase2 .
ARG GIT_COMMIT=""
ENV GIT_COMMIT=${GIT_COMMIT}

USER app

# Sin ENTRYPOINT: el comando se escribe completo al ejecutar, por ejemplo
#   docker run --rm -v "${PWD}/data:/app/data:ro" -v "${PWD}/models:/app/models" nyc-taxi-propinas:fase2 python train.py --data data/yellow_tripdata_2015-01.csv ...
#   docker run --rm nyc-taxi-propinas:fase2 python -m pytest
# El entrenamiento NUNCA se ejecuta durante docker build.
CMD ["python", "predict.py", "--help"]
