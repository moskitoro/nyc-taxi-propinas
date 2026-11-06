# Imagen para entrenar y predecir el modelo de propina generosa (Fase 2)
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Lo que baje "train.py --descargar" queda en el volumen de datos y no se pierde al cerrar el contenedor
    KAGGLEHUB_CACHE=/app/datos/kagglehub

WORKDIR /app

# Primero las dependencias, para aprovechar la caché de Docker
COPY requirements.txt .
RUN pip install -r requirements.txt

# Usuario sin privilegios: el contenedor no corre como root
RUN useradd --create-home --uid 1000 app

# Código, pruebas, datos de ejemplo y versiones del modelo (v1 = Fase 1, v2 = train.py)
COPY --chown=app:app propinas/ propinas/
COPY --chown=app:app train.py predict.py ./
COPY --chown=app:app tests/ tests/
COPY --chown=app:app ejemplos/ ejemplos/
COPY --chown=app:app modelos/ modelos/

RUN mkdir -p /app/datos && chown -R app:app /app
USER app

# Carpetas que se montan como volúmenes desde el computador
VOLUME ["/app/datos", "/app/modelos"]

# Uso:
#   docker run --rm imagen predict.py --entrada ejemplos/viajes_ejemplo.csv
#   docker run --rm -v <carpeta>/datos:/app/datos -v <carpeta>/modelos:/app/modelos imagen train.py --datos datos/archivo.csv
#   docker run --rm imagen -m pytest
ENTRYPOINT ["python"]
CMD ["predict.py", "--help"]
