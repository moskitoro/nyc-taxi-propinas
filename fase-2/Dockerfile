# Imagen para entrenar y predecir el modelo de propina generosa (Fase 2)
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Primero las dependencias, para aprovechar la caché de Docker
COPY requirements.txt .
RUN pip install -r requirements.txt

# Código, pruebas, datos de ejemplo y versiones del modelo (v1 = Fase 1)
COPY propinas/ propinas/
COPY train.py predict.py ./
COPY tests/ tests/
COPY ejemplos/ ejemplos/
COPY modelos/ modelos/

# Carpetas que se montan como volúmenes desde el computador
RUN mkdir -p /app/datos
VOLUME ["/app/datos", "/app/modelos"]

# Uso:
#   docker run --rm imagen train.py --datos datos/yellow_tripdata_2015-01.csv
#   docker run --rm imagen predict.py --entrada ejemplos/viajes_ejemplo.csv
#   docker run --rm imagen -m pytest
ENTRYPOINT ["python"]
CMD ["predict.py", "--help"]
