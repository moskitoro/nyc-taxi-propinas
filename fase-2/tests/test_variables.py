"""Pruebas de la preparación de datos: variable objetivo, variables, limpieza y control de fuga."""
import numpy as np
import pandas as pd
import pytest

from propinas.datos import validar_columnas
from propinas.variables import (PROHIBIDAS, VARIABLES, calcular_basicas, crear_objetivo, crear_variables,
                                filtrar_invalidos, preparar_para_entrenar, preparar_para_predecir)


def test_objetivo_umbral_20_por_ciento_inclusivo():
    df = pd.DataFrame({"fare_amount": [10.0, 10.0, 10.0], "tip_amount": [1.99, 2.0, 3.0]})
    assert crear_objetivo(df).tolist() == [0, 1, 1]


def test_variables_de_un_viaje_conocido(viaje_base):
    d = crear_variables(calcular_basicas(viaje_base)).iloc[0]
    assert d["duracion_min"] == pytest.approx(15)
    assert d["velocidad_mph"] == pytest.approx(8)            # 2 millas en 15 minutos
    assert d["tarifa_por_milla"] == pytest.approx(5)
    assert d["hora_sin"] == pytest.approx(np.sin(2 * np.pi * 18 / 24))
    assert d["dia_semana"] == 3 and d["fin_de_semana"] == 0  # jueves
    assert d["dist_centro_recogida"] == pytest.approx(0, abs=1e-6)  # recoge en Midtown
    assert d["aeropuerto"] == 0 and d["coords_faltantes"] == 0


def test_viaje_al_aeropuerto_jfk(viaje_base):
    v = viaje_base.assign(dropoff_latitude=40.6413, dropoff_longitude=-73.7781)
    assert crear_variables(calcular_basicas(v)).iloc[0]["aeropuerto"] == 1


def test_coordenadas_en_cero_quedan_como_faltantes(viaje_base):
    v = viaje_base.assign(pickup_latitude=0.0, pickup_longitude=0.0)
    d = crear_variables(calcular_basicas(v)).iloc[0]
    assert np.isnan(d["pickup_latitude"]) and d["coords_faltantes"] == 1


def test_limpieza_elimina_registros_imposibles(viaje_base):
    malos = pd.concat([
        viaje_base,
        viaje_base.assign(trip_distance=0.0),                                   # distancia 0
        viaje_base.assign(passenger_count=0),                                   # 0 pasajeros
        viaje_base.assign(tpep_dropoff_datetime="2015-01-15 17:59:00"),         # duración negativa
        viaje_base.assign(RateCodeID=99),                                       # código inexistente
    ], ignore_index=True)
    assert len(filtrar_invalidos(calcular_basicas(malos))) == 1


def test_ninguna_variable_predictora_tiene_fuga():
    assert not set(VARIABLES) & set(PROHIBIDAS)


def test_la_propina_no_influye_en_X(csv_sintetico):
    df = pd.read_csv(csv_sintetico)
    X1, _ = preparar_para_entrenar(df)
    X2 = preparar_para_predecir(df.loc[X1.index].assign(tip_amount=999.0, total_amount=999.0))
    assert list(X1.columns) == VARIABLES
    pd.testing.assert_frame_equal(X1, X2)


def test_prediccion_no_elimina_filas(viaje_base):
    raro = pd.concat([viaje_base, viaje_base.assign(trip_distance=0.0)], ignore_index=True)
    X = preparar_para_predecir(raro)
    assert len(X) == 2 and not np.isinf(X.select_dtypes("number")).any().any()


def test_fecha_vacia_deja_variables_de_tiempo_vacias_para_imputar(viaje_base):
    d = crear_variables(calcular_basicas(viaje_base.assign(tpep_pickup_datetime=None))).iloc[0]
    for col in ["duracion_min", "velocidad_mph", "hora_sin", "dia_semana", "fin_de_semana"]:
        assert np.isnan(d[col]), col


def test_categoria_vacia_o_no_vista_queda_como_menos_uno(viaje_base):
    X = preparar_para_predecir(pd.concat([viaje_base.assign(VendorID=None), viaje_base.assign(VendorID=7)]))
    assert X["vendorid"].tolist() == ["-1", "7"]


def test_columnas_sin_importar_mayusculas_y_repetidas():
    validar_columnas(["VendorID", "Fare_Amount "], ["vendorid", "fare_amount"])     # no falla
    with pytest.raises(ValueError, match="repetidas"):
        validar_columnas(["VendorID", "vendorid"], ["vendorid"])
