"""Carga de datos y partición en entrenamiento / validación / prueba."""
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import N_PRUEBA, N_VALIDACION, OBJETIVO, RUTA_DATOS, SEMILLA


@dataclass
class Particion:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def cargar_datos(ruta=RUTA_DATOS) -> tuple[pd.DataFrame, pd.Series]:
    """Lee el CSV y devuelve (X, y). Elimina la columna 'index' si existe."""
    df = pd.read_csv(ruta)
    if "index" in df.columns:
        df = df.drop(columns=["index"])
    return df.drop(columns=[OBJETIVO]), df[OBJETIVO]


def particionar(X: pd.DataFrame, y: pd.Series) -> Particion:
    """Divide los datos en desarrollo/prueba y luego entrenamiento/validación."""
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=N_PRUEBA, random_state=SEMILLA
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=N_VALIDACION, random_state=SEMILLA
    )
    return Particion(X_train, y_train, X_val, y_val, X_test, y_test)
