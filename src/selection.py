"""Criterios de selección de variables (todos usan solo entrenamiento)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from .config import ALFA, K_CORRELACION
from .metrics import riesgo
from .models import ajustar_y_evaluar


def ranking_correlacion(X_train: pd.DataFrame, y_train: pd.Series) -> pd.Series:
    """Calcula correlaciones de Pearson y ordena por valor absoluto."""
    corr = X_train.corrwith(y_train)
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def seleccion_correlacion(X_train, y_train, k: int = K_CORRELACION) -> list[str]:
    """Devuelve las k variables con mayor correlación absoluta."""
    return list(ranking_correlacion(X_train, y_train).head(k).index)


def pvalores_modelo_completo(X_train, y_train) -> pd.Series:
    """Devuelve el valor p de cada coeficiente (sin la constante)."""
    ols = sm.OLS(y_train, sm.add_constant(X_train)).fit()
    return ols.pvalues.drop("const")


def seleccion_pruebas_t(X_train, y_train, alfa: float = ALFA) -> list[str]:
    """Selecciona variables con p < alfa en la regresión completa."""
    p = pvalores_modelo_completo(X_train, y_train)
    return list(p[p < alfa].index)


def seleccion_forward(X_train, y_train, X_val, y_val) -> pd.DataFrame:
    """Agrega en cada paso la variable que más reduce el riesgo de entrenamiento."""
    seleccionadas: list[str] = []
    filas = []
    for paso in range(1, X_train.shape[1] + 1):
        mejor_var, mejor_riesgo = None, np.inf
        for var in (v for v in X_train.columns if v not in seleccionadas):
            cols = seleccionadas + [var]
            modelo = LinearRegression().fit(X_train[cols], y_train)
            riesgo_train = riesgo(y_train, modelo.predict(X_train[cols]))
            if riesgo_train < mejor_riesgo:
                mejor_var, mejor_riesgo = var, riesgo_train
        seleccionadas.append(mejor_var)
        _, riesgo_train, riesgo_valid = ajustar_y_evaluar(
            X_train[seleccionadas], y_train, X_val[seleccionadas], y_val
        )
        filas.append((paso, mejor_var, riesgo_train, riesgo_valid))
    return pd.DataFrame(
        filas, columns=["N_vars", "Variable_agregada", "Riesgo_train", "Riesgo_valid"]
    )


def mejor_forward(tabla: pd.DataFrame) -> tuple[list[str], float, float]:
    """Devuelve el modelo forward con menor riesgo de validación."""
    idx = tabla["Riesgo_valid"].idxmin()
    n = int(tabla.loc[idx, "N_vars"])
    variables = list(tabla["Variable_agregada"].iloc[:n])
    return variables, tabla.loc[idx, "Riesgo_train"], tabla.loc[idx, "Riesgo_valid"]
