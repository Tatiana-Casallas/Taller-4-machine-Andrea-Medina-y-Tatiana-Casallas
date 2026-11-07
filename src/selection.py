"""Criterios de selección de variables (todos usan solo entrenamiento)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from .config import ALFA, K_CORRELACION
from .metrics import riesgo
from .models import ajustar_y_evaluar


def ranking_correlacion(X_train: pd.DataFrame, y_train: pd.Series) -> pd.Series:
    """Correlación de Pearson con Salary, ordenada por valor absoluto."""
    corr = X_train.corrwith(y_train)
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def seleccion_correlacion(X_train, y_train, k: int = K_CORRELACION) -> list[str]:
    """Las k variables con mayor |correlación|."""
    return list(ranking_correlacion(X_train, y_train).head(k).index)


def pvalores_modelo_completo(X_train, y_train) -> pd.Series:
    """Valor p de la prueba t de cada coeficiente (sin la constante)."""
    ols = sm.OLS(y_train, sm.add_constant(X_train)).fit()
    return ols.pvalues.drop("const")


def seleccion_pruebas_t(X_train, y_train, alfa: float = ALFA) -> list[str]:
    """Variables con p < alfa en la regresión completa."""
    p = pvalores_modelo_completo(X_train, y_train)
    return list(p[p < alfa].index)


def seleccion_forward(X_train, y_train, X_val, y_val) -> pd.DataFrame:
    """Selección hacia adelante guiada por el riesgo de entrenamiento.

    En cada paso agrega la variable que más reduce el riesgo de entrenamiento,
    sin retirar las anteriores, hasta usar todas. Devuelve la tabla del paso a
    paso: N_vars, Variable_agregada, Riesgo_train, Riesgo_valid.
    """
    seleccionadas: list[str] = []
    filas = []
    for paso in range(1, X_train.shape[1] + 1):
        mejor_var, mejor_riesgo = None, np.inf
        for var in (v for v in X_train.columns if v not in seleccionadas):
            cols = seleccionadas + [var]
            m = LinearRegression().fit(X_train[cols], y_train)
            r = riesgo(y_train, m.predict(X_train[cols]))
            if r < mejor_riesgo:
                mejor_var, mejor_riesgo = var, r
        seleccionadas.append(mejor_var)
        _, r_tr, r_val = ajustar_y_evaluar(
            X_train[seleccionadas], y_train, X_val[seleccionadas], y_val
        )
        filas.append((paso, mejor_var, r_tr, r_val))
    return pd.DataFrame(
        filas, columns=["N_vars", "Variable_agregada", "Riesgo_train", "Riesgo_valid"]
    )


def mejor_forward(tabla: pd.DataFrame) -> tuple[list[str], float, float]:
    """Modelo forward con menor riesgo de validación: (variables, r_train, r_val)."""
    idx = tabla["Riesgo_valid"].idxmin()
    n = int(tabla.loc[idx, "N_vars"])
    variables = list(tabla["Variable_agregada"].iloc[:n])
    return variables, tabla.loc[idx, "Riesgo_train"], tabla.loc[idx, "Riesgo_valid"]
