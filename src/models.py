"""Ajuste y evaluación de regresiones lineales con intercepto."""
from sklearn.linear_model import LinearRegression

from .metrics import riesgo


def ajustar_y_evaluar(X_tr, y_tr, X_val, y_val):
    """Ajusta con entrenamiento; devuelve (modelo, riesgo_train, riesgo_val)."""
    modelo = LinearRegression(fit_intercept=True).fit(X_tr, y_tr)
    return (
        modelo,
        riesgo(y_tr, modelo.predict(X_tr)),
        riesgo(y_val, modelo.predict(X_val)),
    )


def riesgo_en_prueba(X_tr, y_tr, X_test, y_test, variables) -> float:
    """Ajusta usando `variables` y evalúa en el conjunto de prueba."""
    modelo = LinearRegression(fit_intercept=True).fit(X_tr[variables], y_tr)
    return riesgo(y_test, modelo.predict(X_test[variables]))
