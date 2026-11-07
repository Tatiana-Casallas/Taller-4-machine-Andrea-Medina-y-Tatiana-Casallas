"""Riesgo cuadrático empírico."""
import numpy as np


def riesgo(y_true, y_pred) -> float:
    """Calcula el error cuadrático medio."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))
