"""Riesgo cuadrático empírico."""
import numpy as np


def riesgo(y_true, y_pred) -> float:
    """R̂_S(h) = (1/|S|) Σ_{i∈S} (y_i − h(x_i))²"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))
