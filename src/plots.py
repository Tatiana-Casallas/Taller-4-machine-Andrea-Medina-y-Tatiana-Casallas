"""Gráficas del taller."""
from pathlib import Path

import matplotlib.pyplot as plt


def graficar_forward(tabla, ruta: Path | None = None, mostrar: bool = False):
    """Riesgo de entrenamiento y validación vs. número de variables."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(tabla["N_vars"], tabla["Riesgo_train"], "o-",
            label="Entrenamiento", color="steelblue")
    ax.plot(tabla["N_vars"], tabla["Riesgo_valid"], "s-",
            label="Validación", color="tomato")
    ax.set_xlabel("Número de variables")
    ax.set_ylabel("Riesgo cuadrático")
    ax.set_title("Selección hacia adelante – Riesgo vs. número de variables")
    ax.set_xticks(tabla["N_vars"])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if ruta is not None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta, dpi=150)
    if mostrar:
        plt.show()
    return fig
