"""Constantes del taller."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ / "data" / "Datos_Hitters.csv"
DIR_TABLAS = RAIZ / "outputs" / "tables"
DIR_FIGURAS = RAIZ / "outputs" / "figures"

OBJETIVO = "Salary"
SEMILLA = 31

N_PRUEBA = 143
N_VALIDACION = 70

K_CORRELACION = 7
ALFA = 0.05
