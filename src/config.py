"""Constantes del taller."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ / "data" / "Datos_Hitters.csv"
DIR_TABLAS = RAIZ / "outputs" / "tables"
DIR_FIGURAS = RAIZ / "outputs" / "figures"

OBJETIVO = "Salary"
SEMILLA = 31

N_PRUEBA = 143       # etapa 1: 120 desarrollo / 143 prueba
N_VALIDACION = 70    # etapa 2: 50 entrenamiento / 70 validación

K_CORRELACION = 7    # nº de variables del criterio de correlación
ALFA = 0.05          # umbral del valor p (prueba t)
