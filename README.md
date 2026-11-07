# Quinta lista de ejercicios – Selección de variables y generalización

Regresión lineal con intercepto sobre el conjunto **Hitters** (263 jugadores de béisbol,
objetivo `Salary`, 19 predictoras). Se comparan cuatro modelos: todas las variables,
mayor correlación (top 7), pruebas t (p < 0.05) y selección hacia adelante.

## Estructura

```
taller_ml/
├── README.md
├── requirements.txt
├── data/
│   └── Datos_Hitters.csv
├── src/
│   ├── config.py      
│   ├── data.py         
│   ├── metrics.py     
│   ├── models.py       
│   ├── selection.py  
│   ├── plots.py       
│   └── main.py       
├── notebooks/
│   └── taller_hitters.ipynb
└── outputs/
    ├── tables/        
    └── figures/       
```

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate       
pip install -r requirements.txt
```

## Uso

Desde la raíz del proyecto:

```bash
python -m src.main              
jupyter notebook notebooks/taller_hitters.ipynb
```

En VS Code: abre la carpeta `taller_ml/`, selecciona el intérprete de `.venv`
y ejecuta el cuaderno con **Run All** desde un kernel reiniciado.

## Partición (random_state=31)

| Conjunto | n |
|---|---|
| Entrenamiento | 50 |
| Validación | 70 |
| Prueba | 143 |

Los ejercicios 1–3 usan solo entrenamiento y validación; la prueba se reserva para el ejercicio 4.

## Resultados (riesgo cuadrático)

| Modelo | Variables | Train | Validación | Prueba |
|---|---|---|---|---|
| Todas las variables | 19 | 68 584.82 | 320 423.19 | 394 308.69 |
| Mayor correlación | 7 | 115 487.25 | 133 035.61 | 145 923.29 |
| Pruebas t | 3 | 161 221.36 | 140 085.19 | 158 772.47 |
| Selección hacia adelante | 3 | 126 982.32 | 130 618.20 | 126 939.60 |

Modelo elegido por validación: **selección hacia adelante** (`CRBI`, `Years`, `CRuns`).
