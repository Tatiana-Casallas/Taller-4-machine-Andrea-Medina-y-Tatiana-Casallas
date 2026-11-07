"""Ejecuta el taller completo (ejercicios 1–4).

Uso (desde la raíz del proyecto):
    python -m src.main
"""
import pandas as pd

from .config import DIR_FIGURAS, DIR_TABLAS
from .data import cargar_datos, particionar
from .models import ajustar_y_evaluar, riesgo_en_prueba
from .plots import graficar_forward
from .selection import (
    mejor_forward,
    ranking_correlacion,
    seleccion_correlacion,
    seleccion_forward,
    seleccion_pruebas_t,
)


def titulo(texto: str) -> None:
    print("=" * 60)
    print(texto)
    print("=" * 60)


def main() -> None:
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)

    # ── 0. Datos y partición ────────────────────────────────
    X, y = cargar_datos()
    p = particionar(X, y)
    print(f"Variables predictoras ({X.shape[1]}): {list(X.columns)}")
    print(f"Total: {len(X)} | Entrenamiento: {len(p.X_train)} | "
          f"Validación: {len(p.X_val)} | Prueba: {len(p.X_test)}\n")

    # ── Ejercicio 1: modelo completo ────────────────────────
    titulo("EJERCICIO 1 – Modelo completo")
    todas = list(p.X_train.columns)
    _, r1_tr, r1_val = ajustar_y_evaluar(p.X_train, p.y_train, p.X_val, p.y_val)
    print(f"Riesgo entrenamiento: {r1_tr:.4f}")
    print(f"Riesgo validación   : {r1_val:.4f}\n")

    # ── Ejercicio 2: selección estadística ──────────────────
    titulo("EJERCICIO 2 – Selección estadística de variables")
    ranking = ranking_correlacion(p.X_train, p.y_train)
    tabla_corr = ranking.rename("Correlacion").to_frame()
    tabla_corr["Correlacion_abs"] = tabla_corr["Correlacion"].abs()
    print("(a) Correlación con Salary (orden por |r|):")
    print(tabla_corr.round(4).to_string(), "\n")
    tabla_corr.to_csv(DIR_TABLAS / "ej2a_correlaciones.csv")

    top7 = seleccion_correlacion(p.X_train, p.y_train)
    _, r2_tr, r2_val = ajustar_y_evaluar(
        p.X_train[top7], p.y_train, p.X_val[top7], p.y_val)
    print(f"(b) Top 7 por correlación: {top7}")
    print(f"    Riesgo train: {r2_tr:.4f} | valid: {r2_val:.4f}\n")

    vars_t = seleccion_pruebas_t(p.X_train, p.y_train)
    _, r3_tr, r3_val = ajustar_y_evaluar(
        p.X_train[vars_t], p.y_train, p.X_val[vars_t], p.y_val)
    print(f"(c) Variables con p < 0.05: {vars_t}")
    print(f"    Riesgo train: {r3_tr:.4f} | valid: {r3_val:.4f}\n")

    comp2 = pd.DataFrame({
        "Modelo": ["Completo (19)", "Top 7 correlación", "Pruebas t (p<0.05)"],
        "N_variables": [len(todas), len(top7), len(vars_t)],
        "Riesgo_train": [r1_tr, r2_tr, r3_tr],
        "Riesgo_valid": [r1_val, r2_val, r3_val],
    })
    print("(d) Comparación:")
    print(comp2.round(4).to_string(index=False), "\n")
    comp2.to_csv(DIR_TABLAS / "ej2d_comparacion.csv", index=False)

    # ── Ejercicio 3: selección hacia adelante ───────────────
    titulo("EJERCICIO 3 – Selección hacia adelante")
    tabla_fwd = seleccion_forward(p.X_train, p.y_train, p.X_val, p.y_val)
    print("(a) Tabla paso a paso:")
    print(tabla_fwd.round(4).to_string(index=False), "\n")
    tabla_fwd.to_csv(DIR_TABLAS / "ej3a_forward.csv", index=False)

    graficar_forward(tabla_fwd, DIR_FIGURAS / "forward_selection.png")
    print(f"(b) Gráfica guardada en {DIR_FIGURAS / 'forward_selection.png'}\n")

    vars_fwd, r4_tr, r4_val = mejor_forward(tabla_fwd)
    print(f"(c) Mejor por validación: {len(vars_fwd)} variables -> {vars_fwd}")
    print(f"    Riesgo train: {r4_tr:.4f} | valid: {r4_val:.4f}\n")

    # ── Ejercicio 4: comparación final ──────────────────────
    titulo("EJERCICIO 4 – Comparación final")
    modelos = {
        "Todas las variables": todas,
        "Mayor correlación": top7,
        "Pruebas t": vars_t,
        "Selección hacia adelante": vars_fwd,
    }
    r_train = [r1_tr, r2_tr, r3_tr, r4_tr]
    r_val = [r1_val, r2_val, r3_val, r4_val]
    r_test = [riesgo_en_prueba(p.X_train, p.y_train, p.X_test, p.y_test, v)
              for v in modelos.values()]
    nombres = list(modelos)

    i_val = min(range(4), key=r_val.__getitem__)
    i_tr = min(range(4), key=r_train.__getitem__)
    i_te = min(range(4), key=r_test.__getitem__)

    print(f"(a) Modelo elegido: {nombres[i_val]} "
          f"(riesgo de validación = {r_val[i_val]:.4f})\n")

    tabla_final = pd.DataFrame({
        "Modelo": nombres,
        "Variables": [len(v) for v in modelos.values()],
        "Riesgo_train": r_train,
        "Riesgo_valid": r_val,
        "Riesgo_prueba": r_test,
    })
    print("(b) Tabla final:")
    print(tabla_final.round(4).to_string(index=False), "\n")
    tabla_final.to_csv(DIR_TABLAS / "ej4b_tabla_final.csv", index=False)

    print(f"(c) Menor riesgo de entrenamiento: {nombres[i_tr]} ({r_train[i_tr]:.4f})")
    print(f"    Menor riesgo de prueba       : {nombres[i_te]} ({r_test[i_te]:.4f})")
    print(f"    ¿Coinciden? {'Sí' if i_tr == i_te else 'No'}\n")

    print(f"(d) Elegido por validación: {nombres[i_val]} "
          f"(valid = {r_val[i_val]:.4f}, prueba = {r_test[i_val]:.4f})")
    print(f"    Menor en prueba       : {nombres[i_te]} (prueba = {r_test[i_te]:.4f})")
    print("    -> " + ("Sí coincide." if i_val == i_te else "No coincide."))


if __name__ == "__main__":
    main()
