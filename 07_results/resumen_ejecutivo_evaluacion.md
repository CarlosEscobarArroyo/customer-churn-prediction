# Resumen ejecutivo — Evaluación del modelo de churn

> **Proyecto**: predicción de *silent churn* de vendedoras de Glamour Perú.
> **Modelo evaluado**: XGBoost tuneado con Optuna (validación cruzada anidada).
> **Fecha**: 2026-06-05 (re-verificado tras re-ejecución completa del pipeline 00→07).
> **Fuente**: notebooks de `06_evaluation/` (`01_feature_importance`, `02_shap`, `03_utilidad_modelo`), sobre el split OOT.

---

## 1. En una línea

El modelo **discrimina bien quién va a dejar de comprar** (AUC OOT 0.765, lift 1.92×) y se apoya en variables de **volumen y frecuencia de actividad reciente** con sentido de negocio. Sirve para **priorizar** a quién contactar, **no** como probabilidad calibrada ni como certeza individual. El valor está en el *targeting*; el techo de señal del problema (~0.77) acota lo que cualquier modelo puede lograr aquí.

---

## 2. Qué se evaluó

- **Algoritmo**: `XGBoost` (gradient boosting), desbalance manejado con `scale_pos_weight` (sin resampling).
- **Hiperparámetros** (Optuna, 60 trials, óptimo por AUC GroupKFold sobre el train-pool — *nested CV*):
  `n_estimators=950, max_depth=3, learning_rate=0.011, subsample=0.82, colsample_bytree=0.80, min_child_weight=15, gamma=2.20, reg_alpha=0.005, reg_lambda=0.26`.
  Configuración fuertemente regularizada (árboles muy poco profundos `depth=3`, `learning_rate` bajo, hojas grandes `min_child_weight=15`, ganancia mínima `gamma=2.2`) → modelo conservador, coherente con un problema de techo de señal.
- **Conjunto de evaluación**: split temporal **OOT** (escenario de producción: entrenar con el pasado, predecir el futuro). Test = 803 filas, prevalencia de churn **26.2 %**.

---

## 3. Desempeño (discriminación)

| Métrica | Valor | Lectura |
|---|---:|---|
| AUC validación interna (GroupKFold train-pool) | **0.7422** | Métrica de selección de hiperparámetros (nested CV) |
| AUC OOT | **0.7648** | Escenario de producción (test aislado, visto una sola vez) |
| PR-AUC (OOT) | 0.5029 | Lift global **1.92×** sobre la prevalencia (0.262) |
| Brier score (OOT) | 0.2012 | ⚠️ Probabilidades **mal calibradas** (ver §6) |

**Vs. baseline sin tunear** (XGBoost baseline: GKF 0.7418 / OOT 0.7540): el tuning aporta **+1.08 pp en OOT** (0.7540 → 0.7648), con mejoras notables en *recall* (0.724 → 0.776) y *lift* del decil superior (2.29× → 2.44×). Ganancia acotada por el techo de señal; el valor del tuning está en el AUC, la **estabilidad temporal** y la **regularización**.

---

## 4. Qué impulsa el churn (importancia + SHAP)

Las dos lentes de importancia (gain de XGBoost y permutación sobre OOT) **coinciden en el top**, y SHAP lo confirma:

| Variable | Lectura de negocio |
|---|---|
| `n_prod_u12`, `n_ped_u12`, `n_ped_u6` | **nº de productos / pedidos** en ventanas de 6–12 meses — amplitud y frecuencia recientes |
| `monto_u6`, `monto_u3`, `n_ped_u3` | **intensidad de compra reciente** (montos y pedidos 3–6 meses) |
| `compras_hist`, `n_ped_acum`, `n_prod_acum` | **historial acumulado** de la relación |
| `meses_desde_compra_previa` | recencia (relevante, pero **no** dominante: rank ~7–12) |

**Hallazgos:**
- El modelo se apoya sobre todo en **volumen y frecuencia de actividad** (RFM en ventanas de 3–12 meses) y en el historial acumulado, más que en la pura recencia. Tiene sentido: una vendedora que baja su frecuencia y diversidad de catálogo es la que está soltando la relación.
- **Discrepancias entre lentes** (señal de redundancia / sobreajuste de entrenamiento, no de generalización): `n_prod_acum` (gain rank 8 vs permutación 64) y `monto_mean_u12` (13 vs 55) pesan en el entrenamiento pero apenas mueven el AUC fuera de muestra — probablemente correlacionadas con otras variables RFM.
- **SHAP** ratifica el top 3 (`compras_hist`, `n_ped_u12`, `n_prod_u12`) y verifica la **dirección correcta**: menor actividad reciente → mayor riesgo. Las explicaciones locales funcionan (caso de mayor riesgo p=0.87 → churn real; caso de menor riesgo p=0.01 → no churn), lo que habilita mensajes de retención personalizados.

---

## 5. Utilidad operativa (targeting)

El uso real es **priorizar** una lista mensual de retención. Ordenando por score y contactando al top X %:

| Regla | Contactadas | Capturados con modelo | Capturados sin modelo (azar) | Capturados de más | Precisión | Recall | Lift vs. prevalencia |
|---|---:|---:|---:|---:|---:|---:|---:|
| Top 5 % | 41 | 23 | 11 | **+12** | 0.56 | 0.11 | 2.15× |
| Top 10 % | 81 | 51 | 21 | **+30** | 0.63 | 0.24 | **2.41×** |
| Top 20 % | 161 | 85 | 42 | **+43** | 0.53 | 0.41 | 2.02× |
| Top 30 % | 241 | 115 | 63 | **+52** | 0.48 | 0.55 | 1.83× |
| Top 50 % | 402 | 165 | 105 | **+60** | 0.41 | 0.79 | 1.57× |
| Umbral t=0.5 | 380 | 163 | 99 | **+64** | 0.43 | 0.78 | 1.64× |

> *Capturados sin modelo (azar)* = `contactadas × prevalencia (0.262)`: los churners que se atraparían contactando al mismo número de vendedoras **elegidas al azar**. *Capturados de más* es el valor incremental del modelo sobre esa línea base (es, justamente, el lift expresado en número de churners).

**Lectura:**
- **Buena priorización en la cima**: en el top 10 % de la base, ~63 % son churners reales (2.4× la prevalencia). Si el equipo tiene presupuesto acotado, ahí está el mayor retorno por contacto. (El top 5 %, con solo 41 casos, es ruidoso.)
- **Cobertura amplia**: contactando al top 50 % se captura el **79 %** de los churners (vs. 50 % si se eligiera al azar) — +29 pp de recall incremental.
- La regla debe elegirse **por capacidad del equipo** (cuántos contactos por mes puede hacer), recorriendo esta tabla.

Matriz de confusión a t=0.5: **163 churners capturados**, 47 perdidos, 217 falsas alarmas, 376 sanos correctos.

---

## 6. Calibración — advertencia importante

El **Brier score (0.2012) es peor que predecir la prevalencia constante** (≈0.193): las probabilidades del modelo están **infladas** (efecto esperado del `scale_pos_weight`, que sube la clase positiva). Implicaciones:

- ✅ **El *ranking* es válido** — el AUC no se ve afectado, así que ordenar por score y contactar al top X % funciona perfectamente.
- ❌ **La probabilidad cruda NO es confiable** — una `p(churn)=0.7` no significa 70 % de riesgo real, y por eso el umbral t=0.5 contacta a casi la mitad de la base.
- **Recomendación**: hacer *targeting* **por percentil de score** (top X %), no por umbral de probabilidad. Si se necesita la probabilidad como tal, **recalibrar** (`CalibratedClassifierCV`, isotónica o Platt) sin reentrenar el modelo.

---

## 7. Costo-beneficio — pendiente de supuestos reales

El notebook incluye un análisis de valor neto, pero corre con **supuestos placeholder** (costo de contacto S/5, valor de retención S/200, tasa de éxito 30 %). Con esos números el óptimo es "contactar a casi todos" (top 81 %), lo que **solo refleja que el contacto es casi gratis frente al valor de retención asumido** — no es una recomendación.

> **Acción requerida**: reemplazar los tres parámetros con cifras reales de Glamour (costo real del contacto/incentivo, margen futuro de una vendedora retenida, tasa de éxito histórica de las campañas de retención) antes de fijar el % de la base a contactar.

---

## 8. Limitaciones

1. **Techo de señal (~0.77 AUC)**: confirmado en el baseline; ningún algoritmo lo supera con los datos actuales. El modelo prioriza, no certifica.
2. **Calibración deficiente**: usar ranking, no probabilidad cruda (§6).
3. **Test OOT pequeño (803 filas)**: las métricas son indicativas; conviene confirmar la estabilidad con monitoreo en producción.
4. **Datos de contexto en snapshot (SCD-1)**: variables como tipo/zona de vendedora no están historizadas; historizarlas (SCD-2) podría abrir señal nueva.

---

## 9. Recomendaciones

**Para usar ya:**
- Desplegar el modelo como **priorizador mensual** por percentil de score; entregar al equipo de retención el top X % según su capacidad.
- Acompañar cada vendedora priorizada con sus **drivers SHAP** para personalizar el contacto.

**Antes de producción:**
- **Recalibrar** las probabilidades si se quieren reportar como tal.
- **Completar el costo-beneficio** con cifras reales para fijar el punto de operación.
- **Reentrenar el modelo final con todos los datos** (el evaluado se entrenó solo con el train del OOT).

**Monitoreo:**
- AUC y recall *rolling* mes a mes; alertar si AUC < 0.70 o recall < 0.65.
- Reentrenamiento periódico (mensual/trimestral) para evitar drift.

---

*Detalle reproducible y gráficos (beeswarm SHAP, curvas ROC/PR, ganancias, calibración) en los notebooks de `06_evaluation/`. Auditoría de leakage en `07_results/auditoria_leakage.md`.*
