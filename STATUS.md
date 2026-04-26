# Contexto del proyecto de churn — Glamour Perú

## Objetivo
Construir un modelo de clasificación de churn para vendedoras de venta directa por catálogo (modelo tipo Avon/Natura).

## Definición de churn (ya decidida)
Una vendedora churnea si **no compra en las próximas 6 campañas consecutivas** después de una campaña de observación.

## Dataset
- Tabla `training_churn_v2` en BigQuery (CSV ~22k filas, ~3,155 vendedoras únicas).
- Granularidad: una fila por (vendedora, campaña) — **panel longitudinal**.
- Filtro aplicado: `compras_historicas >= 4` (excluye vendedoras de historia muy corta para evitar la regla trivial "1 compra = churn").
- Features: RFM clásico (recencia, frecuencia, monetario) en ventanas de 3/6/12 campañas, tendencias, diversidad de producto, contexto de coordinadora y ubicación.
- Target balance: 27.9% churn rate.
- Columnas leakage explícitas: `compro_t1..t6`, `monto_t1..t6` (excluidas del modelo).

## Decisiones metodológicas tomadas
1. **Ventana de churn fija de 6 campañas** (vs 3 original) porque el análisis mostró que 31% de gaps entre compras eran ≥ 4 campañas, generando falsos churn de vendedoras esporádicas legítimas.
2. **Filtro de historia mínima** porque en v1, el 45% de vendedoras tenían 1 sola compra y churneaban al 99.94% (regla trivial dominaba el aprendizaje).
3. **Validación con GroupKFold por vendedora** (no split temporal puro). Razón: el panel longitudinal con múltiples filas por vendedora genera sesgo si la misma ID aparece en train y test. GroupKFold garantiza que todas las observaciones de una vendedora estén del mismo lado.

## Resultados actuales (baselines)
- HistGradientBoosting con `class_weight='balanced'`.
- AUC con GroupKFold: **0.757 ± 0.007** (métrica honesta, la principal a reportar).
- AUC con split temporal: 0.765 (ligeramente inflado por overlap de IDs entre train/test).
- PR-AUC: 0.53 (vs piso de 0.28).
- Recall en clase churn @ threshold 0.5: 78%; precision: 45%.

## Validación contra literatura
Revisé ~12 papers de churn (mayoría telecom y banca). Conclusiones:
- Mi definición de ventana fija prospectiva está alineada con la literatura.
- Mi filtro de historia es más estricto que el promedio pero metodológicamente justificado (Ahmad et al. hace algo similar).
- **Mi diseño de panel longitudinal + GroupKFold es más sofisticado que el promedio** (la literatura usa una fila por cliente, lo que evita el problema en lugar de resolverlo).
- No comparar mi AUC 0.76 con AUC 0.93+ de papers de telecom — esos vienen de datasets balanceados con random splits sin control de leakage.

## Qué tengo
- SQL de construcción del dataset v2.
- Notebook 1: análisis de dispersión justificando los cambios (v1 → v2).
- Notebook 2: validación + baselines con GroupKFold.

## Posibles próximos pasos
- Tuning de hiperparámetros del HGB.
- Probar LightGBM/XGBoost.
- Feature engineering: feature de regularidad (std del gap entre compras).
- Análisis de threshold óptimo según costos de negocio.
- Calibración de probabilidades si se va a usar en producción.

---

# Checklist de leakage y riesgos — temas a profundizar

Cada punto agrupa una preocupación única, indica su estado actual, y deja claro qué falta verificar.

---

## 🔴 1. Leakage por entidad: misma vendedora en train y test
**El problema:** En un split temporal puro, una misma vendedora puede aparecer en train (campañas antiguas) y en test (campañas recientes). Sus rasgos personales fijos (departamento, edad, coordinadora, tipo) están en todas sus filas. El modelo "reconoce" a vendedoras conocidas en lugar de generalizar.

**Evidencia cuantitativa:**
- AUC en vendedoras conocidas (vistas en train): 0.78
- AUC en vendedoras nuevas (no vistas): 0.73
- Diferencia: ~5 puntos = magnitud del sesgo de reconocimiento

**Estado:** ✅ **Resuelto con GroupKFold por vendedora**. Todas las filas de una misma vendedora caen en el mismo fold (todas en train o todas en test, nunca repartidas). AUC honesto: **0.757 ± 0.007**.

**Pendiente:** considerar split combinado temporal + GroupKFold para escenario aún más realista de producción (vendedoras nuevas en campañas futuras).

---

## 🟡 2. Correlación entre filas de la misma vendedora dentro de train
**El problema:** Aunque GroupKFold previene que María esté en train Y test, sus 3 filas siguen estando juntas en train. El modelo "ve" información correlacionada de la misma persona varias veces. Esto rompe el supuesto de independencia entre observaciones.

**Dos efectos distintos:**
- **a) Ponderación implícita:** una vendedora con 20 filas pesa 20× más que una con 1 fila al entrenar. Sesga el modelo hacia el comportamiento de vendedoras muy activas.
- **b) Solapamiento de ventanas de target:** si María tiene observación en C5 (target = compras en C6-C11) y otra en C8 (features incluyen C6-C7 en su ventana u3), el target de la primera se solapa con las features de la segunda. Dentro de train, el modelo puede aprender asociaciones que son estructura del dataset, no patrones generalizables.

**Estado:** ⚠️ **No es leakage técnico** (no infla métricas en test), pero sí afecta qué aprende el modelo. Sin medir cuantitativamente.

**Pendiente:**
- Probar `sample_weight = 1 / n_filas_de_la_vendedora` para mitigar (a).
- Experimento ablation: comparar modelo con panel completo vs modelo con 1 fila por vendedora (la más reciente) para medir el impacto real de (b).

---

## 🟡 3. Constantes y pseudo-features ocultas
**El problema:** `campanas_desde_ultima_compra` siempre vale 0 en el dataset porque el filtro `compro_en_obs=1` lo fuerza. Es una pseudo-feature sin información.

**Estado:** ✅ **Excluida del modelo**. Documentar para futuras versiones del SQL para no recrear el mismo error.

---

## 🟡 4. Patrones de NaN correlacionados con el target
**El problema:** `tendencia_monto_u3_vs_prev3` y similares tienen NaN cuando `monto_u6 - monto_u3 = 0` (vendedora inactiva en las últimas campañas). Justo el caso más correlacionado con churn. HGB maneja NaN nativamente y puede aprender de la presencia/ausencia de NaN.

**Estado:** No es leakage estricto, pero el patrón puede no ser estable en producción si cambia la lógica de cálculo de features.

**Pendiente:** verificar que el patrón de NaN sea consistente entre training set e inferencia productiva.

---

## 🟡 5. Categóricas de alta cardinalidad
**El problema:** `distrito` (cientos de valores) y `ccodubigeo` podrían identificar grupos muy pequeños de vendedoras, acercándose a un identificador. Si se usaran con target encoding sin cuidado, sería leakage.

**Estado:** ✅ **Mitigado**. Solo se usa `departamento` (≤33 categorías). Si se incluyen en el futuro, usar target encoding con CV interna.

---

## 🟢 Cosas que NO son leakage (aclaraciones para no volver a discutirlas)

### Tener múltiples filas del mismo vendedor en el dataset
No es leakage por sí solo. Es la estructura natural del panel longitudinal. Solo se vuelve problema si esas filas se reparten entre train y test sin control (ver punto 1, ya resuelto).

### Features con `compras_historicas`, `tasa_compra_u12`, etc.
No son leakage. El SQL las construye con `ROWS BETWEEN N PRECEDING AND CURRENT ROW`, que solo mira hacia atrás. Cada fila contiene solo información disponible al momento de la observación.

### Que el modelo aprenda patrones por departamento, edad o tipo
No es leakage. Son features legítimas de contexto. Solo sería problema si la cardinalidad fuera tan alta que identifique individuos (por eso se excluyó `distrito`).

---

## Preguntas abiertas para profundizar

1. **¿El punto 2b (solapamiento de ventanas de target) tiene impacto medible?** Hacer ablation con 1 fila por vendedora vs panel completo.
2. **¿Conviene split combinado temporal + GroupKFold** como escenario más conservador para reportar la métrica?
3. **¿`sample_weight = 1/n_filas` mejora o empeora la generalización en GroupKFold?**
4. **¿Cómo presentar la métrica final** ante la profesora? Una tabla con tres escenarios (split temporal, GroupKFold, combinado) puede ser lo más honesto.