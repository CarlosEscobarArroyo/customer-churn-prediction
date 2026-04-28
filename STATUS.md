# Estado del proyecto de churn — Glamour Perú

> **Última actualización**: 2026-04-26.
> Para historia completa de versiones del dataset, ver `VERSIONES.md`.

## Objetivo
Modelo de clasificación de **silent churn** para vendedoras de venta directa por catálogo (modelo tipo Avon/Natura).

## Definición de churn vigente (v5)
Una vendedora churnea en el mes _M_ si **no compra en ninguno de los 6 meses calendario siguientes** (_M+1_ … _M+6_). El horizonte k=6 viene del análisis de codo en NB 06 (drafts/): después de 6 meses sin compra, la probabilidad mensual de retorno cae por debajo del 6%.

## Dataset vigente
- Tabla: `glamour-peru-dw.glamour_dw.training_churn_v5` (BigQuery).
- SQL: `data/qry_churn_v5.sql`.
- ~23 700 filas / ~4 200 vendedoras únicas (panel longitudinal: una fila por (vendedora, mes)).
- Filtro de población: `compras_historicas >= 3` (en mensual).
- Tasa de churn: 27.5%.
- Cobertura temporal: 2017-01 → 2025-10 (106 meses).

## Modelo vigente
- **`HistGradientBoostingClassifier`** con `class_weight='balanced'`.
- Hiperparámetros tuneados con Optuna (50 trials):
  `learning_rate=0.0175, max_iter=750, max_depth=4, max_leaf_nodes=22, min_samples_leaf=100`.
- 39 features tras pulido (originalmente 42; se sacaron `ccodrelacion`, `mes_num`, `anio_mes_num`).
- Notebook entregable: `notebooks/clean/modelo_final_v5.ipynb`.

## Métricas vigentes

| Métrica | Valor | Lectura |
|---|---:|---|
| AUC GroupKFold (mean ± std) | **0.7485 ± 0.008** | Métrica honesta, vendedora nunca en train+test simultáneamente |
| AUC split forward | **0.7537** | Escenario de producción (entrenar con histórico, predecir futuro) |
| PR-AUC GroupKFold OOF | 0.502 | Lift 1.83× sobre prevalencia (0.275) |
| Std AUC por mes (test) | 0.030 | Estable temporalmente |

### Punto operativo recomendado: t = 0.50

| | Valor |
|---|---:|
| Recall | 0.73 |
| Precision | 0.43 |
| Lift precision sobre prevalencia | 1.56× |
| % de población contactada | 47% |

Argumento de viabilidad: contactando al mismo 47% al azar atraparíamos al 47% de los churners; con el modelo, atrapamos al 73%. Esos +26 pp de recall son el valor incremental.

Ver `notebooks/clean/modelo_final_v5.ipynb` §9 para la discusión completa de viabilidad de negocio.

## Decisiones metodológicas vigentes

1. **Granularidad mensual** (panel longitudinal). Una fila por (vendedora, mes). Razón: las "campañas" dejaron de ser unidad temporal consistente desde 2025-06 (de 1/mes a 2-3/mes simultáneas). Detalle en `V4_MENSUAL.md`.
2. **Horizonte de 6 meses** (vigente desde v5). Validado por análisis de codo en NB 06.
3. **Filtro `compras_historicas >= 3`**. Excluye historia muy corta para evitar la regla trivial "1 compra ⇒ churn". Confirmado en el experimento v6_all que sacarlo daña la calidad del target.
4. **GroupKFold por vendedora** (5 folds) como protocolo principal. Vendedora nunca en train+test simultáneamente.
5. **Split temporal forward** como protocolo secundario (escenario de producción). GAP = horizonte+1 = 7 meses entre último mes de train y primero de test.
6. **`class_weight='balanced'`** para sensibilidad a la clase positiva (27.5% prevalencia).

## Trabajo realizado en la última iteración (2026-04-26)

1. **Notebook entregable** (`modelo_final_v5.ipynb`): documento autocontenido con marco teórico ML pedagógico.
2. **Ablation de features temporales** (`ablation_temporal_v5.ipynb`): sacar `mes_num` y `anio_mes_num` no daña forward (lo mejora levemente) y reduce riesgo de extrapolación.
3. **Tuning de hiperparámetros con Optuna** (`tuning_optuna_v5.ipynb`): +1.2 pp AUC GroupKFold, +0.8 pp AUC forward, modelo más estable. `learning_rate` explica 79% de la varianza.
4. **Discusión de viabilidad de negocio** (sec. 9 del notebook final): lift por decil, puntos de operación, recomendación operativa.

## Próximos pasos sugeridos

### Para producción
- **Pipeline mensual**: programar `qry_churn_v5.sql` + scoring + push de lista priorizada al equipo de retención (Cloud Composer / Cloud Scheduler).
- **Dashboard de monitoreo**: AUC rolling + recall @ t=0.50 mes a mes. Alertar si AUC < 0.70 o recall < 0.65.
- **Reentrenamiento mensual o trimestral** para evitar drift silencioso.

### Para mejorar el techo del modelo (mediano plazo)
- **Mejor data temporal (SCD-2)**: hoy `edad_vendedor`, `id_coordinadora`, `tipo_vendedor` son snapshots actuales. Historizarlos abriría señal real.
- **Variables de comportamiento ausentes**: interacciones con la app, asistencia a eventos, devoluciones — no están en el SQL actual.
- **Análisis de supervivencia**: predecir tiempo-hasta-churn en vez de binario. Recupera vendedoras censuradas hoy filtradas por `WHERE churn IS NOT NULL` y permite escalonar urgencia.
- **SHAP a nivel individual**: explicar predicciones específicas para alimentar la campaña de retención con un mensaje personalizado.

---

# Checklist de leakage y riesgos

Cada punto agrupa una preocupación única, indica su estado actual, y deja claro qué falta verificar.

---

## ✅ 1. Leakage por entidad: misma vendedora en train y test
**Resuelto** con `GroupKFold` por `id_vendedor`. Todas las filas de una misma vendedora caen en el mismo fold. AUC honesto: **0.7485 ± 0.008**.

**Pendiente menor**: el split forward incluye 412/556 vendedoras (74%) que también estaban en train. No es leakage técnico (es el escenario realista), pero conviene saberlo al interpretar la métrica.

---

## 🟡 2. Correlación entre filas de la misma vendedora dentro de train
**El problema:** aunque GroupKFold previene que María esté en train Y test, sus filas siguen estando juntas en train. El modelo "ve" información correlacionada de la misma persona varias veces.

**Estado:** ⚠️ **No es leakage técnico** (no infla métricas en test), pero sí afecta qué aprende el modelo. Sin medir cuantitativamente. Detalle en `CORRELACION.md`.

**Pendiente:**
- Probar `sample_weight = 1 / n_filas_de_la_vendedora` para mitigar el sesgo hacia vendedoras muy activas.
- Ablation: panel completo vs 1 fila por vendedora (la más reciente).

---

## ✅ 3. Variables de target intermedio (`compro_t1..t6`, `monto_t1..t6`)
**Resuelto desde v3.** Esas columnas ya no están en el dataset.

---

## ✅ 4. SCD-1 sospechosos (`estado_coordinadora`)
**Resuelto desde v3.** Excluido del schema.

---

## ✅ 5. Categóricas de alta cardinalidad (`distrito`, `ccodubigeo`)
**Mitigado.** Excluidas del modelo. Sólo se usa `provincia` (152 valores) y `departamento` (32 valores).

---

## ✅ 6. ID disfrazado (`ccodrelacion`)
**Resuelto en esta iteración** (2026-04-26). Aparecía top-4 en la importancia por permutación pero quitarlo costó solo 0.005 AUC → era ID, no señal genuina.

---

## ✅ 7. Features de calendario con riesgo de extrapolación (`mes_num`, `anio_mes_num`)
**Resuelto en esta iteración** (`ablation_temporal_v5.ipynb`). En split forward el modelo sin ellas tiene el mismo o mejor AUC y +4.6% PR-AUC. Riesgo de extrapolación para `anio_mes_num` futuros eliminado.

---

## 🟡 8. Patrones de NaN correlacionados con el target
**El problema:** `tendencia_*` tienen NaN cuando la vendedora estuvo inactiva — justo el caso correlacionado con churn. HGB maneja NaN nativamente y puede aprender de su presencia/ausencia.

**Estado:** no es leakage estricto, pero el patrón puede no ser estable en producción si cambia la lógica de cálculo de features.

**Pendiente:** verificar consistencia del patrón de NaN entre training e inferencia productiva.

---

## 🟢 Aclaraciones (cosas que NO son leakage)

### Tener múltiples filas del mismo vendedor en el dataset
No es leakage por sí solo — es la estructura natural del panel longitudinal. Sólo se vuelve problema si se reparten entre train y test sin control (resuelto en #1).

### Features con `compras_historicas`, ventanas u3/u6/u12, etc.
No son leakage. El SQL las construye con `ROWS BETWEEN N PRECEDING AND CURRENT ROW` — sólo miran hacia atrás.

### Que el modelo aprenda patrones por departamento, edad o tipo
No es leakage. Son features legítimas de contexto. Sólo sería problema si la cardinalidad fuera tan alta que identificara individuos (por eso se excluyó `distrito`).
