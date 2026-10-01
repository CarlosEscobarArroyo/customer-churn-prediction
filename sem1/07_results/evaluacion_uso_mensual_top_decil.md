# Evaluación del uso mensual mediante top decile lift

Fecha de evaluación: 2026-09-14. Se recalcularon predicciones de los binarios
`models/{xgboost,rf,logreg}_tuned.joblib` sobre
`data/processed/churn_dataset_features.csv`, usando las asignaciones de test de
`data/processed/oot_split.csv`, unidas por `(id_vendedor, mes_rank)` con validación
de unicidad. No se reentrenaron modelos ni se modificaron notebooks.

## Conclusión

XGBoost tiene utilidad para un piloto de priorización mensual de retención con
capacidad limitada. El top 10 % mensual concentra aproximadamente el doble de
churn que una selección aleatoria del mismo tamaño. Esta evidencia demuestra
capacidad de priorización; la retención incremental y rentabilidad de una campaña
requieren medir el efecto de la intervención.

## Alcance y rendimiento

- Test temporal: octubre de 2025 a enero de 2026; 885 observaciones de 532 vendedoras.
- Entrenamiento hasta marzo de 2025, con gap de seis meses antes del test.
- Población: vendedoras que compraron durante el mes observado (`activo = 1`).
- Objetivo: ninguna compra durante los seis meses siguientes. La frecuencia mensual
  de scoring no convierte el objetivo en churn del mes siguiente.
- XGBoost: AUC OOT **0,7637**, average precision **0,5352**, prevalencia **27,80 %**.
- AUC GroupKFold vigente reportada: **0,7476**; fuente:
  `07_results/cambios_tras_features_campanas.md`. No se volvió a ejecutar CV en esta evaluación.

Los documentos de junio y `CLAUDE.md` contienen resultados de versiones anteriores.
El cociente AP/prevalencia (1,93×) es distinto del lift del decil superior.

## Cálculo mensual

Por cada mes se ordena el score de mayor a menor, se desempata por `id_vendedor`
y se seleccionan `ceil(0.10 * n_mes)` filas. El redondeo permite cubrir al menos
el 10 % de la población elegible.

`lift_mes = (churns en seleccionadas / seleccionadas) / prevalencia_mes`.

| Mes | Elegibles | Churn en la base | Seleccionadas | Churns identificados | Precisión top 10 % | Recall | Lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025-10 | 184 | 21,74 % | 19 | 10 | 52,63 % | 25,00 % | 2,42× |
| 2025-11 | 236 | 29,24 % | 24 | 15 | 62,50 % | 21,74 % | 2,14× |
| 2025-12 | 273 | 31,14 % | 28 | 20 | 71,43 % | 23,53 % | 2,29× |
| 2026-01 | 192 | 27,08 % | 20 | 10 | 50,00 % | 19,23 % | 1,85× |

En conjunto: **91 selecciones mensuales**, 89 vendedoras distintas y **55 casos
positivos**, frente a **25,28** esperados al azar. Precisión: **60,44 %**; recall:
**22,36 %**; quedan fuera **191 de 246** observaciones positivas.

Lift de la política mensual: `55 / sum(k_mes * prevalencia_mes)` = **2,1755×**.
Son aproximadamente **30 casos adicionales identificados**, sin atribuirles una
retención causada por el contacto. Dos vendedoras aparecen seleccionadas en dos
meses; las cifras describen oportunidades de contacto y observaciones, no siempre
personas diferentes.

El ranking agrupado de los cuatro meses da 89 seleccionadas, 56 positivos y lift
**2,2636×**. No es la misma política que seleccionar el 10 % dentro de cada mes.
El lift 2,29× de otros reportes utiliza 88 filas (`floor(n/10)`); la discrepancia
de redondeo no implica otro modelo.

## Capacidad de contacto

| Selección dentro de cada mes | Contactos potenciales | Casos positivos | Precisión | Recall | Lift frente al azar mensual |
|---|---:|---:|---:|---:|---:|
| Top 10 % | 91 | 55 | 60,44 % | 22,36 % | 2,18× |
| Top 20 % | 179 | 101 | 56,42 % | 41,06 % | 2,03× |
| Top 30 % | 267 | 137 | 51,31 % | 55,69 % | 1,85× |

El top 10 % es una opción para capacidad limitada, no un óptimo económico
demostrado. El top 20 % amplía la cobertura manteniendo un lift cercano a 2×.

## Comparación con los modelos disponibles

Con la misma política y los mismos 91 contactos potenciales:

| Modelo | Casos positivos | Precisión | Lift mensual agregado |
|---|---:|---:|---:|
| XGBoost | 55 | 60,44 % | 2,18× |
| Random Forest | 61 | 67,03 % | 2,41× |
| Regresión logística | 58 | 63,74 % | 2,29× |

Random Forest es un candidato a contrastar si el objetivo principal es lift@10 %.
La ventaja observada son seis casos respecto a XGBoost en cuatro meses; no se ha
demostrado significancia estadística ni superioridad futura. Seleccionar un ganador
con este test exige confirmación en nuevas ventanas temporales.

## Condiciones para el uso mensual

1. Calcular scores después del cierre de cada mes sobre la población activa elegible
   y priorizar por percentil. No extrapolar estas métricas a vendedoras ya inactivas.
2. Separar scoring de entrenamiento: puntuar cada mes no exige reentrenar cada mes.
   Para entrenamiento supervisado y evaluación se necesitan etiquetas maduras.
3. Crear una extracción de scoring que conserve las features y filtros de población,
   pero no exija seis meses futuros. El SQL de entrenamiento actual excluye meses
   recientes mediante `mes_rank <= max_rank - 6`; por sí solo no produce el scoring
   del último mes cerrado.
4. Registrar contactos y gestionar repeticiones. Si se excluyen vendedoras recién
   contactadas, hay que reevaluar el ranking sobre esa nueva población elegible.
5. Hacer un piloto con asignación aleatoria a contacto/control dentro de la población
   priorizada, manteniendo la asignación por vendedora entre meses. Medir recompra,
   margen incremental y costo total. Comparar también con la regla operativa vigente:
   superar al azar no demuestra superar al equipo de retención.
6. Monitorear lift@10 %, precisión@10 %, recall@10 %, prevalencia y volumen por mes,
   cuando maduren los seis meses de seguimiento. Antes, monitorear distribución de
   scores, calidad de datos y señales tempranas de recompra, sin confundirlas con
   la etiqueta final de churn.

La evaluación cubre solo cuatro meses y 19–28 seleccionadas por mes. Una positiva
adicional cambia la precisión mensual entre 3,6 y 5,3 puntos porcentuales; el rango
observado de lift es descriptivo, no un intervalo de confianza. Hace falta ampliar
el backtest temporal antes de afirmar estabilidad anual. Las filas repetidas por
vendedora deben tratarse como dependientes al estimar incertidumbre.

Las probabilidades crudas requieren revisión antes de interpretarse como riesgo
absoluto: Brier **0,2054**, frente a **0,2007** para la prevalencia constante del
test (comparación retrospectiva). Brier combina calibración y discriminación;
también debe inspeccionarse la curva de calibración, como explica la
[documentación de scikit-learn](https://scikit-learn.org/1.7/modules/calibration.html).

Un modelo de riesgo no estima por sí solo cuánto cambia el resultado al intervenir;
esa distinción está desarrollada en la
[documentación de inferencia causal de PyWhy](https://www.pywhy.org/dowhy/v0.11/example_notebooks/tutorial-causalinference-machinelearning-using-dowhy-econml.html).
Los supuestos monetarios del notebook son ilustrativos y no permiten asegurar ROI.
