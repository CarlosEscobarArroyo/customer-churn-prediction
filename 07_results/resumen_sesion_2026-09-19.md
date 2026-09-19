# Resumen de la sesión — 19 de septiembre de 2026

## Estado al cierre

Auditamos el proyecto, reparamos el dataset, ejecutamos Optuna con validación temporal, comparamos ventanas recientes y construimos ensembles. **Todos estos entrenamientos terminaron.**

El candidato con mayor AUC temporal medio fue el promedio al 50 % de **regresión logística entrenada con 36 meses y XGBoost entrenado con toda la historia permitida**, con **AUC 0,78961**. No alcanzamos 0,80. Esta es una métrica de selección sobre validaciones reutilizadas, no una evaluación final independiente.

La extracción de datos de BigQuery **no se actualizó**. El siguiente paso pendiente es actualizar los datos y confirmar el candidato en períodos posteriores, manteniéndolo congelado durante la evaluación.

## 1. Auditoría inicial

El dataset tiene **30.356 observaciones y 6.323 vendedoras**. Cada observación representa una vendedora en un mes. La población exige compra en ese mes y al menos un mes con compra previa. Churn significa **no comprar durante los seis meses posteriores**.

Se verificó que las etiquetas coinciden con las compras originales del snapshot local. Se encontraron:

- Una celda con el texto `Directo` en la variable numérica `monto_ult_vs_media`, posición 919 contando desde cero, en el CSV de modelado. Bloqueaba la evaluación directa de XGBoost.
- Documentación que describe versiones diferentes del proyecto.
- Selección supervisada de variables antes de la validación cruzada interna.
- Uso repetido del OOT para comparar alternativas: no era un test intacto visto una sola vez.
- Importancia por permutación basada en accuracy, aunque el gráfico de selección indicaba AUC.
- Ausencia de pruebas automatizadas y de un flujo persistente de entrenamiento.

Evidencia completa: [auditoría del estado](auditoria_estado_2026-09-19.md).

## 2. Reparación de datos

Se reconstruyó `data/processed/churn_dataset_features.csv` desde el preprocesado verificado, conservando las 68 variables seleccionadas anteriormente y su orden para mantener compatibilidad con los modelos existentes.

Se guardaron un respaldo del CSV dañado y manifiestos de reparación con hashes en `reports/data_repair/`. Se comprobó que los cuatro modelos anteriores reproducían sus AUC registrados después de reparar el dato.

Se añadieron pruebas de integridad numérica, separación temporal, exclusión de identificadores y datos maestros, ventanas recientes y combinación de scores.

La reparación no resuelve por sí sola los problemas metodológicos de los notebooks antiguos. El experimento nuevo utiliza un protocolo distinto.

## 3. Optuna con validación temporal

Se ejecutaron **100 trials por algoritmo: 400 en total**, sin fallos. Completaron la evaluación 244 trials y se podaron anticipadamente 156.

El protocolo nuevo utiliza:

- Cuatro bloques de validación de cuatro meses: octubre de 2023 a enero de 2025.
- Entrenamiento expansivo, con seis meses de separación antes de cada bloque.
- Variables transaccionales; exclusión de identificadores, objetivo y datos maestros cuyo estado histórico no estaba garantizado.
- Comparación entre todas las variables y selección por permutación AUC aprendida en una partición temporal interna del entrenamiento.
- Escalado de logística y pesos de clase ajustados con el entrenamiento de cada fold.
- Objetivo: media no ponderada del ROC AUC de los cuatro bloques.
- Estudios SQLite, métricas por trial, checkpoints y pipelines persistidos.

Los mejores resultados fueron:

| Modelo | AUC temporal medio |
|---|---:|
| Random Forest | 0,78833 |
| XGBoost | 0,78815 |
| CatBoost | 0,78719 |
| Regresión logística | 0,78594 |

Los cuatro mejores trials prefirieron **todas las 42 variables transaccionales**. La diferencia entre Random Forest y XGBoost fue muy pequeña; no se demostró superioridad estadística.

Artefactos: `reports/optuna_temporal_v1/`.

## 4. Ventanas recientes y ensembles

Se compararon cuatro ventanas de entrenamiento: toda la historia, 24, 36 y 48 meses. Se mantuvieron los hiperparámetros ganadores anteriores: **esta comparación no incluyó retuning específico por ventana**.

Fueron 16 variantes, con cuatro folds cada una, y posteriormente siete ensembles: las seis parejas de algoritmos y el promedio de los cuatro, usando la mejor ventana de cada algoritmo.

| Modelo | 24 meses | 36 meses | 48 meses | Toda la historia |
|---|---:|---:|---:|---:|
| Regresión logística | 0,78742 | **0,78770** | 0,78656 | 0,78594 |
| Random Forest | 0,78743 | 0,78729 | 0,78758 | **0,78833** |
| XGBoost | 0,78736 | 0,78680 | 0,78707 | **0,78815** |
| CatBoost | 0,78656 | 0,78627 | 0,78645 | **0,78719** |

Las ventanas recientes ayudaron a logística. Los tres modelos de árboles rindieron mejor con toda la historia.

| Candidato | AUC temporal medio |
|---|---:|
| Logística 36 meses + XGBoost toda la historia | **0,78961** |
| Logística 36 meses + Random Forest toda la historia | 0,78958 |
| Promedio de los cuatro modelos | 0,78952 |
| Mejor individual: Random Forest | 0,78833 |

El ensemble ganador promedia los scores al 50 %. Su mejora sobre el mejor individual fue **0,00129 de AUC**. Los scores no se presentan como probabilidades calibradas.

### Qué significa combinar ventanas distintas

Ambos componentes predicen las mismas filas futuras, respetando el mismo corte temporal. Por ejemplo:

| Etapa | Regresión logística | XGBoost |
|---|---|---|
| Entrenamiento | Abril 2021–marzo 2024 | Toda la historia hasta marzo 2024 |
| Separación temporal | Abril–septiembre 2024 | Abril–septiembre 2024 |
| Validación | Octubre 2024–enero 2025 | Octubre 2024–enero 2025 |

Los 36 meses indican qué filas se usan para entrenar. No modifican la definición de churn ni recortan la historia utilizada para construir las features disponibles en cada observación.

Se exportó y verificó la recarga del candidato en `reports/recent_windows_v1/best_candidate.joblib`.

## 5. Priorización mensual y matrices de confusión

Se evaluó el ensemble ganador seleccionando el mayor riesgo **dentro de cada mes**, redondeando hacia abajo. Son 16 meses y **4.379 observaciones vendedora-mes**; una persona puede aparecer varias veces.

| Regla | Contactos mensuales promedio | Precisión | Recall | Lift mensual medio |
|---|---:|---:|---:|---:|
| Top 10 % | 27 | 69,1 % | 22,1 % | 2,28× |
| Top 20 % | 54 | 63,2 % | 40,8 % | 2,08× |
| Top 30 % | 82 | 57,6 % | 55,8 % | 1,89× |

La precisión y el recall anteriores se calculan acumulando observaciones; el lift es el promedio de los lifts mensuales. Por ello no debe calcularse ese lift medio dividiendo directamente la precisión agregada entre la prevalencia agregada.

| Resultado acumulado | Top 10 % | Top 20 % | Top 30 % |
|---|---:|---:|---:|
| Verdaderos positivos | 298 | 550 | 752 |
| Falsos positivos | 133 | 320 | 554 |
| Falsos negativos | 1.049 | 797 | 595 |
| Verdaderos negativos | 2.899 | 2.712 | 2.478 |
| Total priorizado | 431 | 870 | 1.306 |

El top 20 % implica entre **47 y 72 contactos mensuales** en los meses analizados. No es una cuota fija de 50; evaluar top 50 sería otra regla.

### Mes con 72 priorizadas

Fue noviembre de 2023, con 361 elegibles:

| Predicción | Sí hubo churn | No hubo churn |
|---|---:|---:|
| Priorizadas | 46 | 26 |
| No priorizadas | 93 | 196 |

Precisión 63,9 %, recall 33,1 %, lift 1,66× y AUC mensual 0,7564. Seleccionar 72 al azar habría identificado unos 28 casos en expectativa, frente a 46 del modelo.

**Identificar churn no significa evitarlo.** El efecto de contactar a las vendedoras necesita validación mediante una intervención real.

## 6. Métricas mensuales del ensemble: top 20 %

Precisión y lift corresponden a las priorizadas; AUC corresponde a todas las elegibles de cada mes.

| Mes | Priorizadas | Precisión | Lift | AUC |
|---|---:|---:|---:|---:|
| Octubre 2023 | 60 | 53,3 % | 2,08× | 0,7810 |
| Noviembre 2023 | 72 | 63,9 % | 1,66× | 0,7564 |
| Diciembre 2023 | 61 | 78,7 % | 2,13× | 0,8309 |
| Enero 2024 | 50 | 64,0 % | 2,00× | 0,7751 |
| Febrero 2024 | 47 | 44,7 % | 1,87× | 0,7191 |
| Marzo 2024 | 48 | 62,5 % | 2,16× | 0,8146 |
| Abril 2024 | 51 | 70,6 % | 2,22× | 0,8178 |
| Mayo 2024 | 53 | 62,3 % | 2,19× | 0,8190 |
| Junio 2024 | 50 | 46,0 % | 2,05× | 0,7972 |
| Julio 2024 | 56 | 66,1 % | 2,56× | 0,8282 |
| Agosto 2024 | 59 | 54,2 % | 1,87× | 0,7596 |
| Septiembre 2024 | 56 | 58,9 % | 1,74× | 0,7502 |
| Octubre 2024 | 49 | 63,3 % | 2,37× | 0,8034 |
| Noviembre 2024 | 53 | 75,5 % | 2,33× | 0,8006 |
| Diciembre 2024 | 54 | 75,9 % | 2,01× | 0,7995 |
| Enero 2025 | 51 | 68,6 % | 2,03× | 0,7909 |

Tabla exportada: [CSV mensual](../reports/recent_windows_v1/ensemble_top20_metricas_mensuales.csv).

## 7. Perfil de las vendedoras priorizadas

Este análisis corresponde al **Random Forest y su top 10 % mensual**, antes del ensemble. No debe confundirse con el perfil recalculado del ensemble.

| Característica —mediana— | Top 10 % | Resto |
|---|---:|---:|
| Meses desde la compra anterior | 17 | 1 |
| Meses con compras previas | 2 | 9 |
| Pedidos en últimos 12 meses | 1 | 5 |
| Monto últimos 12 meses | S/304 | S/1.993 |

Predominaban compradoras ocasionales que reaparecían después de una pausa larga. Todas compraron en el mes observado, por definición de la población.

Según el snapshot maestro, 95,1 % eran asesoras y 4,9 % líderes. Las asesoras ya representaban el 91,4 % de la base; parte de su predominio refleja esa composición. Esas categorías se usaron solo para descripción, no como entrada del nuevo modelo.

El grupo contenía 431 observaciones y 395 personas distintas; 295 observaciones terminaron en churn (68,4 %). Son asociaciones históricas, no explicaciones causales ni certezas individuales.

## 8. Fechas, límites y conclusiones

El snapshot local contiene compras hasta **mayo de 2026**. Como se necesitan seis meses futuros para madurar una etiqueta, permite evaluar observaciones hasta **noviembre de 2025**.

| Período observado | Uso en el experimento |
|---|---|
| Octubre 2023–enero 2025 | Cuatro validaciones usadas en selección |
| Febrero–julio 2025 | Gap respecto al OOT histórico |
| Agosto–noviembre 2025 | OOT histórico, fuera de la nueva búsqueda |

El OOT ya se había consultado en experimentos anteriores. Puede aportar una comprobación adicional, pero no debe presentarse como un test totalmente intacto.

No se demostró un techo teórico. Se observó una meseta entre algoritmos y una mejora modesta con el ensemble. El 0,78961 tampoco es directamente comparable con el antiguo AUC OOT de aproximadamente 0,765: cambian los períodos, el protocolo y las variables.

### Pendientes

1. Actualizar la extracción de datos y verificar hasta qué mes hay compras completas.
2. Mantener congelado el candidato y evaluar períodos posteriores no utilizados para seleccionarlo.
3. Con compras completas hasta agosto de 2026, se podrían evaluar etiquetas maduras hasta febrero de 2026.
4. Definir capacidad mensual de contacto, costos y valor real de retención.
5. Diseñar un piloto con control para medir retención incremental.
6. Como experimento adicional opcional, reajustar hiperparámetros para ventanas recientes; no se hizo en esta sesión.
7. Sincronizar la documentación antigua y completar la preparación operativa. No se desplegó un sistema de scoring en producción.

## 9. Qué notebooks estudiar y en qué orden

El objetivo es comprender y poder explicar el flujo, no memorizar cada línea. **La referencia del experimento actual son los notebooks 06 y 07 de modelado.** Los notebooks antiguos ayudan a entender el recorrido, pero conservan decisiones y resultados del protocolo anterior.

### Ruta principal

| Orden | Notebook | Qué debes aprender |
|---|---|---|
| 1 | [EDA del dataset](../01_data_understanding/01_churn_dataset_eda.ipynb) | Qué representa una fila, población, prevalencia, nulos, distribución temporal y variables. Separar exploración descriptiva de evidencia final. |
| 2 | [Split temporal original](../02_train_test_oot_splits/01_train_test_oot_splits.ipynb) | Train, gap y OOT; por qué se necesitan seis meses de separación; diferencia entre agrupar por persona y validar futuro. Los cuatro folds actuales se implementan en el notebook 06. |
| 3 | [Verificación de la reparación](../03_preprocessing/02_verificacion_reparacion.ipynb) | Alineación por claves, tipos numéricos, reconstrucción de variables y comprobación de resultados. Este notebook verifica la reparación; no vuelve a escribirla. |
| 4 | [Optuna con validación temporal](../05_modelling/06_optuna_validacion_temporal.ipynb) | **Núcleo del entrenamiento actual:** cuatro folds, selección interna, escalado, pesos de clase, espacios de búsqueda, función objetivo, poda, persistencia y métricas por período. |
| 5 | [Ventanas recientes y ensembles](../05_modelling/07_ventanas_recientes_ensemble.ipynb) | **Núcleo del candidato elegido:** ventanas de 24/36/48 meses, comparación controlada, predicciones fuera del entrenamiento de cada fold, promedio de scores, selección y exportación. |
| 6 | [Perfil del top 10 % mensual](../06_evaluation/04_perfil_top10_mensual.ipynb) | Seleccionar dentro de cada mes, comparar top frente al resto, interpretar precisión y lift. Atención: describe Random Forest, no el ensemble. |

**Si tienes poco tiempo, empieza por 06 y 07 de modelado.** Consulta EDA y splits cuando necesites aclarar el origen de los datos o las fechas. Después recorre la ruta completa.

### Complementarios: útiles para profundizar, no para explicar los resultados actuales como si fueran vigentes

| Notebook | Para qué sirve / advertencia |
|---|---|
| [EDA de tablas originales](../01_data_understanding/00_fact_tables_%26_dimensions_eda.ipynb) | Entender pedidos, detalle, dimensiones y calidad de los datos fuente. |
| [Horizonte de churn](<../01_data_understanding/horizonte_churn_v5 (2).ipynb>) | Entender el análisis del horizonte de inactividad; distinguir su contexto histórico de la población vigente. |
| [Preprocesamiento original](../03_preprocessing/01_preprocessing.ipynb) | Imputación y codificación categórica del flujo anterior. El flujo actual es transaccional y no usa todas esas categorías. |
| [Ingeniería y selección original](../04_feature_engineering%26selection/01_feature_engineering.ipynb) | Razones, tendencias e ingeniería original. Conserva la salvedad de accuracy frente a AUC y la selección previa a CV; no tomarlo como protocolo corregido. |
| [Modelos baseline](../05_modelling/01_modelos_baseline.ipynb) | Qué aporta una referencia inicial y cómo comparar clasificadores. |
| [Tuning XGBoost anterior](../05_modelling/02_tuning_xgboost_optuna.ipynb), [Random Forest anterior](../05_modelling/03_tuning_random_forest_optuna.ipynb), [logística anterior](../05_modelling/04_tuning_logreg_optuna.ipynb), [CatBoost anterior](../05_modelling/05_tuning_catboost_optuna.ipynb) | Consultar detalles de cada algoritmo. **No necesitas aprenderte los cuatro para seguir el experimento actual:** el notebook 06 concentra la búsqueda temporal nueva. |
| [Importancia de variables anterior](../06_evaluation/01_feature_importance.ipynb) | Importancia nativa y por permutación de los modelos anteriores. No es una interpretación recalculada del ensemble. |
| [SHAP anterior](../06_evaluation/02_shap.ipynb) | Explicaciones globales y locales. No corresponde a un análisis SHAP del ensemble nuevo. |
| [Utilidad del modelo anterior](../06_evaluation/03_utilidad_modelo.ipynb) | Conceptos de calibración, ganancias, lift, confusión y costo-beneficio. Usa modelos/resultados anteriores y supuestos económicos; no confundir sus cifras con las de este resumen. |

### Qué debes poder explicar al terminar

- Qué significa churn y qué vendedoras pueden recibir una predicción.
- Qué información está disponible en la fecha de predicción.
- Por qué existe un gap de seis meses y qué diferencia hay entre validación y test final.
- Qué maximiza Optuna y por qué el mejor trial no es una evaluación independiente.
- Por qué los dos componentes del ensemble pueden tener distintas ventanas sin usar el futuro.
- Qué diferencia hay entre AUC, precisión, recall y lift.
- Por qué top 20 % no significa 50 contactos fijos y por qué el ranking debe hacerse por mes.
- Por qué identificar churn no demuestra el efecto de una campaña de retención.

### Archivos de apoyo que no son notebooks

- `00_dataset_construction/qry_churn.sql`: **lectura necesaria** para entender la etiqueta, población y ventanas de features.
- `scripts/temporal_optuna.py`: backend validado y clases importables para guardar/cargar pipelines; el notebook 06 expone el flujo de entrenamiento.
- `scripts/recent_windows.py`: recorte de filas por meses y clase persistible del ensemble.
- `scripts/build_temporal_notebooks.py` y `scripts/build_recent_windows_notebook.py`: generación de notebooks; no hace falta estudiarlos para entender el modelo.
- `tests/test_temporal_optuna.py` y `tests/test_recent_windows.py`: garantías automatizadas de integridad y separación.
- [README](../README.md): comandos, reanudación y mapa del proyecto.

**Limitación de los entregables:** las matrices y tablas del ensemble para top 10/20/30 % y la tabla mensual de top 20 % se calcularon mediante comprobaciones y se exportaron a CSV/JSON en `reports/recent_windows_v1/`. No se creó un notebook específico que reúna esas matrices. El notebook 07 sí contiene el cálculo del lift mensual del top 10 % y la comparación de candidatos.

Los datos, estudios, checkpoints y modelos son artefactos locales ignorados por Git. Los notebooks conservan salidas ejecutadas. Se preservaron los cambios previos del usuario; no se hizo commit ni despliegue durante esta sesión.
