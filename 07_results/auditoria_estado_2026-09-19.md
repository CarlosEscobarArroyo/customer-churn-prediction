# Auditoría del estado del proyecto — 2026-09-19

## Dictamen

El proyecto es un **prototipo experimental avanzado**: dispone de construcción de datos, exploración, partición temporal, preprocesamiento, selección, cuatro modelos optimizados e interpretación. Las métricas almacenadas se pueden verificar con los modelos locales, pero **la ejecución directa actual está afectada por un dato corrupto** y la validación no permite sostener todas las afirmaciones de la documentación. No se encontró un flujo operativo de inferencia listo para producción.

Prioridad: recuperar la reproducibilidad, corregir el protocolo de validación y unificar los entregables antes de ampliar la experimentación.

## Alcance y límites

- Inspección del código SQL/Python, los 14 notebooks, documentación técnica y resultados Markdown, archivos de parámetros, datos y modelos locales; revisión de Git.
- Comprobación de etiquetas contra las tablas locales de pedidos y fechas; verificación de claves, particiones y características.
- Ejecución del preprocesamiento actual en memoria, sin escribir sus CSV, y comparación con el CSV preprocesado existente.
- Inferencia de los cuatro modelos guardados sobre el test temporal; no se repitieron los entrenamientos ni los 60 trials por modelo.
- No se consultó BigQuery, no se verificó la actualidad del warehouse y no se reejecutó el SQL remoto. No se auditó el contenido de los documentos Word ni la bibliografía.
- Se conservaron los cambios previos del usuario. Este informe no implica una reparación de los hallazgos.

## Estado comprobado de los datos

| Elemento | Resultado |
|---|---|
| Dataset original | 30.356 filas, 46 columnas, 6.323 vendedoras |
| Prevalencia | 30,93 % |
| Observaciones | Diciembre de 2016 a noviembre de 2025 |
| Última fecha de pedidos locales | 31 de mayo de 2026 |
| Entrenamiento | 28.311 filas, hasta enero de 2025 |
| Separación temporal | 1.242 filas, febrero–julio de 2025 |
| Test | 803 filas, 492 vendedoras, agosto–noviembre de 2025 |
| Positivos en test | 210; prevalencia 26,15 % |
| Variables | 78 preprocesadas; 68 seleccionadas |

Comprobaciones satisfactorias:

- Cero claves duplicadas `(id_vendedor, mes_rank)` en los cuatro CSV de etapas.
- Claves y etiquetas alineadas entre dataset original, preprocesado, seleccionado y split.
- Cero diferencias entre la partición persistida y la fórmula temporal que usan los modelos, con los parámetros actuales.
- Las 30.356 etiquetas coinciden con la ausencia de compras en los seis meses siguientes, reconstruida desde los pedidos locales.
- Todas las observaciones tienen compra en el mes observado, actividad previa y seis meses de futuro disponibles en el snapshot local.
- El preprocesamiento actual reproduce el CSV preprocesado, dentro de tolerancia numérica, y produce características numéricas sin nulos.
- La lista de 68 características coincide con `selected_features.json`.

La población vigente es **vendedoras que compraron en el mes observado y tienen al menos un mes de compra previo**. No abarca automáticamente a todas las vendedoras inactivas. Esta definición debe acompañar cualquier propuesta de uso mensual.

## Hallazgos prioritarios

### Alta — CSV de modelado corrupto y evaluación directa bloqueada

En `data/processed/churn_dataset_features.csv`, la columna `monto_ult_vs_media` contiene el texto `Directo` en la posición de fila 919, contando desde cero. Corresponde a entrenamiento. Es la única discrepancia numérica material frente a las columnas compartidas con `churn_dataset_processed.csv` (tolerancia 1e-9).

Consecuencias verificadas:

- Pandas lee la columna como `object`.
- XGBoost falla en `predict_proba` por el tipo de columna, aunque la celda corrupta no pertenece al test.
- Los notebooks de evaluación cargan ese CSV sin conversión defensiva, por lo que sus salidas guardadas no prueban que el código actual pueda ejecutarse directamente.
- CatBoost convierte valores no numéricos a NaN. Esto permite continuar, pero no recupera el valor correcto ni demuestra con qué versión del dato se entrenó el binario guardado.

Acción: regenerar el dataset seleccionado desde la etapa anterior y comprobar esquema, finitud y correspondencia con el origen antes de entrenar. Conservar evidencia del defecto y registrar hashes de los nuevos artefactos. No sustituir silenciosamente valores inválidos por NaN como reparación definitiva.

### Alta — La auditoría anterior sobrestima el aislamiento de la validación

`07_results/auditoria_leakage.md` afirma que el OOT se evalúa una sola vez y que las salvedades no tienen impacto material. El código actual no permite sostenerlo:

- La etapa 04 evalúa en OOT las variables base, las ampliadas y las seleccionadas.
- Las etapas 05 y 06 comparan repetidamente modelos sobre el mismo OOT; `03_utilidad_modelo` identifica al ganador por lift en ese bloque.
- El preprocesamiento y, especialmente, la **selección supervisada de variables** se ajustan antes de la CV interna, sobre todo el train-pool. Las etiquetas de las filas de validación de cada fold ya participaron en esa selección.
- El mejor AUC de Optuna se reporta como AUC GroupKFold: es una métrica utilizada para optimizar hiperparámetros, no una estimación independiente posterior a esa búsqueda.

Esto no demuestra fuga directa del objetivo a las features ni invalida toda la señal observada. Sí impide presentar el OOT como test final intacto si se usa para decidir el modelo, y deja un posible optimismo en CV cuyo tamaño no se ha medido.

El código implementa CV interna de tuning más un holdout temporal; no contiene múltiples folds externos para estimar una nested CV completa. La nomenclatura y el alcance de las garantías deben precisarse.

Acción: ajustar transformaciones y selección dentro de cada fold; seleccionar modelo y regla operativa en validación; reservar un período final nuevo sin utilizarlo para decisiones. Incorporar evaluaciones temporales sucesivas y cuantificar incertidumbre considerando las observaciones repetidas por vendedora.

### Media — La selección usa accuracy, aunque el gráfico dice AUC

En `04_feature_engineering&selection/01_feature_engineering.ipynb`, la llamada a `permutation_importance` omite `scoring`. Se comprobó en la instalación local que el valor por defecto usa `estimator.score`; para el Random Forest clasificador corresponde a accuracy. El gráfico etiqueta el resultado como caída de AUC.

Además, la importancia se calcula sobre el mismo entrenamiento usado para ajustar el bosque y se considera “significativa” toda importancia positiva; ese criterio no constituye una prueba de significancia estadística.

Acción: definir explícitamente la métrica de selección y calcularla en validación interna, manteniendo el test final fuera de esa decisión. Cambiar a `scoring='roc_auc'` puede modificar las 68 variables y exige volver a evaluar los modelos.

### Media — Documentación y elección del modelo desactualizadas

- `CLAUDE.md` describe rutas ausentes, un modelo HistGradientBoosting y una población con tres compras históricas, distintos del flujo vigente.
- `pyproject.toml` declara `README.md`, pero el archivo no existe.
- El capítulo de experimentación y el resumen ejecutivo presentan XGBoost como final y omiten parte de la experimentación posterior.
- CatBoost tiene notebook, parámetros y modelo local, pero no está incluido en los tres notebooks de evaluación de la etapa 06.
- La afirmación de un “techo de señal” de 0,77 y la equivalencia estadística de modelos no quedan demostradas por la comparación disponible. La desviación mensual del AUC no basta para demostrar equivalencia.
- La auditoría anterior menciona un snapshot CSV “committeado”; los CSV actuales están ignorados y no aparecen entre los archivos versionados.

Acción: establecer una única definición vigente de población, datos, protocolo, métrica rectora y modelo elegido; actualizar las conclusiones a partir de una evaluación consistente.

### Media — Reproducibilidad dependiente del entorno local

- Existe `uv.lock` y un entorno Python 3.12.3 funcional, pero faltan un comando integral del pipeline y documentación de ejecución vigente.
- Solo el preprocesamiento consume el split persistido; las etapas posteriores duplican la fórmula. Coinciden hoy, pero cambiar la etapa 02 no actualiza automáticamente el resto.
- No se encontró un artefacto persistido que reúna imputación, vocabulario categórico, ingeniería, selección y modelo para inferencia nueva.
- Los modelos se cargan por existencia del archivo, sin verificar compatibilidad mediante hash de datos, versión de código o manifiesto de entrenamiento.
- Los estudios Optuna no se persisten completos: se guardan mejores parámetros y métricas, sin una base de estudios configurada.
- `MedianPruner(n_warmup_steps=5)` recibe pasos de fold 0 a 4: la poda nunca llega a activarse en esos cinco pasos. La documentación atribuye una poda temprana que esa configuración no realiza.

### Media — Falta cerrar el uso operativo

- No se encontró un flujo de scoring periódico, despliegue, monitoreo ni registro de intervenciones.
- El SQL de entrenamiento excluye los seis meses sin etiqueta madura; no sirve sin adaptación para puntuar el mes más reciente.
- El costo-beneficio utiliza supuestos explícitos: contacto S/5, retención S/200 y éxito 30 %. No acredita retorno observado de una campaña.
- Las variables maestras SCD-1 no garantizan el estado histórico de cada observación; queda un riesgo de información no disponible en su fecha que requiere evaluar o eliminar esas variables.
- El snapshot local llega a mayo de 2026. No se verificó si el warehouse tiene datos más recientes.

## Métricas verificadas de los modelos guardados

Mismo OOT de 803 filas. Lift calculado sobre las 80 filas de mayor score (`n // 10`); recall con umbral 0,5. Todos los AUC coinciden con sus JSON de parámetros.

| Modelo | AUC OOT | Average precision | Recall | Lift top 10 % | Brier |
|---|---:|---:|---:|---:|---:|
| Regresión logística | 0,7693 | 0,5082 | 0,7333 | 2,2465 | 0,1935 |
| CatBoost | 0,7652 | 0,4975 | 0,7571 | 2,2943 | 0,2013 |
| XGBoost | 0,7648 | 0,5029 | 0,7762 | 2,4377 | 0,2012 |
| Random Forest | 0,7590 | 0,4916 | 0,7286 | 2,3421 | 0,1975 |

Para verificar XGBoost se convirtió la columna problemática a numérica **solo en memoria**. No hay valores inválidos en test; el CSV no se modificó. Las otras tres inferencias funcionaron con el test leído originalmente. Esto verifica la inferencia y las métricas, no la reproducibilidad de sus entrenamientos.

La regresión logística presenta el mayor AUC observado; XGBoost el mayor lift del decil y recall a 0,5. No hay un ganador universal ni evidencia aquí de superioridad estadística. El Brier de predecir la prevalencia del propio test es 0,1931, una referencia descriptiva; ningún modelo lo mejora en este bloque. Si se necesitan probabilidades interpretables, se requiere evaluar calibración con datos separados del test final.

## Estado de ingeniería y Git

- Último commit local: `eb5bfd2`, 2026-06-06, mensaje `fiz`.
- Antes de esta auditoría había cinco notebooks modificados y cuatro archivos sin seguimiento: notebook y parámetros de CatBoost, más dos Word.
- Los 14 notebooks no contienen errores almacenados, pero todas las celdas de código del preprocesamiento carecen de contador de ejecución; una celda de ingeniería también. Esto no equivale a una reejecución validada.
- El cambio no confirmado de ingeniería incluye una expresión aislada `13`, sin efecto funcional.
- `pytest --collect-only -q`: cero pruebas recogidas, salida 5.
- `ruff check . --output-format concise --statistics`: 187 incidencias, principalmente estilo (151 E702 y 25 E741); también 8 F541, 2 F401 y 1 E402. No son 187 fallos funcionales.
- El código extraído de los 14 notebooks pasó análisis sintáctico Python.
- No se encontraron configuración de CI ni pruebas automatizadas del pipeline.

## Orden recomendado de trabajo

1. **Restablecer reproducibilidad:** regenerar el CSV afectado, validar esquema y splits, registrar hashes y verificar nuevamente inferencia y entrenamiento.
2. **Cerrar metodología:** selección y transformaciones dentro de CV; métrica explícita; reserva temporal final; retirar garantías que no se han comprobado.
3. **Cerrar resultados:** comparar los cuatro modelos con el mismo protocolo, elegir según el objetivo operativo y sincronizar documentación y tablas.
4. **Preparar entrega:** README, configuración única, pipeline de inferencia persistido y verificaciones de datos y artefactos.
5. **Validar negocio:** datos recientes, capacidad mensual de contacto, costos reales y piloto con grupo de control para medir retención incremental.

El trabajo ya realizado permite continuar desde una base experimental sustancial. Los principales pendientes son de integridad, validación y trazabilidad, antes que de incorporar más algoritmos.
