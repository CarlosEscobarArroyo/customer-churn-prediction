# Cambios para la PPT final — `SI2_2026_2_ESCOBAR_v4.pptx`

Brief para actualizar la presentación (44 láminas) al estado real del trabajo al 2026-10-04.
Las láminas 1–16 (problema, objetivos, literatura, marco teórico) casi no cambian; las 17–40
describen la experimentación del **seminario 1** y hay que rehacerlas; las 41–42 (experimento de
campo) ya están al día.

Los números de este documento salen de `sem2_draft/reports/*.md`, `04_modelado/modelo_final.json` y
`tablas_tesis.md`. No inventar ni redondear distinto; si un dato no está aquí, no ponerlo.

## 0. Reglas generales

1. **Mantener el diseño** (plantilla, colores, tarjetas con ícono, barra de «Lectura» al pie). Solo
   cambia el contenido.
2. **No mencionar ensembles** en ninguna lámina. El modelo final es un modelo individual: LightGBM.
3. **No mencionar datos maestros como variables** (sexo, edad, antigüedad, tipo de vendedora,
   departamento, provincia): se excluyeron. Tampoco one-hot ni imputación por mediana.
4. **No mostrar `id_vendedor` ni filas crudas** del dataset.
5. Nombres: «validación temporal (4 bloques)» para la selección y «test out-of-time» para la prueba
   final. El test out-of-time **no interviene en ninguna decisión**; todo lo que diga «se eligió
   porque…» debe apoyarse en la validación temporal.
6. Un solo separador decimal en toda la PPT (hoy mezcla «30,9 %» y «0.7648»). Usar punto, como en
   las tablas.
7. Tiempo: 15 minutos. La sección de experimentación/resultados debe quedar en ~12 láminas (ver §3);
   lo demás va a anexos después de «Gracias».
8. Numeración de secciones: hoy hay «4. Antecedentes» y «4. Marco teórico», y «6. Experimentación»,
   «6. Resultados» y «6. Referencias». Renumerar de corrido. Los números de página de las láminas
   también están desordenados (3, 4, 5, 7, 2, 6, 7, 7…): regenerarlos.

## 1. Imágenes disponibles (`presentation/img/`)

| Archivo | Qué muestra | Dónde va |
|---|---|---|
| `01_metodologia.png` | Flujo metodológico vigente, 5 bloques (4.1–4.5). Es vertical (1125×2125) | Lámina 17. Mejor **redibujarlo nativo en horizontal** con el estilo de la plantilla usando esta imagen como referencia de contenido; si se pega tal cual, no se lee |
| `02_eda_churn_mes.png` | Filas por mes y tasa de churn mensual, dic-2016 → mar-2026 | EDA (reemplaza el gráfico de balance de la lámina 25) |
| `03_eda_auc_univariado.png` | AUC univariado de las 42 variables | EDA (reemplaza la lámina 26). Es alta: recortar al top 10–15 o ponerla a un lado |
| `04_particion_bloques.png` | 4 bloques de validación temporal + test out-of-time, con brecha de 6 meses | Reemplaza la lámina 32 |
| `05_variables_permutacion.png` | Importancia por permutación de las 42 variables (XGBoost de referencia, media de 4 bloques) | Reemplaza la lámina 34. Es alta: a un lado, con el texto al otro |
| `06_variables_ablacion.png` | Ablación forward: AUC al agregar cada variable; en azul las 6 conservadas | Selección de variables |
| `07_modelos_comparacion.png` | AUC ± desv. est. y lift de los 5 modelos en validación temporal; LightGBM resaltado | Comparación de modelos (o rehacer como gráfico nativo con la Tabla C) |
| `08_evaluacion_oot_curvas.png` | ROC, precisión-recall y lift/recall/precisión por % contactado en el test out-of-time | Resultados del modelo final |
| `09_shap_resumen.png` | Gráfico de resumen SHAP (beeswarm), alta resolución | Reemplaza la imagen de la lámina 39 (misma figura, hoy pixelada) |
| `10_shap_dependencia.png` | Dependencia SHAP de las 6 variables | Lámina nueva o anexo |
| `11_shap_local_bajo.png`, `12_shap_local_medio.png`, `13_shap_local_alto.png` | Explicaciones locales: p = 0.05, 0.32 y 0.67 | Reemplazan la imagen de la lámina 40 (usar bajo y alto; medio opcional) |

Imágenes actuales de la PPT que **se conservan**: logos e íconos, las dos de la lámina 3, las fórmulas
del marco teórico (10–16) y el gráfico de hazard de la lámina 18.

## 2. Lámina por lámina

### 1 — Portada
- Dice «SEMINARIO DE INVESTIGACIÓN I» y «2026-1». Debe decir **SEMINARIO DE INVESTIGACIÓN II** y
  **2026-2**.

### 2 — Índice
- Actualizar a la numeración nueva y agregar «Resultados» si queda como sección aparte.

### 3 — Problema · 4 — Objetivos
- Sin cambios. Los cuatro objetivos específicos coinciden con lo hecho (OE1 selección de variables,
  OE2 re-entrenamiento con LightGBM y CatBoost, OE3 SHAP, OE4 experimento de campo).

### 5–7 — Revisión de la literatura
- Sin cambios obligatorios. Opcional en la 5 («Desbalance»): agregar que también se comparan pesos
  de clase y SMOTE (Chawla et al., 2002), porque ahora se evalúan.

### 8–9 — Antecedentes
- Sin cambios.

### 10–16 — Marco teórico
- 11, 12 y 13 son la misma lámina repetida con imágenes que se van sumando. Si no es una animación
  intencional, dejar solo la 13.
- 16: «Accuracy» no se usa en los resultados. Reemplazar esa tarjeta por **PR-AUC** (área bajo la
  curva precisión-recall), que sí se reporta. Precisión y recall se quedan.
- 15: sin cambios (SHAP y TreeSHAP es exactamente lo que se usó).

### 17 — Metodología propuesta — REHACER
La imagen actual es el flujo de seminario 1 (6 fases, con one-hot y solo RL · RF · XGBoost). Flujo vigente:

1. **4.1 Definición del churn, dataset y preprocesamiento** — churn = sin compras en 6 meses; panel
   vendedora-mes multi-slice; actualización con nuevos meses; imputación a cero; estandarización solo
   para regresión logística; pesos de clase vs SMOTE vs submuestreo.
2. **4.2 Selección de variables y del modelo final** — variables LRFM + acumuladas + cambio;
   permutación + ablación forward; LR, RF, XGBoost, LightGBM, CatBoost; validación temporal de 4
   bloques con AUC promedio; Optuna, 100 pruebas por modelo; ventana de 2, 3, 4 años o completa;
   GroupKFold por vendedora; test out-of-time; AUC-ROC, lift, Top Decile Lift, matriz de confusión,
   curva ROC.
3. **4.3 Análisis de explicabilidad SHAP** — global (ranking |SHAP| medio) y local (decil de mayor riesgo).
4. **4.4 Diseño y ejecución del experimento** — 6 cohortes mensuales, ingreso único; asignación
   aleatoria 50/50; tratamiento: top 30 % por modelo; control: 30 % aleatorio; incentivo durante 6 meses.
5. **4.5 Evaluación de resultados del experimento** — tasa de abandono por grupo; regresión
   logística modelo vs. azar; churn, precisión y recall en no contactadas.

### 18 — Definición del churn
- Se conserva (la definición no cambió). El gráfico y el 7.3 % son de seminario 1 y no se
  recalcularon con la extracción nueva: no agregarle cifras nuevas.

### 19–20 — Diseño del dataset multi-slice — CORREGIR CIFRAS

| Hoy dice | Debe decir |
|---|---|
| Panel 2017–2026 | Observaciones de **dic-2016 a mar-2026** (pedidos del DW hasta el 30-sep-2026; los últimos 6 meses solo sirven para etiquetar) |
| ≈ 89 cortes temporales | **112** cortes mensuales |
| 30 356 filas vendedora-mes | **31,235** |
| 6 323 vendedoras únicas | **6,386** |
| Prevalencia 30,9 % | **30.84 %** (9,633 positivos) |
| «estable (0,27–0,37 anual)» | 0.35–0.37 en 2018–2020; **0.27–0.30 desde 2021** |
| «≥ 3 meses de historia exigidos» | **Compra en el mes observado y al menos una compra previa** |

- Lámina 19, «Fuente de datos»: quitar «dimensiones de vendedoras»; el dataset usa solo pedidos (y el
  maestro de productos para las variables de diversidad).
- Agregar una línea: **42 variables**, todas transaccionales.

### 21–23 — Capturas del dataset — ELIMINAR
- Muestran filas crudas con `id_vendedor`, y la 23 muestra datos maestros que ya no se usan.

### 24 — EDA (texto) — REHACER
Las tres tarjetas actuales son de seminario 1 (fecha de nacimiento, productos sin categoría,
campañas, segmentos por tipo y departamento). Contenido nuevo:

- **Estructura del panel**: mediana de 2 filas por vendedora, máximo 88. 12 columnas con nulos
  estructurales (ratios y rezagos sin historia, hasta 48.2 %), 0 tras imputar a cero.
- **Estabilidad temporal**: prevalencia 30.84 %; por año 0.35–0.37 en 2018–2020 y 0.27–0.30 desde
  2021. En el test out-of-time: 27.65 %.
- **Poder predictivo**: `n_prod_u12` (AUC 0.722), `monto_u12` (0.722), `n_ped_u12` (0.718); todas en
  sentido inverso: a mayor actividad, menor abandono.
- Barra inferior (reemplaza «Diferencias por segmento»): **Redundancia**: 11 pares de variables con
  |ρ de Spearman| ≥ 0.9 (ventanas u3/u6/u12 de la misma métrica), lo que motiva la selección de variables.

### 25–28 — EDA (figuras)
- 25 (balance del target): reemplazar por `02_eda_churn_mes.png`.
- 26 (AUC univariado con `edad` y `antiguedad_meses`): reemplazar por `03_eda_auc_univariado.png`.
- 27 (matriz de correlación con `edad` y `antiguedad_meses`): **eliminar**; no hay versión nueva. La
  redundancia queda dicha en la lámina 24.
- 28 (churn por sexo, tipo, departamento, provincia): **eliminar**.

### 29 — Preprocesamiento — REHACER
Tres tarjetas nuevas:

- **Solo información transaccional**: se excluyen los datos maestros de la vendedora (sexo, edad,
  antigüedad, tipo, ubicación) porque el DW guarda solo su valor vigente, sin historial; usarlos
  asignaría información actual a períodos pasados.
- **Imputación y escala**: meses sin compra y valores faltantes se imputan a cero. Estandarización
  solo para la regresión logística, ajustada con el entrenamiento de cada bloque.
- **Corte del último mes**: solo se usan meses completos del DW, para que un mes parcial no contamine
  la etiqueta de churn.
- Barra inferior — **Desbalance** (XGBoost de referencia, 6 variables, validación temporal):

| Estrategia | AUC-ROC promedio | Lift top-10 % |
|---|:--:|:--:|
| **Pesos de clase** | **0.7884** | **2.35** |
| Sin ponderar | 0.7879 | 2.27 |
| Submuestreo | 0.7871 | 2.20 |
| SMOTE | 0.7791 | 2.23 |

  Ganan los pesos de clase; el peso de la clase positiva entra al espacio de búsqueda de Optuna. La
  ventaja sobre no ponderar es marginal (+0.0005 de AUC): decirlo así, sin exagerar.

### 30–31 — Capturas de código (`fit_preprocessor`, `apply_preprocessor`) — ELIMINAR
- Ese preprocesamiento (mediana de antigüedad, one-hot) ya no existe.

### 32 — Partición — REEMPLAZAR
- Imagen: `04_particion_bloques.png`. Título sugerido: «Validación temporal y test out-of-time».
- Texto: test out-of-time = últimos 4 meses de observación, separado por una brecha de 6 meses igual
  al horizonte de churn. Las decisiones se toman con 4 bloques de validación de 4 meses, cada uno
  entrenado con toda la historia previa menos la brecha.

| Bloque | Entrena hasta | Valida | Filas | Churn |
|---|---|---|--:|--:|
| 1 | jul-2023 | feb-2024 → may-2024 | 1,006 | 0.283 |
| 2 | nov-2023 | jun-2024 → sep-2024 | 1,110 | 0.279 |
| 3 | mar-2024 | oct-2024 → ene-2025 | 1,039 | 0.328 |
| 4 | jul-2024 | feb-2025 → may-2025 | 872 | 0.318 |
| **Test out-of-time** | may-2025 | **dic-2025 → mar-2026** | 879 | 0.277 |

  Datos de desarrollo: 29,183 filas.

### 33 — Ingeniería de variables y selección — REHACER
Lo que dice hoy está superado: «84 → 68 variables», permutación con Random Forest, AUC out-of-time
0.7625 → 0.7628 y `ticket_acum` como la más importante.

- **Enfoque LRFM** (se conserva, sin «antigüedad»): Length (historial de compras), Recency,
  Frequency, Monetary, extendido con variables acumuladas.
- **Familias derivadas** (se conserva): ventanas móviles 3/6/12 m, tendencias normalizadas [−1, 1],
  acumuladas de toda la historia y deltas respecto a t−n (n = 1, 3, 6, 9, 12).
- **Selección** (nuevo): importancia por permutación con un XGBoost de referencia, medida dentro del
  entrenamiento y promediada en los 4 bloques; luego ablación forward en orden de importancia: una
  variable se conserva solo si el AUC promedio sube más de 0.001.
- Barra inferior: **42 → 6 variables**: el AUC promedio en validación temporal pasa de 0.7828 a
  **0.7879** y el lift top-10 % de 2.24 a 2.36 (XGBoost de referencia). 37 de 42 variables tienen
  importancia positiva, pero solo 6 aportan en la ablación.

### 34 — Figura de permutación — REEMPLAZAR
- Imagen: `05_variables_permutacion.png` (hoy es la de Random Forest, con `ticket_acum` arriba).
- Agregar `06_variables_ablacion.png` en la misma lámina o en la siguiente, con la Tabla A.

**Tabla A — Variables seleccionadas**

| Orden | Variable | Qué mide | Caída de AUC al permutarla | AUC promedio al agregarla |
|--:|---|---|:--:|:--:|
| 1 | `compras_hist` | Meses con compra en toda la historia previa | 0.0089 | 0.6939 |
| 2 | `n_prod_u12` | Productos distintos en 12 meses | 0.0071 | 0.7760 |
| 3 | `monto_cv_u12` | Coeficiente de variación del monto en 12 meses | 0.0044 | 0.7804 |
| 4 | `n_ped_u12` | Pedidos en 12 meses | 0.0040 | 0.7825 |
| 5 | `monto_u12` | Monto en 12 meses | 0.0035 | 0.7844 |
| 6 | `ticket_prom_u3` | Ticket promedio en 3 meses | 0.0013 | 0.7879 |

### 35 — Tabla de variables — ACTUALIZAR
Quitar las tres filas de «Antigüedad y contexto (datos maestros)» (`antiguedad_meses`, sexo, tipo,
departamento). `compras_hist` pasa a una fila «Longitud». Tabla vigente (42 variables; marcar en
negrita las 6 seleccionadas):

| Tipo | Categoría | Variables | N.º |
|---|---|---|:--:|
| LRFM | Longitud | **`compras_hist`** | 1 |
| | Recencia | `meses_desde_compra_previa`, `recencia_norm` | 2 |
| | Frecuencia | `meses_activos_u{3,6,12}`, `n_ped_u{3,6,`**`12`**`}`, `intensidad_u3`, `tasa_act_reciente_vs_hist` | 8 |
| | Valor monetario | `monto_u{3,6,`**`12`**`}`, `monto_mean_u12`, `monto_std_u12`, **`monto_cv_u12`**, `monto_ult_vs_media`, **`ticket_prom_u3`**, `ticket_prom_u12` | 9 |
| Dinámica temporal | Tendencia normalizada | `tend_monto_u3_vs_prev3`, `tend_nped_u3_vs_prev3` | 2 |
| | Deltas | `d_monto_m{1,3,6,9,12}`, `d_nped_m{1,3,6,9,12}` | 10 |
| Acumuladas | Volumen histórico | `monto_acum`, `n_ped_acum`, `n_prod_acum` | 3 |
| | Ratios históricos | `ticket_acum`, `monto_por_prod_acum`, `monto_mensual_acum` | 3 |
| Diversidad de producto | Amplitud de catálogo | **`n_prod_u12`**, `n_cat_max_u12`, `n_cat_max_acum`, `basket_size_u12` | 4 |

Fuente: datos de pedidos (más el maestro de productos para diversidad). Por tiempo, esta tabla puede
ir a anexos.

### 36 — Entrenamiento y validación — REHACER
Hoy presenta GroupKFold como «Protocolo 1», tres clasificadores y Optuna con 60 evaluaciones en
validación anidada. Vigente:

- **Validación temporal (selección)**: 4 bloques expansivos con brecha de 6 meses; criterio = AUC
  promedio de los 4 bloques; desempate por Top Decile Lift mensual; a igualdad, el modelo más simple.
- **Cinco algoritmos, mismo presupuesto**: regresión logística, Random Forest, XGBoost, LightGBM y
  CatBoost; Optuna, **100 pruebas** por algoritmo, con las 6 variables y con las 42.
- **Ventana de entrenamiento**: con los hiperparámetros fijos, toda la historia frente a los últimos
  2, 3 o 4 años.
- Barra inferior: **GroupKFold por vendedora** pasa a ser verificación complementaria del modelo
  elegido; el **test out-of-time** se reserva para el modelo final.

### 37 — Resultados: comparación de modelos — REHACER
Toda la lámina es de seminario 1 (3 modelos, XGBoost ganador, test de 803 obs.). El gráfico nativo de
barras «Top-Decile Lift por modelo» también: rehacerlo con los 5 modelos o usar
`07_modelos_comparacion.png`.

Subtítulo: «Validación temporal (4 bloques, feb-2024 → may-2025) — tras optimización».

**Tabla C — Comparación de modelos**

| Modelo | Variables | Ventana | AUC-ROC promedio | Desv. est. | Lift top-10 % mensual |
|---|:--:|:--:|:--:|:--:|:--:|
| **LightGBM** | 6 | Toda la historia | **0.7911** | 0.0084 | **2.31** |
| XGBoost | 6 | Toda la historia | 0.7910 | 0.0094 | 2.28 |
| CatBoost | 6 | Toda la historia | 0.7904 | 0.0090 | 2.28 |
| Random Forest | 6 | Toda la historia | 0.7898 | 0.0086 | 2.29 |
| Regresión logística | 42 | 36 meses | 0.7871 | 0.0076 | 2.24 |

- Con 6 variables ganan los cuatro modelos de árboles (+0.003 a +0.005 de AUC frente a 42); la
  regresión logística prefiere las 42 (0.7845 vs 0.7811) y mejora con ventana de 36 meses (0.7871).
- En árboles, toda la historia ≥ 4 años ≥ 3 años > 2 años, con diferencias ≤ 0.003.
- Barra «Selección»: los cinco modelos quedan entre 0.787 y 0.791, una diferencia menor que la
  variación entre bloques (≈ 0.009): **discriminación equivalente**. Se elige **LightGBM**: mayor AUC
  promedio y mayor lift entre los modelos, con solo 6 variables. Lectura honesta: el techo está en la
  información disponible, no en el algoritmo.

### 38 — Resultados: test out-of-time — REHACER
Subtítulo: «Test out-of-time: dic-2025 → mar-2026, 879 observaciones, prevalencia 27.6 %».
Imagen: `08_evaluacion_oot_curvas.png`.

| Métrica (LightGBM, 6 variables) | Valor |
|---|--:|
| AUC-ROC | **0.7931** |
| PR-AUC | 0.574 |
| Top Decile Lift (media mensual) | 2.19 |
| AUC-ROC por mes (dic, ene, feb, mar) | 0.792 / 0.780 / 0.789 / 0.815 |

**Puntos de operación**

| % base contactada | Vendedoras | Churners capturados | Precisión | Recall | Lift |
|--:|--:|--:|:--:|:--:|:--:|
| 5 % | 44 | 31 | 0.705 | 0.128 | 2.55 |
| 10 % | 88 | 55 | 0.625 | 0.226 | 2.26 |
| 20 % | 176 | 106 | 0.602 | 0.436 | 2.18 |
| 30 % | 264 | 144 | 0.545 | 0.593 | 1.97 |
| 50 % | 440 | 204 | 0.464 | 0.840 | 1.68 |

(«Churners capturados» = vendedoras × precisión, sobre 243 churners en total.)

- Matriz de confusión con p ≥ 0.5: TP 123, FP 91, FN 120, TN 545 → precisión 0.575, recall 0.506,
  24 % de la base contactada.
- El lift del top 10 % aparece con dos valores y ambos son correctos: **2.26** con los 4 meses juntos
  y **2.19** como media de los 4 meses por separado. Rotular cuál es cuál.
- Resaltar la fila del 30 %: es la cobertura del experimento de campo.
- Mensaje: el AUC en test (0.7931) queda al nivel del de validación (0.7911) y los cuatro meses son
  homogéneos (0.78–0.81): el modelo generaliza a un período no visto.
- Barra «En respuesta al objetivo general»: reemplazar la frase de XGBoost por: **LightGBM con 6
  variables** alcanza AUC 0.793 en test out-of-time; contactando al 10 % de mayor riesgo se obtiene
  precisión 0.63 y 2.3× más abandonos que al azar, y con el 30 % se captura el 59 %.

**Verificación GroupKFold** (cabe como nota en esta lámina o en anexo):

| Validación | AUC-ROC | Desv. est. |
|---|:--:|:--:|
| GroupKFold (5) por vendedora | 0.7338 | 0.005 |
| StratifiedKFold (5) sin grupos | 0.7334 | 0.007 |

Separar vendedoras no cambia el resultado (+0.0004): el modelo no memoriza individuos. El nivel
0.73 no es comparable con el 0.79 temporal porque el K-fold aleatorio valida sobre toda la historia
2016–2025, incluidos los años 2018–2020 de mayor prevalencia.

### 39 — SHAP global — CORREGIR
- Imagen: reemplazar por `09_shap_resumen.png` (misma figura en alta resolución).
- Aclarar que SHAP se calculó con TreeSHAP sobre el test out-of-time (879 filas), modelo LightGBM.
- Agregar el ranking (% del |SHAP| medio): `n_ped_u12` 21.9, `monto_u12` 20.2, `compras_hist` 20.2,
  `n_prod_u12` 16.0, `monto_cv_u12` 13.8, `ticket_prom_u3` 7.9.
- «`ticket_prom_u3`: efecto mixto, sin dirección clara» → corregir: es la de menor peso y, en
  conjunto, un ticket mayor reduce el riesgo (correlación valor–SHAP −0.69), aunque la relación no
  es monótona.
- «compras_hist: la antigüedad activa también protege» → «el historial de compras también protege»
  (antigüedad era una variable de datos maestros que ya no existe).
- Barra «Lectura»: quitar «El modelo prioriza el comportamiento reciente de compra sobre el perfil de
  la vendedora»: el perfil no se incluyó, así que el modelo no lo pudo priorizar ni descartar.
  Reemplazar por: «cinco de las seis variables reducen el riesgo al aumentar; solo la irregularidad
  del monto (`monto_cv_u12`) lo eleva».
- Opcional (anexo o lámina extra): `10_shap_dependencia.png`. Lectura (valores aproximados, leídos
  del gráfico; no están en los reportes): el efecto de `n_ped_u12`,
  `monto_u12`, `compras_hist` y `n_prod_u12` baja rápido y se satura (≈ 15 pedidos, ≈ S/ 6,000,
  ≈ 30 meses con compra); `monto_cv_u12` cruza a riesgo cerca de 1.0.
- Si preguntan por la consistencia con la permutación: ambos métodos coinciden en `ticket_prom_u3`
  como la última, pero difieren en el orden del resto (Spearman 0.14). No son comparables uno a uno:
  la permutación se midió con el XGBoost de referencia sobre 42 variables correlacionadas y SHAP
  sobre el modelo final de 6. No ponerlo en lámina como «consistentes».

### 40 — SHAP local — CORREGIR
- Imágenes: `11_shap_local_bajo.png` y `13_shap_local_alto.png` (la actual está pixelada).
  `12_shap_local_medio.png` (p = 0.32) solo si hay espacio.
- Los valores de los casos son correctos (bajo p = 0.05: monto 9,070, 38 pedidos, 26 compras
  históricas, 126 productos; alto p = 0.67: 1 compra histórica, 1 pedido, 2 productos, monto 33,
  CV 3.46).
- «Todas las variables empujan a la baja» → «casi todas»: en el caso bajo, `ticket_prom_u3` empuja
  levemente al alza (+0.16).
- «gasto inestable, abandono inminente» → «gasto inestable»; el modelo da una probabilidad de 0.67,
  no una certeza.
- Valor base: «≈ 0.30» → **0.287**; decir «cercano a la prevalencia», no «coincide».
- Indicar que los casos son los percentiles 5, 50 y 95 del puntaje en el test out-of-time; el caso
  alto pertenece al decil de mayor riesgo (umbral del decil: p ≥ 0.631).

### 41 — Diseño y ejecución del experimento — AL DÍA
Coincide con la cohorte real. Ajustes menores:
- Tarjeta «1 · Modelo»: precisar «LightGBM re-entrenado con 31,235 filas (observaciones etiquetadas
  hasta mar-2026, pedidos hasta sep-2026)».
- Agregar el dato que muestra que la selección funciona como se diseñó: probabilidad media de
  abandono de las seleccionadas **0.559 en tratamiento** (mínimo 0.452) frente a **0.365 en control**;
  media de las 251 elegibles 0.316.
- Estado: **1 de 6 cohortes** (sep-2026). Faltan oct-2026 → feb-2027 y la medición a 3 y 6 meses.
  **Todavía no hay resultados de abandono**: no poner ninguna cifra de efecto.

### 42 — Programa «Círculo Impulso» — AL DÍA
- Opcional: el pedido mínimo de rescate es S/ 180 frente a S/ 150 del catálogo general; el premio
  tiene un costo acotado al 8 % del ticket facturado.

### 43 — Referencias
- El título dice «6. Referencias»: renumerar.
- Agregar las citadas en la metodología nueva que no están: Loureiro et al. (2025) — LightGBM y
  selección forward; Yang et al. (2024) — importancia por permutación; Chawla et al. (2002) — SMOTE;
  Akiba et al. (2019) — Optuna; Lundberg & Lee (2017) — SHAP; Jiang et al. (2024) — interpretabilidad
  local con SHAP; Wagh et al. (2024) y Faraji Googerdchi et al. (2024), que ya se citan en la lámina 8.
  **Los datos bibliográficos completos (título, revista) los debe dar Carlos**: no inventarlos.

### 44 — Gracias
- Sin cambios. Después van los anexos.

## 3. Orden sugerido de la sección de experimentación y resultados

1. Metodología (17, rehecha)
2. Definición del churn (18)
3. Dataset multi-slice (19–20, fusionadas en una)
4. EDA (24 con `02_eda_churn_mes.png`)
5. Preprocesamiento y desbalance (29)
6. Validación temporal y test out-of-time (32)
7. Selección de variables: 42 → 6 (33 + 34 + Tabla A)
8. Entrenamiento y comparación de modelos (36 + 37)
9. Test out-of-time del modelo final (38)
10. SHAP global (39)
11. SHAP local (40)
12. Experimento de campo: diseño (41) y programa (42)

Anexos: tabla de las 42 variables (35), AUC univariado, dependencia SHAP, verificación GroupKFold,
matriz de confusión.

## 4. Lo que falta y no se puede poner todavía

- Resultados del experimento de campo (OE4): solo existe la asignación de la primera cohorte.
- Conclusiones: la PPT no tiene lámina de conclusiones. Si se agrega, limitarla a lo medido:
  (1) 6 variables transaccionales igualan o superan a las 42; (2) los cinco algoritmos rinden de forma
  equivalente, el límite está en la información; (3) LightGBM generaliza al test out-of-time (AUC
  0.793); (4) el efecto sobre el abandono real aún no está medido: depende del experimento en curso.
