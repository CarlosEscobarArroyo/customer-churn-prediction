# Avance de sem2 — 2026-10-01

Bitácora de qué se hizo, qué dio y qué falta, por sección de la metodología (`metodologia.md`).
Los números salen de los reportes generados (`reports/*.md`, `02_particion/particion.json`,
`03_variables/variables_seleccionadas.json`); no se escriben a mano.

## Estado por sección

| Sección | Qué pide | Estado | Dónde |
|---|---|---|---|
| 4.1 | Re-extraer del DW; misma definición de churn, población y multi-slice mensual | ✅ | `00_datos/` |
| 4.1 | Imputación a cero; estandarización solo para logística, ajustada en train | ✅ | `src/datos.py`; `StandardScaler` dentro del pipeline |
| 4.1 | Desbalance: pesos de clase vs sin ponderar vs submuestreo vs SMOTE | ✅ | `04_modelado/` |
| 4.2 | Solo transaccional LRFM + acumuladas + cambio; sin datos maestros | ✅ | `00_datos/qry_churn.sql` |
| 4.2 | Permutación + ablación forward | ✅ | `03_variables/` |
| 4.3 | OOT 4 meses + brecha 6 m + 4 bloques de validación expansiva | ✅ | `02_particion/` |
| 4.3 | LogReg, RF, XGBoost, LightGBM, CatBoost con Optuna, mismo presupuesto | ✅ | `04_modelado/` |
| 4.3 | Ventanas de entrenamiento y ensembles | ✅ | `04_modelado/` |
| 4.3 | GroupKFold como verificación complementaria | ✅ | `05_evaluacion/` |
| 4.3 | OOT una sola vez: AUC, PR-AUC, ROC, matriz, precisión/recall/lift por % contactado | ✅ | `05_evaluacion/` |
| 4.4 | SHAP global, dependencia, casos locales del decil superior | ⏳ | `06_interpretacion/` |
| — | EDA (no está en la metodología; se agregó a pedido) | ✅ | `01_eda/` |

## 00_datos — extracción y preprocesamiento (§4.1)

SQL heredado de `sem1/00_dataset_construction/qry_churn.sql` con tres cambios:

1. Sin datos maestros (sexo, edad, antigüedad, tipo, ubicación): solo existe su valor vigente (§4.2).
2. Sin variables de campañas: en `sem1` el ensemble de 42 variables igualó a los modelos de 71 en el OOT.
3. Corte al último mes completo del warehouse: el mes del último pedido se conserva solo si ese pedido
   cae en el último día del mes. Sin este corte, un mes parcial entra en la ventana de etiqueta de los
   últimos meses observados y contamina `churn`. `sem1` no lo controlaba.

Extracción del 2026-10-01 (DW con pedidos hasta el 30-sep-2026):

- **31,235 filas** (vendedora-mes), **6,386 vendedoras**, observaciones **dic-2016 → mar-2026**.
- Prevalencia de churn **30.84 %**. 36 variables del SQL + 6 derivadas en Python = **42**.
- 12 columnas con nulos estructurales (SAFE_DIVIDE / LAG sin historia); 0 nulos tras imputar a cero.

## 01_eda

- Prevalencia por año estable en 0.27–0.30 desde 2021 (2018–2020 más alta, 0.35–0.37).
- Panel: mediana 2 filas por vendedora, máximo 88. Justifica GroupKFold como verificación.
- Señal univariada: `monto_u12`, `n_prod_u12`, `n_ped_u12` con AUC ≈ 0.72 cada una por sí sola.
- Redundancia: 11 pares con |ρ de Spearman| ≥ 0.9 (ventanas u3/u6/u12 de la misma métrica).

Detalle: `reports/eda.md`.

## 02_particion (§4.3)

Una sola definición en `src/particion.py`; fechas efectivas en `02_particion/particion.json`.

| bloque | train hasta | validación | n_val | churn |
|---|---|---|---:|---:|
| val_1 | 2023-07 | 2024-02 → 2024-05 | 1006 | 0.283 |
| val_2 | 2023-11 | 2024-06 → 2024-09 | 1110 | 0.279 |
| val_3 | 2024-03 | 2024-10 → 2025-01 | 1039 | 0.328 |
| val_4 | 2024-07 | 2025-02 → 2025-05 | 872 | 0.318 |
| **oot** | 2025-05 | **2025-12 → 2026-03** | 879 | 0.277 |

Pool de desarrollo: 29,183 filas. El OOT no se usa en ninguna decisión de `03` a `04`.

**Nota de honestidad:** feb y mar 2026 nunca se vieron en `sem1`; dic-2025 y ene-2026 sí aparecieron
en uno de los OOT de `sem1`. El OOT no es 100 % virgen, pero en `sem2` cumple el protocolo (una sola
evaluación, sin decisiones sobre él).

## 03_variables (§4.2)

Modelo de referencia: XGBoost con parámetros fijos moderados y `scale_pos_weight`. Todo sobre los 4
bloques del pool de desarrollo.

1. **Permutación dentro del entrenamiento**: en cada bloque, partición temporal interna con la misma
   brecha de 6 m; `scoring='roc_auc'`, 5 repeticiones; promedio de los 4 bloques. 37 de 42 con
   importancia media > 0. Descartadas: `tend_monto_u3_vs_prev3`, `tend_nped_u3_vs_prev3`,
   `d_monto_m12`, `ticket_prom_u12`, `recencia_norm`.
2. **Ablación forward** en orden de importancia; se conserva si el AUC medio sube > 0.001.
   Se probó antes una regla más estricta (sube el AUC medio y mejora en ≥ 3 de 4 bloques):
   también dio 6 variables, cambiando solo la sexta (`d_monto_m1` por `ticket_prom_u3`).

**Seleccionadas (6):** `compras_hist`, `n_prod_u12`, `monto_cv_u12`, `n_ped_u12`, `monto_u12`,
`ticket_prom_u3`.

| | 42 variables | 6 seleccionadas |
|---|---:|---:|
| XGBoost ref. — AUC medio 4 bloques | 0.7828 | **0.7879** |
| XGBoost ref. — lift top-10 % mensual | 2.24× | **2.36×** |
| Logística (C=0.1, balanced) — AUC medio | **0.7844** | 0.7805 |

Lectura: para árboles, 6 variables bastan y mejoran; la logística rinde algo mejor con 42 (+0.004).
Ninguna otra variable mueve el AUC medio más de 0.001. Es consistente con `sem1`: el techo está en la
información, no en el número de variables. Son métricas de selección, no estimación final.

**Decisión pendiente para `04`:** tunear cada algoritmo con los dos conjuntos (6 y 42) y dejar que el
criterio del §4.3 (AUC medio; desempate por Top Decile Lift; a igualdad, el más simple) decida.

Detalle: `reports/variables.md`.

## 04_modelado (§4.1 desbalance + §4.3 búsqueda, ventanas, ensembles)

Secuencia **greedy**: cada decisión se toma con los 4 bloques y queda fija antes de la siguiente. Criterio
en todas: AUC medio; empate (≤ 0.001) → lift top-10 % mensual; empate (≤ 0.02) → el más simple (menos
miembros, luego menos variables). Corrida del 2026-10-01, 33 min de búsqueda (24 hilos).

1. **Desbalance** (XGBoost de referencia, 6 variables): pesos de clase 0.7884 / lift 2.35; sin ponderar
   0.7879 / 2.27; submuestreo 0.7871 / 2.20; SMOTE 0.7791 / 2.23. **Gana pesos de clase**; en Optuna el
   peso de la clase positiva se tunea en [1, ratio], así cubre el continuo entre "sin ponderar" y "balanceado".
2. **Optuna**: 5 algoritmos × {6, 42} variables, **100 trials** por estudio (TPE, semilla 42), objetivo
   AUC medio de los 4 bloques. Estudios persistidos en `04_modelado/optuna.db` (reanudable);
   trials en `04_modelado/trials/*.csv` (no versionados).

   | algoritmo | AUC 6 vars | AUC 42 vars | elegido | ventana elegida | AUC final | lift |
   |---|---:|---:|---|---|---:|---:|
   | LightGBM | **0.7911** | 0.7862 | 6 | todo | 0.7911 | 2.31 |
   | XGBoost | **0.7910** | 0.7858 | 6 | todo | 0.7910 | 2.28 |
   | CatBoost | **0.7904** | 0.7859 | 6 | todo | 0.7904 | 2.28 |
   | RF | **0.7898** | 0.7869 | 6 | todo | 0.7898 | 2.29 |
   | LogReg | 0.7811 | **0.7845** | 42 | 36 m | 0.7871 | 2.24 |

3. **Conjunto**: 6 variables gana en los cuatro modelos de árboles (+0.003 a +0.005); la logística prefiere
   las 42 (+0.003), igual que en `03`.
4. **Ventanas** (hiperparámetros fijos): en árboles, toda la historia ≥ 48 m ≥ 36 m > 24 m, con diferencias
   ≤ 0.003; la logística mejora recortando a 36 m (0.7871 vs 0.7845 con todo).
5. **Ensembles** (promedio de probabilidades OOF, miembros en orden de AUC): los cinco juntos dan el AUC más
   alto (0.7920), pero empatan (≤ 0.001) con `lightgbm+xgboost` (0.7913), `lightgbm+xgboost+catboost`
   (0.7912), los 4 árboles (0.7911), LightGBM solo (0.7911) y XGBoost solo (0.7910). El desempate por lift lo
   gana **`lightgbm+xgboost+catboost`** (2.34; los demás ≤ 2.32).

**Modelo final: ensemble `lightgbm+xgboost+catboost`**, 6 variables, ventana expansiva, pesos de clase
tuneados (w ≈ 1.05–1.21). AUC medio 0.7912 ± 0.009, lift top-10 % mensual 2.34. Spec completa (variables,
hiperparámetros, ventana) en `04_modelado/modelo_final.json`.

Lecturas:

- Los cinco algoritmos quedan en 0.787–0.791: el techo sigue estando en la información, no en el algoritmo
  (consistente con `sem1`). La diferencia entre el final y LightGBM solo (0.0001 de AUC, +0.03 de lift)
  está dentro del ruido entre bloques (std ≈ 0.009); el ensemble se elige por el criterio declarado, no
  porque sea claramente mejor. Con una tolerancia de lift de 0.04 en vez de 0.02 habría ganado LightGBM solo.
- Varios estudios tienen su mejor trial cerca del final (#90–#99): 100 trials no satura la búsqueda, pero
  la ganancia entre el trial 30 y el 100 es ≤ 0.0009 de AUC en todos los estudios (`trials/*.csv`; gráfico
  de convergencia en el notebook). Más presupuesto no cambiaría la decisión.
- El peso de clase óptimo queda cerca de 1 en los tres miembros: con AUC como objetivo, ponderar aporta poco;
  la ganancia del paso 1 (+0.0005) es marginal.

Detalle: `reports/modelado.md`.

## 05_evaluacion (§4.3 GroupKFold + OOT)

Configuración fija desde `modelo_final.json`; aquí no se decide nada. Corrida del 2026-10-01.

**Verificación complementaria** (pool de desarrollo, 5 folds, ensemble completo re-entrenado por fold):

| validación | AUC | std | lift decil |
|---|---:|---:|---:|
| temporal 4 bloques (selección, `04`) | 0.7912 | 0.009 | 2.34 |
| GroupKFold(5) por `id_vendedor` | 0.7342 | 0.004 | 2.08 |
| StratifiedKFold(5) sin grupos | 0.7340 | 0.007 | 2.08 |

GroupKFold − StratifiedKFold = +0.0002: separar vendedoras no cambia nada, el modelo no memoriza
individuos. El nivel más bajo (0.73) no es comparable con el temporal: el K-fold aleatorio valida sobre
toda la historia (2016–2025), incluidos los años 2018–2020 de prevalencia 0.35–0.37 y dinámica distinta;
sirve para la comparación con/sin grupos, no como estimación de despliegue.

**Test OOT, una sola evaluación** (dic-2025 → mar-2026, 879 filas, prevalencia 0.276; entrenamiento con
todo el pool hasta may-2025):

| métrica | OOT |
|---|---:|
| AUC-ROC | **0.7926** |
| PR-AUC | 0.568 |
| Lift decil superior (media mensual) | 2.27 |
| AUC por mes | 0.790 / 0.782 / 0.791 / 0.814 |
| AUC por miembro (lightgbm / xgboost / catboost) | 0.793 / 0.794 / 0.790 |

Por % de base contactada: top-5 % precisión 0.66, recall 0.12, lift 2.38; top-10 % 0.63 / 0.23 / 2.26;
top-20 % 0.59 / 0.43 / 2.14; top-30 % 0.55 / 0.59 / 1.97. Matriz a p ≥ 0.5: TP 130, FP 98, FN 113,
TN 538 (precisión 0.57, recall 0.54, 26 % contactados).

Lecturas:

- El AUC OOT (0.7926) queda al nivel del AUC de selección (0.7912): en este período no se observa el
  sesgo optimista que esperábamos. Los 4 meses del OOT son homogéneos (0.78–0.81).
- `sem1` reportó 0.7636 en su OOT (oct-2025 → ene-2026, 885 filas). No es el mismo período ni el mismo
  corte del warehouse (`sem2` descarta el mes parcial, ver `00_datos`), así que no es una comparación
  directa; sí indica que el período de prueba pesa más que el modelo.
- Los tres miembros rinden igual en el OOT (0.790–0.794) y el ensemble no los supera: confirma que la
  ganancia del ensemble está dentro del ruido, como ya se leía en `04`.
- Modelo guardado en `models/ensemble_final.joblib` (no versionado); métricas en
  `05_evaluacion/oot_metricas.json`.

Detalle: `reports/evaluacion.md`.

## Infraestructura

- `src/datos.py`: carga, preprocess (imputación + 6 derivadas), cliente BigQuery con cuenta gmail.
- `src/particion.py`: `oot_split`, `temporal_folds`, `describe`.
- `src/evaluacion.py`: `evaluar(make_model, df, feats, folds, ventana_meses)` → AUC por bloque,
  media, std, lift top-10 % mensual, predicciones OOF; `metricas_oof(df, oof, folds)` para ensembles
  por promedio de OOF. Reutilizable en `05`.
- `src/modelos.py`: `build(algo, params)` para los 5 algoritmos; `EnsemblePromedio` (ventana por
  miembro, `proba_miembros`) y `desde_spec(modelo_final.json)`. Lo usan `04`, `05` y `06`.
- `04_modelado/tuning.py`: espacios de búsqueda y `tune(...)` con SQLite.
- Notebooks escritos directo (sin builders); el kernel corre desde la carpeta del notebook.
- `data/`, `models/`, `optuna.db` y `trials/` no se versionan.

## Próximo paso

`06_interpretacion` (§4.4): TreeSHAP sobre cada miembro del ensemble (`models/ensemble_final.joblib`) y
promedio de los valores SHAP; ranking global por |SHAP| medio y gráfico de resumen; contraste con la
importancia por permutación de `03`; gráficos de dependencia de las variables más influyentes; casos
locales de vendedoras del decil superior del OOT (sin mostrar `id_vendedor`).
