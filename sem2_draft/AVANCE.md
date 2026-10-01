# Avance de sem2 — 2026-10-01

Bitácora de qué se hizo, qué dio y qué falta, por sección de la metodología (`metodologia.md`).
Los números salen de los reportes generados (`reports/*.md`, `02_particion/particion.json`,
`03_variables/variables_seleccionadas.json`); no se escriben a mano.

## Estado por sección

| Sección | Qué pide | Estado | Dónde |
|---|---|---|---|
| 4.1 | Re-extraer del DW; misma definición de churn, población y multi-slice mensual | ✅ | `00_datos/` |
| 4.1 | Imputación a cero; estandarización solo para logística, ajustada en train | ✅ | `src/datos.py`; `StandardScaler` dentro del pipeline |
| 4.1 | Desbalance: pesos de clase vs sin ponderar vs submuestreo vs SMOTE | ⏳ | `04_modelado/` |
| 4.2 | Solo transaccional LRFM + acumuladas + cambio; sin datos maestros | ✅ | `00_datos/qry_churn.sql` |
| 4.2 | Permutación + ablación forward | ✅ | `03_variables/` |
| 4.3 | OOT 4 meses + brecha 6 m + 4 bloques de validación expansiva | ✅ | `02_particion/` |
| 4.3 | LogReg, RF, XGBoost, LightGBM, CatBoost con Optuna, mismo presupuesto | ⏳ | `04_modelado/` |
| 4.3 | Ventanas de entrenamiento y ensembles | ⏳ | `04_modelado/` |
| 4.3 | GroupKFold como verificación complementaria | ⏳ | `05_evaluacion/` |
| 4.3 | OOT una sola vez: AUC, PR-AUC, ROC, matriz, precisión/recall/lift por % contactado | ⏳ | `05_evaluacion/` |
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

## Infraestructura

- `src/datos.py`: carga, preprocess (imputación + 6 derivadas), cliente BigQuery con cuenta gmail.
- `src/particion.py`: `oot_split`, `temporal_folds`, `describe`.
- `src/evaluacion.py`: `evaluar(make_model, df, feats, folds, ventana_meses)` → AUC por bloque,
  media, std, lift top-10 % mensual, predicciones OOF. Reutilizable en `04` y `05`.
- Notebooks escritos directo (sin builders); el kernel corre desde la carpeta del notebook.
- `data/` no se versiona.

## Próximo paso

`04_modelado`: desbalance (4 estrategias) → Optuna para 5 algoritmos × {6, 42} variables →
ventanas 24/36/48/todo → ensembles por promedio. Presupuesto de trials por definir (50 o 100).
