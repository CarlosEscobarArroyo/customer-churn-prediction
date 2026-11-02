# Figuras estandarizadas — churn Glamour

Todas las gráficas del pipeline (01→06) regeneradas con **un solo color base:
azul profesional `#4c72b0`**. Series múltiples usan degradados de la misma
familia azul; heatmaps/SHAP usan un colormap azul secuencial.

- Estilo compartido: `_estilo.py` (impórtalo si regeneras alguna:
  `from _estilo import AZUL, aplicar_estilo, paleta, CMAP, guardar`).
- **40 de 48 figuras** son azul monocromo sobre fondo blanco.
- **8 figuras** (diagnósticos de Optuna, prefijos `05b/05c/05d/05e`) se
  **extrajeron tal cual de los notebooks** (fondo gris, línea roja de "Best
  Value"): regenerarlas en azul requeriría re-correr las 4 búsquedas de 60
  trials. Ver nota al final.

## Índice

### 01 · Entendimiento de datos — tablas crudas (`01a_fact_eda__`)
1. `01_volumen_por_tabla` — nº de filas por tabla (escala log).
2. `02_edad_e_ingreso_vendedoras` — histograma de edad + ingresos por año.
3. `03_campanas_por_anio` — nº de campañas por año (cadencia).
4. `04_productos_por_categoria` — top 15 categorías de producto.
5. `05_ubicaciones_por_departamento` — top 15 departamentos.
6. `06_monto_y_pedidos_por_vendedora` — histograma de monto (p99) y pedidos/vendedora.
7. `07_pedidos_por_mes` — serie temporal de pedidos por mes.
8. `08_reconciliacion_cabecera_detalle` — diferencia relativa cabecera vs detalle.

### 01 · Entendimiento de datos — dataset de churn (`01b_churn_eda__`)
1. `01_balance_target` — balance no-churn vs churn (prevalencia 30.9%).
2. `02_estabilidad_temporal` — volumen mensual + churn-rate por mes (drift COVID).
3. `03_filas_por_vendedora` — meses observados por vendedora (eje-y log).
4. `04_nulos_por_columna` — % de nulos por columna.
5. `05_distribuciones_numericas` — grid de histogramas de features (p99).
6. `06_churn_rate_categoricas` — churn-rate por categoría vs prevalencia.
7. `07_auc_univariado` — AUC univariado orientado (dirección +/−).
8. `08_correlacion` — matriz de correlación de features numéricas.

### 01 · Horizonte de churn (`01c_horizonte__`)
1. `01_dist_gaps_cdf` — distribución de gaps entre compras + CDF (ref. k=6).
2. `02_tasa_churn_vs_k` — tasa de churn proxy según horizonte k (1–12 meses).
3. `03_hazard_retorno` — hazard de retorno por mes de silencio.

### 02 · Split temporal (`02_split__`)
1. `01_particion_oot_por_mes` — partición out-of-period (train/gap/test) por mes.

### 04 · Feature engineering (`04_feateng__`)
1. `01_importancia_permutacion_top20` — top 20 importancia por permutación (RF).

### 05 · Modelado — baseline (`05a_baseline__`)
1. `01_auc_lift` — AUC (GroupKFold vs OOT) y lift del decil top por modelo.

### 05 · Modelado — tuning Optuna (`05b/05c/05d/05e_tuning_*__`) ⚠ estilo original
Para XGBoost (`05b`), Random Forest (`05c`), LogReg (`05d`) y CatBoost (`05e`):
1. `01_historia_optimizacion` — evolución del AUC OOF por trial.
2. `02_importancia_hiperparams` — importancia relativa de cada hiperparámetro.

### 06 · Evaluación — importancia de features (`06a_featimp__`)
1. `01_importancia_nativa` — top 15 por modelo (gain / impureza / |coef|).
2. `02_importancia_permutacion` — top 15 caída de AUC por permutación (OOT).

### 06 · Evaluación — SHAP (`06b_shap__`)
1. `01_beeswarm_xgboost` · 2. `02_beeswarm_rf` · 3. `03_beeswarm_logreg` — beeswarm por modelo.
4. `04_barh_shap_relativo` — |SHAP| promedio relativo (top 15) por modelo.
5. `05_dependence_compras_hist` — dependence de la feature más influyente.
6–8. `06/07/08_waterfall_alto_*` — caso de mayor riesgo por modelo.
9–11. `09/10/11_waterfall_bajo_*` — caso de menor riesgo por modelo.

### 06 · Evaluación — utilidad de negocio (`06c_utilidad__`)
1. `01_calibracion` — reliability diagram + Brier.
2. `02_roc_pr` — curvas ROC y Precision-Recall.
3. `03_ganancias_lift` — ganancias acumuladas y lift por % contactado.
4. `04_matriz_confusion` — matriz de confusión a t=0.5 por modelo.
5. `05_costo_beneficio` — valor neto esperado vs % contactado.

## Nota sobre los diagnósticos de Optuna (8 figuras)

Los notebooks de tuning no persisten el objeto `study` de Optuna, así que
regenerar sus 2 diagnósticos en azul obligaría a re-ejecutar cada búsqueda
(60 trials × 5 folds). Para no gastar ese cómputo se reusaron las imágenes ya
renderizadas en los notebooks (datos reales, estilo original de Optuna). Si se
quiere uniformidad total en azul, basta re-correr las 4 búsquedas (~20–30 min)
y volver a graficar `study.trials_dataframe()` y
`optuna.importance.get_param_importances(study)` con el tema de `_estilo.py`.
