# TabFM (BigQuery `AI.PREDICT`) vs XGBoost tuneado

> Generado por `05_modelling/10_tabfm.py` el 2026-09-07 08:48.
> Test OOT: 885 filas, prevalencia 0.2780. Un solo pase, sin tuning y sin GroupKFold.

## Qué es TabFM

**TabFM** es el modelo fundacional pre-entrenado de Google Research para datos tabulares. No se entrena: recibe la tabla de train como ejemplos *in-context* y predice en un solo forward pass. En Google Cloud vive **dentro de BigQuery**, vía la función SQL `AI.PREDICT` (estado: Preview). No genera modelo persistido ni endpoint desplegado.

- Doc: <https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-ai-predict>
- Paper/blog: <https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/>

No confundir con **TimesFM** (series de tiempo) ni con AutoML Tables / BQML `BOOSTED_TREE_CLASSIFIER`, que sí entrenan un modelo propio.

## Limitación que condiciona el experimento

`AI.PREDICT` acepta **como máximo 20 columnas de features**. Con las 110 del pipeline la consulta falla en duro:

```
400 The number of features 110 exceeds the maximum allowed number of features 20.
```

Para superar ese techo hay que escribir a `bqml-feedback@google.com` (no es un flag). Así que TabFM corre sobre las **20 features top por ganancia de XGBoost ajustado solo en train**, y se agrega una fila de XGBoost sobre esas mismas 20 para separar *efecto del modelo* de *efecto del recorte de features*.

Features usadas:

```
 1. monto_u12
 2. n_prod_u12
 3. monto_u6
 4. n_ped_u6
 5. unidades_u12
 6. n_ped_u12
 7. n_ped_u3
 8. monto_acum
 9. monto_u3
10. n_ped_acum
11. camp_part_u12
12. camp_saltadas
13. n_prod_acum
14. compras_hist
15. unidades_u3
16. intensidad_u3
17. tasa_camp_u3
18. tiene_lider
19. meses_activos_u12
20. tasa_camp_u6
```

## Resultados (bloque OOT, 885 filas)

| modelo                      |   n_feats |   oot_AUC |   oot_AUCstd |   oot_PRAUC |   oot_liftPR |   oot_lift10 |   oot_prec |   oot_rec |   oot_F1 |   segundos |
|:----------------------------|----------:|----------:|-------------:|------------:|-------------:|-------------:|-----------:|----------:|---------:|-----------:|
| XGBoost tuneado (110 feats) |       110 |    0.7638 |       0.0202 |      0.536  |       1.9284 |       2.1667 |     0.4397 |    0.7561 |   0.5561 |     1.6762 |
| XGBoost tuneado (20 feats)  |        20 |    0.7684 |       0.0267 |      0.5448 |       1.96   |       2.2076 |     0.4577 |    0.748  |   0.5679 |     0.819  |
| TabFM zero-shot (20 feats)  |        20 |    0.7627 |       0.017  |      0.5257 |       1.8912 |       2.3302 |     0.5521 |    0.4309 |   0.484  |    85.6444 |

`oot_AUCstd` = std del AUC por mes (solo meses con ambas clases). `oot_liftPR` = PR-AUC / prevalencia. `oot_lift10` = lift del decil top. `prec`/`rec`/`F1` al umbral 0.5. `segundos` = fit+predict para XGBoost, latencia de la consulta `AI.PREDICT` para TabFM.

**Ojo con prec/rec/F1 al 0.5**: no son comparables entre filas. XGBoost corre con `scale_pos_weight` (balanceado), o sea que sus probabilidades están infladas hacia la clase positiva y el 0.5 le queda como un umbral agresivo (recall alto, precisión baja). TabFM sale calibrado a la prevalencia real, así que el mismo 0.5 le queda conservador (precisión alta, recall bajo). Las filas se comparan de verdad por las métricas de ordenamiento — AUC, PR-AUC y lift del decil — que son invariantes al umbral; el punto operativo de cada modelo se elige después.

## Costo y tiempo

- Consulta `AI.PREDICT`: **85.6 s**, `total_bytes_billed` = 20,971,520 bytes → **~USD 0.0001** a precio on-demand (6.25 USD/TiB).
- Durante el Preview TabFM se factura como una consulta BigQuery normal (bytes procesados u on-demand slots). **Desde el 30/10/2026 pasa a precio por tokens**: `(filas_train × cols + filas_pred × (cols-1)) × n_ensembles`, sobre el costo normal de BigQuery. Un GroupKFold de 5 folds multiplicaría eso por 5.
- Sin endpoints ni instancias: costo recurrente **cero**. Las tablas temporales se borran al final del script.

## Limitaciones

- **20 features** máximo (el bloqueo real para este dataset).
- Máximo 10 clases en clasificación (irrelevante acá, es binario).
- El label debe ser `BOOL` o `STRING`; con `INT64` hace regresión en silencio.
- Estado **Preview**: sin SLA, la API y el precio pueden cambiar.
- No hay control de semilla ni de `n_ensembles` → las probabilidades no son bit-a-bit reproducibles entre corridas.
- No devuelve importancias ni nada explicable: para la tesis no reemplaza el análisis SHAP/permutación que ya tenemos sobre XGBoost.

## Lectura

TabFM **empata** con el XGBoost tuneado (0.7627 vs 0.7638, -0.0011 de AUC): la diferencia está dentro del ruido de un bloque de 885 filas. Contra el XGBoost con las mismas 20 features (0.7684) la brecha es de -0.0057, que es la comparación limpia modelo-contra-modelo.

**Para la tesis**: sirve como punto de referencia externo — 'un modelo fundacional sin entrenar ni tunear llega hasta acá' — y como evidencia de que el techo de ~0.76 de AUC es del problema y de los datos, no del algoritmo. No sirve como modelo principal: el tope de 20 features obliga a tirar 90 variables construidas a mano, que es justamente el trabajo que la tesis documenta.

**Para producción**: no. El límite de 20 features, el estado Preview sin SLA, la falta de explicabilidad y el cambio a precio por tokens del 30/10/2026 lo hacen peor apuesta que un XGBoost que ya está entrenado, es reproducible y cuesta cero por inferencia. El cuello de botella del proyecto sigue sin ser el algoritmo: es que el uplift de la acción de retención no está medido.
