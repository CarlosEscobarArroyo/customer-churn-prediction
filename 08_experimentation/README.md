# 08_experimentation — nuevas líneas de investigación (2026-09)

Cuatro líneas evaluadas contra el modelo final (XGBoost tuneado, 91 features tras
preprocessing) con los dos protocolos del repo: **GroupKFold(5) por vendedora** (30 821
filas) y **OOT** (train ≤ rank 101, gap 6, test = ranks 108–111, 885 filas). Cada notebook
deja su reporte con números reales en `reports/opcion_X.md` (+ `.csv`).

| Notebook | Pregunta | Reporte | Resultado |
|---|---|---|---|
| `01_opcion_A_lstm_xgboost.ipynb` | ¿Una LSTM sobre los 24 meses del panel denso aporta al XGBoost (stacking, embedding auxiliar, embedding supervisado, meses aplanados)? | `reports/opcion_A_lstm.md` | **Marginal.** LSTM sola 0.745 GKF / 0.767 OOT (6 canales ≈ las 91 features). Stacking p(LSTM) en XGBoost: +0.0017 GKF (IC 0.0008–0.0026) y +0.0018 OOT; embedding auxiliar +0.0005 (IC incluye 0); 24 meses aplanados +0.0002. Spearman base↔LSTM 0.96, top 10 % coincide 71–76 %: redundante. |
| `02_opcion_B_ritmo_pausas_campanas.ipynb` | ¿Ritmo individual, pausas/reactivaciones, rachas y secuencia de campañas agregan sobre las ventanas fijas? | `reports/opcion_B_ritmo.md` | **No aporta.** 29 features `r_*`: base+ritmo +0.0004 GKF (IC −0.0001–0.0010), +0.0017 OOT. `ritmo_solo` 0.727 GKF / 0.765 OOT. Redundancia alta (rho de Spearman máx. 0.68–0.92 con las vigentes); `r_p_gap_le6` es la única `r_*` en el top 10 por permutación. Tiempo hasta la próxima compra: ya empatado en 09_supervivencia. |
| `03_opcion_C_google_trends.ipynb` | ¿Google Trends (Perú) anticipa el nivel mensual de churn o mejora la selección individual? | `reports/opcion_C_google_trends.md` | **No aporta selección individual.** 20 términos, 60 features: +0.0047 OOT (IC −0.0009–0.0106); rolling 12 meses +0.0002 agrupado / +0.0023 dentro del mes. Techo del oráculo mensual +0.0067 agrupado. Pronóstico de la tasa mensual con Trends empeora vs estacionalidad (RMSE 0.066 vs 0.040). Series de calidad desigual (`temu` 72 % ceros; salto 2022 de 0.59× a 1.57×). |
| `04_opcion_D_segmentos.ipynb` | ¿Segmentos de vendedoras (negocio + KMeans) requieren modelos aparte, segmento explícito o fine-tune? | `reports/opcion_D_segmentos.md` | **No: el global gana en todo.** 9 segmentaciones × 4 enfoques. Modelos especializados pierden entre −0.002 (tipo) y −0.085 (nivel de compra) de AUC GKF; el fine-tune reduce la pérdida pero no la anula; el segmento explícito es neutro (±0.0003). Única excepción: flag `red` (Asesora con/sin líder) +0.0024 GKF, +0.0005 OOT (IC cruza 0). Segmentos peor servidos: frecuencia 1–3 meses (0.667), monto bajo (0.622), reactivadas 4+ (0.659); ahí falta información, no modelo. |

## Cómo correr

```bash
# 1) datos auxiliares (una vez; requieren BigQuery con la cuenta con acceso a glamour-peru-dw)
uv run python 08_experimentation/extraer_panel_denso.py     # data/processed/panel_denso_mensual.csv + dim_vendedor.csv
# 2) notebooks, en orden (B usa salidas de A; D usa salidas de B si existen)
uv run python 08_experimentation/build_nb.py                # regenera los .ipynb desde nb_src/*.py
for nb in 01 02 03 04; do
  uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 08_experimentation/${nb}_*.ipynb
done
```

- Los notebooks se generan desde `nb_src/*.py` (formato percent); editar el `.py` y regenerar.
- `exp_utils.py`: carga del baseline (réplica de 02→04 vía `05_modelling/experimentos/pipeline.py`),
  cubo de secuencias del panel denso, métricas extendidas (AUC, PR-AUC, precisión/cobertura
  top 10 % y 30 %, std mensual, Brier, ECE), bootstrap pareado del ΔAUC y la LSTM (torch).
- Tiempos en un portátil de 8 núcleos: A ≈ 30 min (LSTM en CPU), B ≈ 5 min, C ≈ 5 min
  (+ descarga de Trends, con cache en `data/external/`), D ≈ 10 min.
- macOS: torch y xgboost traen cada uno su `libomp` y se bloquean si ambos paralelizan;
  `exp_utils._torch()` importa xgboost primero y deja torch en 1 hilo. En Linux/GPU se
  puede subir `torch.set_num_threads`.

## Criterio de lectura común

Una variante "aporta" solo si el ΔAUC GroupKFold tiene IC bootstrap que excluye 0 **y** el
ΔAUC OOT tiene el mismo signo. El OOT (885 filas) tiene SE ≈ 0.02 y no discrimina
diferencias < 0.02 por sí solo; el GroupKFold sí (SE ≈ 0.003).

## Conclusión (2026-09-15)

| Línea | Veredicto | ΔAUC GKF (IC 95 %) | ΔAUC OOT |
|---|---|---|---|
| A. Stacking p(LSTM) en XGBoost | marginal, consistente | +0.0017 (0.0008 a 0.0026) | +0.0018 |
| A. Embedding auxiliar / 24 meses aplanados | nada | +0.0005 / +0.0002 (IC incluye 0) | −0.003 / +0.003 |
| B. Ritmo, pausas, rachas, campañas | nada | +0.0004 (−0.0001 a 0.0010) | +0.0017 |
| C. Google Trends | nada individual; nivel mensual ≤ oráculo (+0.007) | no aplica (GKF optimista) | +0.0047 (−0.0009 a 0.0106) |
| D. Modelos por segmento | peor que el global | −0.002 a −0.085 | −0.001 a −0.075 |

- **Continuar**: nada como cambio del modelo final. El stacking LSTM es la única mejora con IC
  que excluye 0 y signo coherente en OOT, pero vale 0.002 de AUC y agrega una red y un
  panel denso al pipeline: no compensa.
- **Combinar**: no. Las tres fuentes nuevas (secuencia aprendida, ritmo a mano, Trends) son
  redundantes con las 91 features (Spearman 0.96 entre p(base) y p(LSTM); rho 0.7–0.9 entre
  `r_*` y las vigentes; el techo de cualquier variable mensual es +0.007).
- **Descartar**: modelos por segmento (menos datos por modelo y nada que el global no aprenda
  de las mismas columnas) y Trends como feature del modelo (ruido de escala, cambio de
  metodología 2022, pronóstico mensual peor que la estacionalidad).
- **Lo que sí dejó D**: el mapa de dónde el modelo es débil (vendedoras de baja frecuencia,
  bajo monto o recién reactivadas, AUC 0.62–0.67). Es información nueva sobre esas vendedoras
  lo que falta, no un algoritmo distinto.
