# Horizonte de churn — análisis empírico (v5)

> Generado automáticamente por `notebooks/clean/horizonte_churn_v5.ipynb` el 2026-05-05 21:04.

## Resultado

**k recomendado: 6 meses** (umbral hazard < 8%).
**k vigente en v5: 6 meses**.
**Coincide: sí**.

## Datos

- Eventos analizados: 24,885 (post-filtro `cum_purchases >= 3`).
- Vendedoras únicas: 4,288.
- Sub-muestra fully-observable (≥12 meses futuros): 22,642.
- Blacklist de campañas: (20102, 20201, 23105).

## Distribución de gaps cerrados

- Mediana: 2 meses.
- P75: 3 meses.
- P90: 9 meses.
- % de gaps ≤ 6: 86.4%.
- % de gaps ≥ 6: 16.1%.

## Curva de churn y hazard

|   k | pct_silent   | hazard   |
|----:|:-------------|:---------|
|   1 | 58.63%       | 41.37%   |
|   2 | 43.86%       | 25.19%   |
|   3 | 36.82%       | 16.05%   |
|   4 | 32.64%       | 11.36%   |
|   5 | 29.78%       | 8.76%    |
|   6 | 27.61%       | 7.3%     |
|   7 | 25.99%       | 5.86%    |
|   8 | 24.49%       | 5.76%    |
|   9 | 23.35%       | 4.65%    |
|  10 | 22.54%       | 3.5%     |
|  11 | 21.65%       | 3.92%    |
|  12 | 20.9%        | 3.47%    |

## Justificación

- ~86% de las vendedoras que vuelven lo hacen en ≤ 6 meses.
- El hazard cruza el 8% en k = 6 meses.
- A partir de ahí, esperar un mes más reduce el hazard en menos de 2 puntos.
- `k=6` balancea especificidad (no marcar prematuramente) y anticipación
  (señal de retención antes de que sea irrecuperable).
