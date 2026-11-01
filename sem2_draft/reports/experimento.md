# Experimento de campo — cohortes (§4.5)

Generado por `07_experimento/cohorte.py`; una sección por cohorte.

## Cohorte 2026-09

- Entrenamiento: 31,235 filas, observaciones 2016-12 → 2026-03 (etiquetas con pedidos hasta 2026-09); modelo `lightgbm`.
- Elegibles: **251** (excluidas por cohortes previas: 0). Tratamiento **126**, control **125**; semilla 42 + mes_rank.
- Seleccionadas (30%): tratamiento **38** (mayor probabilidad), control **38** (al azar).
- Probabilidad media de abandono: 0.316 (elegibles).

| grupo | seleccionada | n | p media | p mín | p máx |
|---|---|---:|---:|---:|---:|
| control | no | 87 | 0.301 | 0.028 | 0.706 |
| control | sí | 38 | 0.365 | 0.031 | 0.706 |
| tratamiento | no | 88 | 0.205 | 0.028 | 0.449 |
| tratamiento | sí | 38 | 0.559 | 0.452 | 0.701 |

