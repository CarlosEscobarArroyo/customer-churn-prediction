# Predicción de churn — Glamour

El flujo vigente se estudia y ejecuta desde notebooks, con Python 3.12 y las dependencias de `pyproject.toml` / `uv.lock`. Los scripts sirven de soporte para validaciones, persistencia y ejecución en segundo plano. La auditoría está en `07_results/auditoria_estado_2026-09-19.md`.

## Archivos para leer, en orden

1. [Verificación de la reparación](03_preprocessing/02_verificacion_reparacion.ipynb): entradas, tipos, reconstrucción de variables y compatibilidad con los modelos previos. Ejecutado y con resultados guardados.
2. [Optuna con validación temporal](05_modelling/06_optuna_validacion_temporal.ipynb): notebook principal. Contiene las funciones de partición, selección, espacios de búsqueda, objetivo, exportación y el bucle de entrenamiento, además de tablas y curvas de seguimiento. Por defecto consulta los estudios activos; su sección 8 permite entrenar o reanudar en Jupyter.
3. `00_dataset_construction/qry_churn.sql`: origen de una fila por vendedora/mes. Churn significa ausencia de compras durante los seis meses posteriores. Población: compró en el mes y tiene compra en al menos un mes previo.
4. `reports/optuna_temporal_v1/protocol.json` y `folds.json`: configuración y fechas efectivas; generados y no versionados.

`scripts/temporal_optuna.py` es el backend validado para ejecución desatendida y las dos transformaciones importables que permiten recargar pipelines. `tests/test_temporal_optuna.py` verifica fechas e integridad. `scripts/build_temporal_notebooks.py` genera las celdas a partir del código validado; regenerar elimina salidas, por lo que debe volver a ejecutarse el notebook después. Los notebooks anteriores se conservan como referencia y no son la entrada de Optuna temporal.

Para pasar del proceso de segundo plano al notebook, solicitar parada mediante `STOP`, esperar al final del trial, retirar `STOP` y activar `EJECUTAR_ENTRENAMIENTO=True`. El bloqueo evita dos entrenamientos simultáneos en el mismo estudio; el historial se conserva.

## Entradas y reparación

El entrenamiento nuevo lee `data/processed/churn_dataset.csv`. Excluye identificadores, objetivo y datos maestros (`sexo`, `edad`, `antiguedad_meses`, `tipo_vendedor`, `departamento`, `provincia`) para limitarse a señales transaccionales históricas. La comparación con los modelos anteriores incluye, por tanto, un cambio de variables además de validación.

La reparación del CSV antiguo utiliza `churn_dataset_processed.csv`, `selected_features.json` y `oot_split.csv`. Reconstruye las seis variables derivadas y mantiene las 68 columnas y su orden para conservar compatibilidad con los modelos existentes. Verifica primero que el preprocesado coincide con su reconstrucción desde el original. Si encuentra diferencias, respalda el CSV anterior y registra hashes y posiciones modificadas en `reports/data_repair/`.

```bash
.venv/bin/python -m scripts.temporal_optuna repair
.venv/bin/python -m pytest tests/test_temporal_optuna.py -q
```

## Validación temporal

Se reserva del tuning tanto el OOT histórico (agosto–noviembre de 2025) como el gap de febrero–julio de 2025. El pool de desarrollo termina en enero de 2025. El OOT ya fue consultado en experimentos anteriores: no constituye una prueba final intacta y el nuevo script no lo evalúa.

Cuatro bloques de validación de cuatro meses cubren octubre de 2023 a enero de 2025. Para cada bloque, entrenamiento expansivo con seis meses completos de separación. El objetivo es la **media no ponderada del ROC AUC de los cuatro bloques**, con average precision, lift del decil superior, recall a 0,5 y AUC mensual como diagnósticos. Las etiquetas de bloques de validación próximos pueden compartir meses futuros; no son cuatro experimentos independientes y su desviación no es un intervalo de confianza.

Cada entrenamiento contiene otra partición temporal con el mismo gap para seleccionar variables mediante importancia por permutación, explícitamente `scoring='roc_auc'`. Solo se retienen importancias positivas; esto es una heurística, no una prueba de significancia. Optuna compara esa selección con usar todas las variables transaccionales. El selector fijo se calcula una vez por fold y se reutiliza entre trials. Ninguna etiqueta de la validación exterior participa en la selección. Escalado y pesos de clase se ajustan en cada entrenamiento.

La configuración ganadora sigue seleccionada sobre validación. Sus resultados no sustituyen una evaluación final sobre datos nuevos no utilizados para tomar decisiones.

## Ejecutar, observar y reanudar

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -u -m scripts.temporal_optuna run \
  --trials 100 --threads 4 --output reports/optuna_temporal_v1
```

Los modelos se alternan, un trial por turno: regresión logística, Random Forest, XGBoost y CatBoost. Presupuesto: 100 trials por modelo, contando completados, podados y fallidos; no son 100 adicionales al reanudar. Un único trial se ejecuta a la vez, con hasta cuatro hilos configurados por estimador. La regresión logística no aprovecha necesariamente todos esos hilos. No hay un límite de duración global: los tiempos dependen de las configuraciones probadas.

- `studies.sqlite3`: historial completo de Optuna; se reutiliza al ejecutar el mismo comando.
- `*_trials.csv`: tabla exportada después de cada trial.
- `*_best.json`: parámetros, métricas, variables y hash del mejor candidato exportado.
- `*_pipeline.joblib`: transformación, columnas elegidas y estimador, entrenados hasta enero de 2025. Se actualiza cuando mejora el AUC. No son modelos aprobados para producción.
- `protocol.json`: hashes de datos/código, versiones de librerías y protocolo. Se rechaza reanudar si cambian; usar otro directorio para un experimento nuevo.
- `folds.joblib`: cache local de particiones y selección.
- `process.json`: PID, inicio y presupuesto solicitado.

Para detener limpiamente después del trial actual: `touch reports/optuna_temporal_v1/STOP`. Para reanudar, retirar ese archivo y ejecutar el mismo comando. Un bloqueo de archivo impide dos coordinadores simultáneos sobre el mismo directorio. Trials interrumpidos en estado RUNNING se marcan FAIL al reiniciar.

Si se ejecuta en tmux, la sesión se llama `churn-optuna` y el log queda en `reports/optuna_temporal_v1/run.log`:

```bash
tmux attach -t churn-optuna
tail -n 30 reports/optuna_temporal_v1/run.log
.venv/bin/python -m scripts.optuna_status
```

Tmux mantiene el proceso al desconectar el terminal, pero no ante un apagado o suspensión del equipo. SQLite permite reanudar posteriormente. Los pipelines se cargan desde la raíz del repositorio con `joblib.load(...)`, de modo que Python pueda importar `scripts.temporal_optuna`.

## Ventanas recientes y ensembles

El [notebook 07](05_modelling/07_ventanas_recientes_ensemble.ipynb) compara toda la historia con 24, 36 y 48 meses de entrenamiento. Conserva los cuatro bloques, el gap de seis meses, las 42 variables transaccionales y los hiperparámetros ganadores anteriores: es una comparación controlada de ventanas, sin retuning. Luego compara promedios iguales de todas las parejas y de los cuatro algoritmos, usando la mejor ventana de cada uno.

Los checkpoints y tablas se guardan en `reports/recent_windows_v1/`. Reejecutar reutiliza las predicciones de variantes terminadas; un hash del protocolo impide mezclarlas con código o parámetros distintos. El notebook incluye entrenamiento, selección, gráficas y exportación del candidato. `scripts/recent_windows.py` solo aporta el recorte por meses y la clase persistible del ensemble; el experimento está en las celdas del notebook.

Los AUC de este análisis siguen siendo métricas de selección sobre las mismas validaciones, no una evaluación final independiente.
