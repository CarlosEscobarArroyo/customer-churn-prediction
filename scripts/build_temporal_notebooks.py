"""Build the readable, executable notebooks for the validated temporal experiment."""

import ast
from pathlib import Path
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/temporal_optuna.py").read_text()
TREE = ast.parse(SOURCE)


def definitions(*names):
    return "\n\n".join(ast.get_source_segment(SOURCE, node) for node in TREE.body
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names)


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


SETUP = '''
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if not (ROOT / "pyproject.toml").exists():
    raise FileNotFoundError("Abrir este notebook dentro del repositorio")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROC = ROOT / "data/processed"
'''


def save(name, cells):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.language_info = {"name": "python", "version": "3.12.3"}
    nbf.write(nb, ROOT / name)


audit = [
    md('''
    # 02 · Verificación de la reparación de datos

    **Objetivo:** comprobar que el dataset de 68 variables quedó reparado y conserva
    compatibilidad con los modelos anteriores. Este notebook es de lectura y no sobrescribe datos.

    La reparación encontró el texto `Directo` en `monto_ult_vs_media`, fila 919
    (posición desde cero). El valor correcto se reconstruyó desde el preprocesado;
    se conservó un respaldo y un manifiesto con hashes en `reports/data_repair/`.

    Orden de lectura: entradas → tipos → variables derivadas → comparación → métricas previas.
    El nuevo experimento temporal se estudia en `05_modelling/06_optuna_validacion_temporal.ipynb`.
    '''),
    code(SETUP),
    md('''
    ## 1. Archivos de entrada

    `churn_dataset_processed.csv` es la etapa previa. `selected_features.json` conserva
    el orden de las 68 variables del experimento anterior. La reparación no cambió esa selección.
    '''),
    code('''
    original = pd.read_csv(PROC / "churn_dataset.csv", parse_dates=["mes_obs"])
    previo = pd.read_csv(PROC / "churn_dataset_processed.csv", parse_dates=["mes_obs"])
    reparado = pd.read_csv(PROC / "churn_dataset_features.csv", parse_dates=["mes_obs"])
    seleccion = json.loads((PROC / "selected_features.json").read_text())
    claves = ["id_vendedor", "mes_obs", "mes_rank", "churn"]
    assert original[claves].equals(previo[claves])
    assert original[claves].equals(reparado[claves])
    assert not reparado.duplicated(["id_vendedor", "mes_rank"]).any()
    pd.DataFrame({"etapa": ["original", "preprocesado", "reparado"],
                  "filas": [len(original), len(previo), len(reparado)],
                  "columnas": [original.shape[1], previo.shape[1], reparado.shape[1]]})
    '''),
    md("## 2. Ningún texto, nulo o infinito puede entrar como variable numérica"),
    code('''
    X = reparado[seleccion]
    assert X.select_dtypes(exclude="number").empty
    assert np.isfinite(X.to_numpy(dtype=float)).all()
    pd.Series({"variables": len(seleccion), "nulos": int(X.isna().sum().sum()),
               "columnas_texto": len(X.select_dtypes("object").columns),
               "valor_recuperado_fila_919": reparado.loc[919, "monto_ult_vs_media"]})
    '''),
    md('''
    ## 3. Reconstrucción independiente de las variables derivadas

    Cada razón utiliza información de la misma fila. No se aprende nada del test.
    Si el denominador es cero, se utiliza cero, igual que en la etapa original.
    '''),
    code(definitions("engineer")),
    code('''
    esperado = engineer(previo)[claves + seleccion]
    pd.testing.assert_frame_equal(esperado, reparado, check_dtype=False, rtol=1e-9, atol=1e-9)
    manifiestos = sorted((ROOT / "reports/data_repair").glob("repair_*.json"))
    evidencia = json.loads(manifiestos[-1].read_text())
    print("Reconstrucción coincidente; diferencias corregidas:")
    print(evidencia["changed_positions_zero_based"])
    print("SHA256 del CSV reparado:", evidencia["after_sha256"])
    '''),
    md('''
    ## 4. Compatibilidad con los modelos anteriores

    Se reproduce su inferencia en el OOT **ya consultado**. Esta tabla comprueba la
    reparación; no se utiliza para optimizar el nuevo experimento temporal.
    '''),
    code('''
    import joblib
    from sklearn.metrics import roc_auc_score

    test = reparado.mes_rank >= reparado.mes_rank.max() - 3
    filas = []
    for nombre in ["logreg", "rf", "xgboost", "catboost"]:
        modelo = joblib.load(ROOT / "models" / f"{nombre}_tuned.joblib")
        prob = modelo.predict_proba(reparado.loc[test, seleccion])[:, 1]
        auc = roc_auc_score(reparado.loc[test, "churn"], prob)
        registrado = json.loads((ROOT / "05_modelling" / f"{nombre}_best_params.json").read_text())["oot_auc"]
        assert np.isclose(auc, registrado, atol=1e-12)
        filas.append({"modelo": nombre, "AUC_OOT_reproducido": auc,
                      "diferencia_vs_registrado": auc - registrado})
    pd.DataFrame(filas).set_index("modelo")
    '''),
    md('''
    La reparación está verificada si todas las aserciones pasan. El código que realizó
    la escritura y el respaldo está en `scripts/temporal_optuna.py`, función `repair`.
    No es necesario repetir la reparación para entrenar: el nuevo flujo parte del dataset original.
    ''')]
save("03_preprocessing/02_verificacion_reparacion.ipynb", audit)

imports = '''
import hashlib
import time
import ast
import fcntl
import importlib.metadata
import joblib
import optuna
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Estas dos clases residen en un módulo importable para poder recargar los .joblib
# fuera de Jupyter. El resto de las funciones de entrenamiento se define abajo.
from scripts.temporal_optuna import TransactionFeatures, KeepColumns

ID = ["id_vendedor", "mes_obs", "mes_rank"]
META = ID + ["churn"]
MASTER = ["sexo", "edad", "antiguedad_meses", "tipo_vendedor", "departamento", "provincia"]
ZERO = ["tend_monto_u3_vs_prev3", "tend_nped_u3_vs_prev3", "monto_cv_u12",
        "monto_ult_vs_media", "monto_por_prod_acum"] + [
    f"d_{m}_m{n}" for m in ("monto", "nped") for n in (1, 3, 6, 9, 12)]
MODELS = ["logreg", "rf", "xgboost", "catboost"]

OUT = ROOT / "reports/optuna_temporal_v1"
N_TRIALS = 100                 # total por modelo, incluyendo trials existentes
THREADS = 4
EJECUTAR_ENTRENAMIENTO = False # True para entrenar desde este notebook
'''
cells = [md('''
    # 06 · Optuna con validación temporal

    Este es el notebook principal para **leer el código, examinar datos, entrenar y
    seguir los resultados** del nuevo experimento. Las funciones de partición,
    selección, búsqueda y evaluación aparecen aquí como celdas ejecutables.

    **Estado inicial:** el entrenamiento en segundo plano usa los mismos estudios.
    `EJECUTAR_ENTRENAMIENTO=False` permite ejecutar todo el notebook para revisar el
    trabajo sin lanzar una segunda búsqueda. La celda de entrenamiento incluye un
    bloqueo para impedir dos procesos modificando a la vez los mismos estudios.

    Para continuar aquí: crear `reports/optuna_temporal_v1/STOP`, esperar que termine
    el trial activo, retirar ese archivo y activar `EJECUTAR_ENTRENAMIENTO=True`.
    No se pierden los trials ya completados. Mantener el kernel activo durante la búsqueda.

    No se ha demostrado un techo teórico del modelo. Este experimento mide el
    rendimiento temporal sobre variables transaccionales y no promete una mejora.
    '''),
    md("## 1. Entorno y configuración"), code(SETUP), code(imports),
    md('''
    ## 2. Datos y validación de integridad

    Partimos del original, **no del CSV con 68 variables seleccionadas globalmente**.
    Se excluyen IDs, target y datos maestros para evitar información de snapshots
    históricos no garantizados. Los nulos estructurales conocidos se imputan a cero;
    cualquier texto numérico inválido o infinito produce un error.
    '''),
    code(definitions("digest", "write_json", "check_numeric", "engineer", "load_raw")),
    code('''
    raw = load_raw()
    oot_start = int(raw.mes_rank.max()) - 3
    train = raw[raw.mes_rank <= oot_start - 7].reset_index(drop=True)
    pd.DataFrame({"conjunto": ["dataset original", "pool de desarrollo"],
                  "filas": [len(raw), len(train)],
                  "inicio": [raw.mes_obs.min(), train.mes_obs.min()],
                  "fin": [raw.mes_obs.max(), train.mes_obs.max()]})
    '''),
    md('''
    ### Transformaciones persistibles

    `TransactionFeatures` elimina columnas no transaccionales, imputa únicamente
    nulos estructurales, añade las seis razones y valida tipos. `KeepColumns` conserva
    la selección. Se importan desde el módulo para que los pipelines guardados puedan
    abrirse en otro proceso. Esta celda muestra exactamente sus implementaciones:
    '''),
    code('''
    import inspect
    print(inspect.getsource(TransactionFeatures))
    print(inspect.getsource(KeepColumns))
    '''),
    md('''
    ## 3. Particiones temporales y selección interna

    Validación exterior: cuatro bloques de cuatro meses (octubre de 2023–enero de 2025).
    Cada entrenamiento termina **siete ranks antes del primer mes de validación**:
    así quedan seis meses de gap y las etiquetas de entrenamiento ya están disponibles.

    Selección: dentro de cada entrenamiento se reserva otro bloque temporal de cuatro
    meses con el mismo gap. Un bosque se ajusta en su pasado y se calcula importancia
    por permutación **AUC** en ese bloque interno. Optuna compara esa selección con
    todas las variables. La validación exterior nunca selecciona directamente variables.

    Las importancias positivas son una heurística, no significancia estadística.
    El OOT histórico y su gap no participan en esta búsqueda.
    '''),
    code(definitions("temporal_folds", "select_inside_training", "prepare")),
    code('''
    resumen_folds = []
    for i, (tr, va) in enumerate(temporal_folds(train)):
        resumen_folds.append({"fold": i, "train_hasta": train.iloc[tr].mes_obs.max(),
                              "valid_desde": train.iloc[va].mes_obs.min(),
                              "valid_hasta": train.iloc[va].mes_obs.max(),
                              "n_train": len(tr), "n_valid": len(va),
                              "prevalencia_valid": train.iloc[va].churn.mean()})
    pd.DataFrame(resumen_folds).set_index("fold")
    '''),
    md('''
    ## 4. Espacios de búsqueda y construcción de los modelos

    Optuna elige hiperparámetros, todas/seleccionadas y ponderación de clases sí/no.
    La regresión logística lleva `StandardScaler` dentro de su pipeline: se ajusta
    en cada entrenamiento. Los pesos de clase se calculan usando solo sus etiquetas.
    No se ejecuta ningún entrenamiento al definir estas funciones.
    '''),
    code(definitions("suggest", "build_model")),
    md('''
    ## 5. Función objetivo: lo que Optuna intenta mejorar

    Cada trial entrena y evalúa los cuatro bloques. Se maximiza su AUC medio sin
    ponderar por tamaño. También se guardan AP, lift top 10 %, recall a 0,5 y AUC mensual.
    Después del segundo bloque puede podarse un trial, una vez existan diez trials
    completos de referencia. No se usa el OOT para early stopping ni para elegir parámetros.

    La dispersión entre bloques ayuda a describir estabilidad; no es un intervalo de
    confianza. Las etiquetas de bloques cercanos pueden compartir meses futuros.
    '''),
    code(definitions("objective")),
    md('''
    ## 6. Guardar el mejor candidato

    Cuando mejora el AUC medio, se ajusta un pipeline en todo el pool de desarrollo
    (hasta enero de 2025). Si usa selección, se vuelve a seleccionar dentro de ese
    pool con su partición temporal interna. Se guarda todo lo necesario para aplicar
    las transformaciones a nuevas filas. Este candidato aún necesita evaluación final.
    '''),
    code(definitions("export_best")),
    md('''
    ## 7. Consultar los estudios activos

    Esta sección es de lectura: puedes ejecutarla repetidamente mientras sigue el
    proceso en segundo plano. Los resultados mostrados son una fotografía del momento
    de ejecución. Que una curva suba en validación no demuestra mejora en datos nuevos.
    '''),
    code('''
    db = OUT / "studies.sqlite3"
    studies = {}
    if db.exists():
        storage_url = f"sqlite:///{db}"
        for summary in optuna.get_all_study_summaries(storage_url):
            name = summary.study_name.removeprefix("temporal_v1_")
            studies[name] = optuna.load_study(study_name=summary.study_name, storage=storage_url)
    else:
        print("Todavía no hay estudios. Activar la sección de entrenamiento para crearlos.")

    filas = []
    for name, study in studies.items():
        trials = study.get_trials()
        completos = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        filas.append({"modelo": name, "completos": len(completos),
                      "podados": sum(t.state == optuna.trial.TrialState.PRUNED for t in trials),
                      "fallidos": sum(t.state == optuna.trial.TrialState.FAIL for t in trials),
                      "en_curso": sum(t.state == optuna.trial.TrialState.RUNNING for t in trials),
                      "mejor_AUC_temporal": study.best_value if completos else np.nan})
    display(pd.DataFrame(filas))
    '''),
    code('''
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, study in studies.items():
        completos = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completos:
            ax.plot([t.number for t in completos], np.maximum.accumulate([t.value for t in completos]),
                    marker=".", label=name)
    ax.set(xlabel="Trial", ylabel="Mejor AUC medio temporal hasta ese trial",
           title="Avance de Optuna — validación, no test final")
    if studies:
        ax.legend()
    ax.grid(alpha=0.2)
    plt.show()
    '''),
    code('''
    detalle = []
    for name, study in studies.items():
        if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            continue
        for fold, metricas in enumerate(study.best_trial.user_attrs["fold_metrics"]):
            detalle.append({"modelo": name, "trial": study.best_trial.number, "fold": fold,
                            **{k: v for k, v in metricas.items() if k != "monthly_auc"}})
    display(pd.DataFrame(detalle))
    '''),
    md('''
    ## 8. Entrenar o reanudar desde el notebook

    Esta celda contiene el bucle de entrenamiento real: prepara los folds, abre SQLite,
    alterna un trial por algoritmo y exporta las mejoras. **No invoca `run()` ni un
    comando externo.** Por defecto se omite para poder ejecutar el notebook completo
    sin competir con el proceso activo.

    Si se modifica la metodología, usar otro directorio `OUT`. Para reanudar los estudios
    ya creados, mantener las definiciones originales y guardar el notebook antes de ejecutar.
    Se verifica que las funciones guardadas coincidan con el protocolo del backend validado.
    El bloqueo se libera al acabar o interrumpir el kernel. `STOP` permite salir entre trials.
    '''),
    code('''
    if not EJECUTAR_ENTRENAMIENTO:
        print("Modo consulta. El proceso de segundo plano puede seguir entrenando.")
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        backend = ROOT / "scripts/temporal_optuna.py"
        # Este notebook es una interfaz ejecutable del mismo protocolo validado.
        # Evita mezclar cambios de funciones guardadas con un estudio existente.
        notebook_path = ROOT / "05_modelling/06_optuna_validacion_temporal.ipynb"
        notebook_json = json.loads(notebook_path.read_text())
        nb_source = "\\n".join("".join(c["source"]) for c in notebook_json["cells"] if c["cell_type"] == "code")
        backend_defs = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(backend.read_text()).body
                        if isinstance(n, ast.FunctionDef)}
        nb_defs = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(nb_source).body
                   if isinstance(n, ast.FunctionDef)}
        different = [name for name, definition in nb_defs.items() if backend_defs.get(name) != definition]
        if different:
            raise ValueError(f"Funciones modificadas: {different}. Actualizar y validar el protocolo antes de reanudar.")
        protocol = {"version": 1, "code_sha256": digest(backend),
                    "raw_sha256": digest(PROC / "churn_dataset.csv"),
                    "train_rows": len(train), "train_end": str(train.mes_obs.max().date()),
                    "folds": 4, "validation_months": 4, "gap_months": 6,
                    "objective": "unweighted mean validation block ROC AUC",
                    "excluded_master_columns": MASTER, "seed": 42, "threads": THREADS,
                    "packages": {p: importlib.metadata.version(p) for p in
                                 ["numpy", "pandas", "scikit-learn", "optuna", "xgboost", "catboost"]}}
        with (OUT / "runner.lock").open("w") as lock:
            # Si el proceso tmux sigue activo, esta línea falla sin tocar sus trials.
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = OUT / "protocol.json"
            if manifest.exists() and json.loads(manifest.read_text()) != protocol:
                raise ValueError("Cambió el protocolo/datos: usar otro directorio OUT")
            write_json(manifest, protocol)
            cache = OUT / "folds.joblib"
            if cache.exists():
                folds = joblib.load(cache)
            else:
                folds = prepare(train, THREADS)
                joblib.dump(folds, cache)
                write_json(OUT / "folds.json", [f["info"] for f in folds])
            storage = optuna.storages.RDBStorage(url=f"sqlite:///{OUT / 'studies.sqlite3'}",
                                               engine_kwargs={"connect_args": {"timeout": 60}})
            studies = {}
            for name in MODELS:
                study = optuna.create_study(storage=storage, study_name=f"temporal_v1_{name}",
                                           direction="maximize", load_if_exists=True,
                                           sampler=optuna.samplers.TPESampler(seed=42),
                                           pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
                for trial in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
                    study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
                studies[name] = study
            seguir = True
            while seguir and not (OUT / "STOP").exists():
                seguir = False
                for name, study in studies.items():
                    if (OUT / "STOP").exists():
                        break
                    if sum(t.state.is_finished() for t in study.trials) >= N_TRIALS:
                        continue
                    seguir = True
                    study.sampler = optuna.samplers.TPESampler(seed=42 + len(study.trials))
                    prev = study.best_trial.number if any(t.state == optuna.trial.TrialState.COMPLETE
                                                         for t in study.trials) else None
                    study.optimize(lambda t: objective(t, name, folds, THREADS), n_trials=1)
                    if prev != study.best_trial.number:
                        export_best(study, name, train, THREADS, OUT)
                    study.trials_dataframe().to_csv(OUT / f"{name}_trials.csv", index=False)
                    print(name, len(study.trials), "/", N_TRIALS, "mejor AUC:", study.best_value)
            print("Entrenamiento terminado o pausado. Reejecutar sección 7 para actualizar tablas.")
    '''),
    md('''
    ## 9. Qué concluir al terminar

    Comparar el AUC medio junto con el comportamiento por período y el lift operativo.
    No comparar directamente este AUC temporal con el AUC del único OOT antiguo:
    corresponden a períodos, protocolo y variables distintos.

    Los mejores trials están seleccionados sobre estas validaciones. Para afirmar
    mejora en despliegue se necesita una evaluación final en datos nuevos y después
    un piloto de retención con costos reales. El resultado de Optuna es un candidato,
    no una prueba de rentabilidad ni del techo teórico del problema.
    ''')]
save("05_modelling/06_optuna_validacion_temporal.ipynb", cells)
print("Notebooks generados. Ejecutar ambos para validar y guardar sus salidas.")
