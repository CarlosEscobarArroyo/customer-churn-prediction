"""Read Optuna progress without starting training: python -m scripts.optuna_status."""

import argparse
from pathlib import Path

import optuna


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/optuna_temporal_v1")
    args = parser.parse_args()
    database = Path(args.output).resolve() / "studies.sqlite3"
    if not database.exists():
        parser.error(f"Study database does not exist: {database}")
    storage = f"sqlite:///{database}"
    for summary in optuna.get_all_study_summaries(storage):
        study = optuna.load_study(study_name=summary.study_name, storage=storage)
        trials = study.get_trials()
        counts = {state.name: sum(t.state == state for t in trials)
                  for state in optuna.trial.TrialState}
        best = summary.best_trial
        score = f"{best.value:.6f} (trial {best.number})" if best else "pending"
        print(f"{study.study_name}: {counts}; best temporal AUC={score}")


if __name__ == "__main__":
    main()
