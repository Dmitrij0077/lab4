"""Воспроизводимый запуск всех экспериментов лабораторной №4.

    python run_experiments.py

Делает: стратифицированный split 80/20 (seed=42) -> для каждой модели 5x2
repeated stratified CV ROC-AUC на train (критерий выбора) -> обучение на всём
train и метрики на test (только для отчёта) -> вызов контрактной функции ->
запись experiment_results.json.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

from ensemble_models import (
    _voting, build_candidates, make_preprocessor, run_ensemble_experiment,
)
from grader_contracts.ensembles import EnsembleInput

warnings.filterwarnings("ignore")

SEED = 42
ROOT = Path(__file__).parent


def metrics(y_true, proba) -> dict[str, float]:
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
    }


def main() -> None:
    df = pd.read_csv(ROOT / "german.csv", sep=";")
    y = df.pop("Creditability")
    x_train, x_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, stratify=y, random_state=SEED
    )
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)

    candidates = build_candidates(SEED)
    # Абляция: итоговый ансамбль без производных признаков.
    candidates["Voting soft, без доп. признаков (абляция)"] = make_pipeline(
        make_preprocessor(True, use_features=False), _voting(SEED)
    )

    rows = []
    for name, pipe in candidates.items():
        start = time.time()
        cv_scores = cross_val_score(pipe, x_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(x_train, y_train)
        m = metrics(y_test, pipe.predict_proba(x_test)[:, 1])
        row = {
            "model": name,
            "cv_roc_auc_mean": float(cv_scores.mean()),
            "cv_roc_auc_std": float(cv_scores.std()),
            **{f"test_{k}": float(v) for k, v in m.items()},
            "seconds": round(time.time() - start, 1),
        }
        rows.append(row)
        print(f"{name:45s} cv={row['cv_roc_auc_mean']:.4f}±{row['cv_roc_auc_std']:.3f} "
              f"test_auc={row['test_roc_auc']:.4f} ({row['seconds']}s)")

    # Критерий выбора — средний CV ROC-AUC на train (test не участвует).
    best = max(rows, key=lambda r: r["cv_roc_auc_mean"])
    print("\nВыбрано по CV:", best["model"])

    # Итоговый прогон именно через контрактную функцию.
    result = run_ensemble_experiment(
        EnsembleInput(x_train, y_train, x_test, y_test, random_seed=SEED)
    )
    final = {
        "model_name": result.model_name,
        "roc_auc": result.roc_auc,
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
    }
    print("Контрактная функция:", final)

    out = {
        "random_seed": SEED,
        "split": {"test_size": 0.2, "stratified": True,
                  "n_train": int(len(x_train)), "n_test": int(len(x_test))},
        "selection_criterion": "mean ROC-AUC, 5x2 repeated stratified CV на train",
        "selected_by_cv": best["model"],
        "final_model": final,
        "experiments": rows,
    }
    (ROOT / "experiment_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
