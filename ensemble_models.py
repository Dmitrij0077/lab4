"""Эксперимент с ансамблем: soft-voting (LogisticRegression + RandomForest + GradientBoosting).

Модуль содержит:
* предобработку и инженерию признаков для german.csv;
* фабрику моделей-кандидатов (`build_candidates`), которую использует
  `run_experiments.py` для сравнения;
* контрактную функцию `run_ensemble_experiment`.

Обучение идёт только на train-части; `test_target` используется исключительно
для вычисления итоговых метрик.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from grader_contracts.ensembles import EnsembleInput, EnsembleResult

# Порядок признаков в german.csv (без целевой Creditability).
FEATURE_COLUMNS = [
    "Account_Balance", "Duration_of_Credit_monthly",
    "Payment_Status_of_Previous_Credit", "Purpose", "Credit_Amount",
    "Value_Savings_Stocks", "Length_of_current_employment",
    "Instalment_per_cent", "Sex_Marital_Status", "Guarantors",
    "Duration_in_Current_address", "Most_valuable_available_asset",
    "Age_years", "Concurrent_Credits", "Type_of_apartment",
    "No_of_Credits_at_this_Bank", "Occupation", "No_of_dependents",
    "Telephone", "Foreign_Worker",
]

# Категориальные признаки, закодированные числами без осмысленного порядка
# (либо с немонотонной связью с риском) — кодируем one-hot.
NOMINAL = [
    "Account_Balance", "Payment_Status_of_Previous_Credit", "Purpose",
    "Value_Savings_Stocks", "Length_of_current_employment",
    "Sex_Marital_Status", "Most_valuable_available_asset",
    "Concurrent_Credits", "Type_of_apartment", "Occupation",
]


def _as_frame(features) -> pd.DataFrame:
    """Приводит вход к DataFrame с именами столбцов german.csv."""
    if isinstance(features, pd.DataFrame):
        return features.reset_index(drop=True)
    arr = np.asarray(features)
    return pd.DataFrame(arr, columns=FEATURE_COLUMNS[: arr.shape[1]])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет производные признаки (используют только саму строку)."""
    out = df.copy()
    out["log_amount"] = np.log(out["Credit_Amount"])
    out["log_duration"] = np.log(out["Duration_of_Credit_monthly"])
    out["amount_per_month"] = out["Credit_Amount"] / out["Duration_of_Credit_monthly"]
    out["monthly_burden"] = out["amount_per_month"] * out["Instalment_per_cent"]
    return out


def make_preprocessor(scale: bool, use_features: bool = True) -> Pipeline:
    """Инженерия признаков + one-hot для номинальных + (опционально) масштабирование."""
    columns = list(FEATURE_COLUMNS)
    if use_features:
        columns += ["log_amount", "log_duration", "amount_per_month", "monthly_burden"]
    numeric = [c for c in columns if c not in NOMINAL]
    transformer = ColumnTransformer(
        [
            ("nominal", OneHotEncoder(handle_unknown="ignore"), NOMINAL),
            ("numeric", StandardScaler() if scale else "passthrough", numeric),
        ]
    )
    steps = [FunctionTransformer(add_features)] if use_features else []
    return make_pipeline(*steps, transformer)


def _logreg(seed: int) -> LogisticRegression:
    return LogisticRegression(C=0.3, max_iter=2000, random_state=seed)


def _forest(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, max_features="sqrt",
        random_state=seed, n_jobs=1,
    )


def _gboost(seed: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=3,
        subsample=0.7, random_state=seed,
    )


def _voting(seed: int) -> VotingClassifier:
    return VotingClassifier(
        [("logreg", _logreg(seed)), ("forest", _forest(seed)), ("gboost", _gboost(seed))],
        voting="soft",
    )


def build_candidates(seed: int) -> dict[str, Pipeline]:
    """Все модели, участвующие в сравнении (полные pipeline: препроцессинг + модель)."""
    return {
        "LogisticRegression (baseline)": make_pipeline(make_preprocessor(True), _logreg(seed)),
        "RandomForest": make_pipeline(make_preprocessor(False), _forest(seed)),
        "ExtraTrees": make_pipeline(
            make_preprocessor(False),
            ExtraTreesClassifier(
                n_estimators=300, min_samples_leaf=3, max_features=0.4,
                random_state=seed, n_jobs=1,
            ),
        ),
        "AdaBoost": make_pipeline(
            make_preprocessor(False),
            AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=seed),
        ),
        "GradientBoosting": make_pipeline(make_preprocessor(False), _gboost(seed)),
        "HistGradientBoosting": make_pipeline(
            make_preprocessor(False),
            HistGradientBoostingClassifier(
                learning_rate=0.03, max_iter=200, max_depth=3,
                l2_regularization=1.0, random_state=seed,
            ),
        ),
        "MLP (32,) alpha=1.0": make_pipeline(
            make_preprocessor(True),
            MLPClassifier(
                hidden_layer_sizes=(32,), alpha=1.0, max_iter=1000, random_state=seed,
            ),
        ),
        "Stacking (LR+RF+GB -> LR)": make_pipeline(
            make_preprocessor(True),
            StackingClassifier(
                [("logreg", _logreg(seed)), ("forest", _forest(seed)), ("gboost", _gboost(seed))],
                final_estimator=LogisticRegression(C=1.0, max_iter=2000),
                cv=5, stack_method="predict_proba",
            ),
        ),
        "Voting soft (LR+RF+GB)": make_pipeline(make_preprocessor(True), _voting(seed)),
    }


def build_final_model(seed: int) -> Pipeline:
    """Выбранная по CV на train модель: soft-voting LR + RF + GB."""
    return make_pipeline(make_preprocessor(True), _voting(seed))


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучает soft-voting ансамбль на train и считает метрики на test."""
    x_train = _as_frame(data.train_features)
    x_test = _as_frame(data.test_features)
    y_train = np.asarray(data.train_target).ravel()
    y_test = np.asarray(data.test_target).ravel()

    model = build_final_model(data.random_seed)
    model.fit(x_train, y_train)  # только train

    probabilities = model.predict_proba(x_test)[:, 1]  # вероятность класса 1
    predictions = (probabilities >= 0.5).astype(int)

    return EnsembleResult(
        model_name="Soft Voting: LogisticRegression + RandomForest + GradientBoosting",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, zero_division=0)),
        recall=float(recall_score(y_test, predictions, zero_division=0)),
        f1=float(f1_score(y_test, predictions, zero_division=0)),
    )
