"""Train and compare three phishing website classifiers.

Dataset: dataset/upload.csv
Target convention: -1 = phishing, 1 = legitimate
The holdout test set is not used for cross-validation or model selection.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "upload.csv"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
TEST_SIZE = 0.20
PHISHING_LABEL = -1


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    required = {"Id", "Result"}
    if not required.issubset(data.columns):
        raise ValueError("Expected dataset columns 'Id' and 'Result'.")

    # Id is a row identifier, not a predictive feature. Result is the target.
    X = data.drop(columns=["Id", "Result"])
    y = data["Result"].astype(int)
    if X.shape[1] != 30:
        raise ValueError(f"Expected 30 input features, found {X.shape[1]}.")
    if set(y.unique()) != {-1, 1}:
        raise ValueError(f"Unexpected target labels: {sorted(y.unique())}")
    if X.isna().any().any():
        raise ValueError("Dataset contains missing feature values; review preprocessing.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_features="sqrt", random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    # Explicit scorer functions preserve -1 as the positive (phishing) class.
    from sklearn.metrics import make_scorer
    scoring = {
        "accuracy": "accuracy",
        "phishing_precision": make_scorer(precision_score, pos_label=PHISHING_LABEL),
        "phishing_recall": make_scorer(recall_score, pos_label=PHISHING_LABEL),
        "phishing_f1": make_scorer(f1_score, pos_label=PHISHING_LABEL),
    }

    fitted_models = {}
    summary_rows = []
    detailed_results = {}

    for name, estimator in models.items():
        print(f"Training {name} ...", flush=True)
        cv_result = cross_validate(
            clone(estimator), X_train, y_train, cv=cv,
            scoring=scoring, n_jobs=1, return_train_score=False,
        )

        fit_start = time.perf_counter()
        estimator.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_start

        pred_start = time.perf_counter()
        y_pred = estimator.predict(X_test)
        predict_seconds = time.perf_counter() - pred_start
        classes = list(estimator.classes_)
        phishing_probability = estimator.predict_proba(X_test)[:, classes.index(PHISHING_LABEL)]

        metrics = {
            "model": name,
            "train_accuracy": float(accuracy_score(y_train, estimator.predict(X_train))),
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_phishing_precision": float(precision_score(y_test, y_pred, pos_label=PHISHING_LABEL)),
            "test_phishing_recall": float(recall_score(y_test, y_pred, pos_label=PHISHING_LABEL)),
            "test_phishing_f1": float(f1_score(y_test, y_pred, pos_label=PHISHING_LABEL)),
            "test_roc_auc_phishing": float(roc_auc_score((y_test == PHISHING_LABEL).astype(int), phishing_probability)),
            "cv_accuracy_mean": float(np.mean(cv_result["test_accuracy"])),
            "cv_accuracy_std": float(np.std(cv_result["test_accuracy"])),
            "cv_phishing_precision_mean": float(np.mean(cv_result["test_phishing_precision"])),
            "cv_phishing_recall_mean": float(np.mean(cv_result["test_phishing_recall"])),
            "cv_phishing_f1_mean": float(np.mean(cv_result["test_phishing_f1"])),
            "cv_phishing_f1_std": float(np.std(cv_result["test_phishing_f1"])),
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
            "confusion_matrix_labels_minus1_plus1": confusion_matrix(
                y_test, y_pred, labels=[-1, 1]
            ).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, labels=[-1, 1],
                target_names=["Phishing (-1)", "Legitimate (1)"],
                output_dict=True,
            ),
        }
        fitted_models[name] = estimator
        detailed_results[name] = metrics
        summary_rows.append({k: v for k, v in metrics.items() if k not in {
            "confusion_matrix_labels_minus1_plus1", "classification_report"
        }})

        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, labels=[-1, 1],
            display_labels=["Phishing", "Legitimate"], cmap="Blues", values_format="d",
        )
        plt.title(f"{name} – Test Confusion Matrix")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        plt.savefig(RESULTS_DIR / f"{safe_name}_confusion_matrix.png", dpi=160)
        plt.close()

        print(
            f"  test accuracy={metrics['test_accuracy']:.4f}; "
            f"phishing F1={metrics['test_phishing_f1']:.4f}; "
            f"5-fold CV phishing F1={metrics['cv_phishing_f1_mean']:.4f}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("cv_phishing_f1_mean", ascending=False)
    summary.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    # Select using cross-validation on the training partition only, not test metrics.
    winner_name = summary.iloc[0]["model"]
    selected_model = fitted_models[winner_name]

    # Holdout test metrics are now final. Refit the selected model on all labelled
    # rows for deployment, without changing the reported holdout evaluation.
    selected_model.fit(X, y)
    with open(ROOT / "model.pkl", "wb") as model_file:
        pickle.dump(selected_model, model_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(MODELS_DIR / "selected_model_name.txt", "w", encoding="utf-8") as f:
        f.write(winner_name + "\n")

    # ROC curves for the held-out test set.
    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    for name, estimator in fitted_models.items():
        RocCurveDisplay.from_estimator(
            estimator, X_test, y_test, pos_label=PHISHING_LABEL,
            name=name, ax=ax,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_title("ROC Curve Comparison – Phishing as Positive Class")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curve_comparison.png", dpi=160)
    plt.close()

    report = {
        "dataset_rows": int(len(data)),
        "input_features": int(X.shape[1]),
        "class_counts": {str(int(k)): int(v) for k, v in y.value_counts().items()},
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "positive_class_for_phishing_metrics": PHISHING_LABEL,
        "selection_rule": "Highest mean 5-fold cross-validation phishing F1 on training partition",
        "selected_model": winner_name,
        "models": detailed_results,
    }
    with open(RESULTS_DIR / "detailed_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Feature importance is available for the two tree ensembles.
    for name in ["Random Forest", "Gradient Boosting"]:
        fitted = fitted_models[name]
        importance = pd.DataFrame({
            "feature": X.columns,
            "importance": fitted.feature_importances_,
        }).sort_values("importance", ascending=False)
        importance.to_csv(
            RESULTS_DIR / f"{name.lower().replace(' ', '_')}_feature_importance.csv",
            index=False,
        )

    print("\nComparison saved to results/model_comparison.csv")
    print(f"Selected deployment model: {winner_name}")
    print("Saved deployment model to model.pkl")


if __name__ == "__main__":
    main()
