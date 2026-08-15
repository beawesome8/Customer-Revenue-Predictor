"""
Phase 1 baseline: Logistic Regression with class weighting.

Why this model first, before XGBoost or a neural net: every later model in
this project has to beat this number, or it doesn't belong in the repo.
Building the fanciest model first and skipping the baseline is how you end
up unable to tell an interviewer whether your XGBoost model is actually
good, or just better than nothing.

Why PR-AUC is the headline metric, not accuracy: Revenue==True is 15.5% of
the data. A model that always predicts False scores 84.5% accuracy while
being useless. PR-AUC (average precision) is the right metric for
rare-positive-class problems because it doesn't reward exploiting the
imbalance.
"""
import json
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report

sys.path.insert(0, ".")
from src.features import add_month_cyclical, build_preprocessor

RANDOM_STATE = 42


def load_data(path: str = "online_shoppers_intention.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return add_month_cyclical(df)


def main():
    df = load_data()
    y = df["Revenue"].astype(int)
    X = df.drop(columns=["Revenue"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
    ])
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    naive_pr_auc = y_test.mean()

    print("=" * 60)
    print("PHASE 1 BASELINE: Logistic Regression (class_weight=balanced)")
    print("=" * 60)
    print(f"Test set size: {len(y_test)}  (positive class rate: {y_test.mean():.3%})")
    print(f"PR-AUC: {pr_auc:.4f}  (naive baseline: {naive_pr_auc:.4f}, lift: {pr_auc/naive_pr_auc:.2f}x)")
    print(f"ROC-AUC: {roc_auc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["No Purchase", "Purchase"]))

    metrics = {
        "model": "logistic_regression_baseline",
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "naive_pr_auc_baseline": round(naive_pr_auc, 4),
        "test_set_size": len(y_test),
        "positive_class_rate": round(float(y_test.mean()), 4),
    }
    with open("baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved metrics to baseline_metrics.json")


if __name__ == "__main__":
    main()
