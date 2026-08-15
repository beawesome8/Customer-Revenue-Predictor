"""
Phase 2: XGBoost and a Keras NN, both benchmarked against the Phase 1
Logistic Regression baseline on identical train/test splits.
"""
import json
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

sys.path.insert(0, ".")
from src.features import add_month_cyclical, build_preprocessor

RANDOM_STATE = 42


def load_data(path="online_shoppers_intention.csv"):
    df = pd.read_csv(path)
    return add_month_cyclical(df)


def run_xgboost(X_train, X_test, y_train, y_test, preprocessor):
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])
    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    return {
        "model": "xgboost",
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "report": classification_report(y_test, y_pred, target_names=["No Purchase", "Purchase"]),
    }


def run_keras_nn(X_train, X_test, y_train, y_test, preprocessor):
    import tensorflow as tf
    tf.random.set_seed(RANDOM_STATE)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()
        X_test_proc = X_test_proc.toarray()
    pos = y_train.sum()
    neg = len(y_train) - pos
    class_weight = {0: 1.0, 1: neg / pos}
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_proc.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                   metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")])
    model.fit(X_train_proc, y_train, validation_split=0.15, epochs=30,
              batch_size=64, class_weight=class_weight, verbose=0)
    y_proba = model.predict(X_test_proc, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "model": "keras_nn",
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "report": classification_report(y_test, y_pred, target_names=["No Purchase", "Purchase"]),
    }


def main():
    df = load_data()
    y = df["Revenue"].astype(int)
    X = df.drop(columns=["Revenue"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    baseline_pr_auc = json.load(open("baseline_metrics.json"))["pr_auc"]
    xgb_result = run_xgboost(X_train, X_test, y_train, y_test, build_preprocessor())
    nn_result = run_keras_nn(X_train, X_test, y_train, y_test, build_preprocessor())

    print("=" * 60)
    print("PHASE 2: XGBoost vs Keras NN vs Phase 1 Baseline")
    print("=" * 60)
    print(f"Baseline (Logistic Regression) PR-AUC: {baseline_pr_auc}\n")
    for r in [xgb_result, nn_result]:
        print(f"--- {r['model']} ---")
        print(f"PR-AUC: {r['pr_auc']}  (delta vs baseline: {r['pr_auc']-baseline_pr_auc:+.4f})")
        print(f"ROC-AUC: {r['roc_auc']}")
        print(r["report"], "\n")

    winner = max([xgb_result, nn_result], key=lambda r: r["pr_auc"])
    print(f"WINNER by PR-AUC: {winner['model']} ({winner['pr_auc']})")

    summary = {
        "baseline_pr_auc": baseline_pr_auc,
        "xgboost": {"pr_auc": xgb_result["pr_auc"], "roc_auc": xgb_result["roc_auc"]},
        "keras_nn": {"pr_auc": nn_result["pr_auc"], "roc_auc": nn_result["roc_auc"]},
        "winner": winner["model"],
    }
    with open("phase2_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
