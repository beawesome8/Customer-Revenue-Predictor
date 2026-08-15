"""
Phase 3 step 1: train the champion model (XGBoost, per Phase 2 evidence)
on the FULL dataset and persist it as a versioned artifact.

Why retrain on 100% of the data instead of reusing the Phase 2 model:
Phase 2's split existed to produce an honest, unbiased performance
estimate - that's what goes on the resume and in the README. This step
trains the model that actually ships, on every row available, since
holding back 20% of training signal in production for no reason leaves
performance on the table. Keep the two mental models separate, or you'll
misreport which number means what.
"""
import json
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.insert(0, ".")
from src.features import add_month_cyclical, build_preprocessor

RANDOM_STATE = 42
MODEL_VERSION = "v1.0.0"


def main():
    df = pd.read_csv("online_shoppers_intention.csv")
    df = add_month_cyclical(df)
    y = df["Revenue"].astype(int)
    X = df.drop(columns=["Revenue"])

    pos = y.sum()
    neg = len(y) - pos
    scale_pos_weight = neg / pos

    pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("clf", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])
    pipeline.fit(X, y)
    joblib.dump(pipeline, "model_artifact.joblib")

    eval_metrics = json.load(open("phase2_comparison.json"))
    manifest = {
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "XGBoost",
        "trained_on_rows": len(df),
        "held_out_eval_pr_auc": eval_metrics["xgboost"]["pr_auc"],
        "held_out_eval_roc_auc": eval_metrics["xgboost"]["roc_auc"],
        "note": "held_out_eval metrics are from the Phase 2 80/20 split, NOT this artifact (trained on 100% of data). Report the held_out numbers as the honest performance estimate.",
    }
    with open("model_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved model_artifact.joblib ({MODEL_VERSION})")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
