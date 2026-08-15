"""
The safety gate: retrains XGBoost on every PR and FAILS the build if
PR-AUC drops below a floor tied to the committed Phase 2 result.

Why a floor and not "must exactly match Phase 2's number exactly": tree
models have minor run-to-run variance across library versions and CI
runners. A floor with headroom below the known-good result catches real
regressions (a broken feature, a flipped label, a bad config change)
without failing the build on noise - same category of gate PromptGuard
uses to block LLM prompt regressions.

MIN_PR_AUC = 0.68 sits with margin below the known-good 0.7438-0.7469
range, and above the old Keras NN baseline (0.6686-0.6824) - so it also
catches "someone accidentally shipped the worse model."
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

sys.path.insert(0, ".")
from src.features import add_month_cyclical, build_preprocessor

RANDOM_STATE = 42
MIN_PR_AUC = 0.68


def test_xgboost_meets_pr_auc_floor():
    df = pd.read_csv("online_shoppers_intention.csv")
    df = add_month_cyclical(df)
    y = df["Revenue"].astype(int)
    X = df.drop(columns=["Revenue"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos

    pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("clf", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])
    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"\nRegression gate: PR-AUC = {pr_auc:.4f} (floor: {MIN_PR_AUC})")
    assert pr_auc >= MIN_PR_AUC, (
        f"PR-AUC regression detected: {pr_auc:.4f} is below the {MIN_PR_AUC} floor. "
        f"This blocks the merge - check recent changes to features.py or the training config."
    )
