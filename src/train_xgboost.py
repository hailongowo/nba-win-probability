import joblib
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
from xgboost import XGBClassifier

from config import TRAINING_FILE, XGBOOST_MODEL_FILE


FEATURE_COLUMNS = [
    "period",
    "seconds_remaining",
    "score_diff",
    "abs_score_diff",
    "score_diff_per_minute_remaining",
    "is_second_half",
    "is_fourth_quarter",
    "is_clutch_time",
    "scoreHome",
    "scoreAway",
]

TARGET_COLUMN = "home_win"
GROUP_COLUMN = "game_id"


def train_xgboost() -> None:
    """
    Train an XGBoost win probability model.

    XGBoost usually performs better than logistic regression,
    but it is slightly harder to explain.
    """

    df = pd.read_csv(TRAINING_FILE)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    groups = df[GROUP_COLUMN]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    pred_proba = model.predict_proba(X_test)[:, 1]
    pred_class = (pred_proba >= 0.5).astype(int)

    print("XGBoost Evaluation")
    print("------------------")
    print(f"Log loss:     {log_loss(y_test, pred_proba):.4f}")
    print(f"Brier score:  {brier_score_loss(y_test, pred_proba):.4f}")
    print(f"ROC AUC:      {roc_auc_score(y_test, pred_proba):.4f}")
    print(f"Accuracy:     {accuracy_score(y_test, pred_class):.4f}")

    # Show feature importance.
    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("\nFeature importance:")
    print(importance)

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
    }

    joblib.dump(artifact, XGBOOST_MODEL_FILE)

    print(f"Saved XGBoost model to {XGBOOST_MODEL_FILE}")


if __name__ == "__main__":
    train_xgboost()