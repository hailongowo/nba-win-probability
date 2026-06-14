import joblib
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
from xgboost import XGBClassifier

from config import TRAINING_FILE, XGBOOST_MODEL_FILE, XGBOOST_MODEL_FILE_B


FEATURE_COLUMNS = [
    "period",
    "seconds_remaining_in_period",
    "regulation_seconds_remaining",
    "overtime_number",
    "posession", # 1 if home team has possession, 0 if away team has possession
    "score_diff",
    "abs_score_diff",
    # "score_diff_per_minute_remaining",
    "is_clutch_time",
    "scoreHome",
    "scoreAway",
    "is_playoffs",
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

    train_df = df[
        (
            (df["game_id"] >= 21500001) &
            (df["game_id"] < 22300000)
        )
        |
        (
            (df["game_id"] >= 41500001) &
            (df["game_id"] < 42300000)
        )
    ]

    validate_df = df[
        (
            (df["game_id"] >= 22300001) &
            (df["game_id"] < 22400000)
        )
        |
        (
            (df["game_id"] >= 42300001) &
            (df["game_id"] < 42400000)
        )
    ]

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_validate = validate_df[FEATURE_COLUMNS]
    y_validate = validate_df[TARGET_COLUMN]

    print("Train rows:", len(train_df))
    print("Validation rows:", len(validate_df))

    print("Train games:", train_df["game_id"].nunique())
    print("Validation games:", validate_df["game_id"].nunique())

    # X = df[FEATURE_COLUMNS]
    # y = df[TARGET_COLUMN]
    # groups = df[GROUP_COLUMN]

    # splitter = GroupShuffleSplit(
    #     n_splits=1,
    #     test_size=0.2,
    #     random_state=42,
    # )

    # train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    # X_train = X.iloc[train_idx]
    # X_test = X.iloc[test_idx]
    # y_train = y.iloc[train_idx]
    # y_test = y.iloc[test_idx]

    model = XGBClassifier(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    pred_proba = model.predict_proba(X_validate)[:, 1]
    pred_class = (pred_proba >= 0.5).astype(int)

    print("XGBoost Evaluation")
    print("------------------")
    print(f"Log loss:     {log_loss(y_validate, pred_proba):.4f}")
    print(f"Brier score:  {brier_score_loss(y_validate, pred_proba):.4f}")
    print(f"ROC AUC:      {roc_auc_score(y_validate, pred_proba):.4f}")
    print(f"Accuracy:     {accuracy_score(y_validate, pred_class):.4f}")

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

    joblib.dump(artifact, XGBOOST_MODEL_FILE_B)

    print(f"Saved XGBoost model to {XGBOOST_MODEL_FILE_B}")


if __name__ == "__main__":
    train_xgboost()