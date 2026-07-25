from pathlib import Path

import joblib
import pandas as pd

from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score

from config import TRAINING_FILE, LOGISTIC_MODEL_FILE

FEATURE_COLUMNS = [
    # "period",
    # "seconds_remaining_in_period",
    # "regulation_seconds_remaining",
    "overtime_number",
    "effective_seconds_remaining",
    "score_diff",
    # "abs_score_diff",
    "score_diff_per_minute_remaining",
    # "scoreHome",
    # "scoreAway",
    "is_clutch_time",
    "posession",
    "is_playoffs",
]

TARGET_COLUMN = "home_win"
GROUP_COLUMN = "game_id"

# df = pd.read_csv(TRAINING_FILE)
# print(df.head())


def train_logistic_regression() -> None:
    """
    Train logistic regression model.
    """

    df = pd.read_csv(TRAINING_FILE)
    
    train_df = df[
        (
            (df["game_id"] >= 22000001) &
            (df["game_id"] < 22400000)
        )
        |
        (
            (df["game_id"] >= 42000001) &
            (df["game_id"] < 42400000)
        )
    ]

    validate_df = df[
        (
            (df["game_id"] >= 22400001) &
            (df["game_id"] < 22500000)
        )
        |
        (
            (df["game_id"] >= 42400001) &
            (df["game_id"] < 42500000)
        )
    ]

    # train_df = shuffle(train_df, random_state=42).reset_index(drop=True)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_validate = validate_df[FEATURE_COLUMNS]
    y_validate = validate_df[TARGET_COLUMN]

    print("Train rows:", len(train_df))
    print("Validation rows:", len(validate_df))

    print("Train games:", train_df["game_id"].nunique())
    print("Validation games:", validate_df["game_id"].nunique())

    # 1. Scale numeric features.
    # 2. Train logistic regression.

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    # Predict probabilities for home team win.
    pred_proba = model.predict_proba(X_validate)[:, 1]

    # Convert probabilities to class predictions.
    pred_class = (pred_proba >= 0.5).astype(int)

    print("Logistic Regression Evaluation\n")
    print("--------------------------------")
    print(f"Log loss:     {log_loss(y_validate, pred_proba):.4f}")
    print(f"Brier score:  {brier_score_loss(y_validate, pred_proba):.4f}")
    print(f"ROC AUC:      {roc_auc_score(y_validate, pred_proba):.4f}")
    print(f"Accuracy:     {accuracy_score(y_validate, pred_class):.4f}")

    # Save model and feature list together.
    # This prevents mistakes during live prediction.
    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
    }

    joblib.dump(artifact, LOGISTIC_MODEL_FILE)

    print(f"Saved logistic regression model to {LOGISTIC_MODEL_FILE}")


if __name__ == "__main__":
    train_logistic_regression()