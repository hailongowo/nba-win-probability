import joblib
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score

from config import TRAINING_FILE, LOGISTIC_MODEL_FILE


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


def train_logistic_regression() -> None:
    """
    Train logistic regression model.

    Important:
    We split by game_id, not by row.

    Why?
    Because one game produces many training rows.
    If rows from the same game appear in both train and test sets,
    evaluation becomes too optimistic.
    """

    df = pd.read_csv(TRAINING_FILE)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    groups = df[GROUP_COLUMN]

    # GroupShuffleSplit keeps all rows from one game together.
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

    # Pipeline means:
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
    pred_proba = model.predict_proba(X_test)[:, 1]

    # Convert probabilities to class predictions.
    pred_class = (pred_proba >= 0.5).astype(int)

    print("Logistic Regression Evaluation")
    print("--------------------------------")
    print(f"Log loss:     {log_loss(y_test, pred_proba):.4f}")
    print(f"Brier score:  {brier_score_loss(y_test, pred_proba):.4f}")
    print(f"ROC AUC:      {roc_auc_score(y_test, pred_proba):.4f}")
    print(f"Accuracy:     {accuracy_score(y_test, pred_class):.4f}")

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