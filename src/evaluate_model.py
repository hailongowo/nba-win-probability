import joblib
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score

from config import TRAINING_FILE, LOGISTIC_MODEL_FILE, XGBOOST_MODEL_FILE


def evaluate_model(model_path) -> None:
    """
    Evaluate a saved model artifact.
    """

    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    df = pd.read_csv(TRAINING_FILE)

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

    X_validate = validate_df[feature_columns]
    y_validate = validate_df["home_win"]

    print("Validation rows:", len(validate_df))
    print("Validation games:", validate_df["game_id"].nunique())

    pred_proba = model.predict_proba(X_validate)[:, 1]
    pred_class = (pred_proba >= 0.5).astype(int)

    print(f"Evaluating model: {model_path}")
    print("--------------------------------")
    print(f"Log loss:     {log_loss(y_validate, pred_proba):.4f}")
    print(f"Brier score:  {brier_score_loss(y_validate, pred_proba):.4f}")
    print(f"ROC AUC:      {roc_auc_score(y_validate, pred_proba):.4f}")
    print(f"Accuracy:     {accuracy_score(y_validate, pred_class):.4f}")


if __name__ == "__main__":
    evaluate_model(LOGISTIC_MODEL_FILE)
    print()
    evaluate_model(XGBOOST_MODEL_FILE)