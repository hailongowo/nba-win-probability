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

    X = df[feature_columns]
    y = df["home_win"]
    groups = df["game_id"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    pred_proba = model.predict_proba(X_test)[:, 1]
    pred_class = (pred_proba >= 0.5).astype(int)

    print(f"Evaluating model: {model_path}")
    print("--------------------------------")
    print(f"Log loss:     {log_loss(y_test, pred_proba):.4f}")
    print(f"Brier score:  {brier_score_loss(y_test, pred_proba):.4f}")
    print(f"ROC AUC:      {roc_auc_score(y_test, pred_proba):.4f}")
    print(f"Accuracy:     {accuracy_score(y_test, pred_class):.4f}")


if __name__ == "__main__":
    evaluate_model(LOGISTIC_MODEL_FILE)
    print()
    evaluate_model(XGBOOST_MODEL_FILE)