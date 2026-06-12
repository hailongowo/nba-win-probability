import joblib

from config import LOGISTIC_MODEL_FILE
from live_features import build_features_from_values


artifact = joblib.load(LOGISTIC_MODEL_FILE)
model = artifact["model"]
feature_columns = artifact["feature_columns"]

# Example fake game state:
# Home team leads by 5 in Q4 with 2 minutes left.
features = build_features_from_values(
    period=4,
    clock="PT02M00.00S",
    home_score=100,
    away_score=95,
)

X = features[feature_columns]
prob = model.predict_proba(X)[0, 1]

print(f"Home win probability: {prob:.2%}")
print(f"Away win probability: {1 - prob:.2%}")