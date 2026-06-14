from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

# Data folders.
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Raw data subfolders.
RAW_GAMES_DIR = RAW_DIR / "games"
RAW_PBP_DIR = RAW_DIR / "play_by_play"
RAW_GAMES_TESTING_DIR = RAW_DIR / "games_testing"
RAW_PBP_TESTING_DIR = RAW_DIR / "play_by_play_testing"
RAW_REGULAR_PBP_DIR = RAW_DIR / "play_by_play_regular"
RAW_PLAYOFF_PBP_DIR = RAW_DIR / "play_by_play_playoffs"
MODIFIED_PBP_REGULAR_DIR = RAW_DIR / "play_by_play_modified/regular"
MODIFIED_PBP_PLAYOFFS_DIR = RAW_DIR / "play_by_play_modified/playoffs"

# Processed file paths.
GAMES_FILE = PROCESSED_DIR / "games.csv"
GAMES_TESTING_FILE = PROCESSED_DIR / "games_testing.csv"
TRAINING_FILE = PROCESSED_DIR / "training_rows.csv"
TESTING_FILE = PROCESSED_DIR / "testing_rows.csv"
REGULAR_GAMES_FILE = PROCESSED_DIR / "regular_games.csv"
PLAYOFF_GAMES_FILE = PROCESSED_DIR / "playoff_games.csv"

# Saved model paths.
LOGISTIC_MODEL_FILE = MODELS_DIR / "logistic_regression.joblib"
XGBOOST_MODEL_FILE = MODELS_DIR / "xgboost_model.joblib"
XGBOOST_MODEL_FILE_B = MODELS_DIR / "xgboost_model_b.joblib"

# Create folders automatically if they do not exist.
for folder in [
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    RAW_GAMES_DIR,
    RAW_PBP_DIR,
    RAW_GAMES_TESTING_DIR,
    RAW_PBP_TESTING_DIR,
    RAW_REGULAR_PBP_DIR,
    RAW_PLAYOFF_PBP_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)