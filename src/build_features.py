import re
import pandas as pd
import numpy as np

from config import MODIFIED_PBP_PLAYOFFS_DIR, MODIFIED_PBP_REGULAR_DIR, TRAINING_FILE, TESTING_FILE

def parse_pctimestring(pctimestring: str) -> int:
    """
    Convert NBA play-by-play clock format into seconds elapsed within the period.

    NBA play-by-play commonly stores time as a string like:
        "PT10M32.00S"

    This means 10 minutes and 32 seconds remaining in the current period.

    We return seconds remaining in the period.

    Example:
        "PT10M32.00S" -> 632
    """

    if pd.isna(pctimestring):
        return np.nan

    text = str(pctimestring)

    # Match strings like PT10M32.00S or PT0M5.20S.
    match = re.match(r"PT(\d+)M([\d\.]+)S", text)

    if not match:
        return np.nan

    minutes = int(match.group(1))
    seconds = float(match.group(2))

    return int(minutes * 60 + seconds)

def get_seconds_remaining(period: int, seconds_remaining_in_period: int) -> int:
    """
    Convert period and period clock into total seconds remaining in the game.

    NBA regulation game:
        4 quarters, 12 minutes each = 48 minutes = 2880 seconds.

    Overtime:
        5 minutes each.

    For simplicity:
        - Periods 1-4 use normal regulation time.
        - Period 5+ are treated as overtime.

    Examples:
        Q1, 12:00 left -> 2880 seconds remaining
        Q4, 01:00 left -> 60 seconds remaining
    """

    if pd.isna(period) or pd.isna(seconds_remaining_in_period):
        return np.nan

    period = int(period)
    seconds_remaining_in_period = int(seconds_remaining_in_period)

    if period <= 4:
        # Number of full regulation periods after the current one.
        periods_left_after_current = 4 - period
        return periods_left_after_current * 12 * 60 + seconds_remaining_in_period

    # For overtime, we estimate only the current OT period.
    # This is not perfect, but it is acceptable for version 1.
    return seconds_remaining_in_period

def build_rows_for_game(pbp: pd.DataFrame, game_id: str) -> pd.DataFrame:
    """
    Convert one game's play-by-play into many model training rows.

    Each row represents the game state after a scoring event.

    The label is whether the home team eventually won.
    """

    if pbp.empty:
        return pd.DataFrame()

    df = pbp.copy()
    df = df.sort_values("actionNumber").copy()

    # Events to keep
    important_actions = [
        "Made Shot",
        "Foul",
        "Free Throw",
        "Turnover",
        "Rebound",
        "Violation",
    ]

    df = df[
        df["actionType"].isin(important_actions)
    ]
    
    
    if df.empty:
        return pd.DataFrame()

    # Parse period clock.
    df["seconds_remaining_in_period"] = df["clock"].apply(parse_pctimestring)

    df["regulation_seconds_remaining"] = np.where(
        df["period"] <= 4,
        (4 - df["period"]) * 720 + df["seconds_remaining_in_period"],
        0,
    )

    df["overtime_number"] = np.maximum(df["period"] - 4, 0)

    df["seconds_elapsed"] = np.where(
        df["period"] <= 4,
        (df["period"] - 1) * 720 + (720 - df["seconds_remaining_in_period"]),
        2880
        + (df["period"] - 5) * 300
        + (300 - df["seconds_remaining_in_period"]),
    )

    # Remove invalid parsed rows.
    df = df.dropna(subset=["scoreHome", "scoreAway", "seconds_remaining_in_period"])

    if df.empty:
        return pd.DataFrame()

    # Final score is the last available score in the game.
    final_home_score = df.iloc[-1]["scoreHome"]
    final_visitor_score = df.iloc[-1]["scoreAway"]

    # Skip rare tied/invalid cases.
    if final_home_score == final_visitor_score:
        return pd.DataFrame()

    home_win = int(final_home_score > final_visitor_score)

    # Score difference from home team's perspective.
    # Positive means home team is winning.
    df["score_diff"] = df["scoreHome"] - df["scoreAway"]

    # Helpful engineered features.
    df["abs_score_diff"] = df["score_diff"].abs()

    # Clutch time: last 5 minutes of Q4 or overtime, score within 5.
    df["is_clutch_time"] = (
        (df["regulation_seconds_remaining"] <= 5 * 60)
        & (df["abs_score_diff"] <= 5)
        & (df["period"] >= 4)
    ).astype(int)

    # Avoid division by zero.
    df["minutes_remaining"] = df["regulation_seconds_remaining"] / 60
    df["score_diff_per_minute_remaining"] = df["score_diff"] / df["minutes_remaining"].replace(0, 0.1)

    # Add target.
    df["home_win"] = home_win
    df["game_id"] = game_id

    # Remove dupes
    df = df.drop_duplicates(
    subset=[
        "period",
        "seconds_remaining_in_period",
        "scoreHome",
        "scoreAway",
        "actionType",
    ]
)

    # We only keep the useful columns for model training.
    feature_rows = df[
        [
            "game_id",
            "period",
            "seconds_remaining_in_period",
            "regulation_seconds_remaining",
            "overtime_number",
            "posession", # 1 if home team has possession, 0 if away team has possession
            "score_diff",
            "abs_score_diff",
            "score_diff_per_minute_remaining",
            "is_clutch_time",
            "scoreHome",
            "scoreAway",
            "is_playoffs",
            "home_win",
        ]
    ].copy()

    return feature_rows

def build_training_dataset() -> pd.DataFrame:
    """
    Build training dataset from all downloaded play-by-play files.
    """

    all_rows = []
    pbp_files = sorted(MODIFIED_PBP_REGULAR_DIR.glob("*.csv"))
    print(f"Found {len(pbp_files)} regular season play-by-play files")

    for file_path in pbp_files:
        game_id = file_path.stem

        try:
            pbp = pd.read_csv(file_path)
            rows = build_rows_for_game(pbp, game_id)

            if not rows.empty:
                all_rows.append(rows)

        except Exception as e:
            print(f"Failed to process {game_id}: {e}")
            continue

    pbp_files = sorted(MODIFIED_PBP_PLAYOFFS_DIR.glob("*.csv"))
    print(f"Found {len(pbp_files)} playoffs play-by-play files")

    for file_path in pbp_files:
        game_id = file_path.stem

        try:
            pbp = pd.read_csv(file_path)
            rows = build_rows_for_game(pbp, game_id)

            if not rows.empty:
                all_rows.append(rows)

        except Exception as e:
            print(f"Failed to process {game_id}: {e}")
            continue

    if not all_rows:
        raise ValueError("No training rows were created. Check your play-by-play files.")

    training_df = pd.concat(all_rows, ignore_index=True)

    # Remove missing and infinite values.
    training_df = training_df.replace([np.inf, -np.inf], np.nan)
    training_df = training_df.dropna()

    training_df.to_csv(TRAINING_FILE, index=False)

    print(f"Saved {len(training_df)} training rows to {TRAINING_FILE}")
    print(training_df.head())

    return training_df

# def build_testing_dataset() -> pd.DataFrame:
#     """
#     Build testing dataset from 2025-26 season downloaded play-by-play files.
#     """

#     all_rows = []

#     pbp_files = sorted(RAW_PBP_TESTING_DIR.glob("*.csv"))

#     print(f"Found {len(pbp_files)} play-by-play files")

#     for file_path in pbp_files:
#         game_id = file_path.stem

#         try:
#             pbp = pd.read_csv(file_path)
#             rows = build_rows_for_game(pbp, game_id, is_playoffs=True)

#             if not rows.empty:
#                 all_rows.append(rows)

#         except Exception as e:
#             print(f"Failed to process {game_id}: {e}")
#             continue

#     if not all_rows:
#         raise ValueError("No training rows were created. Check your play-by-play files.")

#     testing_df = pd.concat(all_rows, ignore_index=True)

#     # Remove missing and infinite values.
#     testing_df = testing_df.replace([np.inf, -np.inf], np.nan)
#     testing_df = testing_df.dropna()

#     testing_df.to_csv(TESTING_FILE, index=False)

#     print(f"Saved {len(testing_df)} testing rows to {TESTING_FILE}")
#     print(testing_df.head())

#     return testing_df


if __name__ == "__main__":
    # build_training_dataset()
    build_training_dataset()