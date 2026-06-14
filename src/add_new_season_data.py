from nba_api.stats.endpoints import leaguegamelog, playbyplayv3
from collect_games import convert_team_rows_to_games
from build_features import build_rows_for_game

import time
import pandas as pd
from pathlib import Path
import numpy as np


new_seasons = ["2025-26"]
output_folder = Path("data/raw/formatting_out")
output_folder.mkdir(exist_ok=True)


def get_new_games (seasons: list[str], season_type: str = "Regular Season"):
    all_rows = []

    for season in seasons:
        print(f"Collecting games for season {season}...")

        result = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star=season_type,
            player_or_team_abbreviation="T",
            timeout=60,
        )

        df = result.get_data_frames()[0]
        df["SEASON"] = season
        df["SEASON_TYPE"] = season_type 
        processed_df = convert_team_rows_to_games(df, season=season, season_type=season_type)
        all_rows.append(processed_df)

        time.sleep(1.5)

    games = pd.concat(all_rows, ignore_index=True)
    # games.to_csv(output_folder / "games.csv", index=False)
    return games

def collect_process_pbp (game_id, games):
    # print (game_id in games["GAME_ID"].values)
    if game_id not in games["GAME_ID"].values:
        print(f"Game ID {game_id} not found in games dataset. Skipping.")
        return None
    
    home_team = games.loc[games["GAME_ID"] == game_id]["HOME_TEAM_ID"].values[0]

    pbp = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=60)
    raw_df = pbp.get_data_frames()[0]
    raw_df.to_csv(output_folder / "temp_pbp.csv", index=False)

    df = pd.read_csv(output_folder / "temp_pbp.csv")

    is_playoffs = 0
    if int(game_id) > 40000000:
        is_playoffs = 1
    
    df["is_playoffs"] = is_playoffs
    df["posession"] = 0

    values = {
        "scoreHome": 0,
        "scoreAway": 0,
    }
    df.fillna(values, inplace=True)

    # print(df["scoreAway"].values)

    for index, row in df.iterrows():
        if index == 0:
            continue

        if row["scoreHome"] == 0 and row["scoreAway"] == 0:
            df.at[index, "scoreHome"] = df.at[index - 1, "scoreHome"]
            df.at[index, "scoreAway"] = df.at[index - 1, "scoreAway"]
        df.at[index, "pointsTotal"] = df.at[index, "scoreHome"] + df.at[index, "scoreAway"]

        action_by = 1 if row["teamId"] == home_team else 0
        if row["actionType"] in ["Made Shot", "Free Throw", "Rebound"]:
            df.at[index, "posession"] = action_by
        elif row["actionType"] in ["Turnover", "Violation", "Foul"]:
            df.at[index, "posession"] = 1 - action_by

    print (game_id)
    rows = build_rows_for_game(df, game_id)

    output_path = output_folder / "test.csv"
    rows.to_csv(output_path, index=False)
    return rows

# def add_training_data(games, game_id):
#     all_rows = []
#     # for game_id in games["GAME_ID"].values:
#     #     try:
#     #         pbp = collect_process_pbp(game_id, games)
#     #         rows = build_rows_for_game(pbp, game_id)
#     #         if not rows.empty:
#     #             all_rows.append(rows)
#     #     except Exception as e:
#     #         print(f"Failed to process {game_id}: {e}")
#     #         continue

#     try:
#         pbp = collect_process_pbp(game_id, games)
#         rows = build_rows_for_game(pbp, game_id)
#         if not rows.empty:
#             all_rows.append(rows)
#     except Exception as e:
#         print(f"Failed to process {game_id}: {e}")
#         return None
    
#     if not all_rows:
#         raise ValueError("No training rows were created. Check your play-by-play files.")

#     training_df = pd.concat(all_rows, ignore_index=True)

#     # Remove missing and infinite values.
#     training_df = training_df.replace([np.inf, -np.inf], np.nan)
#     training_df = training_df.dropna()

#     training_df.to_csv(output_folder / "test.csv", index=False)

#     print(f"Saved {len(training_df)} training rows to {output_folder / "test.csv"}")
#     print(training_df.head())

#     return training_df


games = get_new_games(new_seasons, "Regular Season")
collect_process_pbp("0022500001", games)
# add_training_data(games, "0022500001")