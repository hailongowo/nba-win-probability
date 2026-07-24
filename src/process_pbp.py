import pandas as pd
from pathlib import Path
from config import RAW_PBP_DIR, PBP_DIR, GAMES_FILE

# Change these paths
input_folder = RAW_PBP_DIR
output_folder = PBP_DIR

games = pd.read_csv(GAMES_FILE, dtype={"GAME_ID": str})
print (games.iloc[0]["GAME_ID"])

for csv_path in input_folder.glob("*.csv"):
    print(f"Processing {csv_path.name}...")
    game_id = csv_path.stem

    # print(game_id)

    if game_id not in games["GAME_ID"].values:
        print(f"Game ID {game_id} not found in games dataset. Skipping.")
        continue

    home_team = games.loc[games["GAME_ID"] == game_id]["HOME_TEAM_ID"].values[0]
    away_team = games.loc[games["GAME_ID"] == game_id]["AWAY_TEAM_ID"].values[0]
    home_win = games.loc[games["GAME_ID"] == game_id]["HOME_WIN"].values[0]
    # print(f"Home team: {home_team}, Away team: {away_team}")

    df = pd.read_csv(csv_path)
    df["posession"] = 0  # Fill missing possession values with 0 (away team)
    df["is_playoffs"] = 1 if game_id[0:3] == "004" else 0
    df["home_win"] = home_win

    values = {
        "scoreHome": 0,
        "scoreAway": 0,
    }
    df.fillna(values, inplace=True)

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

        


    # Save to output folder with same filename
    output_path = output_folder / csv_path.name
    df.to_csv(output_path, index=False)

print("Done.")