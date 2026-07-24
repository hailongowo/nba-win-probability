import pandas as pd

from config import RAW_GAMES_DIR, GAMES_FILE

def convert_team_rows_to_games():
    """
    Convert LeagueGameLog's two team rows per game into one game-level row.
    """
    raw_games = []

    for file in RAW_GAMES_DIR.glob('*.csv'):
        df = pd.read_csv(file)
        raw_games.append(df)

    raw_games = pd.concat(raw_games, ignore_index=True)

    games = []

    for game_id, game_rows in raw_games.groupby("GAME_ID"):
        home_rows = game_rows[
            game_rows["MATCHUP"].str.contains("vs.", regex=False, na=False)
        ]

        away_rows = game_rows[
            game_rows["MATCHUP"].str.contains("@", regex=False, na=False)
        ]


        # is_neutral_site = 0
        if len(home_rows) != 1 or len(away_rows) != 1:
            print(f"Skipping {game_id}: could not identify home and away teams")
            # is_neutral_site = 1
            continue

        home = home_rows.iloc[0]
        away = away_rows.iloc[0]

        games.append(
            {
                "GAME_ID": str(game_id).zfill(10),
                "HOME_TEAM_ID": int(home["TEAM_ID"]),
                "HOME_TEAM": home["TEAM_ABBREVIATION"],
                "AWAY_TEAM_ID": int(away["TEAM_ID"]),
                "AWAY_TEAM": away["TEAM_ABBREVIATION"],
                "HOME_POINTS": int(home["PTS"]),
                "AWAY_POINTS": int(away["PTS"]),
                "HOME_WIN": int(home["WL"] == "W"),
            }
        )

    pd.DataFrame(games).to_csv(GAMES_FILE, index = False)

convert_team_rows_to_games()