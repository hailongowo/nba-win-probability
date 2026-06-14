import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

from config import RAW_GAMES_DIR, REGULAR_GAMES_FILE, PLAYOFF_GAMES_FILE

def convert_team_rows_to_games(
    team_rows: pd.DataFrame,
    season: str,
    season_type: str,
) -> pd.DataFrame:
    """
    Convert LeagueGameLog's two team rows per game into one game-level row.
    """

    games = []

    for game_id, game_rows in team_rows.groupby("GAME_ID"):
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
                "SEASON": season,
                "SEASON_TYPE": season_type,
                "GAME_ID": str(game_id).zfill(10),
                "GAME_DATE": home["GAME_DATE"],
                "HOME_TEAM_ID": int(home["TEAM_ID"]),
                "HOME_TEAM": home["TEAM_ABBREVIATION"],
                "AWAY_TEAM_ID": int(away["TEAM_ID"]),
                "AWAY_TEAM": away["TEAM_ABBREVIATION"],
                "HOME_POINTS": int(home["PTS"]),
                "AWAY_POINTS": int(away["PTS"]),
                "HOME_WIN": int(home["WL"] == "W"),
                # "IS_NEUTRAL_SITE": int(is_neutral_site),
            }
        )

    return pd.DataFrame(games)

regular_games = []
playoff_games = []

def collect_games(seasons: list[str], season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Collect NBA game logs for multiple seasons.

    Parameters
    ----------
    seasons:
        List of NBA season strings.
        Example: ["2022-23", "2023-24", "2024-25"]

    season_type:
        Usually "Regular Season" or "Playoffs".

    Returns
    -------
    pd.DataFrame
        A cleaned dataframe containing unique games.
    """

    all_rows = []

    for season in seasons:
        print(f"Collecting games for season {season}...")

        # LeagueGameLog returns one row per team per game.
        # That means each game usually appears twice:
        # one row for the home team and one row for the away team.
        result = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star=season_type,
            player_or_team_abbreviation="T",
            timeout=60,
        )

        df = result.get_data_frames()[0]

        # Save raw season file in case we need to debug later.
        raw_path = RAW_GAMES_DIR / f"league_game_log_{season.replace('-', '_')}_{season_type.replace(' ', '_')}.csv"
        df.to_csv(raw_path, index=False)

        df["SEASON"] = season
        df["SEASON_TYPE"] = season_type 
        processed_df = convert_team_rows_to_games(df, season=season, season_type=season_type)
        all_rows.append(processed_df)

        # Be polite to NBA.com servers.
        time.sleep(1.5)

    games = pd.concat(all_rows, ignore_index=True)

    if season_type == "Regular Season":
        regular_games.append(games)
        games.to_csv(REGULAR_GAMES_FILE, index=False)
        print(f"Saved {len(games)} unique games to {REGULAR_GAMES_FILE}")
    elif season_type == "Playoffs":
        playoff_games.append(games)
        games.to_csv(PLAYOFF_GAMES_FILE, index=False)
        print(f"Saved {len(games)} unique games to {PLAYOFF_GAMES_FILE}")

    return games


if __name__ == "__main__":
    # Start with fewer seasons while testing.
    # Later, add more seasons for better model quality.
    seasons = [
        "2015-16",
        "2016-17",
        "2017-18",
        "2018-19",
        "2019-20",
        "2020-21",
        "2021-22",
        "2022-23",
        "2023-24",
        "2024-25",
    ]

    collect_games(seasons=seasons, season_type="Playoffs")

    collect_games(seasons=seasons, season_type="Regular Season")

    # regular_games["is_playoffs"] = 0
    # playoff_games["is_playoffs"] = 1

    # all_games = pd.concat(
    #     [regular_games, playoff_games],
    #     ignore_index=True,
    # )

    # all_games = all_games.sort_values(["game_date", "game_id"])
    # all_games.to_csv("data/processed/all_games.csv", index=False)
