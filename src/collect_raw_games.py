import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

from config import RAW_GAMES_DIR

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

        # Be polite to NBA.com servers.
        time.sleep(1.5)

if __name__ == "__main__":
    # Start with fewer seasons while testing.
    # Later, add more seasons for better model quality.
    seasons = [
        "2020-21",
        "2021-22",
        "2022-23",
        "2023-24",
        "2024-25",
        "2025-26",
    ]

    collect_games(seasons=seasons, season_type="Playoffs")
    collect_games(seasons=seasons, season_type="Regular Season")
