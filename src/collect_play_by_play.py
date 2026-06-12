import time
import pandas as pd
from nba_api.stats.endpoints import playbyplayv3

from config import GAMES_FILE, RAW_PBP_DIR


def collect_play_by_play(max_games: int | None = None) -> None:
    """
    Download play-by-play data for every game in games.csv.

    Parameters
    ----------
    max_games:
        Optional limit for testing.
        Example: max_games=20 downloads only 20 games.
    """

    games = pd.read_csv(GAMES_FILE)

    if max_games is not None:
        games = games.head(max_games)

    for index, row in games.iterrows():
        game_id = str(row["GAME_ID"]).zfill(10)
        output_path = RAW_PBP_DIR / f"{game_id}.csv"

        # Skip if already downloaded.
        # This is important because downloading all games can take a long time.
        if output_path.exists():
            print(f"Skipping {game_id}, already exists")
            continue

        print(f"Downloading play-by-play for {game_id} ({index + 1}/{len(games)})")

        try:
            pbp = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=60)
            df = pbp.get_data_frames()[0]

            df.to_csv(output_path, index=False)
            print(f"Saved {output_path}")

            # Sleep to reduce the chance of getting rate limited.
            time.sleep(1.5)

        except Exception as e:
            print(f"Failed to download {game_id}: {e}")
            # Continue instead of crashing the whole collection process.
            continue


if __name__ == "__main__":
    # Use a small number first for testing.
    # After it works, change max_games=None.
    collect_play_by_play(max_games=None)