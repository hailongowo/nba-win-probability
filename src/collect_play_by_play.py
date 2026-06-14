import time
import requests
import pandas as pd
from nba_api.stats.endpoints import playbyplayv3

from config import REGULAR_GAMES_FILE, PLAYOFF_GAMES_FILE, RAW_PLAYOFF_PBP_DIR, RAW_REGULAR_PBP_DIR


def retry(func, retries=3, delay=30):
    def retry_wrapper(*args, **kwargs):
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(
                    f"{func.__name__} failed "
                    f"({attempt + 1}/{retries}): {e}"
                )

                if attempt == retries - 1:
                    raise

                time.sleep(delay)

    return retry_wrapper

@retry
def download_game(game_id, output_path):
    pbp = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=60)
    df = pbp.get_data_frames()[0]
    df.to_csv(output_path, index=False)

def collect_play_by_play(max_games: int | None = None, is_playoffs = False) -> None:
    """
    Download play-by-play data for every game in games.csv.

    Parameters
    ----------
    max_games:
        Optional limit for testing.
        Example: max_games=20 downloads only 20 games.
    is_playoffs:
        Optional filter for playoff games.
        Example: is_playoffs=True downloads only playoff games.
    """
    GAMES_FILE = PLAYOFF_GAMES_FILE if is_playoffs else REGULAR_GAMES_FILE
    RAW_PBP_DIR = RAW_PLAYOFF_PBP_DIR if is_playoffs else RAW_REGULAR_PBP_DIR

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
            download_game(game_id, output_path)
            print(f"Saved {output_path}")

            # Sleep to reduce the chance of getting rate limited.
            # time.sleep(1.5)

        except Exception as e:
            print(f"Failed to download {game_id}: {e}")
            # Continue instead of crashing the whole collection process.
            continue


if __name__ == "__main__":
    # Use a small number first for testing.
    # After it works, change max_games=None.
    collect_play_by_play(max_games=None, is_playoffs=False)
    collect_play_by_play(max_games=None, is_playoffs=True)