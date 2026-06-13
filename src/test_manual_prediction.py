import joblib
import pandas as pd
import matplotlib.pyplot as plt

from config import LOGISTIC_MODEL_FILE, TESTING_FILE


def plot_win_probability(game_id: str) -> None:
    """
    Plot home/away win probability for one game.

    The TESTING_FILE should contain rows created from play-by-play data.
    Each row should represent one game state.
    """

    # Load trained model artifact.
    artifact = joblib.load(LOGISTIC_MODEL_FILE)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    # Load testing data.
    df = pd.read_csv(TESTING_FILE)

    # Make sure game_id is compared as string.
    df["game_id"] = df["game_id"].astype(str)
    game_id = str(game_id)

    # Keep only rows from the selected game.
    game_df = df[df["game_id"] == game_id].copy()

    if game_df.empty:
        raise ValueError(f"No rows found for game_id={game_id}")

    # Sort from start of game to end of game.
    # seconds_remaining is high at the start and low near the end.
    game_df = game_df.sort_values("seconds_remaining", ascending=False)

    # Select model input features.
    X = game_df[feature_columns]

    # Predict probability for every row.
    game_df["home_win_probability"] = model.predict_proba(X)[:, 1]
    game_df["away_win_probability"] = 1 - game_df["home_win_probability"]

    # Create elapsed time for x-axis.
    # Regulation game = 2880 seconds.
    # If you include overtime, this is a simplified version.
    game_df["seconds_elapsed"] = 2880 - game_df["seconds_remaining"]
    game_df["minutes_elapsed"] = game_df["seconds_elapsed"] / 60

    # Plot graph.
    plt.figure(figsize=(12, 6))

    plt.plot(
        game_df["minutes_elapsed"],
        game_df["home_win_probability"] * 100,
        label="Home win probability",
        linewidth=2,
    )

    # plt.plot(
    #     game_df["minutes_elapsed"],
    #     game_df["away_win_probability"] * 100,
    #     label="Away win probability",
    #     linewidth=2,
    # )

    plt.axhline(50, linestyle="--", linewidth=1, label="50%")

    plt.title(f"Win Probability Graph - Game {game_id}")
    plt.xlabel("Game time elapsed")
    plt.ylabel("Win probability (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()

    # Optional: print final prediction row.
    final_row = game_df.iloc[-1]

    print("Final model state:")
    print(f"Home win probability: {final_row['home_win_probability']:.2%}")
    print(f"Away win probability: {final_row['away_win_probability']:.2%}")
    print(f"Actual home_win label: {final_row['home_win']}")


if __name__ == "__main__":
    plot_win_probability("22500001")
    # df = pd.read_csv(TESTING_FILE)
    # print (df["game_id"].unique())