import pandas as pd
from pathlib import Path

# Change these paths
input_folder = Path("play_by_play")
output_folder = Path("play_by_play_modified")

output_folder.mkdir(exist_ok=True)

for csv_path in input_folder.glob("*.csv"):
    print(f"Processing {csv_path.name}...")

    df = pd.read_csv(csv_path)

    # Make sure required columns exist
    required_cols = ["scoreHome", "scoreAway"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        print(f"Skipped {csv_path.name}: missing columns {missing}")
        continue

    # Fill scoreHome and scoreAway using previous known values
    df["scoreHome"] = df["scoreHome"].ffill().fillna(0).astype(int)
    df["scoreAway"] = df["scoreAway"].ffill().fillna(0).astype(int)

    # Use existing total column if present
    if "totalPoints" in df.columns:
        total_col = "totalPoints"
    elif "pointsTotal" in df.columns:
        total_col = "pointsTotal"
    else:
        total_col = "totalPoints"

    # Update total points
    df[total_col] = df["scoreHome"] + df["scoreAway"]

    # Save to output folder with same filename
    output_path = output_folder / csv_path.name
    df.to_csv(output_path, index=False)

print("Done.")