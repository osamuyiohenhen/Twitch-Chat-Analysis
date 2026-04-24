import pandas as pd
import os
import random

# CONFIG
INPUT_FILE = "twitch_chat_labels.csv"
OUTPUT_FILE = "labeled_data_v2.csv"


def main():
    # 1. Load Raw Data
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    # Engine='python' handles messy chat logs better
    df = pd.read_csv(
        INPUT_FILE,
        header=None,
        usecols=[0, 1],
        names=["channel", "message"],
        on_bad_lines="skip",
        engine="python",
    )
    total_rows = len(df)

    # 2. Load Existing Progress
    if os.path.exists(OUTPUT_FILE):
        df_labeled = pd.read_csv(OUTPUT_FILE)
        if "original_index" in df_labeled.columns:
            seen_indices = set(df_labeled["original_index"].tolist())
        else:
            seen_indices = set()
        print(f"Loaded {len(seen_indices)} labeled rows.")
    else:
        # Create new file with these columns
        df_labeled = pd.DataFrame(
            columns=["channel", "message", "label", "original_index"]
        )
        seen_indices = set()
        df_labeled.to_csv(OUTPUT_FILE, index=False)

    print("\n--- Randomized Labeler ---")
    print("KEYS: [1] Negative  [2] Neutral  [3] Positive")
    print("      [s] Skip (Garbage)  [q] Quit")
    print(f"Pool Size: {total_rows} messages")
    print("-------------------------------------------------------")

    new_rows = []

    try:
        while True:
            # 3. Filter out rows we have already seen/skipped
            available_indices = list(set(df.index) - seen_indices)

            if not available_indices:
                print("All messages have been processed.")
                break

            # 4. PICK A RANDOM MESSAGE
            random_idx = random.choice(available_indices)
            row = df.loc[random_idx]

            channel_name = str(row["channel"]).strip()
            msg_text = str(row["message"]).strip()

            # Auto-skip empty stuff
            if len(msg_text) < 1 or msg_text.lower() == "nan":
                seen_indices.add(random_idx)
                continue

            # 5. Display
            print(f"\n[{channel_name}]")
            print(f"{msg_text}")

            # 6. Get Input
            while True:
                choice = input("Label? > ").lower()

                match choice:
                    case "q":
                        raise KeyboardInterrupt
                    
                    case "s":
                        print("-> Skipped")
                        seen_indices.add(random_idx) 
                        break
                    
                    case "1":
                        new_rows.append({
                            "channel": channel_name,
                            "message": msg_text,
                            "label": 0,
                            "original_index": random_idx,
                        })
                        print("-> Negative")
                        seen_indices.add(random_idx)
                        break
                    
                    case "2":
                        new_rows.append({
                            "channel": channel_name,
                            "message": msg_text,
                            "label": 1,
                            "original_index": random_idx,
                        })
                        print("-> Neutral")
                        seen_indices.add(random_idx)
                        break
                    
                    case "3":
                        new_rows.append({
                            "channel": channel_name,
                            "message": msg_text,
                            "label": 2,
                            "original_index": random_idx,
                        })
                        print("-> Positive")
                        seen_indices.add(random_idx)
                        break
                    
                    case _:
                        print("Invalid. Use 1, 2, 3, or s (skip).")

            # 7. Auto-save every 5 LABELED rows
            if len(new_rows) >= 5:
                pd.DataFrame(new_rows).to_csv(
                    OUTPUT_FILE, mode="a", header=False, index=False
                )
                new_rows = []
                print(f"-- Saved (Progress: {len(seen_indices)} processed) --")

    except KeyboardInterrupt:
        print("\nPausing...")

    # Final Save
    if new_rows:
        pd.DataFrame(new_rows).to_csv(OUTPUT_FILE, mode="a", header=False, index=False)
        print("Final batch saved.")


if __name__ == "__main__":
    main()
