# run.py
import argparse
from primary import load_model, start_backend  # Imports from your primary.py

if __name__ == "__main__":
    # 1. Accept channel name from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=str, required=True)
    args = parser.parse_args()

    print(f"--- LAUNCHING BACKEND FOR {args.channel} ---")

    # 2. Load the heavy model (happens in this separate process)
    # We pass 'None' for the queue because we are using CSV mode.
    clf = load_model()
    start_backend(args.channel, None, clf)
