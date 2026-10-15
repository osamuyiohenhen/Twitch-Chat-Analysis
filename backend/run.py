# run.py
import argparse
from primary import load_model, start_backend  # Imports from primary.py

if __name__ == "__main__":
    # Accept channel name from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=str, required=True)
    args = parser.parse_args()

    print(f"--- Launching Backend for {args.channel} ---")

    # Load the heavy model (happens in this separate process)
    # pass 'None' for the queue because we are using CSV mode.
    clf = load_model()
    start_backend(args.channel, None, clf)
