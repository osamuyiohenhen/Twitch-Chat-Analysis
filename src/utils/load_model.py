import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM

parser = argparse.ArgumentParser()
parser.add_argument("--repo", default="muyihenhen/twitch-roberta-base")
parser.add_argument("--dest", default="./twitch-roberta-base")
args = parser.parse_args()

AutoTokenizer.from_pretrained(args.repo).save_pretrained(args.dest)
AutoModelForMaskedLM.from_pretrained(args.repo).save_pretrained(args.dest)