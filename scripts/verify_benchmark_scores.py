import json
import argparse


def verify_scores(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    scores = data["texify"]["scores"]

    if scores["bleu"] <= 0.6 or scores["meteor"] <= 0.6 or scores["edit"] > 0.2:
        print(scores)
        raise ValueError("Scores do not meet the required threshold")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify benchmark scores")
    parser.add_argument("file_path", type=str, help="Path to the json file")
    args = parser.parse_args()
    verify_scores(args.file_path)
