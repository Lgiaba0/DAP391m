import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.recommend_pipeline import recommend


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="Tim phong khach san Da Nang cho 2 nguoi, gan bien, 1-2 trieu, co ho boi")
    return parser.parse_args()


def main():
    args = parse_args()
    response = recommend(args.query)
    print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
