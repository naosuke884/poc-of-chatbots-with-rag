"""1 問をどちらかの方式で答える CLI。

    uv run python -m rag_poc.ask --method pageindex "質問"
"""

import argparse
import json
import sys

from rag_poc.methods import METHODS, pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(METHODS), required=True)
    parser.add_argument("question")
    args = parser.parse_args()
    json.dump(pipeline(args.method)(args.question).to_dict(), sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    main()
