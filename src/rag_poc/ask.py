"""1 問をどちらかの方式で答える CLI。

    uv run python -m rag_poc.ask --method pageindex "質問"
"""

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["vector", "pageindex"], required=True)
    parser.add_argument("question")
    args = parser.parse_args()
    if args.method == "vector":
        from rag_poc.vector_rag import ask
    else:
        from rag_poc.pageindex_rag import ask
    json.dump(ask(args.question).to_dict(), sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    main()
