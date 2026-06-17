#!/usr/bin/env python3
import sys

import config


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python research.py \"your query here\"", file=sys.stderr)
        sys.exit(1)

    query = sys.argv[1].strip()

    if not query:
        print("Error: query cannot be empty", file=sys.stderr)
        sys.exit(1)

    if len(query) > config.MAX_QUERY_LENGTH:
        print(
            f"Error: query exceeds maximum length of {config.MAX_QUERY_LENGTH} characters",
            file=sys.stderr,
        )
        sys.exit(1)

    config.validate()

    from pipeline import run_pipeline
    answer = run_pipeline(query)
    print(answer)


if __name__ == "__main__":
    main()
