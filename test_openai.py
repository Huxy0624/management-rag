#!/usr/bin/env python
from __future__ import annotations

import os


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        print("OPENAI_API_KEY read successfully.")
    else:
        print("OPENAI_API_KEY not found.")


if __name__ == "__main__":
    main()
