#!/usr/bin/env python
from __future__ import annotations

import json

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
PAYLOAD = {
    "model": "qwen2.5:3b",
    "prompt": "请用一句话解释什么是管理。",
    "stream": False,
}
TIMEOUT = (10, 180)


def main() -> None:
    print(f"Request URL: {OLLAMA_URL}")
    print("Payload:")
    print(json.dumps(PAYLOAD, ensure_ascii=False, indent=2))
    print(f"Timeout: {TIMEOUT}")
    print()

    try:
        response = requests.post(
            OLLAMA_URL,
            json=PAYLOAD,
            timeout=TIMEOUT,
        )
    except requests.exceptions.Timeout:
        print("Error: request timed out.")
        print("Hint: the model may still be loading, or the Python HTTP call path is blocked/slow.")
        return
    except requests.exceptions.ConnectionError:
        print("Error: cannot connect to Ollama.")
        print("Hint: make sure Ollama is running and http://localhost:11434 is reachable.")
        return
    except requests.RequestException as exc:
        print(f"Error: request failed: {exc}")
        return

    print(f"HTTP status code: {response.status_code}")
    print("Raw response text:")
    print(response.text)
    print()

    try:
        data = response.json()
    except ValueError:
        print("Error: response is not valid JSON.")
        return

    answer = data.get("response")
    if answer:
        print("Parsed answer:")
        print(answer.strip())
    else:
        print("Parsed answer: <empty>")


if __name__ == "__main__":
    main()
