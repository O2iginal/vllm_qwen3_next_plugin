#!/usr/bin/env python3
"""
对本地 vLLM OpenAI API server 做最小自然语言生成验收。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


DEFAULT_CASES = [
    {
        "name": "zh_intro",
        "prompt": "你好，请用一句话介绍你自己。",
        "max_tokens": 16,
    },
    {
        "name": "en_math",
        "prompt": "What is 2 + 3? Answer briefly.",
        "max_tokens": 16,
    },
]


def request_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def is_readable_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    bad_markers = ["<unk><unk>", "\x00"]
    return not any(marker in stripped for marker in bad_markers)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18011)
    parser.add_argument("--model", default="qwen3-next-plugin-test")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/v1/completions"
    failed = False

    for case in DEFAULT_CASES:
        payload = {
            "model": args.model,
            "prompt": case["prompt"],
            "max_tokens": args.max_tokens or case["max_tokens"],
            "temperature": args.temperature,
        }
        try:
            result = request_json(url, payload)
        except urllib.error.URLError as exc:
            print(f"[FAIL] {case['name']}: request error: {exc}")
            failed = True
            continue

        choices = result.get("choices", [])
        text = choices[0].get("text", "") if choices else ""
        ok = is_readable_text(text)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case['name']}")
        print(f"prompt: {case['prompt']}")
        print(f"text: {text!r}")
        print()
        failed = failed or (not ok)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
