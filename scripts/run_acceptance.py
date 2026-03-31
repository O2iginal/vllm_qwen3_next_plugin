#!/usr/bin/env python3
"""
串联当前插件的最小验收链路。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, cmd: list[str], extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    print(f"=== {name} ===")
    print("cmd:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18011)
    parser.add_argument("--model", default="qwen3-next-plugin-test")
    parser.add_argument(
        "--skip-runtime",
        action="store_true",
        help="只检查本地导入与 registry，不检查服务端生成。",
    )
    args = parser.parse_args()

    run_step("pytest", [sys.executable, "-m", "pytest", "tests", "-q"])
    run_step("plugin_registry", [sys.executable, str(REPO_ROOT / "scripts" / "check_plugin_import.py")])

    if not args.skip_runtime:
        run_step(
            "generation_validation",
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "validate_generation.py"),
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--model",
                args.model,
            ],
        )

    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
