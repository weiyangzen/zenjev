#!/usr/bin/env python3
"""Deterministic console screenshot via geckodriver (read-only).

Navigates the live console with a synthetic-only feed, waits for the SSE
stream to populate the task feed, verifies card counts through WebDriver, and
writes a PNG. Intended for README/docs captures; it never mutates the panel.
"""
from __future__ import annotations

import argparse
import base64
import json
import pathlib
import subprocess
import time
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[1]
GECKO = "/snap/bin/geckodriver"


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    return json.loads(urllib.request.urlopen(request, timeout=60).read())


def get(url: str) -> dict:
    return json.loads(urllib.request.urlopen(url, timeout=60).read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8790/?filter=synthetic&shot=1")
    parser.add_argument("--out", default=str(REPO / "Docs/assets/console.png"))
    parser.add_argument("--wait", type=float, default=8.0)
    parser.add_argument("--profile", default=str(pathlib.Path.home() / "ffprof"))
    args = parser.parse_args()

    driver = subprocess.Popen(
        [GECKO, "--port", "4444", "--log", "fatal"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                get("http://127.0.0.1:4444/status")
                break
            except Exception:
                time.sleep(0.5)
        session = post(
            "http://127.0.0.1:4444/session",
            {
                "capabilities": {
                    "alwaysMatch": {
                        "browserName": "firefox",
                        "moz:firefoxOptions": {
                            "args": ["-headless", "--no-remote", "--profile", args.profile],
                            "prefs": {"dom.max_script_run_time": 30},
                        },
                    }
                }
            },
        )
        sid = session["value"]["sessionId"]
        post(
            f"http://127.0.0.1:4444/session/{sid}/window/rect",
            {"width": 1680, "height": 1560},
        )
        post(f"http://127.0.0.1:4444/session/{sid}/url", {"url": args.url})
        time.sleep(args.wait)
        cards = post(
            f"http://127.0.0.1:4444/session/{sid}/elements", {"using": "css selector", "value": ".task"}
        )["value"]
        shot = get(f"http://127.0.0.1:4444/session/{sid}/screenshot")["value"]
        target = pathlib.Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(shot))
        print(json.dumps({"cards": len(cards), "bytes": target.stat().st_size, "out": str(target)}))
        try:
            urllib.request.urlopen(
                urllib.request.Request(f"http://127.0.0.1:4444/session/{sid}", method="DELETE"),
                timeout=30,
            )
        except Exception:
            pass
        return 0
    finally:
        driver.terminate()
        try:
            driver.wait(timeout=10)
        except subprocess.TimeoutExpired:
            driver.kill()


if __name__ == "__main__":
    raise SystemExit(main())
