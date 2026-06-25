#!/usr/bin/env python3
"""
vldrive_pipeline.py

Monitors outputs/latest_bev_grid.jpg produced by eval_nuscenes.py.
Each time the file is updated (new mtime), encodes it to base64 and
asks a local Ollama vision model to issue a structured driving decision.

Usage:
    python vldrive_pipeline.py
    python vldrive_pipeline.py --image_path simple_bev/outputs/latest_bev_grid.jpg
    python vldrive_pipeline.py --model qwen2-vl:7b --poll_interval 0.3
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

# Core instruction exactly as specified — output-format clause appended so the
# model returns a machine-parseable DECISION token.
_SYSTEM_PROMPT = (
    "Analyze this top-down Bird's Eye View map. "
    "The red shapes represent vehicles. "
    "Determine if there is a blocking hazard ahead and decide if the vehicle should: "
    "[PROCEED, SLOW_DOWN, YIELD, STOP].\n\n"
    "The image shows two panels side-by-side: "
    "LEFT = model prediction, RIGHT = ground truth. "
    "The ego vehicle is positioned at the centre-bottom; "
    "forward direction is toward the top of the image.\n\n"
    "Reply in EXACTLY this format (no extra lines):\n"
    "REASONING: <one sentence explaining the scene>\n"
    "DECISION: <PROCEED | SLOW_DOWN | YIELD | STOP>"
)

_VALID_DECISIONS = {"PROCEED", "SLOW_DOWN", "YIELD", "STOP"}

# ---------------------------------------------------------------------------
# Decision labels (printed to stdout)
# ---------------------------------------------------------------------------
_LABELS = {
    "PROCEED":   "[  PROCEED  ] Path ahead clear.",
    "SLOW_DOWN": "[ SLOW DOWN ] Vehicles ahead — reduce speed.",
    "YIELD":     "[   YIELD   ] Cross-traffic entering lane.",
    "STOP":      "[   STOP    ] Immediate blocking hazard.",
}


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def encode_image(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Ollama API
# ---------------------------------------------------------------------------

def query_ollama(b64_image: str, model: str, base_url: str, timeout: int) -> str:
    """POST to /api/generate and return the response text.

    Raises:
        urllib.error.URLError  — Ollama is not reachable (connection refused, DNS, etc.)
        RuntimeError           — Ollama is reachable but returned an HTTP error
                                 (e.g. model not pulled, bad request).
    """
    payload = json.dumps({
        "model": model,
        "prompt": _SYSTEM_PROMPT,
        "images": [b64_image],
        "stream": False,
        "options": {
            "temperature": 0.1,   # low temperature → consistent decisions
            "num_predict": 120,   # cap token budget; we only need two lines
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body.get("response", "")
    except urllib.error.HTTPError as exc:
        # Read the body — Ollama puts the real reason here (e.g. "model not found").
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            msg = json.loads(raw).get("error", raw)
        except Exception:
            msg = raw or exc.reason
        raise RuntimeError(f"HTTP {exc.code} from Ollama: {msg}") from None


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(text: str) -> tuple:
    """Return (reasoning: str, decision: str) from model output."""
    reasoning = ""
    decision = "UNKNOWN"
    for line in text.strip().splitlines():
        upper = line.strip().upper()
        if line.strip().startswith("REASONING:"):
            reasoning = line.strip()[len("REASONING:"):].strip()
        elif line.strip().startswith("DECISION:"):
            token = line.strip()[len("DECISION:"):].strip().upper().rstrip(".")
            if token in _VALID_DECISIONS:
                decision = token
            else:
                # Fallback: scan the token for any valid keyword
                for d in _VALID_DECISIONS:
                    if d in token:
                        decision = d
                        break
        # Also catch bare decision words on a line of their own
        elif upper.rstrip(".") in _VALID_DECISIONS and decision == "UNKNOWN":
            decision = upper.rstrip(".")
    return reasoning, decision


# ---------------------------------------------------------------------------
# Main monitor loop
# ---------------------------------------------------------------------------

def monitor(image_path: str, model: str, base_url: str,
            poll_interval: float, timeout: int, log_path: str | None) -> None:

    abs_path = os.path.abspath(image_path)
    print(f"[VLDrive] image   : {abs_path}")
    print(f"[VLDrive] model   : {model}")
    print(f"[VLDrive] ollama  : {base_url}")
    print(f"[VLDrive] poll    : {poll_interval}s")
    if log_path:
        print(f"[VLDrive] log     : {log_path}")
    print("-" * 64)

    log_fh = open(log_path, "a") if log_path else None
    last_mtime: float | None = None

    try:
        while True:
            # ---- wait for file to appear / change -------------------------
            if not os.path.isfile(abs_path):
                time.sleep(poll_interval)
                continue

            mtime = os.stat(abs_path).st_mtime
            if mtime == last_mtime:
                time.sleep(poll_interval)
                continue

            last_mtime = mtime
            ts = datetime.fromtimestamp(mtime).strftime("%H:%M:%S.%f")[:-3]
            print(f"\n[{ts}] New BEV frame — querying {model} ...", flush=True)

            # ---- encode and query -----------------------------------------
            try:
                b64 = encode_image(abs_path)
            except OSError as exc:
                print(f"  [ERROR] Cannot read image: {exc}")
                time.sleep(poll_interval)
                continue

            t0 = time.monotonic()
            try:
                raw_text = query_ollama(b64, model, base_url, timeout)
            except urllib.error.URLError as exc:
                print(f"  [ERROR] Ollama unreachable ({base_url}): {exc.reason}")
                print("         Is 'ollama serve' running?")
                time.sleep(poll_interval)
                continue
            except Exception as exc:
                print(f"  [ERROR] Query failed: {exc}")
                time.sleep(poll_interval)
                continue
            latency = time.monotonic() - t0

            # ---- parse and display ----------------------------------------
            reasoning, decision = parse_response(raw_text)
            label = _LABELS.get(decision, f"[ {decision:^9} ]")

            print(f"  Reasoning : {reasoning or '(not parsed)'}")
            print(f"  Decision  : {label}")
            print(f"  Latency   : {latency:.2f}s")
            print("-" * 64, flush=True)

            # ---- optional log ---------------------------------------------
            if log_fh:
                entry = {
                    "timestamp": ts,
                    "mtime": mtime,
                    "decision": decision,
                    "reasoning": reasoning,
                    "latency_s": round(latency, 3),
                    "raw": raw_text.strip(),
                }
                log_fh.write(json.dumps(entry) + "\n")
                log_fh.flush()

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n[VLDrive] Stopped by user.")
    finally:
        if log_fh:
            log_fh.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="VLDrive: BEV image → Ollama qwen2-vl → driving decision"
    )
    ap.add_argument(
        "--image_path",
        default="simple_bev/outputs/latest_bev_grid.jpg",
        help="Path to the BEV JPEG written by eval_nuscenes.py",
    )
    ap.add_argument(
        "--ollama_url",
        default="http://localhost:11434",
        help="Base URL of the local Ollama server",
    )
    ap.add_argument(
        "--model",
        default="qwen2.5vl:7b",
        help="Ollama vision model tag",
    )
    ap.add_argument(
        "--poll_interval",
        type=float,
        default=0.5,
        help="File-poll interval in seconds (default 0.5)",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP request timeout for Ollama in seconds (default 60)",
    )
    ap.add_argument(
        "--log",
        default=None,
        metavar="FILE",
        help="Optional path to append JSONL decision log (e.g. decisions.jsonl)",
    )
    args = ap.parse_args()

    monitor(
        image_path=args.image_path,
        model=args.model,
        base_url=args.ollama_url,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
        log_path=args.log,
    )


if __name__ == "__main__":
    main()
