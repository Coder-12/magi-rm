#!/usr/bin/env python3
"""
src/utils/logger.py — lightweight structured experiment logger.
"""

import os, json, datetime

class Logger:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.log_file = os.path.join(run_dir, "train.log")
        self.json_file = os.path.join(run_dir, "metrics.jsonl")
        with open(self.log_file, "a") as f:
            f.write(f"\n=== Run started {datetime.datetime.utcnow().isoformat()}Z ===\n")

    def log(self, msg: str):
        print(msg)
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")

    def log_metrics(self, step: int, metrics: dict):
        record = {"step": step, **metrics}
        with open(self.json_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Safe pretty-print: handle None and non-floats
        def fmt_val(v):
            if v is None:
                return "None"
            try:
                return f"{float(v):.4f}"
            except Exception:
                return repr(v)

        try:
            summary = " | ".join(f"{k}={fmt_val(v)}" for k, v in metrics.items())
        except Exception:
            # very defensive fallback
            summary = json.dumps(metrics)
        print(f"[LOG] step={step} | {summary}")
