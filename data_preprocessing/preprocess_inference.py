#!/usr/bin/env python3
"""
preprocessing_inference.py

Real-time preprocessing helper for inference service.

- Exposes FeatureProcessor and FeatureManager classes to maintain per-key state
  (machine_name, program_name, tool).
- For each incoming event (single JSON), compute features **based on past state only**
  then update the state.
- Designed to be imported into your inference service (FastAPI/Flask/async worker).
- Also has a CLI for local testing / batch processing of a few events.

Important:
- This module is stateful. In a production inference server you must keep a single
  long-lived FeatureManager instance (do NOT create a new manager per HTTP request),
  so expanding/rolling stats persist across requests.
- Optionally persist manager state to disk (save_states/load_states) when gracefully
  shutting down / starting up the service.

  ********************
- Instantiate only one FeatureManager per unique M-P-T key to maintain correct state 
  across sequential requests within each key.
  ******************** 


Usage Notes: 
- # Process a single event from stdin
echo '{
"machine_name":"M1",
"program_name":"P1",
"tool":"T1",
"timestamp":"2025-09-17T10:00:00Z",
"load":10,
"feed":0.2,
"speed":2000,
"cumulative_load":5
}' \
  | python preprocessing_inference.py -i - --state-out state.json

# Restart later and keep state:
python preprocessing_inference.py -i events.ndjson --state-in state.json --state-out state.json
"""

import argparse
import json
import os
import sys
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Tuple

ROLLING_WINDOWS = [5, 10, 25, 50]
COLUMNS = ["load", "feed", "speed", "cumulative_load"]

FEATURE_SCHEMA: List[str] = []
FEATURE_SCHEMA.extend(COLUMNS)
for c in COLUMNS:
    FEATURE_SCHEMA.append(f"d_{c}")
    FEATURE_SCHEMA.append(f"dd_{c}")
for c in COLUMNS:
    FEATURE_SCHEMA.append(f"avg_{c}_till_now")
    FEATURE_SCHEMA.append(f"max_{c}_till_now")
for c in COLUMNS:
    for w in ROLLING_WINDOWS:
        FEATURE_SCHEMA.append(f"avg_{c}_last_{w}")
        FEATURE_SCHEMA.append(f"std_{c}_last_{w}")
        FEATURE_SCHEMA.append(f"max_{c}_last_{w}")

def parse_timestamp(ts: str) -> int:
    if ts is None:
        raise ValueError("timestamp missing")
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

class FeatureProcessor:
    """Stateful processor for a single key (machine-program-tool)."""

    def __init__(self):
        self.last_ts = None
        self.last_vals = {c: None for c in COLUMNS}
        self.last_derivs = {c: 0.0 for c in COLUMNS}
        self.sums = {c: 0.0 for c in COLUMNS}
        self.counts = {c: 0 for c in COLUMNS}
        self.maxes = {c: float("-inf") for c in COLUMNS}
        self.windows = {c: {w: deque(maxlen=w) for w in ROLLING_WINDOWS} for c in COLUMNS}

    def process(self, event: Dict) -> Dict:
        """
        Compute features for the incoming event (based on past state), update the
        state with the current event, and return a result dict containing:
          - duration (seconds)
          - features (list in FEATURE_SCHEMA order)
        """
        # compute summary info based on prior state
        ts = parse_timestamp(event["timestamp"])
        duration = (ts - self.last_ts) / 1000.0 if self.last_ts is not None else 0.0
        vals = {c: safe_float(event.get(c, 0.0)) for c in COLUMNS}

        # derivatives
        d = {}
        dd = {}
        for c in COLUMNS:
            prev = self.last_vals[c]
            prev_d = self.last_derivs[c]
            if prev is None or duration == 0.0:
                d[c] = 0.0
                dd[c] = 0.0
            else:
                d[c] = (vals[c] - prev) / duration
                dd[c] = (d[c] - prev_d) / duration

        # expanding (till now)
        avg_till = {}
        max_till = {}
        for c in COLUMNS:
            if self.counts[c] == 0:
                avg_till[c] = 0.0
                max_till[c] = 0.0
            else:
                avg_till[c] = self.sums[c] / self.counts[c]
                max_till[c] = self.maxes[c]

        # rolling (till now)
        rolling = {}
        for c in COLUMNS:
            for w, q in self.windows[c].items():
                if len(q) == 0:
                    mean, std, mx = 0.0, 0.0, 0.0
                else:
                    mean = sum(q) / len(q)
                    var = sum((x - mean) ** 2 for x in q) / len(q)
                    std = var ** 0.5
                    mx = max(q)
                rolling[(c, w)] = (mean, std, mx)

        # assemble feature vector in schema order
        feature_vector = []
        for f in FEATURE_SCHEMA:
            if f in COLUMNS:
                feature_vector.append(vals[f])
            elif f.startswith("d_"):
                feature_vector.append(d[f[2:]])
            elif f.startswith("dd_"):
                feature_vector.append(dd[f[3:]])
            elif f.endswith("_till_now") and f.startswith("avg_"):
                c = f.split("_")[1]
                feature_vector.append(avg_till[c])
            elif f.endswith("_till_now") and f.startswith("max_"):
                c = f.split("_")[1]
                feature_vector.append(max_till[c])
            elif "_last_" in f:
                parts = f.split("_")
                c = parts[1]
                w = int(parts[-1])
                mean, std, mx = rolling[(c, w)]
                if f.startswith("avg_"):
                    feature_vector.append(mean)
                elif f.startswith("std_"):
                    feature_vector.append(std)
                elif f.startswith("max_"):
                    feature_vector.append(mx)
            else:
                feature_vector.append(0.0)

        # --- update state with current event for future calls ---
        for c in COLUMNS:
            self.sums[c] += vals[c]
            self.counts[c] += 1
            self.maxes[c] = max(self.maxes[c], vals[c])
            for w in ROLLING_WINDOWS:
                self.windows[c][w].append(vals[c])

        # update last_ts, last_vals, last_derivs
        prev_ts = self.last_ts
        prev_vals = self.last_vals.copy()
        self.last_ts = ts
        for c in COLUMNS:
            # compute derivative to store as last_derivs (for next dd computation)
            if prev_vals[c] is None or prev_ts is None:
                self.last_derivs[c] = 0.0
            else:
                dt = (ts - prev_ts) / 1000.0
                self.last_derivs[c] = (vals[c] - prev_vals[c]) / dt if dt != 0 else 0.0
            self.last_vals[c] = vals[c]

        return {
            "timestamp": event["timestamp"],
            "duration": duration,
            "features": feature_vector,
        }

    def to_serializable(self) -> Dict:
        """Return a JSON-serializable snapshot of the state (for persistence)."""
        return {
            "last_ts": self.last_ts,
            "last_vals": self.last_vals,
            "last_derivs": self.last_derivs,
            "sums": self.sums,
            "counts": self.counts,
            "maxes": self.maxes,
            "windows": {c: {w: list(self.windows[c][w]) for w in self.windows[c]} for c in self.windows},
        }

    @classmethod
    def from_serializable(cls, data: Dict):
        """Construct a FeatureProcessor from a serialized snapshot."""
        p = cls()
        p.last_ts = data.get("last_ts")
        p.last_vals = data.get("last_vals", {c: None for c in COLUMNS})
        p.last_derivs = data.get("last_derivs", {c: 0.0 for c in COLUMNS})
        p.sums = data.get("sums", {c: 0.0 for c in COLUMNS})
        p.counts = data.get("counts", {c: 0 for c in COLUMNS})
        p.maxes = data.get("maxes", {c: float("-inf") for c in COLUMNS})
        # restore windows
        for c in COLUMNS:
            for w in ROLLING_WINDOWS:
                p.windows[c][w].clear()
                vals = data.get("windows", {}).get(c, {}).get(str(w), None)
                # tolerate numeric keys as well:
                if vals is None:
                    vals = data.get("windows", {}).get(c, {}).get(int(w), [])
                if vals is None:
                    vals = []
                for v in vals:
                    p.windows[c][w].append(v)
        return p

class FeatureManager:
    """
    Convenience manager to keep FeatureProcessor instances per key in memory.
    Use a single FeatureManager in your inference service to maintain state across requests.
    """
    def __init__(self, key_fields: Tuple[str, str, str] = ("machine_name", "program_name", "tool")):
        self.key_fields = key_fields
        self.processors = {}  # key_tuple -> FeatureProcessor

    def _key_from_event(self, ev: Dict) -> Tuple[str, str, str]:
        return tuple(ev.get(k, "") for k in self.key_fields)

    def process_event(self, ev: Dict) -> Dict:
        key = self._key_from_event(ev)
        if key not in self.processors:
            self.processors[key] = FeatureProcessor()
        result = self.processors[key].process(ev)
        # attach key fields to output for convenience
        return {
            **{self.key_fields[i]: key[i] for i in range(3)},
            **result
        }

    def save_states(self, path: str):
        """Persist processors' states to disk (json)."""
        serial = {",".join(k): p.to_serializable() for k, p in self.processors.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serial, f)

    def load_states(self, path: str):
        """Load states persisted by save_states."""
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            serial = json.load(f)
        for kstr, pdata in serial.items():
            key = tuple(kstr.split(","))
            fp = FeatureProcessor.from_serializable(pdata)
            self.processors[key] = fp

# -------------------------
# CLI for quick testing
# -------------------------
def read_event_from_file(path: str):
    if path == "-":
        txt = sys.stdin.read().strip()
        if not txt:
            return []
        if txt.startswith("["):
            return json.loads(txt)
        else:
            # assume single JSON or NDJSON
            lines = txt.splitlines()
            out = []
            for l in lines:
                if l.strip():
                    out.append(json.loads(l))
            return out
    else:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            if not txt:
                return []
            if txt.startswith("["):
                return json.loads(txt)
            else:
                lines = txt.splitlines()
                out = []
                for l in lines:
                    if l.strip():
                        out.append(json.loads(l))
                return out

def main():
    parser = argparse.ArgumentParser(description="Preprocessing for inference (single events).")
    parser.add_argument("--input", "-i", default="-", help="Input JSON (single event or NDJSON). Use '-' for stdin.")
    parser.add_argument("--state-in", default=None, help="Optional path to load manager state from.")
    parser.add_argument("--state-out", default=None, help="Optional path to save manager state to after processing.")
    args = parser.parse_args()

    evs = read_event_from_file(args.input)
    manager = FeatureManager()
    if args.state_in:
        manager.load_states(args.state_in)

    for ev in evs:
        out = manager.process_event(ev)
        print(json.dumps(out))

    if args.state_out:
        manager.save_states(args.state_out)

if __name__ == "__main__":
    main()

