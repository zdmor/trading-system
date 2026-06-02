"""Minimal system_state snapshot helper — used by backup scripts."""
import sys, os, json
from datetime import datetime

TS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TS_DIR)


def snapshot_all(filepath: str = None):
    """Save current system_state to a timestamped JSON file."""
    from system_state import read_state

    state = read_state()
    out_dir = os.path.join(TS_DIR, "output")
    os.makedirs(out_dir, exist_ok=True)

    if filepath is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        filepath = os.path.join(out_dir, f"snapshot_{ts}.json")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str, ensure_ascii=False)
    print(f"Snapshot saved: {filepath}")
    return filepath


if __name__ == "__main__":
    snapshot_all()
