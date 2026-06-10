"""A10 — dataset versioning. Log every recording session to dataset/registry.json.

    python -m ml.a10_registry.registry --drone DJI_Tello --distance 20 \
        --location Lab_Room3 --weather indoor_clear --files 30 --notes "first run"

Or import register_session(...) and call it from your capture script.
"""
from __future__ import annotations
import os, sys, json, argparse, datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.config import REGISTRY_PATH


def register_session(drone_model, distance_m, location, weather, n_files,
                     notes="", when=None):
    entry = {
        "date": (when or datetime.datetime.now()).isoformat(timespec="seconds"),
        "drone": drone_model, "distance_m": distance_m, "location": location,
        "weather": weather, "files_added": n_files, "notes": notes,
    }
    registry = []
    if os.path.exists(REGISTRY_PATH):
        registry = json.load(open(REGISTRY_PATH))
    registry.append(entry)
    json.dump(registry, open(REGISTRY_PATH, "w"), indent=2)
    print(f"Registered. Total sessions: {len(registry)} "
          f"({sum(e['files_added'] for e in registry)} files)")
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drone", required=True)
    ap.add_argument("--distance", type=float, required=True)
    ap.add_argument("--location", required=True)
    ap.add_argument("--weather", default="")
    ap.add_argument("--files", type=int, required=True)
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    register_session(a.drone, a.distance, a.location, a.weather, a.files, a.notes)


if __name__ == "__main__":
    main()
