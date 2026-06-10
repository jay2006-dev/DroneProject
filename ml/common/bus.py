"""Tiny message-bus helper.

The full CampusShield system publishes sensor verdicts over MQTT. For ML dev we
don't want a broker to be a hard dependency, so this publishes if paho-mqtt +
a reachable broker exist, and otherwise no-ops (printing in verbose mode).
"""
from __future__ import annotations
import json


def publish(topic: str, payload: dict, hostname: str = "localhost",
            verbose: bool = True) -> bool:
    """Best-effort MQTT publish. Returns True if sent, False if skipped."""
    try:
        import paho.mqtt.publish as pub  # type: ignore
        pub.single(topic, json.dumps(payload), hostname=hostname)
        return True
    except Exception as e:  # broker down or paho not installed -> just log
        if verbose:
            print(f"[bus] (offline) {topic} <- {json.dumps(payload)}  ({type(e).__name__})")
        return False
