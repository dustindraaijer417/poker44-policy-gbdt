"""Diagnostic statistics about the live requests this miner receives.

Purpose and boundary
--------------------
The model is trained on the public benchmark, but it is *served* on validator
traffic, and the two differ in shape: chunk length, table size, bet scale and
action mix. Those differences are exactly what breaks a model that looked good
offline. This module records **summary statistics only** -- counts, means,
quantiles -- so that shift can be measured directly instead of assumed.

It deliberately does NOT persist hands, chunks, payloads or any validator
content, and nothing it produces is used for training. It exists so that model
development can be checked against reality, and so that a distribution shift
shows up as a number rather than as an unexplained score drop. The published
training-data statement therefore remains accurate: the model is trained on the
public benchmark only.

Writes a small rolling JSON summary; failures are swallowed so diagnostics can
never affect serving.
"""

from __future__ import annotations

import json
import os
import statistics
import threading
import time
from typing import Any, Dict, List

_LOCK = threading.Lock()
_PATH = os.environ.get("POKER44_PROBE_PATH", "/root/subnet126/poker44-miner/data/live_probe.json")
_MAX_REQUESTS = 500


def _q(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return float(s[i])


def summarize_request(chunks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Shape statistics for one incoming request. No content is retained."""
    sizes = [len(c) for c in chunks]
    actions_per_hand: List[int] = []
    seats: List[int] = []
    amounts: List[float] = []
    pots: List[float] = []
    kinds: Dict[str, int] = {}
    hero_visible = 0
    hands_seen = 0

    for chunk in chunks:
        for hand in chunk:
            if not isinstance(hand, dict):
                continue
            hands_seen += 1
            meta = hand.get("metadata") or {}
            acts = hand.get("actions") or []
            actions_per_hand.append(len(acts))
            try:
                seats.append(int(meta.get("max_seats") or 0))
            except (TypeError, ValueError):
                pass
            hero = meta.get("hero_seat")
            if any(a.get("actor_seat") == hero for a in acts if isinstance(a, dict)):
                hero_visible += 1
            for a in acts:
                if not isinstance(a, dict):
                    continue
                k = str(a.get("action_type") or "")
                kinds[k] = kinds.get(k, 0) + 1
                try:
                    amounts.append(float(a.get("normalized_amount_bb") or 0.0))
                    pots.append(float(a.get("pot_before") or 0.0))
                except (TypeError, ValueError):
                    pass

    total_actions = max(1, sum(kinds.values()))
    return {
        "ts": int(time.time()),
        "n_chunks": len(chunks),
        "hands_per_chunk_mean": round(statistics.fmean(sizes), 2) if sizes else 0,
        "hands_per_chunk_min": min(sizes) if sizes else 0,
        "hands_per_chunk_max": max(sizes) if sizes else 0,
        "actions_per_hand_mean": round(statistics.fmean(actions_per_hand), 2) if actions_per_hand else 0,
        "actions_per_hand_max": max(actions_per_hand) if actions_per_hand else 0,
        "max_seats_mode": max(set(seats), key=seats.count) if seats else 0,
        "max_seats_max": max(seats) if seats else 0,
        # how often the focal player is even visible in the truncated window --
        # the statistic that explains why hero-conditioned features fail live
        "hero_visible_rate": round(hero_visible / max(1, hands_seen), 4),
        "amount_bb_mean": round(statistics.fmean(amounts), 3) if amounts else 0,
        "amount_bb_q90": round(_q(amounts, 0.9), 3),
        "pot_before_q50": round(_q(pots, 0.5), 4),
        "action_mix": {k: round(v / total_actions, 4) for k, v in sorted(kinds.items())},
    }


def record(chunks: List[List[Dict[str, Any]]]) -> None:
    """Append one request summary. Never raises."""
    try:
        summary = summarize_request(chunks)
        with _LOCK:
            data = []
            if os.path.exists(_PATH):
                try:
                    with open(_PATH) as fh:
                        data = json.load(fh)
                except Exception:
                    data = []
            data.append(summary)
            data = data[-_MAX_REQUESTS:]
            os.makedirs(os.path.dirname(_PATH), exist_ok=True)
            tmp = _PATH + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(data, fh)
            os.replace(tmp, _PATH)
    except Exception:
        pass
