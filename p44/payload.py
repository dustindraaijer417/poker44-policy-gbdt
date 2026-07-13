"""Project hands into the view a miner actually receives.

The validator passes every hand through
poker44.validator.payload_view.prepare_hand_for_miner before sending it, which
retains only a short window of each hand's actions. Training data is projected
through the same function so the training and inference distributions match.

resample_action_window re-draws which window is exposed, and is used as a training
augmentation so the model does not depend on any single draw.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List

from poker44.validator.payload_view import (
    _MINER_ACTION_WINDOW_MAX,
    _MINER_ACTION_WINDOW_MIN,
    prepare_hand_for_miner,
)

__all__ = ["to_live_view", "resample_action_window", "LIVE_WINDOW_MIN", "LIVE_WINDOW_MAX"]

LIVE_WINDOW_MIN = _MINER_ACTION_WINDOW_MIN
LIVE_WINDOW_MAX = _MINER_ACTION_WINDOW_MAX


def to_live_view(hand: Dict[str, Any]) -> Dict[str, Any]:
    """Project a benchmark hand through the validator's canonicalizer."""
    return prepare_hand_for_miner(hand)


def resample_action_window(hand: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Re-draw which action window the miner sees, for training augmentation.

    ``prepare_hand_for_miner`` picks its window from a hash of the hand, so it is
    a single fixed draw. Live, we get one draw per hand and cannot choose it. To
    stop the model latching onto whichever actions one particular draw exposed, we
    train over many draws of the same hand: the model then has to rely on signal
    that survives an arbitrary window, which is the only kind that generalizes.
    """
    actions = hand.get("actions") or []
    if len(actions) <= LIVE_WINDOW_MIN:
        return hand

    window = rng.randint(LIVE_WINDOW_MIN, LIVE_WINDOW_MAX)
    if len(actions) <= window:
        return hand

    # Mirror the validator's shape: always keep first and last, sample the middle,
    # and give each street a chance to be represented.
    keep = {0, len(actions) - 1}
    by_street: Dict[str, List[int]] = {}
    for idx in range(1, len(actions) - 1):
        by_street.setdefault(str(actions[idx].get("street", "")), []).append(idx)
    for street in sorted(by_street):
        if len(keep) >= window:
            break
        keep.add(rng.choice(by_street[street]))

    middle = [i for i in range(1, len(actions) - 1) if i not in keep]
    rng.shuffle(middle)
    for idx in middle:
        if len(keep) >= window:
            break
        keep.add(idx)

    out = dict(hand)
    out["actions"] = [dict(actions[i]) for i in sorted(keep)]
    for new_id, action in enumerate(out["actions"], start=1):
        action["action_id"] = str(new_id)
    return out
