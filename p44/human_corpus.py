"""Real human hands, rebuilt as focal-player chunks.

The corpus shipped with the Poker44 subnet repository records every seat's actions,
so any player appearing in enough hands can be treated as the focal player: take
their hands and point metadata.hero_seat at their seat.

max_chunks_per_player caps how many chunks any one player contributes. The corpus
has a single player present in every hand, who would otherwise supply most of the
class.
"""

from __future__ import annotations

import gzip
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

# Path to the Poker44 subnet checkout that ships the human hand corpus.
SUBNET_PATH = Path(os.environ.get("POKER44_SUBNET_PATH", "../Poker44-subnet"))
CORPUS = SUBNET_PATH / "hands_generator" / "human_hands" / "poker_hands_combined.json.gz"

MIN_HANDS = 30
MAX_HANDS = 40


def load_raw() -> List[dict]:
    with gzip.open(CORPUS, "rt") as fh:
        return json.load(fh)


def human_chunks(min_hands: int = MIN_HANDS, max_hands: int = MAX_HANDS,
                 seed: int = 0, max_chunks: int | None = None,
                 max_chunks_per_player: int = 3) -> List[List[dict]]:
    """Focal-player chunks for every human with enough hands.

    Each chunk is one real player's hands with hero_seat pointed at them, matching
    the shape of a live evaluation chunk.

    max_chunks_per_player matters: one player (the corpus 'hero') appears in all
    32k hands and would otherwise supply ~800 of the chunks, so the model would
    learn that individual rather than humans in general. Capping per player buys
    diversity, which is the whole point of using this corpus.
    """
    hands = load_raw()
    rng = random.Random(seed)

    # player_uid -> [(hand, that player's seat)]
    by_player: Dict[str, List[tuple]] = defaultdict(list)
    for hand in hands:
        for p in hand.get("players") or []:
            uid = p.get("player_uid")
            seat = p.get("seat")
            if uid and seat:
                by_player[uid].append((hand, seat))

    chunks: List[List[dict]] = []
    for uid, entries in by_player.items():
        if len(entries) < min_hands:
            continue
        rng.shuffle(entries)
        made = 0
        # A player with many hands yields several independent chunks, up to the cap.
        for start in range(0, len(entries), max_hands):
            if made >= max_chunks_per_player:
                break
            window = entries[start:start + max_hands]
            if len(window) < min_hands:
                break
            made += 1
            chunk = []
            for hand, seat in window:
                h = json.loads(json.dumps(hand))     # deep copy; we rewrite hero_seat
                h["metadata"] = dict(h.get("metadata") or {})
                h["metadata"]["hero_seat"] = seat
                h.pop("label", None)
                chunk.append(h)
            chunks.append(chunk)
            if max_chunks and len(chunks) >= max_chunks:
                return chunks
    return chunks
