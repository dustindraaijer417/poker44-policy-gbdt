"""Context-conditioned policy features for the focal player.

Each feature is a conditional response rate, a bet-sizing ratio, or a dispersion
of one of those, measured within a (street, facing-a-bet) context -- for example
P(fold | facing a bet on the flop).

Conditioning on the situation is what makes these features describe the player
rather than the table: the composition of a table changes how OFTEN a given
situation arises, but not how a particular player responds once it does. These are
the same quantities a poker tracking tool uses to profile an opponent (VPIP, PFR,
fold-to-bet, aggression factor, sizing distribution).

No counts, no table aggregates, and no absolute money amounts appear here.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List

STREETS = ("preflop", "flop", "turn", "river")
RESPONSES = ("fold", "check", "call", "bet", "raise")
FACING = ("open", "facing")   # open = hero may check; facing = a bet/raise is live


def _entropy(counts) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    e = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            e -= p * math.log(p + 1e-12)
    return e


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _street_of(action: Dict[str, Any]) -> str:
    s = str(action.get("street") or "").lower()
    return s if s in STREETS else "preflop"


def extract_policy(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    """Conditional policy features for the focal player of one chunk."""
    # context -> response -> count
    resp: Dict[tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # context -> list of bet-size-as-fraction-of-pot when hero puts money in
    sizing: Dict[tuple, List[float]] = defaultdict(list)
    # per-hand aggression, to measure drift across a session
    per_hand_aggro: List[float] = []

    for hand in hands:
        meta = hand.get("metadata") or {}
        hero = meta.get("hero_seat")
        actions = hand.get("actions") or []

        h_aggr, h_n = 0, 0
        for i, act in enumerate(actions):
            if act.get("actor_seat") != hero:
                continue

            street = _street_of(act)
            # is a bet/raise live on this street when the hero acts?
            facing = "open"
            for prior in reversed(actions[:i]):
                if _street_of(prior) != street:
                    break
                if prior.get("action_type") in ("bet", "raise"):
                    facing = "facing"
                    break

            kind = str(act.get("action_type") or "")
            if kind not in RESPONSES:
                continue

            ctx = (street, facing)
            resp[ctx][kind] += 1

            h_n += 1
            if kind in ("bet", "raise"):
                h_aggr += 1

            pot_before = float(act.get("pot_before", 0.0) or 0.0)
            amount = float(act.get("amount", 0.0) or 0.0)
            if kind in ("bet", "raise", "call") and amount > 0 and pot_before > 0:
                sizing[ctx].append(min(5.0, amount / pot_before))

        if h_n:
            per_hand_aggro.append(h_aggr / h_n)

    feats: Dict[str, float] = {}

    for street in STREETS:
        for facing in FACING:
            ctx = (street, facing)
            dist = resp.get(ctx, {})
            total = sum(dist.values())
            tag = f"{street}_{facing}"

            # response distribution: the hero's policy in this situation
            for r in RESPONSES:
                feats[f"pol__{tag}__p_{r}"] = (dist.get(r, 0) / total) if total else 0.0

            # how mixed is the policy here? a fixed-rule bot is sharper than a human
            feats[f"pol__{tag}__entropy"] = _entropy([dist.get(r, 0) for r in RESPONSES]) if total else 0.0
            feats[f"pol__{tag}__determinism"] = (max(dist.values()) / total) if total else 0.0

            # aggression within the context (bet+raise vs call+check+fold)
            aggr = dist.get("bet", 0) + dist.get("raise", 0)
            feats[f"pol__{tag}__aggression"] = (aggr / total) if total else 0.0

            # sizing policy: bots pick from a small, stable menu of pot fractions
            sizes = sizing.get(ctx, [])
            feats[f"pol__{tag}__size_mean"] = _mean(sizes)
            feats[f"pol__{tag}__size_std"] = _std(sizes)
            feats[f"pol__{tag}__size_cv"] = _std(sizes) / (_mean(sizes) + 1e-6) if sizes else 0.0
            if sizes:
                grid = defaultdict(int)
                for s in sizes:
                    grid[round(s, 1)] += 1
                feats[f"pol__{tag}__size_grid_entropy"] = _entropy(list(grid.values()))
                feats[f"pol__{tag}__size_grid_top"] = max(grid.values()) / len(sizes)
            else:
                feats[f"pol__{tag}__size_grid_entropy"] = 0.0
                feats[f"pol__{tag}__size_grid_top"] = 0.0

            # observed(1)/unobserved(0): lets the model discount empty contexts without
            # leaking how often the context arose (which is table-driven).
            feats[f"pol__{tag}__seen"] = 1.0 if total else 0.0

    # session drift: humans tilt, get bored, change gears. A rule bot does not.
    feats["pol__drift__aggro_std"] = _std(per_hand_aggro)
    feats["pol__drift__aggro_mean"] = _mean(per_hand_aggro)

    # classic HUD ratios, all conditional so they stay villain-invariant
    pf_open = resp.get(("preflop", "open"), {})
    pf_face = resp.get(("preflop", "facing"), {})
    n_open = sum(pf_open.values())
    n_face = sum(pf_face.values())
    feats["pol__hud__pfr_open"] = ((pf_open.get("raise", 0) + pf_open.get("bet", 0)) / n_open) if n_open else 0.0
    feats["pol__hud__fold_to_open"] = (pf_face.get("fold", 0) / n_face) if n_face else 0.0
    feats["pol__hud__3bet"] = (pf_face.get("raise", 0) / n_face) if n_face else 0.0

    post_face = defaultdict(int)
    for s in ("flop", "turn", "river"):
        for r, c in resp.get((s, "facing"), {}).items():
            post_face[r] += c
    n_pf = sum(post_face.values())
    feats["pol__hud__fold_to_bet_postflop"] = (post_face.get("fold", 0) / n_pf) if n_pf else 0.0
    feats["pol__hud__raise_bet_postflop"] = (post_face.get("raise", 0) / n_pf) if n_pf else 0.0

    return feats
