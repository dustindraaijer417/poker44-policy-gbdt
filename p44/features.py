"""Chunk-level feature extraction for Poker44 bot detection.

Features are grouped into families so they can be ablated independently. The
families exist because of what the release audit tells us:

``pooled``      The audit's own family: seat-agnostic means/stds over the chunk.
                The generator randomizes bot policy per release, so the *direction*
                of these features flips (mean_starting_stack: 15 direct vs 16
                inverse across 31 releases). Expected to be near-worthless out of
                sample. Kept only as a baseline to prove that point.

``hero``        The same behavioral stats, but conditioned on the focal seat
                (``metadata.hero_seat``), which is what the label is actually
                about. The audit does not cover these.

``consistency`` Dispersion/entropy of the hero's policy rather than its level. A
                bot executes a fixed policy, so it is self-consistent regardless
                of whether this release made it tight or loose. Direction-stable
                by construction, which is the property the pooled family lacks.

``relative``    Hero minus the villains at the same tables. Any per-release drift
                that moves every player equally cancels, so these survive the
                generator's re-randomization.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Sequence

MONEY_ACTIONS = ("bet", "raise", "call")
AGGRESSIVE = ("bet", "raise")
ALL_ACTIONS = ("fold", "check", "call", "bet", "raise")
STREETS = ("preflop", "flop", "turn", "river")


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den else default


def _entropy(counts: Sequence[float]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log(p + 1e-12)
    return ent


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _action_stats(actions: List[Dict[str, Any]]) -> Dict[str, float]:
    """Rates, sizing and dispersion for one set of actions (hero's or villains')."""
    counts = {a: 0 for a in ALL_ACTIONS}
    sizes: List[float] = []
    pot_ratios: List[float] = []
    by_street: Dict[str, int] = defaultdict(int)

    for act in actions:
        kind = act.get("action_type")
        if kind in counts:
            counts[kind] += 1
        by_street[str(act.get("street", ""))] += 1
        amt = float(act.get("normalized_amount_bb", 0.0) or 0.0)
        if kind in MONEY_ACTIONS and amt > 0:
            sizes.append(amt)
            pot_before = float(act.get("pot_before", 0.0) or 0.0)
            if pot_before > 0:
                # amount is in table currency, pot in the same units
                pot_ratios.append(min(5.0, _safe_div(float(act.get("amount", 0.0) or 0.0), pot_before)))

    n = sum(counts.values())
    out: Dict[str, float] = {}
    for a in ALL_ACTIONS:
        out[f"rate_{a}"] = _safe_div(counts[a], n)
    aggro = counts["bet"] + counts["raise"]
    out["aggression"] = _safe_div(aggro, n)
    out["aggression_factor"] = _safe_div(aggro, counts["call"] + counts["fold"] + 1)
    out["action_entropy"] = _entropy([counts[a] for a in ALL_ACTIONS])
    out["n_actions"] = float(n)

    out["size_mean"] = _mean(sizes)
    out["size_std"] = _std(sizes)
    out["size_cv"] = _safe_div(_std(sizes), _mean(sizes) + 1e-6)
    out["potratio_mean"] = _mean(pot_ratios)
    out["potratio_std"] = _std(pot_ratios)

    # Sizing quantization: a bot picking from a fixed sizing grid produces few
    # distinct pot-fractions; a human's are smeared. The validator's bucket noise
    # blurs this but does not erase the relative concentration.
    if pot_ratios:
        grid = defaultdict(int)
        for r in pot_ratios:
            grid[round(r, 1)] += 1
        out["size_grid_entropy"] = _entropy(list(grid.values()))
        out["size_grid_distinct"] = _safe_div(len(grid), len(pot_ratios))
        out["size_grid_top"] = _safe_div(max(grid.values()), len(pot_ratios))
    else:
        out["size_grid_entropy"] = 0.0
        out["size_grid_distinct"] = 0.0
        out["size_grid_top"] = 0.0

    for s in STREETS:
        out[f"street_share_{s}"] = _safe_div(by_street.get(s, 0), n)
    return out


def _facing_bet_context(actions: List[Dict[str, Any]], idx: int) -> bool:
    """True if a bet/raise is live on this street when action idx is taken."""
    street = actions[idx].get("street")
    for prior in actions[:idx][::-1]:
        if prior.get("street") != street:
            break
        if prior.get("action_type") in AGGRESSIVE:
            return True
    return False


def extract(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    """Featurize one chunk (a list of hands for one focal player)."""
    hero_actions: List[Dict[str, Any]] = []
    villain_actions: List[Dict[str, Any]] = []

    per_hand_aggro: List[float] = []
    per_hand_hero_n: List[float] = []
    hero_absent = 0
    vpip_hands = 0
    pfr_hands = 0

    # policy(context -> action) counts, for conditional entropy
    policy: Dict[tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    starting_stacks: List[float] = []
    player_counts: List[float] = []
    street_counts: List[float] = []
    action_counts: List[float] = []
    pot_befores: List[float] = []
    pot_afters: List[float] = []
    all_sizes: List[float] = []
    zero_amt = 0
    all_n = 0

    for hand in hands:
        meta = hand.get("metadata") or {}
        hero_seat = meta.get("hero_seat")
        actions = hand.get("actions") or []
        players = hand.get("players") or []

        player_counts.append(float(len(players)))
        street_counts.append(float(len(hand.get("streets") or [])))
        action_counts.append(float(len(actions)))
        for p in players:
            starting_stacks.append(float(p.get("starting_stack", 0.0) or 0.0))

        h_acts = []
        for i, act in enumerate(actions):
            all_n += 1
            amt = float(act.get("normalized_amount_bb", 0.0) or 0.0)
            if amt <= 0:
                zero_amt += 1
            else:
                all_sizes.append(amt)
            pot_befores.append(float(act.get("pot_before", 0.0) or 0.0))
            pot_afters.append(float(act.get("pot_after", 0.0) or 0.0))

            if act.get("actor_seat") == hero_seat:
                h_acts.append(act)
                kind = str(act.get("action_type") or "")
                ctx = (str(act.get("street") or ""), _facing_bet_context(actions, i))
                policy[ctx][kind] += 1
            else:
                villain_actions.append(act)

        hero_actions.extend(h_acts)
        per_hand_hero_n.append(float(len(h_acts)))
        if not h_acts:
            hero_absent += 1
        kinds = [a.get("action_type") for a in h_acts]
        if any(k in MONEY_ACTIONS for k in kinds):
            vpip_hands += 1
        if any(
            a.get("action_type") in AGGRESSIVE and a.get("street") == "preflop"
            for a in h_acts
        ):
            pfr_hands += 1
        n_h = len(h_acts)
        if n_h:
            per_hand_aggro.append(
                _safe_div(sum(1 for k in kinds if k in AGGRESSIVE), n_h)
            )

    n_hands = max(1, len(hands))
    feats: Dict[str, float] = {}

    # ---------------- pooled: the audit's own family (baseline) ----------------
    pooled = _action_stats(hero_actions + villain_actions)
    for k, v in pooled.items():
        feats[f"pooled__{k}"] = v
    feats["pooled__mean_starting_stack"] = _mean(starting_stacks)
    feats["pooled__mean_player_count"] = _mean(player_counts)
    feats["pooled__mean_street_count"] = _mean(street_counts)
    feats["pooled__mean_action_count"] = _mean(action_counts)
    feats["pooled__mean_pot_before"] = _mean(pot_befores)
    feats["pooled__mean_pot_after"] = _mean(pot_afters)
    feats["pooled__zero_amount_share"] = _safe_div(zero_amt, all_n)
    feats["pooled__std_normalized_amount_bb"] = _std(all_sizes)
    feats["pooled__mean_normalized_amount_bb"] = _mean(all_sizes)

    # ---------------- hero: conditioned on the focal seat ----------------
    hero = _action_stats(hero_actions)
    for k, v in hero.items():
        feats[f"hero__{k}"] = v
    feats["hero__actions_per_hand"] = _safe_div(len(hero_actions), n_hands)
    feats["hero__absent_rate"] = _safe_div(hero_absent, n_hands)
    feats["hero__vpip"] = _safe_div(vpip_hands, n_hands)
    feats["hero__pfr"] = _safe_div(pfr_hands, n_hands)
    feats["hero__pfr_vpip"] = _safe_div(pfr_hands, max(1, vpip_hands))

    # ---------------- consistency: dispersion, not level ----------------
    # A bot runs a fixed policy, so its conditional action distribution is sharp
    # whether this release made it tight or loose. That is what makes these
    # direction-stable while the pooled family flips.
    cond_ents, cond_w, det_scores = [], [], []
    for ctx, dist in policy.items():
        tot = sum(dist.values())
        if tot < 2:
            continue
        cond_ents.append(_entropy(list(dist.values())))
        cond_w.append(tot)
        det_scores.append(max(dist.values()) / tot)
    feats["consistency__policy_entropy"] = (
        _safe_div(sum(e * w for e, w in zip(cond_ents, cond_w)), sum(cond_w))
        if cond_w
        else 0.0
    )
    feats["consistency__policy_determinism"] = _mean(det_scores)
    feats["consistency__policy_entropy_std"] = _std(cond_ents)
    feats["consistency__n_contexts"] = float(len(cond_ents))
    feats["consistency__aggro_std"] = _std(per_hand_aggro)
    feats["consistency__aggro_mean"] = _mean(per_hand_aggro)
    feats["consistency__hero_n_std"] = _std(per_hand_hero_n)
    feats["consistency__size_cv"] = hero.get("size_cv", 0.0)
    feats["consistency__grid_entropy"] = hero.get("size_grid_entropy", 0.0)

    # ---------------- relative: hero minus villains at the same tables ----------------
    # Per-release drift that shifts every player equally cancels here.
    villains = _action_stats(villain_actions)
    for key in (
        "rate_fold", "rate_check", "rate_call", "rate_bet", "rate_raise",
        "aggression", "aggression_factor", "action_entropy",
        "size_mean", "size_std", "size_cv",
        "potratio_mean", "potratio_std",
        "size_grid_entropy", "size_grid_distinct", "size_grid_top",
    ):
        feats[f"relative__d_{key}"] = hero.get(key, 0.0) - villains.get(key, 0.0)
        v = villains.get(key, 0.0)
        feats[f"relative__r_{key}"] = _safe_div(hero.get(key, 0.0), v, default=0.0) if abs(v) > 1e-9 else 0.0

    return feats


FAMILIES = ("pooled", "hero", "consistency", "relative")


def family_of(name: str) -> str:
    return name.split("__", 1)[0]
