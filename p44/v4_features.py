"""Chunk-size-invariant, live-robust features.

Three measured failures of the previous feature set drove this rewrite:

1. CHUNK SIZE. Benchmark chunks are 30-40 hands; live validator chunks are
   80-100. Our previous model scored AUC 0.98 at 39 hands but 0.81 at 119 --
   because chunk-level order statistics (min/max/q10/q90 over hands) and raw
   counts drift systematically with the number of hands. Everything here is a
   per-hand rate, a per-action rate, or a statistic computed on a fixed-size
   subsample, so it does not move with chunk length.

2. ABSOLUTE MONEY. Absolute bet/pot/stack magnitudes in big blinds are heavily
   out-of-distribution on the live feed (live pots and bets run roughly half the
   benchmark scale). Trees split on benchmark-scale thresholds and then misfire.
   Only *ratios* (bet relative to pot, stack relative to pot) appear here.

3. TABLE SHAPE. Live tables reach 9 seats; the benchmark caps at 6. Raw seat
   counts are therefore not comparable, so seat information enters only as
   normalized position or as a share.

What replaces them: action-sequence n-grams with a frozen vocabulary (a bot
replays the same action patterns at the same pot fractions, which shows up as
concentrated n-gram mass) and sequence-repetition statistics computed on a fixed
subsample.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any, Dict, List

STREETS = ("preflop", "flop", "turn", "river")
ACTIONS = ("fold", "check", "call", "bet", "raise")
_ACT_CODE = {"fold": "F", "check": "K", "call": "C", "bet": "B", "raise": "R"}
SIGNATURE_SAMPLE = 30      # fixed subsample so repetition stats are size-invariant


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


def _size_bucket(amount: float, pot_before: float) -> str:
    """Bet size as a fraction of pot -- scale-free, unlike absolute bb."""
    if amount <= 0:
        return "0"
    if pot_before <= 0:
        return "?"
    r = amount / pot_before
    if r < 0.4:
        return "s"
    if r < 0.9:
        return "m"
    if r < 1.5:
        return "p"
    return "o"


def _hand_tokens(hand: Dict[str, Any]) -> List[str]:
    """Ordered action tokens: street + action + pot-relative size bucket."""
    toks = []
    for a in hand.get("actions") or []:
        kind = str(a.get("action_type") or "")
        if kind not in _ACT_CODE:
            continue
        street = str(a.get("street") or "preflop").lower()
        try:
            amt = float(a.get("amount") or 0.0)
            pot = float(a.get("pot_before") or 0.0)
        except (TypeError, ValueError):
            amt = pot = 0.0
        toks.append(f"{street[0]}{_ACT_CODE[kind]}{_size_bucket(amt, pot)}")
    return toks


def _hand_rates(hand: Dict[str, Any]) -> Dict[str, float]:
    """Per-hand behaviour, all as rates/ratios. No absolute money, no raw counts."""
    actions = hand.get("actions") or []
    meta = hand.get("metadata") or {}
    hero = meta.get("hero_seat")

    counts = Counter()
    street_counts = Counter()
    actors = []
    ratios = []
    hero_actions = 0
    n = 0
    for a in actions:
        kind = str(a.get("action_type") or "")
        if kind not in ACTIONS:
            continue
        n += 1
        counts[kind] += 1
        street_counts[str(a.get("street") or "preflop").lower()] += 1
        actors.append(a.get("actor_seat"))
        if a.get("actor_seat") == hero:
            hero_actions += 1
        try:
            amt = float(a.get("amount") or 0.0)
            pot = float(a.get("pot_before") or 0.0)
        except (TypeError, ValueError):
            amt = pot = 0.0
        if amt > 0 and pot > 0:
            ratios.append(min(6.0, amt / pot))

    out: Dict[str, float] = {}
    d = max(1, n)
    for k in ACTIONS:
        out[f"rate_{k}"] = counts[k] / d
    out["aggression"] = (counts["bet"] + counts["raise"]) / d
    out["passive"] = (counts["call"] + counts["check"]) / d
    for s in STREETS:
        out[f"street_{s}"] = street_counts.get(s, 0) / d
    out["action_entropy"] = _entropy([counts[k] for k in ACTIONS])
    out["actor_entropy"] = _entropy(list(Counter(actors).values()))
    # sequence texture
    switches = sum(1 for i in range(1, len(actors)) if actors[i] != actors[i - 1])
    out["actor_switch_rate"] = switches / max(1, len(actors) - 1)
    out["hero_action_share"] = hero_actions / d
    # pot-relative sizing only
    out["potratio_mean"] = _mean(ratios)
    out["potratio_std"] = _std(ratios)
    out["potratio_cv"] = _std(ratios) / (_mean(ratios) + 1e-6) if ratios else 0.0
    out["sized_action_share"] = len(ratios) / d
    return out


def extract_v4(chunk: List[Dict[str, Any]]) -> Dict[str, float]:
    """Size-invariant chunk features."""
    hands = [h for h in chunk if isinstance(h, dict)]
    if not hands:
        return {}
    n = len(hands)

    # ---- per-hand rates, pooled by mean/std ONLY (order stats drift with n) ----
    per_hand = [_hand_rates(h) for h in hands]
    keys = sorted(per_hand[0].keys())
    feats: Dict[str, float] = {}
    for k in keys:
        vals = [ph.get(k, 0.0) for ph in per_hand]
        feats[f"h_{k}_mean"] = _mean(vals)
        feats[f"h_{k}_std"] = _std(vals)

    # ---- action-token n-grams, normalized per hand (scale-free) ----
    tok_lists = [_hand_tokens(h) for h in hands]
    uni, bi = Counter(), Counter()
    total_tokens = 0
    for toks in tok_lists:
        total_tokens += len(toks)
        uni.update(toks)
        bi.update(f"{toks[i]}|{toks[i+1]}" for i in range(len(toks) - 1))
    # normalize by token count -> a distribution, independent of chunk length
    tt = max(1, total_tokens)
    for t in _UNI_VOCAB:
        feats[f"ng1_{t}"] = uni.get(t, 0) / tt
    for t in _BI_VOCAB:
        feats[f"ng2_{t.replace('|','_')}"] = bi.get(t, 0) / tt
    feats["ng_uni_entropy"] = _entropy(list(uni.values()))
    feats["ng_bi_entropy"] = _entropy(list(bi.values()))
    feats["ng_uni_distinct_rate"] = len(uni) / tt
    feats["ng_top1_share"] = (max(uni.values()) / tt) if uni else 0.0
    feats["tokens_per_hand"] = total_tokens / n

    # ---- sequence repetition on a FIXED subsample (size-invariant) ----
    # A bot replays identical action sequences; sampling a constant number of
    # hands keeps the statistic comparable across 34- and 100-hand chunks.
    idx = _stable_sample(n, SIGNATURE_SAMPLE, tok_lists)
    sigs = ["|".join(tok_lists[i]) for i in idx]
    m = max(1, len(sigs))
    sc = Counter(sigs)
    feats["sig_top_share"] = max(sc.values()) / m
    feats["sig_unique_share"] = len(sc) / m
    feats["sig_entropy"] = _entropy(list(sc.values()))
    # prefix repetition: first 3 actions identical across hands
    p3 = Counter("|".join(tok_lists[i][:3]) for i in idx)
    feats["sig_prefix3_top_share"] = max(p3.values()) / m
    feats["sig_prefix3_unique_share"] = len(p3) / m

    return feats


def _stable_sample(n: int, k: int, tok_lists) -> List[int]:
    """Deterministic subsample of hand indices, independent of chunk order."""
    if n <= k:
        return list(range(n))
    # order by a content hash so the same hands are picked regardless of position
    order = sorted(range(n), key=lambda i: hashlib.sha1(
        "|".join(tok_lists[i]).encode("utf-8", "ignore")).digest())
    return order[:k]


def _build_vocab():
    """Frozen token vocabulary so the feature schema never drifts."""
    uni = []
    for s in STREETS:
        for a in _ACT_CODE.values():
            for b in ("0", "?", "s", "m", "p", "o"):
                uni.append(f"{s[0]}{a}{b}")
    # bigrams limited to same-street common pairs to keep the count sane
    common = [t for t in uni if t[2] in ("0", "s", "m", "p")]
    bi = [f"{a}|{b}" for a in common for b in common if a[0] == b[0]]
    return tuple(uni), tuple(bi)


_UNI_VOCAB, _BI_VOCAB = _build_vocab()
