"""Additional bot policies used to broaden the training distribution.

SandboxPokerBot is rule-based and close to deterministic given a situation. Real
bots may be built on other principles, so these policies cover paradigms it does
not:

  SolverBot   Mixed strategies. Randomizes between actions at strength-dependent
              frequencies, and bets from a fixed grid of pot fractions.
  MimicBot    Rule-based, but perturbs its policy per hand and drifts its
              aggression across a session.
  StealthBot  Mixed strategies with continuous (non-grid) bet sizing and session
              drift.

Including several paradigms in training is what allows leave-one-family-out
validation to say something about bots the model has not seen.
"""

from __future__ import annotations

import random
from typing import Optional

from hands_generator.bot_hands.sandbox_poker_bot import (
    ActionType,
    BotDecision,
    BotProfile,
    GameState,
    LegalActions,
    SandboxPokerBot,
    Street,
)

# Pot fractions a solver-style bot bets from, with the frequency it picks each.
SOLVER_SIZING_GRID = ((0.33, 0.45), (0.66, 0.35), (1.00, 0.20))


class SolverBot(SandboxPokerBot):
    """Mixed-strategy bot: human-looking entropy, machine-looking bet sizing."""

    def __init__(self, profile: BotProfile, rng_seed: Optional[int] = None):
        super().__init__(profile, rng_seed=rng_seed)
        self.mix = random.Random((rng_seed or 0) ^ 0x5010E2)

    def _sample(self, weights: dict) -> str:
        total = sum(w for w in weights.values() if w > 0)
        if total <= 0:
            return "fold"
        r = self.mix.random() * total
        acc = 0.0
        for action, w in weights.items():
            if w <= 0:
                continue
            acc += w
            if r <= acc:
                return action
        return next(iter(weights))

    def _grid_size(self, pot: int) -> int:
        r = self.mix.random()
        acc = 0.0
        for frac, w in SOLVER_SIZING_GRID:
            acc += w
            if r <= acc:
                return max(1, int(round(pot * frac)))
        return max(1, int(round(pot * SOLVER_SIZING_GRID[-1][0])))

    def act(self, state: GameState, legal: LegalActions) -> BotDecision:
        if state.hole_cards:
            s = self._get_hand_strength_from_csv(state.hole_cards)
            if s is not None:
                state.hand_strength = s

        pos = self._position_factor(state.position_index, state.num_players)
        hs = state.hand_strength if state.hand_strength is not None else 0.5
        aggr = self.profile.aggression

        # Frequencies vary smoothly with strength and position -- no hard rule
        # boundaries, and the bot mixes rather than committing. This is what makes
        # its per-context action entropy look human.
        if state.street == Street.PREFLOP:
            w = {
                "raise": max(0.0, (hs ** 2) * (0.55 + 0.35 * aggr) + 0.06 * pos),
                "call": max(0.0, hs * 0.35 + 0.10 * pos),
                "fold": max(0.0, (1.0 - hs) ** 1.5 * 1.15 - 0.20 * pos),
                "check": 0.0,
            }
        else:
            w = {
                "raise": max(0.0, (hs ** 2) * (0.42 + 0.40 * aggr) + 0.05 * pos),
                "call": max(0.0, hs * 0.40),
                "fold": max(0.0, (1.0 - hs) ** 1.5 * 1.00),
                "check": max(0.0, 0.45 - 0.30 * hs),
            }

        if not legal.can_check:
            w["check"] = 0.0
        if not legal.can_call:
            w["call"] = 0.0
        if not (legal.can_raise or legal.can_bet):
            w["raise"] = 0.0
        if not legal.can_fold or legal.can_check:
            w["fold"] = 0.0   # never fold when checking is free

        choice = self._sample(w)

        if choice == "raise" and (legal.can_raise or legal.can_bet):
            size = self._grid_size(max(1, state.pot))
            if legal.can_raise:
                amt = self._clamp(max(size, legal.min_raise), legal.min_raise, legal.max_raise)
                return BotDecision(action=ActionType.RAISE, amount=amt)
            amt = self._clamp(max(size, legal.min_bet), legal.min_bet, legal.max_bet)
            return BotDecision(action=ActionType.BET, amount=amt)
        if choice == "call" and legal.can_call:
            return BotDecision(action=ActionType.CALL, amount=legal.call_amount)
        if choice == "check" and legal.can_check:
            return BotDecision(action=ActionType.CHECK, amount=0)
        if legal.can_check:
            return BotDecision(action=ActionType.CHECK, amount=0)
        return BotDecision(action=ActionType.FOLD, amount=0)


class StealthBot(SolverBot):
    """The hardest adversary we can construct: attacks every top signal at once.

    - mixed strategies          -> per-context entropy looks human (beats determinism)
    - CONTINUOUS bet sizing     -> no fixed grid (beats size_grid_entropy / size_grid_top)
    - session aggression drift  -> beats pol__drift__aggro_std

    If the detector still finds this, its signal is not resting on any single tell.
    If it does not, we have found the ceiling honestly, in private, rather than on-chain.
    """

    def __init__(self, profile: BotProfile, rng_seed: Optional[int] = None):
        super().__init__(profile, rng_seed=rng_seed)
        self.base_aggression = profile.aggression

    def _grid_size(self, pot: int) -> int:
        # Continuous, human-like spread of pot fractions instead of a fixed menu.
        frac = min(1.5, max(0.15, self.mix.gauss(0.62, 0.28)))
        return max(1, int(round(pot * frac)))

    def act(self, state: GameState, legal: LegalActions) -> BotDecision:
        drift = 0.16 * self.mix.gauss(0.0, 1.0)
        self.profile.aggression = min(0.95, max(0.25, self.base_aggression + drift))
        return super().act(state, legal)


class MimicBot(SandboxPokerBot):
    """Rule-bot that fakes human inconsistency: per-hand noise plus session drift.

    Directly attacks pol__drift__aggro_std and the determinism features.
    """

    def __init__(self, profile: BotProfile, rng_seed: Optional[int] = None):
        super().__init__(profile, rng_seed=rng_seed)
        self.noise = random.Random((rng_seed or 0) ^ 0x717C)
        self.hand_seen = 0
        self.base_aggression = profile.aggression

    def act(self, state: GameState, legal: LegalActions) -> BotDecision:
        self.hand_seen += 1

        # Drift aggression across the session the way a tilting human would, and
        # jitter it per hand so the policy is never quite the same twice.
        drift = 0.18 * self.noise.gauss(0.0, 1.0)
        wobble = 0.10 * self.noise.gauss(0.0, 1.0)
        self.profile.aggression = min(0.95, max(0.25, self.base_aggression + drift + wobble))

        decision = super().act(state, legal)

        # Occasionally take a deliberately off-policy line, as a human misclick or
        # mood swing would.
        if self.noise.random() < 0.07:
            if legal.can_check:
                return BotDecision(action=ActionType.CHECK, amount=0)
            if legal.can_call:
                return BotDecision(action=ActionType.CALL, amount=legal.call_amount)
        return decision
