"""Training set construction: real human hands versus generated bot hands.

The human class is drawn from the public hand corpus shipped with the Poker44
subnet repository. It contains one long-running player plus thousands of
opponents; any player with enough hands is rebuilt as a focal player by pointing
metadata.hero_seat at their seat, which matches the shape of an evaluation chunk.
Chunks per player are capped so that no single player dominates the class.

The bot class is generated with the Poker44 SandboxPokerBot across a range of
behavior profiles, plus the additional policies in p44.adversarial_bots. Table
stakes and seat counts are sampled from the human corpus so both classes share one
table distribution.

Several distinct bot families are generated so the model can be validated
leave-one-family-out: trained on some bot types and evaluated on a bot type it has
never seen.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from typing import Dict, List, Tuple

from hands_generator.bot_hands.generate_poker_data import BotProfile, PokerHandGenerator
from p44.human_corpus import human_chunks

# Distinct bot archetypes. Spread across the tightness/aggression/bluff space so
# that holding one out is a genuine test of generalization, not a re-run.
BOT_FAMILIES: Dict[str, BotProfile] = {
    "tight_aggressive": BotProfile(name="tight_aggressive", tightness=0.72, aggression=0.78,
                                   bluff_freq=0.05, bet_pot_fraction_medium=0.60),
    "loose_aggressive": BotProfile(name="loose_aggressive", tightness=0.38, aggression=0.84,
                                   bluff_freq=0.14, bet_pot_fraction_medium=0.75),
    "tight_passive": BotProfile(name="tight_passive", tightness=0.70, aggression=0.33,
                                bluff_freq=0.03, bet_pot_fraction_medium=0.45),
    "loose_passive": BotProfile(name="loose_passive", tightness=0.40, aggression=0.30,
                                bluff_freq=0.07, bet_pot_fraction_medium=0.40),
    "balanced": BotProfile(name="balanced", tightness=0.55, aggression=0.55,
                           bluff_freq=0.08, bet_pot_fraction_medium=0.55),
    "nit": BotProfile(name="nit", tightness=0.80, aggression=0.45, bluff_freq=0.02,
                      max_risk_fraction_of_stack=0.12, bet_pot_fraction_medium=0.50),
    "maniac": BotProfile(name="maniac", tightness=0.30, aggression=0.92, bluff_freq=0.20,
                         max_risk_fraction_of_stack=0.30, bet_pot_fraction_medium=0.85),
    "trappy": BotProfile(name="trappy", tightness=0.62, aggression=0.42, bluff_freq=0.06,
                         trap_frequency=0.35, bet_pot_fraction_medium=0.50),
}


def _bot_chunks_for_family(
    name: str,
    profile: BotProfile,
    n_chunks: int,
    seed: int,
    min_hands: int = 30,
    max_hands: int = 40,
    bot_class=None,
) -> List[List[dict]]:
    """Generate bot hands for one family and slice them into focal-player chunks.

    bot_class swaps the policy the generator seats at the table. The generator
    hard-references SandboxPokerBot, so we rebind the symbol in its module for the
    duration of the call -- that is how the solver/mimic adversaries get played.
    """
    rng = random.Random(seed)
    need = n_chunks * max_hands
    gen = PokerHandGenerator(seed=seed)

    tmp = os.path.join(tempfile.gettempdir(), f"p44_bot_{name}_{seed}.json")
    import hands_generator.bot_hands.generate_poker_data as gpd
    original = gpd.SandboxPokerBot
    if bot_class is not None:
        gpd.SandboxPokerBot = bot_class
    try:
        gen.generate_hands(
            num_hands_to_play=int(need * 1.6),
            num_hands_to_select=need,
            bot_profiles=[profile],
            output_file=tmp,
            hands_per_session=40,
        )
    finally:
        gpd.SandboxPokerBot = original

    hands = json.load(open(tmp))
    os.unlink(tmp)

    for h in hands:
        h.pop("label", None)      # never let the label ride along in the payload

    chunks = []
    i = 0
    while i + min_hands <= len(hands) and len(chunks) < n_chunks:
        size = rng.randint(min_hands, max_hands)
        chunk = hands[i:i + size]
        if len(chunk) >= min_hands:
            chunks.append(chunk)
        i += size
    return chunks


def all_families() -> Dict[str, tuple]:
    """name -> (profile, bot_class). bot_class None = the stock rule-based bot."""
    from p44.adversarial_bots import MimicBot, SolverBot, StealthBot

    fams: Dict[str, tuple] = {n: (p, None) for n, p in BOT_FAMILIES.items()}
    # Solver/GTO paradigm: mixed strategies (human-looking entropy), fixed sizing grid.
    fams["solver_balanced"] = (
        BotProfile(name="solver_balanced", tightness=0.55, aggression=0.55), SolverBot)
    fams["solver_aggressive"] = (
        BotProfile(name="solver_aggressive", tightness=0.48, aggression=0.80), SolverBot)
    fams["solver_tight"] = (
        BotProfile(name="solver_tight", tightness=0.68, aggression=0.45), SolverBot)
    # Deliberately fakes human inconsistency and session drift.
    fams["mimic"] = (
        BotProfile(name="mimic", tightness=0.55, aggression=0.55, bluff_freq=0.10), MimicBot)
    # Attacks every top signal at once: mixed strategy + continuous sizing + drift.
    fams["stealth"] = (
        BotProfile(name="stealth", tightness=0.58, aggression=0.60), StealthBot)
    return fams


RULE_FAMILIES = set(BOT_FAMILIES)
ADVERSARIAL_FAMILIES = {"solver_balanced", "solver_aggressive", "solver_tight", "mimic", "stealth"}


def build(
    n_bot_chunks_per_family: int = 200,
    seed: int = 11,
) -> Tuple[List[List[dict]], List[int], List[str]]:
    """Return (chunks, labels, family) — family is 'human' for real-human chunks."""
    chunks: List[List[dict]] = []
    labels: List[int] = []
    family: List[str] = []

    for i, (name, (profile, bot_class)) in enumerate(sorted(all_families().items())):
        bot = _bot_chunks_for_family(
            name, profile, n_bot_chunks_per_family, seed + i, bot_class=bot_class)
        chunks.extend(bot)
        labels.extend([1] * len(bot))
        family.extend([name] * len(bot))

    # Match the human count to the bot count so the set stays balanced.
    n_bots = len(chunks)
    humans = human_chunks(seed=seed, max_chunks=n_bots, max_chunks_per_player=3)
    chunks.extend(humans)
    labels.extend([0] * len(humans))
    family.extend(["human"] * len(humans))

    return chunks, labels, family
