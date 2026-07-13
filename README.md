# poker44-policy-gbdt

An open-source bot-detection miner for the [Poker44](https://github.com/Poker44/Poker44-subnet)
Bittensor subnet (netuid 126).

The validator sends chunks of poker hands. Each chunk is one focal player's hand batch;
the miner returns one bot-risk score per chunk. This repository contains the complete
model flow: training-data construction, features, training, the trained artifact, and the
miner that serves it.

## Features

The model uses only the **focal player's own behavior**. The feature set is:

- **Policy features** (`p44/policy_features.py`) — context-conditioned response rates such
  as `P(fold | facing a bet on the flop)`, measured per `(street, facing-a-bet)` context,
  along with bet-sizing ratios, sizing dispersion, per-context entropy, and session
  aggression drift. These are the quantities a poker tracking tool uses to profile a
  player: VPIP, PFR, fold-to-bet, aggression factor, sizing distribution.
- **Hero rate features** (`p44/features.py`) — the focal seat's action mix and sizing
  statistics.

Table-level aggregates (statistics pooled across all seats) and hero action *counts* are
deliberately excluded, in `p44/model.py:CONTAMINATED` and by construction in
`chunk_features()`. Both describe the composition and dynamics of the table rather than the
player the label refers to. Conditioning on the situation is what keeps a feature about the
player: table composition changes how *often* a situation arises, not how a given player
responds once it does.

## Model

An ensemble of five shallow gradient-boosted trees (depth 3) and one regularized logistic
regression. Depth is kept low given the size of the training set.

## Calibration

The validator reward mixes rank statistics (average precision, recall at a bounded
false-positive rate) with terms evaluated at a hard `0.5` decision threshold. Rank
statistics are invariant under any monotone remap of the scores, so `p44/model.py:calibrate`
remaps each batch monotonically so that a fixed top fraction (`FLAG_RATE = 0.07`) lands at
or above `0.5`. This sets the operating point explicitly without changing the model's
ranking.

## Training data

- **Humans** — the public hand corpus shipped with the subnet repo
  (`hands_generator/human_hands/poker_hands_combined.json.gz`). Any player with enough hands
  is rebuilt as a focal player by pointing `metadata.hero_seat` at their seat. Chunks per
  player are capped so no single player dominates the class.
- **Bots** — hands generated with Poker44's `SandboxPokerBot` across a range of behavior
  profiles, plus the additional policies in `p44/adversarial_bots.py` (mixed-strategy,
  drifting, and continuous-sizing variants). Table stakes and seat counts are sampled from
  the human corpus so both classes share one table distribution.

Every hand is projected through the validator's own canonicalizer
(`poker44.validator.payload_view.prepare_hand_for_miner`) before featurization, so the
training distribution matches what the miner receives at inference.

## Validation

Leave-one-bot-family-out: the model is trained on real humans plus all but one bot family,
then evaluated on real humans plus the held-out family, which it has never seen. Scored with
the validator's own `reward()`, imported from the subnet package.

Run `python3 scripts/train.py` to reproduce the table.

## Reproduce

```bash
git clone https://github.com/Poker44/Poker44-subnet   # provides poker44 + the human corpus
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ./Poker44-subnet

export POKER44_SUBNET_PATH=./Poker44-subnet
python3 scripts/train.py        # build training set, validate, write artifacts/detector.joblib
```

## Run

```bash
WALLET_NAME=<coldkey> \
HOTKEY=<hotkey> \
AXON_PORT=8091 \
POKER44_MODEL_REPO_URL=https://github.com/dustindraaijer417/poker44-policy-gbdt \
./run_miner.sh
```

## License

MIT
