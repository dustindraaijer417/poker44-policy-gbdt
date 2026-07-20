"""Poker44 bot-detection miner.

Returns one bot-risk score per chunk. Chunks arrive already projected through the
validator canonicalizer, so hands are featurized as received; the training pipeline
applies the same projection so the distributions match.

Scores are rank-calibrated per batch (see p44.model.calibrate).
"""

import time
import traceback
from pathlib import Path
from typing import List, Tuple

import bittensor as bt
import joblib
import numpy as np

from p44.rank_norm import chunk_tie_key, exact_rank_map, rank_normalize
from p44.v4_features import extract_v4
from poker44.base.miner import BaseMinerNeuron
from poker44.base.neuron import BaseNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse


def _install_config_shim() -> None:
    """Backfill the custom config fields dropped by newer bittensor.

    The subnet's argparse defines ``--netuid`` and a ``--neuron.*`` namespace, but
    bittensor >= 10.4 builds ``bt.Config`` in a way that discards these custom
    entries (its own namespaces such as ``axon``/``wallet`` survive). The result:
    ``config.netuid`` is ``None`` -- so ``metagraph(None)`` is sent to the chain
    and its runtime traps in ``get_neurons_lite`` -- and ``config.neuron`` is
    ``None``, which crashes ``check_config``. We wrap ``BaseNeuron.config`` to
    reconstruct both from the subnet's own argument defaults (parsing ``--netuid``
    from argv). On older bittensor, where these survive, the shim is a no-op.
    """
    import argparse
    import sys

    original = BaseNeuron.config.__func__

    def _argv() -> argparse.Namespace:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--netuid", type=int, default=126)
        parser.add_argument("--wallet.name", dest="wallet_name", default=None)
        parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", default=None)
        parser.add_argument("--blacklist.force_validator_permit",
                            dest="force_permit", action="store_true", default=True)
        parser.add_argument("--no-blacklist.force_validator_permit",
                            dest="force_permit", action="store_false")
        parser.add_argument("--blacklist.allow_non_registered",
                            dest="allow_non_registered", action="store_true", default=False)
        parser.add_argument("--blacklist.allowed_validator_hotkeys",
                            dest="allowed_hotkeys", nargs="*", default=[])
        known, _ = parser.parse_known_args(sys.argv[1:])
        return known

    def config_with_shim(cls):
        cfg = original(cls)
        argv = _argv()
        if cfg.get("netuid") is None:
            cfg.netuid = int(argv.netuid)
        # bittensor >= 10.4 leaves wallet name/hotkey at "default" instead of the
        # value passed on the command line; restore them from argv.
        if argv.wallet_name and cfg.wallet.get("name") in (None, "default"):
            cfg.wallet.name = argv.wallet_name
        if argv.wallet_hotkey and cfg.wallet.get("hotkey") in (None, "default"):
            cfg.wallet.hotkey = argv.wallet_hotkey
        # The whole ``blacklist`` namespace is a custom one and gets dropped too;
        # BaseMinerNeuron reads config.blacklist.force_validator_permit on init.
        if cfg.get("blacklist") is None:
            blacklist = bt.Config()
            blacklist.force_validator_permit = bool(argv.force_permit)
            blacklist.allow_non_registered = bool(argv.allow_non_registered)
            blacklist.allowed_validator_hotkeys = list(argv.allowed_hotkeys or [])
            cfg.blacklist = blacklist
        if cfg.get("neuron") is None:
            neuron = bt.Config()
            neuron.name = "poker44-miner"
            neuron.device = "cpu"
            neuron.epoch_length = 50
            neuron.disable_set_weights = False
            neuron.moving_average_alpha = 0.1
            neuron.num_concurrent_forwards = 1
            neuron.timeout = 180.0
            neuron.axon_off = False
            neuron.wait_for_inclusion = False
            neuron.wait_for_finalization = False
            cfg.neuron = neuron
        return cfg

    BaseNeuron.config = classmethod(config_with_shim)


_install_config_shim()

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = REPO_ROOT / "artifacts" / "detector_v5.joblib"
IMPLEMENTATION_FILES = [
    REPO_ROOT / "neurons" / "miner.py",
    REPO_ROOT / "p44" / "v4_features.py",
    REPO_ROOT / "p44" / "rank_norm.py",
    REPO_ROOT / "p44" / "model.py",
    REPO_ROOT / "p44" / "payload.py",
]


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)

        blob = joblib.load(ARTIFACT)
        self.models = blob["models"]
        self.feature_names = blob["feature_names"]
        self.flag_fraction = float(blob.get("flag_fraction", 0.10))
        bt.logging.info(
            f"Loaded {blob.get('kind', 'detector')}: {len(self.feature_names)} features, "
            f"{len(self.models)} ensemble members"
        )

        self.model_manifest = self._build_manifest()
        compliance = evaluate_manifest_compliance(self.model_manifest)
        bt.logging.info(
            f"Manifest transparency={compliance['status']} "
            f"missing={compliance['missing_fields']} "
            f"violations={compliance['policy_violations']} "
            f"digest={manifest_digest(self.model_manifest)}"
        )
        if compliance["status"] != "transparent":
            bt.logging.warning(
                "Manifest is NOT transparent -- miners with opaque manifests risk being "
                "zeroed. Fix before relying on emissions."
            )

    def _build_manifest(self) -> dict:
        manifest = build_local_model_manifest(
            repo_root=REPO_ROOT,
            implementation_files=IMPLEMENTATION_FILES,
            defaults={
                "model_name": "poker44-rank-robust",
                "model_version": "5.0",
                "framework": "scikit-learn",
                "license": "MIT",
                "open_source": True,
                "inference_mode": "remote",
                "notes": (
                    "Chunk-level bot detector. Chunk-size-invariant behavioral features "
                    "(per-hand action-mix and entropy rates, pot-relative bet sizing, "
                    "action-token n-grams with a frozen vocabulary, and sequence-repetition "
                    "statistics on a fixed subsample). Features are rank-normalized within "
                    "each request so the serving distribution matches training regardless of "
                    "chunk length or bet scale. LightGBM ensemble plus a regularized linear "
                    "model. Output is an order-preserving rank map that places a fixed top "
                    "fraction above the 0.5 decision threshold, with ties broken by a "
                    "content hash so scores do not depend on request ordering."
                ),
                "training_data_statement": (
                    "Trained on the public Poker44 training benchmark served by "
                    "https://api.poker44.net/api/v1/benchmark (all released chunk groups, "
                    "2026-05-30..2026-07-20), using the published groundTruth labels. Hands "
                    "are projected through poker44.validator.payload_view.prepare_hand_for_miner "
                    "before featurization, and training chunks are merged to the chunk sizes "
                    "seen in production. Validation is leave-one-release-out at production "
                    "chunk size: mean AUC 0.887, AP 0.912 over 15 held-out release dates. No "
                    "validator-only or private evaluation data was used."
                ),
                "training_data_sources": [
                    "https://api.poker44.net/api/v1/benchmark",
                ],
                "private_data_attestation": (
                    "This model does not train on validator-only evaluation data, and does "
                    "not use any private, scraped, or non-public poker data."
                ),
            },
        )
        # build_local_model_manifest omits data_attestation, but validator policy
        # checks for it, so it is injected here after the build.
        manifest["data_attestation"] = (
            "All training data is the public Poker44 training benchmark, retrieved from the "
            "public benchmark API. No private hand histories, no player PII, no validator "
            "evaluation material."
        )
        return manifest

    def score_chunks(self, chunks: List[List[dict]]) -> List[float]:
        """One calibrated bot-risk score per chunk.

        Features are rank-normalized *within this request*, which is how the model
        was trained (ranks computed inside each release). That makes the serving
        distribution match training regardless of how many hands the validator puts
        in a chunk or what scale its pots and bets are on.

        The final mapping places exactly the top `flag_fraction` above 0.5 while
        preserving the model's ordering, so the reward's rank terms are untouched
        and its hard-threshold terms are pinned.
        """
        if not chunks:
            return []
        rows = [[f.get(n, 0.0) for n in self.feature_names]
                for f in (extract_v4(c) for c in chunks)]
        X = np.nan_to_num(np.asarray(rows, dtype=np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
        Xr = rank_normalize(X)
        raw = np.mean([m.predict_proba(Xr)[:, 1] for m in self.models], axis=0)
        return exact_rank_map(raw, self.flag_fraction,
                              tie_keys=[chunk_tie_key(c) for c in chunks])

    def _fallback(self, chunks: List[List[dict]]) -> List[float]:
        """Deterministic degraded path.

        Rank-calibrated like the main path, so a scoring failure still yields a
        usable score distribution rather than a constant.
        """
        crude = []
        for chunk in chunks:
            hero_aggro, n = 0.0, 0
            for hand in chunk:
                hs = (hand.get("metadata") or {}).get("hero_seat")
                for act in hand.get("actions") or []:
                    if act.get("actor_seat") == hs:
                        n += 1
                        if act.get("action_type") in ("bet", "raise"):
                            hero_aggro += 1.0
            crude.append(hero_aggro / n if n else 0.5)
        return exact_rank_map(crude, getattr(self, "flag_fraction", 0.10),
                              tie_keys=[chunk_tie_key(c) for c in chunks])

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []
        started = time.time()
        try:
            scores = self.score_chunks(chunks)
        except Exception:
            bt.logging.error(f"Scoring failed, using fallback:\n{traceback.format_exc()}")
            scores = self._fallback(chunks)

        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)

        elapsed = time.time() - started
        flagged = sum(synapse.predictions)
        bt.logging.info(
            f"Scored {len(chunks)} chunks in {elapsed:.2f}s | flagged {flagged} "
            f"({flagged / max(1, len(chunks)):.1%}) | "
            f"range [{min(scores):.3f}, {max(scores):.3f}]" if scores else "empty request"
        )
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Poker44 miner running.")
        while True:
            bt.logging.info(
                f"UID {miner.uid} | incentive={miner.metagraph.I[miner.uid]:.6f} "
                f"| block={miner.block}"
            )
            time.sleep(300)
