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

from p44.model import calibrate, ensemble_predict, full_matrix
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
ARTIFACT = REPO_ROOT / "artifacts" / "detector_benchmark.joblib"
IMPLEMENTATION_FILES = [
    REPO_ROOT / "neurons" / "miner.py",
    REPO_ROOT / "p44" / "features.py",
    REPO_ROOT / "p44" / "policy_features.py",
    REPO_ROOT / "p44" / "model.py",
    REPO_ROOT / "p44" / "payload.py",
]


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)

        blob = joblib.load(ARTIFACT)
        self.models = blob["models"]
        self.feature_names = blob["feature_names"]
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
                "model_name": "poker44-benchmark-gbdt",
                "model_version": "3.0",
                "framework": "scikit-learn",
                "license": "MIT",
                "open_source": True,
                "inference_mode": "remote",
                "notes": (
                    "Chunk-level bot detector. Full behavioral feature set over the "
                    "miner-visible payload: action-mix and street distribution, bet-sizing "
                    "level and dispersion, hero-conditioned rates, hero-vs-table contrast, "
                    "and context-conditioned policy features (P(response | street, "
                    "facing-bet)). Gradient-boosted tree ensemble plus a regularized linear "
                    "model; per-batch rank calibration against the 0.5 decision threshold. "
                    "Hands are projected through the validator canonicalizer and training "
                    "chunks are augmented to varying sizes."
                ),
                "training_data_statement": (
                    "Trained on the public Poker44 training benchmark served by "
                    "https://api.poker44.net/api/v1/benchmark (all released chunk groups, "
                    "2026-05-30..2026-07-20), using the published groundTruth labels. Every "
                    "hand is projected through poker44.validator.payload_view."
                    "prepare_hand_for_miner before featurization. Validation is "
                    "leave-one-release-out: mean AUC 0.90 / AP 0.92 on held-out release "
                    "dates. No validator-only or private evaluation data was used."
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
        """One calibrated bot-risk score per chunk."""
        if not chunks:
            return []
        X = full_matrix(chunks, self.feature_names)
        return [round(float(s), 6) for s in calibrate(ensemble_predict(self.models, X))]

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
        return [round(float(s), 6) for s in calibrate(np.asarray(crude))]

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
