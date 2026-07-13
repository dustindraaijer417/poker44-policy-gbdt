"""Train the detector and validate it leave-one-bot-family-out.

The model is trained on real human hands versus generated bot hands (see
p44.realistic). Validation holds out one bot family at a time: the model is trained
on the remaining families and evaluated against real humans and the held-out bot
type, which it has never seen. Training chunks are augmented to a range of sizes so
the model scores short chunks as well as full-size ones.

Scored with the validator's own reward() function, imported from the subnet package
so it cannot drift from upstream.
"""

from __future__ import annotations

import json
import os
import pickle

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from p44.model import FLAG_RATE, BotDetector, augment_chunks, calibrate, chunk_features, matrix
from p44.payload import to_live_view
from p44.realistic import all_families, build as build_realistic
from poker44.score.scoring import reward

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "artifacts")
CHUNKS = os.path.join(ROOT, "data", "adversarial_chunks.pkl")


def get_chunks():
    if os.path.exists(CHUNKS):
        with open(CHUNKS, "rb") as fh:
            return pickle.load(fh)
    chunks, labels, family = build_realistic(n_bot_chunks_per_family=150)
    views = [[to_live_view(h) for h in c] for c in chunks]
    blob = (views, labels, family)
    with open(CHUNKS, "wb") as fh:
        pickle.dump(blob, fh)
    return blob


def main():
    os.makedirs(OUT, exist_ok=True)
    views, labels, family = get_chunks()
    y = np.asarray(labels, int)
    fam = np.asarray(family)

    names = sorted(chunk_features(views[0]).keys())
    X = matrix(views, names)
    print(f"chunks={len(views)}  bots={int((y==1).sum())}  real-humans={int((y==0).sum())}")
    print(f"features={len(names)}  (policy + clean hero; no table features)\n")

    human = np.flatnonzero(y == 0)
    rng = np.random.default_rng(0)
    rng.shuffle(human)
    h_te, h_tr = human[: len(human) // 2], human[len(human) // 2:]

    print("=== leave-one-bot-family-out (real humans vs an UNSEEN bot type) ===")
    print(f"{'held-out bot family':22} {'AP':>6} {'AUC':>6} {'recall':>7} {'sanity':>7} {'REWARD':>7}")
    print("-" * 60)
    rows = []
    for f in sorted(all_families()):
        tr = np.r_[np.flatnonzero((y == 1) & (fam != f)), h_tr]
        te = np.r_[np.flatnonzero(fam == f), h_te]
        det = BotDetector(names).fit(X[tr], y[tr])
        p = det.predict(X[te])
        rew, m = reward(p, y[te])
        rows.append((rew, m["ap_score"], roc_auc_score(y[te], p), m["bot_recall"],
                     m["human_safety_penalty"]))
        print(f"{f:22} {m['ap_score']:6.3f} {rows[-1][2]:6.3f} {m['bot_recall']:7.3f} "
              f"{m['human_safety_penalty']:7.2f} {rew:7.3f}")
    a = np.array(rows)
    print("-" * 60)
    print(f"{'MEAN':22} {a[:,1].mean():6.3f} {a[:,2].mean():6.3f} {a[:,3].mean():7.3f} "
          f"{a[:,4].mean():7.2f} {a[:,0].mean():7.3f}")
    print(f"{'WORST':22} {a[:,1].min():6.3f} {a[:,2].min():6.3f} {a[:,3].min():7.3f} "
          f"{a[:,4].min():7.2f} {a[:,0].min():7.3f}")

    # ---- floor: what if none of it transfers to a real bot? -----------------
    print("\n=== transfer floor ===")
    rng2 = np.random.default_rng(0)
    n = 150
    yq = np.r_[np.ones(n // 2, int), np.zeros(n // 2, int)]
    print(f"  random ranking + our calibration : reward = "
          f"{reward(calibrate(rng2.random(n)), yq)[0]:.3f}")
    print("  (the live field currently sits at ~0.54, i.e. exactly this floor)")

    # ---- final model on everything, short-chunk hardened --------------------
    import random as _random
    av, ay, _ = augment_chunks(views, y.tolist(), fam.tolist(), _random.Random(99))
    det = BotDetector(names).fit(matrix(av, names), np.asarray(ay, int))
    joblib.dump({"model": det, "feature_names": names}, os.path.join(OUT, "detector.joblib"))
    json.dump({
        "trained_on": "real human corpus vs SandboxPokerBot (upstream generator)",
        "n_chunks": len(views),
        "n_features": len(names),
        "bot_families": sorted(all_families()),
        "lofo_mean_reward": float(a[:, 0].mean()),
        "lofo_mean_ap": float(a[:, 1].mean()),
        "lofo_worst_ap": float(a[:, 1].min()),
        "flag_rate": FLAG_RATE,
        "excludes": "all table/pooled features and hero counts (table-composition artifacts)",
    }, open(os.path.join(OUT, "metadata.json"), "w"), indent=2)
    print(f"\nsaved -> {os.path.abspath(os.path.join(OUT, 'detector.joblib'))}")


if __name__ == "__main__":
    main()
