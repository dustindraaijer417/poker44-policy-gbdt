"""Benchmark-supervised detector — the approach the live board says actually works.

Why this replaces the previous strategy
---------------------------------------
The prior model trained on real human hands vs a locally-generated bot, on the
theory that the public benchmark was a decoy. Live evidence refuted that:

  * our deployed model scores AUC 0.5168 on held-out benchmark labels (no signal)
  * it scores composite 0.4575 live, rank 63 -- which is the signal-free floor
  * four of the live top ten are "benchmark-supervised" models at 0.57-0.58

So the benchmark does carry signal that transfers live, and the earlier
exclusions (drop the benchmark, drop table-level features) were both derived from
a *simulated* bot distribution rather than from live behavior. Here we train
directly on the benchmark, with the full feature set, and validate the only way
that has proven meaningful: leave-one-release-out on releases the model never saw.

What is kept from the prior work
--------------------------------
  * live-view projection (validator truncates hands to a 5-8 action window)
  * short-chunk hardening (chunk size is not known in advance)
  * rank calibration (pure math about the reward's hard 0.5 threshold, not a thesis)
"""

from __future__ import annotations

import json
import os
import random

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from p44.dataset import load_groups
from p44.features import extract
from p44.model import AUG_SIZES, calibrate
from p44.payload import to_live_view
from p44.policy_features import extract_policy

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "artifacts")
SEEDS = (0, 1, 2, 3, 4)


def full_features(chunk):
    """Everything: pooled/table + hero + consistency + relative + policy."""
    f = dict(extract(chunk))
    f.update(extract_policy(chunk))
    return f


def build_matrix(chunks, names):
    rows = [[f.get(n, 0.0) for n in names] for f in (full_features(c) for c in chunks)]
    return np.nan_to_num(np.asarray(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def fit_ensemble(X, y):
    models = []
    for s in SEEDS:
        m = HistGradientBoostingClassifier(
            max_depth=4, max_iter=350, learning_rate=0.06,
            l2_regularization=1.0, min_samples_leaf=20, random_state=s)
        m.fit(X, y)
        models.append(m)
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=3000, C=0.2, class_weight="balanced"))
    lr.fit(X, y)
    models.append(lr)
    return models


def predict(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def augment(chunks, y, rng, per=2):
    """Variable-size sub-slices so short chunks score well too."""
    cs, ys = [], []
    for c, yy in zip(chunks, y):
        cs.append(c); ys.append(yy)
        for _ in range(per):
            size = rng.choice(AUG_SIZES)
            if len(c) <= size:
                cs.append(c)
            else:
                st = rng.randint(0, len(c) - size)
                cs.append(c[st:st + size])
            ys.append(yy)
    return cs, np.asarray(ys, int)


def main():
    os.makedirs(OUT, exist_ok=True)
    G = load_groups()
    print(f"benchmark groups: {len(G)} across {len(set(g[2] for g in G))} releases")

    chunks = [[to_live_view(h) for h in hands] for hands, _, _, _ in G]
    y = np.array([lab for _, lab, _, _ in G])
    dates = np.array([d for _, _, d, _ in G])
    names = sorted(full_features(chunks[0]).keys())
    X = build_matrix(chunks, names)
    print(f"features: {len(names)} (pooled + hero + consistency + relative + policy)\n")

    # Honest protocol: hold out one release at a time (model never saw that day).
    test_days = sorted([d for d in set(dates.tolist())
                        if int((dates == d).sum()) >= 100])
    print("=== leave-one-release-out (held-out day never seen in training) ===")
    print(f"{'held-out release':18} {'AUC':>7} {'AP':>7}")
    print("-" * 36)
    aucs, aps = [], []
    rng = random.Random(7)
    for d in test_days:
        te = dates == d
        tr = ~te
        tr_chunks = [chunks[i] for i in np.flatnonzero(tr)]
        a_chunks, a_y = augment(tr_chunks, y[tr], rng)
        Xa = build_matrix(a_chunks, names)
        models = fit_ensemble(Xa, a_y)
        p = predict(models, X[te])
        auc = roc_auc_score(y[te], p); ap = average_precision_score(y[te], p)
        aucs.append(auc); aps.append(ap)
        print(f"{d:18} {auc:7.4f} {ap:7.4f}")
    print("-" * 36)
    print(f"{'MEAN':18} {np.mean(aucs):7.4f} {np.mean(aps):7.4f}")
    print(f"{'WORST':18} {np.min(aucs):7.4f} {np.min(aps):7.4f}")

    # Final model on everything, augmented.
    a_chunks, a_y = augment(chunks, y, rng)
    Xa = build_matrix(a_chunks, names)
    models = fit_ensemble(Xa, a_y)
    joblib.dump({"models": models, "feature_names": names, "kind": "benchmark-supervised"},
                os.path.join(OUT, "detector_benchmark.joblib"))
    json.dump({"trained_on": "public Poker44 benchmark (all releases), live-view projected",
               "n_groups": len(G), "n_features": len(names),
               "loro_mean_auc": float(np.mean(aucs)), "loro_mean_ap": float(np.mean(aps)),
               "loro_worst_auc": float(np.min(aucs))},
              open(os.path.join(OUT, "metadata_benchmark.json"), "w"), indent=2)
    print(f"\nsaved -> artifacts/detector_benchmark.joblib")


if __name__ == "__main__":
    main()
