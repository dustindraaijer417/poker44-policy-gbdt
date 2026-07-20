"""Train v5: size-invariant features + within-batch rank normalization.

Validation mirrors serving exactly: the held-out release is scored as a *batch*,
with feature ranks computed inside that batch, at live chunk size. A model that
only works when it can see benchmark-scale absolute values cannot look good here.

Reports, per held-out release:
  * AUC at benchmark chunk size (30-40 hands)
  * AUC at LIVE chunk size (70-120 hands)  <- the number that matters
  * reward under the validator's own reward(), after exact_rank_map calibration
"""

from __future__ import annotations

import json
import os
import random

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

from p44.dataset import load_groups
from p44.payload import to_live_view
from p44.rank_norm import chunk_tie_key, exact_rank_map, rank_normalize, rank_normalize_grouped
from p44.v4_features import extract_v4
from poker44.score.scoring import reward

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "artifacts")
LIVE_MIN, LIVE_MAX = 70, 120
FLAG_FRACTION = 0.10          # rivals' sweep found 0.10 optimal, 0.20 always worse
SEEDS = (0, 1, 2)


def mat(chunks, names):
    rows = [[f.get(n, 0.0) for n in names] for f in (extract_v4(c) for c in chunks)]
    return np.nan_to_num(np.asarray(rows, float), nan=0.0, posinf=0.0, neginf=0.0)


def merge_to_live(chunks, y, rng):
    by = {0: [], 1: []}
    for c, yy in zip(chunks, y):
        by[int(yy)].append(c)
    oc, oy = [], []
    for lab, pool in by.items():
        pool = pool[:]; rng.shuffle(pool)
        i = 0
        while i < len(pool):
            target = rng.randint(LIVE_MIN, LIVE_MAX)
            merged = []
            while i < len(pool) and len(merged) < target:
                merged.extend(pool[i]); i += 1
            if len(merged) >= LIVE_MIN // 2:
                oc.append(merged); oy.append(lab)
    return oc, np.array(oy)


def fit(X, y):
    models = []
    for s in SEEDS:
        m = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=63,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                               reg_lambda=1.0, min_child_samples=20,
                               random_state=s, verbose=-1)
        m.fit(X, y); models.append(m)
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=3000, C=0.2, class_weight="balanced"))
    lr.fit(X, y); models.append(lr)
    return models


def pred(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def main():
    os.makedirs(OUT, exist_ok=True)
    G = load_groups()
    chunks = [[to_live_view(h) for h in hands] for hands, _, _, _ in G]
    y = np.array([lab for _, lab, _, _ in G])
    dates = np.array([d for _, _, d, _ in G])
    names = sorted(extract_v4(chunks[0]).keys())
    print(f"groups={len(chunks)} features={len(names)} (rank-normalized within batch)")

    days = sorted([d for d in set(dates.tolist()) if int((dates == d).sum()) >= 100])
    rng = random.Random(11)
    print(f"\n{'held-out':13} {'AUC@bench':>10} {'AUC@LIVE':>9} {'AP@LIVE':>8} {'REWARD':>8}")
    print("-" * 52)
    ab, al, pl, rw = [], [], [], []
    for d in days:
        te = dates == d; tr = ~te
        tr_chunks = [chunks[i] for i in np.flatnonzero(tr)]
        tr_y, tr_dates = y[tr], dates[tr]
        # augment with live-scale merges, grouped under the same date
        mc, my = merge_to_live(tr_chunks, tr_y, rng)
        md = np.array([f"{d}-merged"] * len(mc))
        Xtr = np.vstack([mat(tr_chunks, names), mat(mc, names)])
        ytr = np.concatenate([tr_y, my])
        gtr = np.concatenate([tr_dates, md])
        models = fit(rank_normalize_grouped(Xtr, gtr), ytr)   # rank within date

        te_chunks = [chunks[i] for i in np.flatnonzero(te)]
        te_y = y[te]
        # benchmark-size batch
        a_b = roc_auc_score(te_y, pred(models, rank_normalize(mat(te_chunks, names))))
        # live-size batch -- ranks computed inside the batch, exactly as at serve
        lc, ly = merge_to_live(te_chunks, te_y, random.Random(5))
        p = pred(models, rank_normalize(mat(lc, names)))
        a_l = roc_auc_score(ly, p); ap_l = average_precision_score(ly, p)
        cal = np.array(exact_rank_map(p, FLAG_FRACTION,
                                      tie_keys=[chunk_tie_key(c) for c in lc]))
        r = reward(cal, ly)[0]
        ab.append(a_b); al.append(a_l); pl.append(ap_l); rw.append(r)
        print(f"{d:13} {a_b:10.4f} {a_l:9.4f} {ap_l:8.4f} {r:8.4f}", flush=True)
    print("-" * 52)
    print(f"{'MEAN':13} {np.mean(ab):10.4f} {np.mean(al):9.4f} {np.mean(pl):8.4f} {np.mean(rw):8.4f}")
    print(f"{'WORST':13} {np.min(ab):10.4f} {np.min(al):9.4f} {np.min(pl):8.4f} {np.min(rw):8.4f}")

    mc, my = merge_to_live(chunks, y, rng)
    X = np.vstack([mat(chunks, names), mat(mc, names)])
    yy = np.concatenate([y, my])
    gg = np.concatenate([dates, np.array(["merged"] * len(mc))])
    models = fit(rank_normalize_grouped(X, gg), yy)
    joblib.dump({"models": models, "feature_names": names, "kind": "v5-rank-normalized",
                 "flag_fraction": FLAG_FRACTION},
                os.path.join(OUT, "detector_v5.joblib"))
    json.dump({"loro_auc_bench": float(np.mean(ab)), "loro_auc_live": float(np.mean(al)),
               "loro_ap_live": float(np.mean(pl)), "loro_reward_live": float(np.mean(rw)),
               "n_features": len(names), "flag_fraction": FLAG_FRACTION},
              open(os.path.join(OUT, "metadata_v5.json"), "w"), indent=2)
    print("\nsaved -> artifacts/detector_v5.joblib")


if __name__ == "__main__":
    main()
