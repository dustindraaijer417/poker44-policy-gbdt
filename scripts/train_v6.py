"""Train v6: diverse rank-blended ensemble, selected for stability, not peak score.

v5 averaged AUC@live 0.8873 but ranged 0.777-0.980 across held-out releases. A
rank that holds every cycle needs the floor raised, not just the mean, so v6
changes three things:

1. BRANCH DIVERSITY. v5 was three LightGBMs plus a logistic -- highly correlated,
   so they fail together. v6 runs five different model families (boosted trees,
   two randomized forests, a linear model). Decorrelated errors are what shrink
   fold-to-fold variance.

2. RANK BLENDING, not probability averaging. If one branch collapses on a shifted
   release, probability-averaging drags the whole ensemble down with it. Blending
   the branches' *ranks* bounds any single branch's influence: a collapsed branch
   contributes a uniform ranking, which is neutral rather than destructive.

3. HUMAN AUGMENTATION. The reward punishes false positives hard (the hard-FPR
   term), and the benchmark is only ~50% human. Extra synthetic all-human chunks,
   built by recombining hands from human chunks, push the decision boundary away
   from humans where the penalty lives.

Selection reports mean, std and worst fold. A candidate that wins on mean but
loses on the floor is not an improvement for our purpose.
"""

from __future__ import annotations

import json
import os
import random

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

from p44.dataset import load_groups
from p44.payload import to_live_view
from p44.rank_norm import chunk_tie_key, exact_rank_map, rank_normalize, rank_normalize_grouped
from p44.v4_features import extract_v4
from poker44.score.scoring import reward

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "artifacts")
LIVE_MIN, LIVE_MAX = 70, 120
FLAG_FRACTION = 0.10
HUMAN_AUG_RATIO = 0.35      # extra synthetic human chunks, as a fraction of real ones


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


def synth_humans(chunks, y, rng, ratio=HUMAN_AUG_RATIO):
    """Extra all-human chunks by recombining hands drawn from human chunks.

    Every hand keeps its real provenance -- only the grouping is synthetic -- so
    these are genuinely human-behaviour chunks, just novel combinations.
    """
    pool = [h for c, yy in zip(chunks, y) if yy == 0 for h in c]
    if not pool:
        return [], np.array([])
    n_new = int(len([1 for yy in y if yy == 0]) * ratio)
    out = []
    for _ in range(n_new):
        k = rng.randint(LIVE_MIN, LIVE_MAX)
        out.append([pool[rng.randrange(len(pool))] for _ in range(k)])
    return out, np.zeros(len(out), dtype=int)


def branches(seed=0):
    return [
        ("lgbm", lgb.LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=63,
                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                                    reg_lambda=1.0, min_child_samples=20,
                                    random_state=seed, verbose=-1)),
        ("xgb", xgb.XGBClassifier(n_estimators=700, learning_rate=0.035, max_depth=6,
                                  subsample=0.85, colsample_bytree=0.7, reg_lambda=1.5,
                                  tree_method="hist", random_state=seed,
                                  eval_metric="logloss")),
        ("extratrees", ExtraTreesClassifier(n_estimators=600, max_depth=12,
                                            max_features="sqrt", min_samples_leaf=2,
                                            class_weight="balanced_subsample",
                                            random_state=seed, n_jobs=-1)),
        ("rf", RandomForestClassifier(n_estimators=500, max_depth=12, max_features="sqrt",
                                      min_samples_leaf=2, class_weight="balanced_subsample",
                                      random_state=seed, n_jobs=-1)),
        ("logistic", make_pipeline(StandardScaler(),
                                   LogisticRegression(max_iter=3000, C=0.2,
                                                      class_weight="balanced"))),
    ]


def fit_branches(X, y, seed=0):
    out = []
    for name, m in branches(seed):
        m.fit(X, y)
        out.append((name, m))
    return out


def _to_rank(v):
    v = np.asarray(v, float)
    n = v.size
    if n <= 1:
        return np.full(n, 0.5)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(n, float)
    r[order] = np.arange(n, dtype=float)
    return r / (n - 1)


def predict_blend(fitted, X, mode="rank"):
    probs = [m.predict_proba(X)[:, 1] for _, m in fitted]
    if mode == "prob" or X.shape[0] < 8:
        return np.mean(probs, axis=0)
    return np.mean([_to_rank(p) for p in probs], axis=0)


def main():
    os.makedirs(OUT, exist_ok=True)
    G = load_groups()
    chunks = [[to_live_view(h) for h in hands] for hands, _, _, _ in G]
    y = np.array([lab for _, lab, _, _ in G])
    dates = np.array([d for _, _, d, _ in G])
    names = sorted(extract_v4(chunks[0]).keys())
    days = sorted([d for d in set(dates.tolist()) if int((dates == d).sum()) >= 100])
    print(f"groups={len(chunks)} features={len(names)} folds={len(days)}")
    print("branches: lgbm, xgb, extratrees, rf, logistic (rank-blended)\n")

    rng = random.Random(11)
    print(f"{'held-out':13} {'AUC@LIVE':>9} {'AP@LIVE':>8} {'REWARD':>8}")
    print("-" * 42)
    al, pl, rw = [], [], []
    for d in days:
        te = dates == d; tr = ~te
        tr_chunks = [chunks[i] for i in np.flatnonzero(tr)]
        tr_y, tr_dates = y[tr], dates[tr]
        mc, my = merge_to_live(tr_chunks, tr_y, rng)
        hc, hy = synth_humans(tr_chunks, tr_y, rng)
        Xtr = np.vstack([mat(tr_chunks, names), mat(mc, names), mat(hc, names)])
        ytr = np.concatenate([tr_y, my, hy])
        gtr = np.concatenate([tr_dates,
                              np.array([f"{d}-m"] * len(mc)),
                              np.array([f"{d}-h"] * len(hc))])
        fitted = fit_branches(rank_normalize_grouped(Xtr, gtr), ytr)

        te_chunks = [chunks[i] for i in np.flatnonzero(te)]
        lc, ly = merge_to_live(te_chunks, y[te], random.Random(5))
        p = predict_blend(fitted, rank_normalize(mat(lc, names)))
        a = roc_auc_score(ly, p); ap = average_precision_score(ly, p)
        cal = np.array(exact_rank_map(p, FLAG_FRACTION,
                                      tie_keys=[chunk_tie_key(c) for c in lc]))
        r = reward(cal, ly)[0]
        al.append(a); pl.append(ap); rw.append(r)
        print(f"{d:13} {a:9.4f} {ap:8.4f} {r:8.4f}", flush=True)

    al, pl, rw = np.array(al), np.array(pl), np.array(rw)
    print("-" * 42)
    print(f"{'MEAN':13} {al.mean():9.4f} {pl.mean():8.4f} {rw.mean():8.4f}")
    print(f"{'STD':13} {al.std():9.4f} {pl.std():8.4f} {rw.std():8.4f}")
    print(f"{'WORST':13} {al.min():9.4f} {pl.min():8.4f} {rw.min():8.4f}")
    print(f"\nstability objective (mean - 0.5*std) on AUC@live: {al.mean() - 0.5*al.std():.4f}")
    print("v5 reference: mean 0.8873, worst 0.7767")

    # final fit on everything
    mc, my = merge_to_live(chunks, y, rng)
    hc, hy = synth_humans(chunks, y, rng)
    X = np.vstack([mat(chunks, names), mat(mc, names), mat(hc, names)])
    yy = np.concatenate([y, my, hy])
    gg = np.concatenate([dates, np.array(["m"] * len(mc)), np.array(["h"] * len(hc))])
    fitted = fit_branches(rank_normalize_grouped(X, gg), yy)
    joblib.dump({"models": [m for _, m in fitted],
                 "branch_names": [n for n, _ in fitted],
                 "feature_names": names, "kind": "v6-rank-blend",
                 "blend": "rank", "flag_fraction": FLAG_FRACTION},
                os.path.join(OUT, "detector_v6.joblib"))
    json.dump({"loro_auc_live_mean": float(al.mean()), "loro_auc_live_std": float(al.std()),
               "loro_auc_live_worst": float(al.min()), "loro_ap_live_mean": float(pl.mean()),
               "loro_reward_mean": float(rw.mean()), "n_features": len(names),
               "branches": [n for n, _ in fitted], "flag_fraction": FLAG_FRACTION},
              open(os.path.join(OUT, "metadata_v6.json"), "w"), indent=2)
    print("\nsaved -> artifacts/detector_v6.joblib")


if __name__ == "__main__":
    main()
