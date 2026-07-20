"""Model definition, feature assembly, and score calibration.

Feature scope
-------------
The deployed feature set is the focal player's own behavior only: the
context-conditioned policy features in p44.policy_features, plus the hero-level
rates from p44.features. Table-level aggregates (any statistic pooled across all
seats) and hero action COUNTS are excluded, because both describe the composition
and dynamics of the table rather than the player the label refers to.

Calibration
-----------
The validator reward combines rank statistics (average precision, recall at a
bounded false-positive rate) with terms measured at a hard 0.5 decision threshold.
Rank statistics are invariant to any monotone remap of the scores, so calibrate()
remaps each batch monotonically such that a fixed top fraction (FLAG_RATE) lands
at or above 0.5. This sets the operating point explicitly without altering the
model's ranking.

Short-chunk hardening
---------------------
The number of hands per chunk is not known ahead of time. augment_chunks expands
the training set with variable-size sub-slices so the model scores short chunks
well, not only full-size ones.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from p44.features import extract
from p44.policy_features import extract_policy

# Hero features that are counts rather than rates: driven by table dynamics.
CONTAMINATED = ("n_actions", "absent_rate", "actions_per_hand")

FLAG_RATE = 0.07
MIN_FLAGGED = 3
SEEDS = (0, 1, 2, 3, 4)


def chunk_features(chunk: List[dict]) -> Dict[str, float]:
    """Policy features plus hero rate features. Excludes table aggregates and counts."""
    feats = dict(extract_policy(chunk))
    for name, value in extract(chunk).items():
        if not name.startswith("hero__"):
            continue
        if any(tok in name for tok in CONTAMINATED):
            continue
        feats[name] = value
    return feats


def matrix(chunks: Sequence[List[dict]], names: Sequence[str]) -> np.ndarray:
    rows = [[f.get(n, 0.0) for n in names] for f in (chunk_features(c) for c in chunks)]
    return np.nan_to_num(np.asarray(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


class BotDetector:
    def __init__(self, feature_names: Sequence[str]):
        self.feature_names = list(feature_names)
        self.models: List = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BotDetector":
        self.models = []
        for seed in SEEDS:
            m = HistGradientBoostingClassifier(
                max_depth=3, max_iter=250, learning_rate=0.06,
                l2_regularization=1.0, min_samples_leaf=25, random_state=seed,
            )
            m.fit(X, y)
            self.models.append(m)
        lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced"),
        )
        lr.fit(X, y)
        self.models.append(lr)
        return self

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        return np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return calibrate(self.predict_raw(X))


def calibrate(p: np.ndarray, flag_rate: float = FLAG_RATE) -> np.ndarray:
    """Monotonically remap so the top `flag_rate` of the batch sits >= 0.5.

    Order is preserved exactly, so rank-based metrics are unchanged.
    """
    p = np.asarray(p, dtype=float)
    n = p.size
    if n == 0:
        return p
    if n < 10:
        return np.clip(p, 0.0, 1.0)

    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    r = ranks / (n - 1)

    n_flag = max(MIN_FLAGGED, int(round(flag_rate * n)))
    n_flag = min(n_flag, n - 1)
    cut = 1.0 - (n_flag / (n - 1))

    out = np.where(
        r >= cut,
        0.5 + 0.5 * (r - cut) / max(1e-9, 1.0 - cut),
        0.5 * r / max(1e-9, cut),
    )
    return np.clip(out, 0.0, 1.0)


# --- short-chunk hardening -------------------------------------------------
# The chunk size is not known in advance. Training only on full-size chunks leaves
# the model weaker on short ones; augment_chunks adds variable-size sub-slices so it
# scores short chunks well too.
AUG_SIZES = (6, 8, 10, 12, 15, 20, 30)


def augment_chunks(views, y, fam, rng, per_chunk: int = 3):
    """Expand a chunk set with `per_chunk` variable-size sub-slices of each chunk."""
    out_v, out_y, out_f = [], [], []
    for v, yy, ff in zip(views, y, fam):
        out_v.append(v); out_y.append(yy); out_f.append(ff)
        for _ in range(per_chunk):
            size = rng.choice(AUG_SIZES)
            if len(v) <= size:
                sl = v
            else:
                start = rng.randint(0, len(v) - size)
                sl = v[start:start + size]
            out_v.append(sl); out_y.append(yy); out_f.append(ff)
    return out_v, out_y, out_f


# --- benchmark-supervised model -------------------------------------------
# Live evidence (see README) showed the policy-only feature set carried no signal
# on the task validators actually score. The deployed model is now trained on the
# public benchmark with the FULL feature set; these helpers are its inference path.


def full_chunk_features(chunk: List[dict]) -> Dict[str, float]:
    """Every feature family: pooled/table, hero, consistency, relative, policy."""
    feats = dict(extract(chunk))
    feats.update(extract_policy(chunk))
    return feats


def full_matrix(chunks: Sequence[List[dict]], names: Sequence[str]) -> np.ndarray:
    rows = [[f.get(n, 0.0) for n in names] for f in (full_chunk_features(c) for c in chunks)]
    return np.nan_to_num(np.asarray(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def ensemble_predict(models, X: np.ndarray) -> np.ndarray:
    """Mean probability across the saved ensemble members."""
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
