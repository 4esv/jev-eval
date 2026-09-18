"""Metrics over result records. Every function takes plain lists so tests can hand-compute them.

Confidence everywhere is the top-label probability (`conf` in the records), not TypeSafe's
derived `confidence` field, so it can be checked for calibration against accuracy.
"""

import math

import numpy as np

THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95)


def accuracy(correct: list[bool]) -> float:
    return float(np.mean(correct))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (mid - half, mid + half)


def macro_f1(gold: list[str], pred: list[str]) -> float:
    """Macro-F1 over the labels that appear in gold."""
    f1s = []
    for c in sorted(set(gold)):
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def ece(conf: list[float], correct: list[bool], bins: int = 10) -> float:
    """Expected calibration error: bin-size-weighted |accuracy - mean confidence|, equal-width bins."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() / len(conf) * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability(conf: list[float], correct: list[bool], bins: int = 10) -> list[tuple[float, float, int]]:
    """(mean confidence, accuracy, count) per non-empty bin, for plotting."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return [(float(conf[idx == b].mean()), float(correct[idx == b].mean()), int((idx == b).sum()))
            for b in range(bins) if (idx == b).any()]


def brier(conf: list[float], correct: list[bool]) -> float:
    """Top-label Brier score: mean (confidence - correct)^2. Lower is better."""
    return float(np.mean((np.asarray(conf, float) - np.asarray(correct, float)) ** 2))


def auroc(conf: list[float], correct: list[bool]) -> float:
    """How well confidence separates right from wrong answers (0.5 = no signal). Ties count half."""
    pos = [c for c, ok in zip(conf, correct) if ok]
    neg = [c for c, ok in zip(conf, correct) if not ok]
    if not pos or not neg:
        return math.nan
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def selective(conf: list[float], correct: list[bool], thresholds=THRESHOLDS) -> list[tuple[float, float, float]]:
    """(threshold, coverage, accuracy on covered) when acting only at conf >= threshold."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    out = []
    for t in thresholds:
        m = conf >= t
        out.append((t, float(m.mean()), float(correct[m].mean()) if m.any() else math.nan))
    return out


def ordinal_mae(gold: list[str], pred: list[str], levels: list[str]) -> float:
    pos = {l: i for i, l in enumerate(levels)}
    return float(np.mean([abs(pos[g] - pos[p]) for g, p in zip(gold, pred)]))


def percentile(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q))


def agreement(a: dict[str, str], b: dict[str, str]) -> float:
    """Share of shared ids where two runs picked the same label."""
    ids = a.keys() & b.keys()
    return sum(a[i] == b[i] for i in ids) / len(ids)
