"""Learner-observable PFSP scoring."""

from __future__ import annotations

import math
import numpy as np


def negotiation_score(agreement_rate, utility, length, length_weight=-0.005, epsilon=0.01):
    agreement_rate = min(max(float(agreement_rate), 0.0), 1.0 - float(epsilon))
    utility = float(utility)
    length = float(length)
    if not all(math.isfinite(value) for value in (agreement_rate, utility, length, length_weight, epsilon)):
        raise ValueError("Score inputs must be finite")
    return (1.0 - agreement_rate) ** (-utility) + float(length_weight) * length


def pfsp_probabilities(entries, alpha=2.0, uniform_mix=0.1):
    if not entries:
        raise ValueError("Empty opponent pool")
    if not math.isfinite(alpha) or alpha < 0 or not 0.0 <= uniform_mix <= 1.0:
        raise ValueError("Invalid PFSP parameters")
    difficulties = np.asarray([entry.difficulty for entry in entries], dtype=np.float64)
    if not np.all(np.isfinite(difficulties)):
        raise ValueError("PFSP difficulties must be finite")
    shifted = np.clip(alpha * (difficulties - difficulties.max()), -700, 0)
    weights = np.exp(shifted)
    probabilities = weights / weights.sum()
    probabilities = (1.0 - uniform_mix) * probabilities + uniform_mix / len(entries)
    probabilities /= probabilities.sum()
    if np.any(probabilities < 0) or not np.all(np.isfinite(probabilities)):
        raise ValueError("Invalid PFSP distribution")
    return probabilities
