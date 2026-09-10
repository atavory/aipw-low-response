#!/usr/bin/env python3
"""Reference implementation of the paper's candidate-selection calculation."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionResult:
    gamma: float
    losses: dict[float, float]
    eligible: tuple[float, ...]


def select_gamma(
    outcomes: Sequence[float],
    responses: Sequence[int | bool],
    response_predictions: Sequence[float],
    candidate_predictions: Mapping[float, Sequence[float]],
    gamma_grid: Iterable[float] = (0.0, 0.25, 0.5, 1.0),
    standard_error_multiplier: float = 1.0,
) -> SelectionResult:
    """Select a residual-update magnitude by weighted held-out loss.

    This implements the convention used for the reported replay: losses and
    their paired standard errors are calculated among respondents. The common
    respondent fraction does not affect candidate-loss ordering.
    """
    grid = tuple(float(value) for value in gamma_grid)
    if not grid or 0.0 not in grid:
        raise ValueError("gamma_grid must contain 0")
    n = len(outcomes)
    if len(responses) != n or len(response_predictions) != n:
        raise ValueError("input lengths differ")
    for gamma in grid:
        if gamma not in candidate_predictions or len(candidate_predictions[gamma]) != n:
            raise ValueError(f"missing or malformed predictions for gamma={gamma}")

    respondent_indices = [index for index, value in enumerate(responses) if value]
    if not respondent_indices:
        return SelectionResult(0.0, {gamma: math.inf for gamma in grid}, (0.0,))

    vectors: dict[float, list[float]] = {}
    losses: dict[float, float] = {}
    for gamma in grid:
        vector = []
        for index in respondent_indices:
            probability = float(response_predictions[index])
            if not 0.0 < probability <= 1.0:
                raise ValueError("response predictions must lie in (0, 1]")
            residual = float(outcomes[index]) - float(candidate_predictions[gamma][index])
            vector.append((1.0 - probability) * residual * residual / probability**2)
        vectors[gamma] = vector
        losses[gamma] = statistics.fmean(vector)

    eligible = [0.0]
    reference = vectors[0.0]
    for gamma in grid:
        if gamma == 0.0:
            continue
        improvement = [base - candidate for base, candidate in zip(reference, vectors[gamma])]
        if len(improvement) < 2:
            continue
        standard_error = statistics.stdev(improvement) / math.sqrt(len(improvement))
        if statistics.fmean(improvement) > standard_error_multiplier * standard_error:
            eligible.append(gamma)

    selected = min(eligible, key=lambda gamma: (losses[gamma], gamma))
    return SelectionResult(selected, losses, tuple(eligible))

