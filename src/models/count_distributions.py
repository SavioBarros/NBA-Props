"""Distribuições discretas para a probabilidade de superar uma linha."""
from __future__ import annotations

import math


def poisson_over(line: float, mean: float) -> float:
    """Calcula P(X > linha) para uma contagem de Poisson."""
    if line < 0:
        return 1.0
    mass = math.exp(-mean)
    cdf = mass
    for count in range(1, math.floor(line) + 1):
        mass *= mean / count
        cdf += mass
    return max(0.0, min(1.0, 1.0 - cdf))


def negative_binomial_over(line: float, mean: float, variance: float) -> float:
    """Ajusta a binomial negativa por momentos e usa Poisson sem sobredispersão."""
    if line < 0:
        return 1.0
    if mean <= 0 or variance <= mean:
        return poisson_over(line, mean)
    size = mean**2 / (variance - mean)
    tail_ratio = 1.0 - mean / variance
    mass = math.exp(size * math.log1p(-tail_ratio))
    cdf = mass
    for count in range(1, math.floor(line) + 1):
        mass *= (count - 1 + size) * tail_ratio / count
        cdf += mass
    return max(0.0, min(1.0, 1.0 - cdf))
