from __future__ import annotations

import math

import pandas as pd


def estimate_weighted_distribution(
    recent: pd.Series,
    season: pd.Series,
    w_recent: float = 0.65,
) -> tuple[float, float] | None:
    recent = pd.to_numeric(recent, errors="coerce").dropna()
    season = pd.to_numeric(season, errors="coerce").dropna()

    if recent.empty and season.empty:
        return None
    if recent.empty:
        mu = float(season.mean())
        var = float(season.var(ddof=1)) if len(season) > 1 else 4.0
        return mu, math.sqrt(max(1.0, var))
    if season.empty:
        mu = float(recent.mean())
        var = float(recent.var(ddof=1)) if len(recent) > 1 else 4.0
        return mu, math.sqrt(max(1.0, var))

    mu_recent = float(recent.mean())
    mu_season = float(season.mean())
    var_recent = float(recent.var(ddof=1)) if len(recent) > 1 else 4.0
    var_season = float(season.var(ddof=1)) if len(season) > 1 else 4.0

    mu = w_recent * mu_recent + (1.0 - w_recent) * mu_season
    var_within = w_recent * var_recent + (1.0 - w_recent) * var_season
    var_between = w_recent * (mu_recent - mu) ** 2 + (1.0 - w_recent) * (mu_season - mu) ** 2
    return mu, math.sqrt(max(1.0, var_within + var_between))


def estimate_line_probabilities(line: float, mean: float, sd: float) -> tuple[float, float, float]:
    """Return Over, Under and push probabilities for a nonnegative integer stat."""
    if line < 0:
        return 1.0, 0.0, 0.0

    scale = max(1e-9, sd) * math.sqrt(2.0)
    # The normal approximation uses unit-width bins and folds the negative tail into zero.
    over_boundary = math.floor(line) + 0.5
    under_boundary = math.ceil(line) - 0.5
    prob_over = 0.5 * math.erfc((over_boundary - mean) / scale)
    prob_under = 0.5 * math.erfc((mean - under_boundary) / scale) if line > 0 else 0.0
    prob_push = max(0.0, 1.0 - prob_over - prob_under) if over_boundary != under_boundary else 0.0
    return prob_over, prob_under, prob_push


def line_result_counts(series: pd.Series, line: float) -> tuple[int, int, int]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return int((numeric > line).sum()), int((numeric < line).sum()), int((numeric == line).sum())


def side_hit_rate(series: pd.Series, line: float, side: str) -> float | None:
    over, under, _ = line_result_counts(series, line)
    decisions = over + under
    hits = over if side == "Over" else under
    return hits / decisions * 100.0 if decisions else None
