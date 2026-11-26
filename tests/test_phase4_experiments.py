import math
import unittest

import pandas as pd

from src.features.context import build_pregame_context, normalize_context_games
from src.models.count_distributions import negative_binomial_over, poisson_over


class CountDistributionTests(unittest.TestCase):
    def test_poisson_probability_uses_the_integer_threshold(self):
        self.assertAlmostEqual(poisson_over(0.5, 2), 1 - math.exp(-2))
        self.assertAlmostEqual(poisson_over(1, 2), 1 - 3 * math.exp(-2))
        self.assertEqual(poisson_over(0.5, 0), 0)

    def test_negative_binomial_matches_the_geometric_case(self):
        self.assertAlmostEqual(negative_binomial_over(0.5, 2, 6), 2 / 3)
        self.assertAlmostEqual(negative_binomial_over(1.5, 2, 6), 4 / 9)
        self.assertEqual(negative_binomial_over(1.5, 2, 1), poisson_over(1.5, 2))


def example_games() -> pd.DataFrame:
    rows = []
    for index, day in enumerate(pd.date_range("2025-10-21", periods=22)):
        for player, team, opponent in [(1, "BOS", "LAL"), (2, "LAL", "BOS")]:
            rows.append({
                "player_id": player, "game_id": str(index), "season": "2025-26", "game_date": day,
                "team": team, "opponent": opponent, "pts": 20 + index % 4, "minutes": 30 + index % 3,
                "fga": 15, "fta": 4, "oreb": 1, "tov": 3,
            })
    return pd.DataFrame(rows)


class PregameContextTests(unittest.TestCase):
    def test_same_day_and_future_box_scores_cannot_change_features(self):
        source = example_games()
        baseline = build_pregame_context(normalize_context_games(source)).set_index(["player_id", "game_id"])
        changed = source.copy()
        future = changed["game_date"].ge(pd.Timestamp("2025-11-08"))
        changed.loc[future, ["minutes", "pts", "fga", "fta", "oreb", "tov"]] = [40, 50, 32, 9, 3, 8]
        recalculated = build_pregame_context(normalize_context_games(changed)).set_index(["player_id", "game_id"])
        pd.testing.assert_frame_equal(baseline.loc[[(1, "18"), (2, "18")]], recalculated.loc[[(1, "18"), (2, "18")]])
        self.assertTrue(baseline.loc[(1, "18")].notna().all())
        self.assertNotEqual(baseline.loc[(1, "19"), "minutes_projection"], recalculated.loc[(1, "19"), "minutes_projection"])
