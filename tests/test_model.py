import math
import unittest

import pandas as pd

from nba_props_data import dedupe_props, head_to_head_games, no_vig_prob_over
from nba_props_model import estimate_line_probabilities, estimate_weighted_distribution, line_result_counts, side_hit_rate


class ModelTests(unittest.TestCase):
    def test_weighting_and_empty_history(self):
        recent = pd.Series([20, 22, 24])
        past = pd.Series([8, 10, 12])
        self.assertEqual(estimate_weighted_distribution(recent, past, 0.75)[0], 19)
        self.assertIsNone(estimate_weighted_distribution(pd.Series(dtype=float), pd.Series(dtype=float)))

    def test_push_and_missing_results(self):
        values = pd.Series([19, "20", 21, None])
        self.assertEqual(line_result_counts(values, 20), (1, 1, 1))
        self.assertEqual(side_hit_rate(values, 20, "Under"), 50)
        self.assertIsNone(side_hit_rate(pd.Series([20]), 20, "Over"))
        for line in (0, 20, 20.5):
            over, under, push = estimate_line_probabilities(line, 20, 4)
            self.assertTrue(math.isclose(over + under + push, 1))
            self.assertEqual(push > 0, line != 20.5)
        self.assertEqual(estimate_line_probabilities(0, 1, 1)[1], 0)

    def test_odds_fair_probability_is_from_the_same_book(self):
        key = {"player": "Example", "stat": "PTS", "line": 20.5, "game": "BOS @ LAL"}
        rows = [
            dict(key, bookmaker=name, odds_over=over, odds_under=under, prob_book_fair_over=no_vig_prob_over(over, under))
            for name, over, under in (("A", 2.1, 1.7), ("B", 1.8, 2.05))
        ]
        [best] = dedupe_props(rows)
        self.assertEqual(best["bookmaker_over"], "A")
        self.assertEqual(best["bookmaker_under"], "B")
        self.assertAlmostEqual(best["prob_book_fair_over"], no_vig_prob_over(2.1, 1.7))
        self.assertAlmostEqual(best["prob_book_fair_under"], 100 - no_vig_prob_over(1.8, 2.05))

    def test_head_to_head_filters_exact_opponent(self):
        current = pd.DataFrame({
            "GAME_DATE": pd.to_datetime(["2025-11-03", "2025-11-01", "2025-11-02", "2025-11-04"]),
            "MATCHUP": ["BOS @ LAL", "BOS @ NYK", "BOS vs. LAC", "BOS vs. MIA"],
        })
        past = pd.DataFrame({"GAME_DATE": pd.to_datetime(["2024-10-31"]), "MATCHUP": ["MIA vs. LAL"]})
        games = head_to_head_games(current, past, "Boston Celtics", "Los Angeles Lakers")
        self.assertEqual(games["MATCHUP"].tolist(), ["MIA vs. LAL", "BOS @ LAL"])
