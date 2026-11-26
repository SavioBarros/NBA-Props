import tempfile
import unittest
from pathlib import Path

from nba_history import load_logs, save_logs
from nba_props_backtest import backtest_points, demo_history, summarize


class BacktestTests(unittest.TestCase):
    def test_forecast_ignores_current_and_future_games(self):
        games = demo_history()
        original = backtest_points(games, window=5)
        date = original.iloc[0]["date"]
        changed_games = games.copy()
        changed_games.loc[changed_games["GAME_DATE"] >= date, "PTS"] = 100
        changed = backtest_points(changed_games, window=5)
        first = original.loc[original["date"].eq(date)].reset_index(drop=True)
        comparison = changed.loc[changed["date"].eq(date)].reset_index(drop=True)
        self.assertTrue(first["model_projection"].equals(comparison["model_projection"]))
        self.assertTrue(first["baseline_projection"].equals(comparison["baseline_projection"]))
        self.assertTrue((comparison["actual_pts"] == 100).all())

    def test_sqlite_reload_and_metrics(self):
        games = demo_history()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "snapshot.sqlite"
            self.assertEqual(save_logs(path, [games]), len(games))
            save_logs(path, [games])
            snapshot = load_logs(path)
            self.assertEqual(len(snapshot), len(games))
            predictions = backtest_points(snapshot)
        self.assertEqual(len(predictions), 68)
        metrics, calibration = summarize(predictions)
        self.assertEqual(len(metrics), 2)
        self.assertTrue(metrics["brier"].between(0, 1).all())
        self.assertEqual(calibration.groupby("metodo")["jogos"].sum().tolist(), [68, 68])
