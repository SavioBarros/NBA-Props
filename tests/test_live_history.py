import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd
from requests import Timeout

from src.data.sources import fetch_season_player_logs, load_player_histories, normalize_season_logs


def sample_season():
    return pd.DataFrame(
        [
            (10, "1", "2025-10-23", "BOS @ NYK", 22, 5, 6),
            (10, "1", "2025-10-23", "BOS @ NYK", 22, 5, 6),
            (10, "2", "Oct 21, 2025", "BOS vs. LAL", 22, 5, 6),
            (20, "1", "2025-10-23", "BOS @ NYK", 7, 6, 1),
            (30, "1", "2025-10-23", "BOS @ NYK", None, None, None),
        ],
        columns=["PLAYER_ID", "GAME_ID", "GAME_DATE", "MATCHUP", "PTS", "REB", "AST"],
    )


class LiveHistoryTests(unittest.TestCase):
    def test_loads_two_seasons_once_for_109_players(self):
        games = normalize_season_logs(sample_season())
        with patch("src.data.sources.fetch_season_player_logs", return_value=(games, "NBA API")) as fetch:
            result = load_player_histories(iter(range(1, 110)), ["2025-26", "2024-25", "2025-26"])
        self.assertCountEqual([call.args[0] for call in fetch.call_args_list], ["2025-26", "2024-25"])
        self.assertEqual(result.logs[10, "2025-26"]["MATCHUP"].tolist(), ["BOS vs. LAL", "BOS @ NYK"])
        self.assertNotIn((30, "2025-26"), result.logs)
        self.assertEqual(result.sources["2025-26"], ("NBA API", date(2025, 10, 23)))

    def test_uses_archive_when_nba_api_times_out(self):
        with patch("src.data.sources.leaguegamelog.LeagueGameLog", side_effect=Timeout) as nba, patch(
            "src.data.sources.fetch_archive_player_logs", return_value=sample_season()
        ) as archive:
            frame, source = fetch_season_player_logs("2090-91")
        nba.assert_called_once()
        archive.assert_called_once_with("2090-91")
        self.assertEqual(source, "arquivo público")
        self.assertEqual(len(frame), 3)

    def test_records_unavailable_season_and_keeps_available_games(self):
        def season_data(season):
            if season == "2025-26":
                raise Timeout
            return normalize_season_logs(sample_season()), "arquivo público"

        with patch("src.data.sources.fetch_season_player_logs", side_effect=season_data):
            result = load_player_histories([10], ["2025-26", "2024-25"])
        self.assertEqual(result.unavailable, ["2025-26"])
        self.assertIn((10, "2024-25"), result.logs)
