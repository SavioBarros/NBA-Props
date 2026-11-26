import unittest
from hashlib import md5
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


class DemoAppTests(unittest.TestCase):
    def test_demo_runs_offline_and_reuses_projection_in_detail(self):
        path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
        with (
            patch("requests.sessions.Session.get", side_effect=AssertionError("demo queried odds")) as odds,
            patch("nba_api.stats.endpoints.playergamelog.PlayerGameLog", side_effect=AssertionError("demo queried NBA")) as stats,
        ):
            app = AppTest.from_file(str(path), default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.dataframe)
            app.slider[0].set_value(5)
            next(button for button in app.button if button.label == "Aplicar filtros").click().run()
            self.assertFalse(app.exception)
            row = app.dataframe[-1].value.iloc[0]
            self.assertEqual(row["Jogos recentes usados"], 5)
            key = f"{row['Jogador']}|{row['Stat']}|{row['Linha']}|{row['Jogo']}|{row['Lado']}"
            app.session_state["selected_rec_id"] = md5(key.encode()).hexdigest()[:12]
            app.run()
            self.assertFalse(app.exception)
            detail = [caption.value for caption in app.caption if "Probabilidade do modelo para" in caption.value]
            self.assertTrue(detail)
            self.assertTrue(all(f"**{row['Prob (modelo) %']:.1f}%**" in value for value in detail))
            odds.assert_not_called()
            stats.assert_not_called()
