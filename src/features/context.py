"""Features de contexto disponíveis antes de cada partida da NBA."""
from __future__ import annotations

import pandas as pd

PLAYER_COLUMNS = ["player_id", "game_id", "season", "game_date", "team", "opponent", "pts", "minutes"]
BOX_SCORE_COLUMNS = ["fga", "fta", "oreb", "tov"]
CONTEXT_COLUMNS = PLAYER_COLUMNS + BOX_SCORE_COLUMNS


def normalize_context_games(frame: pd.DataFrame) -> pd.DataFrame:
    """Padroniza o arquivo histórico e valida as chaves mostradas no box score."""
    games = frame.rename(columns=str.lower).loc[:, CONTEXT_COLUMNS].copy()
    games["game_id"] = games["game_id"].astype("string")
    games["game_date"] = pd.to_datetime(games["game_date"], format="mixed").dt.normalize()
    for col in ("player_id", "pts", "minutes", *BOX_SCORE_COLUMNS):
        games[col] = pd.to_numeric(games[col], errors="coerce")
    games = games.dropna(subset=CONTEXT_COLUMNS).drop_duplicates()
    games = games.loc[games["minutes"] > 0].sort_values("game_date", kind="stable")
    if games.duplicated(["player_id", "game_date"]).any():
        raise ValueError("Há múltiplos registros por jogador e data no arquivo de contexto.")
    return games


def _rolling_previous(values: pd.Series, window: int, min_games: int) -> pd.Series:
    return values.shift().rolling(window, min_periods=min_games).mean()


def _team_box_scores(games: pd.DataFrame) -> pd.DataFrame:
    keys = ["game_date", "game_id", "team", "opponent"]
    totals = games.groupby(keys, as_index=False)[["pts", "minutes", *BOX_SCORE_COLUMNS]].sum()
    totals["possessions"] = totals["fga"] + 0.44 * totals["fta"] - totals["oreb"] + totals["tov"]
    opponents = totals[["game_id", "team", "possessions", "pts"]].rename(
        columns={"team": "opponent", "possessions": "opp_possessions", "pts": "opp_points"}
    )
    totals = totals.merge(opponents, on=["game_id", "opponent"], how="inner", validate="one_to_one")
    possessions = (totals["possessions"] + totals["opp_possessions"]) / 2
    possessions = possessions.where(possessions > 0)
    totals["pace"] = possessions * 240 / totals["minutes"]
    totals["defensive_rating"] = 100 * totals["opp_points"] / possessions
    if totals.duplicated(["team", "game_date"]).any():
        raise ValueError("Há múltiplos registros por equipe e data no arquivo de contexto.")
    return totals.sort_values("game_date", kind="stable")


def build_pregame_context(games: pd.DataFrame) -> pd.DataFrame:
    """Calcula minutos, ritmo e força defensiva com janelas estritamente anteriores."""
    team_box = _team_box_scores(games)
    by_team = team_box.groupby("team", sort=False)
    team_box["prior_pace"] = by_team["pace"].transform(_rolling_previous, 10, 5)
    team_box["prior_defense"] = by_team["defensive_rating"].transform(_rolling_previous, 10, 5)

    league = team_box.groupby("game_date", as_index=False)["defensive_rating"].mean()
    league["league_defense"] = _rolling_previous(league["defensive_rating"], 10, 5)
    opponent_context = team_box[["game_id", "team", "prior_pace", "prior_defense"]].rename(
        columns={"team": "opponent", "prior_pace": "opponent_pace", "prior_defense": "opponent_defense"}
    )
    features = games[["player_id", "game_id", "game_date", "season", "team", "opponent", "pts", "minutes"]].merge(
        team_box[["game_id", "team", "pace", "prior_pace"]], on=["game_id", "team"], how="left", validate="many_to_one"
    )
    features = features.merge(opponent_context, on=["game_id", "opponent"], how="left", validate="many_to_one")
    features = features.merge(
        league[["game_date", "league_defense"]], on="game_date", how="left", validate="many_to_one"
    ).sort_values("game_date", kind="stable")

    by_player = features.groupby(["player_id", "season"], sort=False)
    expected_minutes = by_player["minutes"].transform(_rolling_previous, 5, 5)
    prior_points = by_player["pts"].transform(_rolling_previous, 15, 15)
    prior_minutes = by_player["minutes"].transform(_rolling_previous, 15, 15)
    recent_game_pace = by_player["pace"].transform(_rolling_previous, 15, 15)
    expected_pace = (features["prior_pace"] + features["opponent_pace"]) / 2

    minutes_only = expected_minutes * prior_points / prior_minutes
    minutes_pace = minutes_only * expected_pace / recent_game_pace
    minutes_pace_defense = minutes_pace * features["opponent_defense"] / features["league_defense"]
    return features[["player_id", "game_id"]].assign(
        minutes_projection=minutes_only,
        pace_projection=minutes_pace,
        defense_projection=minutes_pace_defense,
    )
