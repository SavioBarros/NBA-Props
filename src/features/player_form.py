"""Features históricas usadas na análise das linhas de jogadores."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from src.data.sources import resolve_team_abbr


def stat_values(games: pd.DataFrame | None, stat: str) -> pd.Series:
    """Extrai as observações numéricas em ordem cronológica, ignorando registros inválidos."""
    if games is None or games.empty:
        return pd.Series(dtype=float)
    return pd.to_numeric(games[stat], errors="coerce").dropna()


def head_to_head_games(
    current: Optional[pd.DataFrame],
    previous: Optional[pd.DataFrame],
    away_name: str,
    home_name: str,
) -> pd.DataFrame:
    """Filtra jogos contra o adversário, inferindo a equipe do jogador pelo registro mais recente."""
    frames = [frame for frame in (previous, current) if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()

    away = resolve_team_abbr(away_name)
    home = resolve_team_abbr(home_name)
    history = pd.concat(frames, ignore_index=True).sort_values("GAME_DATE", kind="stable")
    matchups = history["MATCHUP"].astype("string").str.upper().str.extract(
        r"^\s*([A-Z0-9]+)\s+(?:VS\.?|@)\s+([A-Z0-9]+)\s*$"
    )
    player_teams = matchups[0].dropna()
    if not away or not home or player_teams.empty:
        return history.iloc[:0]

    player_team = player_teams.iloc[-1]
    if player_team not in (away, home):
        return history.iloc[:0]

    opponent = home if player_team == away else away
    return history.loc[matchups[1].eq(opponent)].copy()



def line_result_counts(series: pd.Series, line: float) -> tuple[int, int, int]:
    """Retorna os totais de Over, Under e devoluções entre os jogos válidos."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return int((numeric > line).sum()), int((numeric < line).sum()), int((numeric == line).sum())



def side_hit_rate(series: pd.Series, line: float, side: str) -> float | None:
    """Calcula a taxa histórica de acerto em percentual entre as partidas decididas."""
    over, under, _ = line_result_counts(series, line)
    decisions = over + under
    hits = over if side == "Over" else under
    return hits / decisions * 100.0 if decisions else None



def h2h_result_text(games: pd.DataFrame, stat: str, line: float, side: str) -> str:
    """Resume a taxa de acertos nos confrontos diretos e sinaliza devoluções."""
    if games.empty:
        return "-"
    over, under, pushes = line_result_counts(games[stat], line)
    decisions = over + under
    hits = over if side == "Over" else under
    summary = f"{hits}/{decisions} ({hits / decisions:.1%})" if decisions else "-"
    if pushes:
        summary += f" · {pushes} devol."
    return summary
