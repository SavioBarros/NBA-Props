"""Geração das linhas avaliadas pelo painel, sem dependência da interface."""
from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from src.features.player_form import h2h_result_text, side_hit_rate, stat_values
from src.models.projection import (
    compute_star_rating,
    estimate_line_probabilities,
    estimate_weighted_distribution,
    implied_percent,
)


def format_stars(count: int) -> str:
    count = max(1, min(5, int(count)))
    return "★" * count + "☆" * (5 - count)



def build_rec_id(player: str, stat: str, line_value: float, game: str, side: str) -> str:
    base = f"{player}|{stat}|{line_value}|{game}|{side}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()[:12]



def evaluate_prop(
    prop: dict[str, Any],
    player_id: int,
    df_current: pd.DataFrame | None,
    df_compare: pd.DataFrame | None,
    df_h2h: pd.DataFrame,
    *,
    recent_games: int,
    recent_weight: float,
) -> dict[str, Any] | None:
    """Avalia uma linha e devolve as projeções e métricas históricas usadas na tabela."""
    stat = str(prop["stat"])
    line_value = float(prop["line"])
    current_stats = stat_values(df_current, stat)
    ser_recent = current_stats.tail(recent_games)
    ser_comp = stat_values(df_compare, stat)

    distribution = estimate_weighted_distribution(ser_recent, ser_comp, w_recent=recent_weight)
    if not distribution:
        return None
    mu, sd = distribution

    p_over, p_under, p_push = estimate_line_probabilities(line_value, mu, sd)

    imp_over = implied_percent(float(prop["odds_over"]))
    imp_under = implied_percent(float(prop["odds_under"]))
    # Devoluções têm retorno líquido zero. Compare as odds com a probabilidade entre jogos decididos.
    p_decision = p_over + p_under
    if not p_decision:
        return None
    val_over = p_over / p_decision * 100.0 - imp_over
    val_under = p_under / p_decision * 100.0 - imp_under

    side, model_value = ("Over", val_over) if val_over >= val_under else ("Under", val_under)
    chosen_prob = p_over * 100.0 if side == "Over" else p_under * 100.0
    chosen_odds = float(prop["odds_over"] if side == "Over" else prop["odds_under"])
    chosen_bookmaker = prop["bookmaker_over"] if side == "Over" else prop["bookmaker_under"]
    fair_side_prob = float(prop["prob_book_fair_over"] if side == "Over" else prop["prob_book_fair_under"])

    recent_rate = side_hit_rate(ser_recent, line_value, side)
    l5_rate = side_hit_rate(current_stats.tail(5), line_value, side)
    l10_rate = side_hit_rate(current_stats.tail(10), line_value, side)
    comp_rate = side_hit_rate(ser_comp, line_value, side)
    h2h_txt = h2h_result_text(df_h2h, stat, line_value, side)

    stars = compute_star_rating(model_value, chosen_prob)
    rec_id = build_rec_id(prop["player"], stat, line_value, prop["game"], side)
    return {
        "rec_id": rec_id,
        "event_id": prop["event_id"],
        "Jogo": prop["game"],
        "Jogador": prop["player"],
        "player_id": player_id,
        "Stat": stat,
        "Linha": line_value,
        "Lado": side,
        "Pick": f"{side} {line_value:g}",
        "Odds": chosen_odds,
        "Melhor Book": chosen_bookmaker,
        "Book Over": prop["bookmaker_over"],
        "Book Under": prop["bookmaker_under"],
        "Odds Over": float(prop["odds_over"]),
        "Odds Under": float(prop["odds_under"]),
        "Prob (modelo) %": round(chosen_prob, 1),
        "Prob devolução %": round(p_push * 100.0, 1),
        "Prob justa book %": round(fair_side_prob, 1),
        "Projeção": round(mu, 2),
        "Diff vs linha": round(mu - line_value, 2),
        "Média modelada": round(mu, 2),
        "Desvio modelado": round(sd, 2),
        "Estrelas": stars,
        "Rating": format_stars(stars),
        "L5 %": round(l5_rate, 1) if l5_rate is not None else None,
        "L10 %": round(l10_rate, 1) if l10_rate is not None else None,
        "Taxa lado recente %": round(recent_rate, 1) if recent_rate is not None else None,
        "Taxa lado comp %": round(comp_rate, 1) if comp_rate is not None else None,
        "H2H": h2h_txt,
        "Value (modelo) %": round(model_value, 1),
        "model_edge": model_value,
        "Jogos recentes usados": int(len(ser_recent)),
        "away_team": prop["away_team"],
        "home_team": prop["home_team"],
        "distribution": distribution,
    }
