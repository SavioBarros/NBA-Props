"""Evaluate points projections chronologically against a rolling-average baseline."""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from src.data.history import DEFAULT_DB, load_logs, normalize_logs
from src.models.projection import estimate_line_probabilities, estimate_weighted_distribution

MODEL_LABELS = {"baseline": "Média dos últimos jogos", "model": "Projeção ponderada"}


def backtest_points(frame: pd.DataFrame, window: int = 15, weight: float = 0.65) -> pd.DataFrame:
    """Forecast each game using dates strictly before the game's date."""
    if window < 2 or not 0 <= weight <= 1:
        raise ValueError("Use uma janela de pelo menos 2 jogos e um peso entre 0 e 1.")
    history = normalize_logs(frame)
    predictions = []
    empty = pd.Series(dtype=float)
    for _, player_games in history.groupby("PLAYER_ID", sort=False):
        by_season: dict[str, list[int]] = defaultdict(list)
        for _, daily_games in player_games.groupby("GAME_DATE", sort=False):
            for game in daily_games.itertuples(index=False):
                year = int(game.SEASON[:4])
                previous_season = f"{year - 1}-{year % 100:02d}"
                recent = by_season[game.SEASON][-window:]
                previous = by_season[previous_season]
                if len(recent) < window or len(previous) < window:
                    continue
                current = pd.Series(recent, dtype=float)
                baseline_mean, baseline_sd = estimate_weighted_distribution(current, empty)
                model_mean, model_sd = estimate_weighted_distribution(current, pd.Series(previous, dtype=float), weight)
                # A half-point line derived before tip-off: this is a simulated benchmark, not archived odds.
                line = math.floor(baseline_mean) + 0.5
                predictions.append({
                    "player_id": game.PLAYER_ID, "game_id": game.GAME_ID, "date": game.GAME_DATE,
                    "season": game.SEASON, "actual_pts": game.PTS, "benchmark_line": line,
                    "above_line": int(game.PTS > line), "baseline_projection": baseline_mean,
                    "model_projection": model_mean,
                    "baseline_probability": estimate_line_probabilities(line, baseline_mean, baseline_sd)[0],
                    "model_probability": estimate_line_probabilities(line, model_mean, model_sd)[0],
                })
            for game in daily_games.itertuples(index=False):
                by_season[game.SEASON].append(int(game.PTS))
    if not predictions:
        raise ValueError("A amostra não tem jogos avaliáveis. Inclua duas temporadas consecutivas com histórico suficiente.")
    return pd.DataFrame(predictions).sort_values(["date", "player_id", "game_id"]).reset_index(drop=True)


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    calibration = []
    for name, label in MODEL_LABELS.items():
        error = predictions[f"{name}_projection"] - predictions["actual_pts"]
        probs = predictions[f"{name}_probability"]
        observed = predictions["above_line"]
        metrics.append({
            "metodo": label, "jogos": len(predictions), "mae": error.abs().mean(),
            "rmse": math.sqrt(error.pow(2).mean()), "brier": (probs - observed).pow(2).mean(),
        })
        bins = pd.cut(probs, bins=[index / 10 for index in range(11)], include_lowest=True)
        grouped = pd.DataFrame({"estimada": probs, "observada": observed}).groupby(bins, observed=True)
        result = grouped.agg(media_estimada=("estimada", "mean"), taxa_observada=("observada", "mean"), jogos=("observada", "size"))
        calibration.append(result.assign(metodo=label, faixa=result.index.astype(str)).reset_index(drop=True))
    return pd.DataFrame(metrics), pd.concat(calibration, ignore_index=True)


def plot_report(metrics: pd.DataFrame, calibration: pd.DataFrame, demo: bool = False) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Erro em pontos", "Calibração do Over"))
    for label, color in zip(MODEL_LABELS.values(), ("#64748b", "#2563eb")):
        scores = metrics.loc[metrics["metodo"].eq(label)].iloc[0]
        fig.add_trace(go.Bar(name=label, x=["MAE", "RMSE"], y=[scores.mae, scores.rmse], marker_color=color), row=1, col=1)
        bins = calibration.loc[calibration["metodo"].eq(label)]
        fig.add_trace(go.Scatter(
            x=bins.media_estimada, y=bins.taxa_observada, customdata=bins.jogos,
            mode="markers+lines", marker_color=color, name=label, showlegend=False,
            hovertemplate="Estimada %{x:.0%}<br>Observada %{y:.0%}<br>Jogos %{customdata}<extra></extra>",
        ), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash", "color": "#94a3b8"},
                             name="Calibração ideal", showlegend=False), row=1, col=2)
    fig.update_xaxes(range=[0, 1], tickformat=".0%", row=1, col=2)
    fig.update_yaxes(range=[0, 1], tickformat=".0%", row=1, col=2)
    label = "dados simulados" if demo else "histórico da NBA"
    return fig.update_layout(title=f"Backtest de pontos · {label}", template="plotly_white", height=430)


def demo_history() -> pd.DataFrame:
    from src.data.demo import demo_player_logs

    return pd.concat([
        games.assign(PLAYER_ID=player_id, SEASON=season)
        for (player_id, season), games in demo_player_logs().items()
    ], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--demo", action="store_true", help="Usa histórico fictício para exercitar a metodologia")
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--weight", type=float, default=0.65)
    parser.add_argument("--output", type=Path, default=Path("reports/backtest"))
    options = parser.parse_args()
    data = demo_history() if options.demo else load_logs(options.db)
    predictions = backtest_points(data, options.window, options.weight)
    metrics, calibration = summarize(predictions)
    options.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(options.output / "previsoes.csv", index=False)
    metrics.to_csv(options.output / "metricas.csv", index=False)
    calibration.to_csv(options.output / "calibracao.csv", index=False)
    plot_report(metrics, calibration, options.demo).write_html(options.output / "resultado.html", include_plotlyjs=True)
    source = "simulados" if options.demo else "do histórico informado"
    print(f"Dados {source}. Linha de avaliação simulada.\n{metrics.to_string(index=False)}")


if __name__ == "__main__":
    main()
