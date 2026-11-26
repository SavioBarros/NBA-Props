"""Experimentos de probabilidades e contexto pré-jogo para o backtest de pontos."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from nba_props_backtest import backtest_points, summarize
from src.data.history import download_archive_season
from src.features.context import CONTEXT_COLUMNS, build_pregame_context, normalize_context_games
from src.models.count_distributions import negative_binomial_over, poisson_over
from src.models.projection import estimate_line_probabilities

EXPERIMENT_LABELS = {
    "baseline": "Média L15",
    "model": "Ponderada · normal",
    "poisson": "Ponderada · Poisson",
    "nb": "Ponderada · binomial negativa",
    "minutes": "Minutos esperados",
    "pace": "Minutos + ritmo",
    "defense": "Minutos + ritmo + defesa",
}
CONTEXT_MODELS = ("minutes", "pace", "defense")


def run_experiments(raw: pd.DataFrame) -> pd.DataFrame:
    """Compara todos os métodos nas mesmas linhas, jogadores e datas."""
    games = normalize_context_games(raw)
    predictions = backtest_points(games.rename(columns=str.upper))
    predictions = predictions.merge(
        build_pregame_context(games), on=["player_id", "game_id"], how="left", validate="one_to_one"
    ).dropna(subset=[f"{model}_projection" for model in CONTEXT_MODELS]).reset_index(drop=True)
    if predictions.empty:
        raise ValueError("A amostra não contém jogos com todas as features pré-jogo.")

    predictions["poisson_projection"] = predictions["model_projection"]
    predictions["nb_projection"] = predictions["model_projection"]
    rows = predictions[["benchmark_line", "model_projection", "model_sd"]].itertuples(index=False, name=None)
    predictions["poisson_probability"] = [poisson_over(line, mean) for line, mean, _ in rows]
    rows = predictions[["benchmark_line", "model_projection", "model_sd"]].itertuples(index=False, name=None)
    predictions["nb_probability"] = [negative_binomial_over(line, mean, sd**2) for line, mean, sd in rows]
    predictions["nb_used_poisson"] = predictions["model_sd"].pow(2).le(predictions["model_projection"])

    for model in CONTEXT_MODELS:
        rows = predictions[["benchmark_line", f"{model}_projection", "model_sd"]].itertuples(index=False, name=None)
        predictions[f"{model}_probability"] = [estimate_line_probabilities(line, mean, sd)[0] for line, mean, sd in rows]
    return predictions


def experiment_chart(metrics: pd.DataFrame, calibration: pd.DataFrame) -> go.Figure:
    """Gera a comparação de erro, Brier e calibração para a amostra comum."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=("MAE em pontos", "Brier score", "Calibração"))
    colors = ["#283d66", "#64748b", "#8b5cf6", "#c2417e", "#168059", "#229eaf", "#e27234"]
    for row, color in zip(metrics.itertuples(index=False), colors):
        label = row.metodo
        fig.add_trace(go.Bar(x=[label], y=[row.mae], marker_color=color, name=label), row=1, col=1)
        fig.add_trace(go.Bar(x=[label], y=[row.brier], marker_color=color, name=label, showlegend=False), row=1, col=2)
        bins = calibration.loc[calibration["metodo"].eq(label)]
        fig.add_trace(
            go.Scatter(x=bins.media_estimada, y=bins.taxa_observada, customdata=bins.jogos, mode="markers+lines",
                       name=label, marker_color=color, showlegend=False,
                       hovertemplate="Estimada %{x:.0%}<br>Observada %{y:.0%}<br>Jogos %{customdata}<extra></extra>"),
            row=1, col=3,
        )
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash", "color": "#94a3b8"},
                             name="Calibração ideal", showlegend=False), row=1, col=3)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(range=[0, 1], tickformat=".0%", title="Probabilidade estimada", row=1, col=3)
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title="Frequência observada", row=1, col=3)
    return fig.update_layout(
        title=f"Experimentos pré-jogo · {int(metrics.iloc[0].jogos):,} observações".replace(",", "."),
        template="plotly_white", height=470, legend={"orientation": "h", "y": -0.18},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--parquet", type=Path, nargs="+", help="Arquivos locais do histórico público da NBA")
    source.add_argument("--archive-seasons", nargs="+", help="Temporadas do arquivo público, eg. 2024-25 2025-26")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, default=Path("reports/experimentos"))
    options = parser.parse_args()
    inputs = options.parquet or [download_archive_season(season, options.revision) for season in options.archive_seasons]
    raw = pd.concat([pd.read_parquet(item, columns=CONTEXT_COLUMNS) for item in inputs], ignore_index=True)
    predictions = run_experiments(raw)
    metrics, calibration = summarize(predictions, EXPERIMENT_LABELS)
    options.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(options.output / "previsoes.csv", index=False)
    metrics.to_csv(options.output / "metricas.csv", index=False)
    calibration.to_csv(options.output / "calibracao.csv", index=False)
    experiment_chart(metrics, calibration).write_html(options.output / "comparacao.html", include_plotlyjs=True)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
