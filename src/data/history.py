"""Load player game logs into a reproducible SQLite snapshot."""
from __future__ import annotations

import argparse
import re
import sqlite3
from contextlib import closing
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from nba_api.stats.endpoints import leaguegamelog

COLUMNS = ["PLAYER_ID", "GAME_ID", "SEASON", "GAME_DATE", "PTS"]
DEFAULT_DB = Path("data/nba_history.sqlite")
ARCHIVE = "https://huggingface.co/datasets/cbratkovics/nba-game-logs"


def normalize_logs(frame: pd.DataFrame) -> pd.DataFrame:
    """Valida, deduplica e ordena as observações históricas por data e partida."""
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {', '.join(sorted(missing))}")
    data = frame.loc[:, COLUMNS].copy()
    data["GAME_DATE"] = pd.to_datetime(data["GAME_DATE"], errors="raise", format="mixed").dt.normalize()
    data["PLAYER_ID"] = pd.to_numeric(data["PLAYER_ID"], errors="raise").astype("Int64")
    data["GAME_ID"] = data["GAME_ID"].astype("string").str.strip()
    data["SEASON"] = data["SEASON"].astype("string").str.strip()
    data["PTS"] = pd.to_numeric(data["PTS"], errors="raise").astype("Int64")
    if (
        data.isna().any().any()
        or data["GAME_ID"].eq("").any()
        or not data["SEASON"].str.fullmatch(r"\d{4}-\d{2}").all()
        or data["PTS"].lt(0).any()
    ):
        raise ValueError("O histórico contém campos inválidos.")
    data = data.drop_duplicates()
    if data.duplicated(["PLAYER_ID", "GAME_ID"]).any():
        raise ValueError("Há registros conflitantes do mesmo jogador e jogo no lote.")
    return data.sort_values(["GAME_DATE", "PLAYER_ID", "GAME_ID"]).reset_index(drop=True)


def fetch_season(season: str) -> pd.DataFrame:
    """Obtém os jogos de temporada regular pela NBA API."""
    frame = leaguegamelog.LeagueGameLog(
        player_or_team_abbreviation="P",
        season=season,
        season_type_all_star="Regular Season",
        timeout=30,
    ).get_data_frames()[0]
    if frame.empty:
        raise ValueError(f"A NBA API retornou um histórico vazio para {season}.")
    return normalize_logs(frame.assign(SEASON=season))


def read_archive_parquet(source: Path | BytesIO) -> pd.DataFrame:
    raw = pd.read_parquet(source, columns=[column.lower() for column in COLUMNS])
    return normalize_logs(raw.rename(columns=str.upper))


def download_archive_season(
    season: str, revision: str = "main", *, timeout: tuple[int, int] = (10, 45)
) -> BytesIO:
    """Baixa uma temporada do arquivo público, sem cota da The Odds API."""
    if not re.fullmatch(r"\d{4}-\d{2}", season) or not re.fullmatch(r"[\w.-]+", revision):
        raise ValueError("Temporada ou revisão inválida.")
    url = f"{ARCHIVE}/resolve/{revision}/game_logs/game_logs_{season}.parquet"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return BytesIO(response.content)


def fetch_archive_season(season: str, revision: str = "main") -> pd.DataFrame:
    return read_archive_parquet(download_archive_season(season, revision))


def save_logs(path: Path, frames: list[pd.DataFrame]) -> int:
    """Grava o lote em SQLite de forma idempotente pela chave de jogador e partida."""
    data = normalize_logs(pd.concat(frames, ignore_index=True))
    if data.empty:
        raise ValueError("O lote de atualização está vazio.")
    data["GAME_DATE"] = data["GAME_DATE"].dt.strftime("%Y-%m-%d")
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS player_games (
                PLAYER_ID INTEGER NOT NULL, GAME_ID TEXT NOT NULL, SEASON TEXT NOT NULL,
                GAME_DATE TEXT NOT NULL, PTS INTEGER NOT NULL, PRIMARY KEY (PLAYER_ID, GAME_ID)
            )
        """)
        connection.executemany("""
            INSERT INTO player_games (PLAYER_ID, GAME_ID, SEASON, GAME_DATE, PTS)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (PLAYER_ID, GAME_ID) DO UPDATE SET
                SEASON = excluded.SEASON, GAME_DATE = excluded.GAME_DATE, PTS = excluded.PTS
        """, ((int(player), game, season, date, int(pts)) for player, game, season, date, pts in data.itertuples(index=False, name=None)))
    return len(data)


def load_logs(path: Path) -> pd.DataFrame:
    """Lê o histórico local com o mesmo esquema usado no backtest."""
    if not path.is_file():
        raise FileNotFoundError(f"Histórico indisponível em {path}. Carregue os dados com python -m nba_history.")
    with closing(sqlite3.connect(path)) as connection:
        return normalize_logs(pd.read_sql_query("SELECT * FROM player_games", connection))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--seasons", nargs="+", help="Temporadas da NBA API no formato AAAA-AA")
    source.add_argument("--csv", type=Path, help="CSV com PLAYER_ID, GAME_ID, SEASON, GAME_DATE e PTS")
    source.add_argument("--archive-seasons", nargs="+", help="Temporadas do arquivo público cbratkovics/nba-game-logs")
    source.add_argument("--parquet", type=Path, nargs="+", help="Arquivos locais do mesmo arquivo histórico")
    parser.add_argument("--revision", default="main", help="Revisão do arquivo remoto em --archive-seasons")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    options = parser.parse_args()
    try:
        if options.csv:
            frames = [pd.read_csv(options.csv, dtype={"GAME_ID": "string", "SEASON": "string"})]
        elif options.parquet:
            frames = [read_archive_parquet(path) for path in options.parquet]
        elif options.archive_seasons:
            frames = [fetch_archive_season(season, options.revision) for season in options.archive_seasons]
        else:
            frames = [fetch_season(season) for season in options.seasons]
    except requests.RequestException as error:
        parser.exit(1, f"Falha de rede ao carregar o histórico ({type(error).__name__}). "
                       "Importe os arquivos com --parquet ou --csv.\n")
    total = save_logs(options.db, frames)
    print(f"{total} registros de jogador por jogo atualizados em {options.db}.")
