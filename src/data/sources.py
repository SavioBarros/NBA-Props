"""Clientes, cache e normalização dos dados ao vivo."""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from threading import local
from typing import Any, Iterable, Optional

import pandas as pd
import requests
import streamlit as st
from nba_api.stats.endpoints import leaguegamelog, playergamelog
from nba_api.stats.static import players, teams
from requests.adapters import HTTPAdapter, Retry

from src.data.history import download_archive_season
from src.models.projection import no_vig_prob_over

# ====== opcionais
try:
    from rapidfuzz import fuzz, process

    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False


BASE_URL = "https://api.the-odds-api.com/v4"
REQUEST_TIMEOUT = 18
MAX_ODDS_WORKERS = 6
MAX_STATS_WORKERS = 8
MARKETS = "player_points,player_rebounds,player_assists"
MARKET_MAP = {
    "player_points": "PTS",
    "player_rebounds": "REB",
    "player_assists": "AST",
}
THREAD_LOCAL = local()


def current_nba_season(today: Optional[datetime] = None) -> str:
    today = today or datetime.now(timezone.utc).replace(tzinfo=None)
    start_year = today.year if today.month >= 10 else today.year - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def previous_nba_season(season: str) -> str:
    start_year = int(season[:4]) - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def _norm_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def build_session() -> requests.Session:
    """Configura tentativas limitadas para erros transitórios e limites de requisições da API."""
    session = requests.Session()
    retries = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(
        max_retries=retries,
        pool_connections=max(8, MAX_STATS_WORKERS * 2),
        pool_maxsize=max(8, MAX_STATS_WORKERS * 2),
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_session() -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = build_session()
        THREAD_LOCAL.session = session
    return session


@st.cache_data(ttl=86_400, show_spinner=False)
def all_players_index() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    roster = players.get_players()
    index: dict[str, dict[str, Any]] = {}
    for player in roster:
        index[_norm_text(player["full_name"])] = player
    return roster, index


@st.cache_data(ttl=86_400, show_spinner=False)
def team_abbr_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for team in teams.get_teams():
        variants = {
            team["full_name"],
            team["nickname"],
            f"{team['city']} {team['nickname']}",
            team["abbreviation"],
        }
        for variant in variants:
            mapping[_norm_text(variant)] = team["abbreviation"]
    return mapping


def resolve_team_abbr(name: str) -> Optional[str]:
    mapping = team_abbr_map()
    key = _norm_text(name)
    if key in mapping:
        return mapping[key]
    if HAS_RAPIDFUZZ and key:
        hit = process.extractOne(key, list(mapping.keys()), scorer=fuzz.WRatio, score_cutoff=80)
        if hit:
            return mapping[hit[0]]
    return mapping.get(_norm_text(name.split()[-1])) if name else None


@st.cache_data(ttl=86_400, show_spinner=False)
def resolve_player_id(name: str) -> Optional[int]:
    roster, index = all_players_index()
    key = _norm_text(name)
    if key in index:
        return int(index[key]["id"])
    if HAS_RAPIDFUZZ and key:
        hit = process.extractOne(key, list(index.keys()), scorer=fuzz.WRatio, score_cutoff=86)
        if hit:
            return int(index[hit[0]]["id"])
    return None


def parse_commence_time(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        return None


def format_api_error(response: requests.Response, api_key: str) -> str:
    """Transforma a resposta em uma mensagem curta e omite a chave da The Odds API."""
    descriptions = {
        401: "Verifique a chave ou a cota da The Odds API.",
        403: "A The Odds API recusou o acesso.",
        429: "Limite de requisições da The Odds API atingido.",
    }
    status = response.status_code
    if status in descriptions:
        return descriptions[status]
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        for field in ("message", "error", "detail"):
            if payload.get(field):
                message = str(payload[field])
                return (message.replace(api_key, "[ocultado]") if api_key else message)[:220]
    return f"HTTP {status}"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_events(api_key: str) -> dict[str, Any]:
    """Consulta o calendário da NBA na The Odds API com cache de cinco minutos."""
    url = f"{BASE_URL}/sports/basketball_nba/events"
    try:
        response = get_session().get(url, params={"apiKey": api_key}, timeout=REQUEST_TIMEOUT)
        if not response.ok:
            return {"events": [], "error": format_api_error(response, api_key), "status_code": response.status_code}
        data = response.json()
        return {"events": data if isinstance(data, list) else [], "error": None, "status_code": response.status_code}
    except Exception as exc:
        return {"events": [], "error": f"Falha ao consultar a The Odds API ({type(exc).__name__}).", "status_code": None}


def fetch_event_odds(event_id: str, api_key: str) -> dict[str, Any]:
    url = f"{BASE_URL}/sports/basketball_nba/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": MARKETS,
        "oddsFormat": "decimal",
    }
    try:
        response = get_session().get(url, params=params, timeout=REQUEST_TIMEOUT)
        if response.status_code == 404:
            return {"payload": None, "error": "evento sem odds disponíveis", "status_code": 404}
        if not response.ok:
            return {"payload": None, "error": format_api_error(response, api_key), "status_code": response.status_code}
        return {"payload": response.json(), "error": None, "status_code": response.status_code}
    except Exception as exc:
        return {"payload": None, "error": f"Falha ao consultar a The Odds API ({type(exc).__name__}).", "status_code": None}


def extract_props_from_event(event_row: dict[str, Any], odds_payload: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normaliza linhas com pares completos de cotações para o mesmo jogador e mercado."""
    if not odds_payload or "bookmakers" not in odds_payload:
        return []

    event_id = event_row.get("id")
    home = event_row.get("home_team") or ""
    away = event_row.get("away_team") or ""
    props: list[dict[str, Any]] = []

    for bookmaker in odds_payload.get("bookmakers", []):
        book_name = bookmaker.get("title") or bookmaker.get("key") or "book"
        for market in bookmaker.get("markets", []):
            stat = MARKET_MAP.get(market.get("key"))
            if not stat:
                continue

            bucket: dict[tuple[str, float], dict[str, float]] = {}
            for outcome in market.get("outcomes", []):
                player_name = outcome.get("description") or outcome.get("name") or ""
                line_value = outcome.get("point")
                side = str(outcome.get("name") or "").lower()
                price = outcome.get("price")
                if not player_name or line_value is None or price is None:
                    continue

                key = (player_name, float(line_value))
                bucket.setdefault(key, {})
                if "over" in side:
                    bucket[key]["over"] = float(price)
                elif "under" in side:
                    bucket[key]["under"] = float(price)

            for (player_name, line_value), prices in bucket.items():
                if "over" not in prices or "under" not in prices:
                    continue
                props.append(
                    {
                        "event_id": event_id,
                        "player": player_name,
                        "stat": stat,
                        "line": float(line_value),
                        "odds_over": prices["over"],
                        "odds_under": prices["under"],
                        "bookmaker": book_name,
                        "game": f"{away} @ {home}",
                        "away_team": away,
                        "home_team": home,
                        "prob_book_fair_over": no_vig_prob_over(prices["over"], prices["under"]),
                    }
                )
    return props


@st.cache_data(ttl=300, show_spinner=False)
def get_player_props(api_key: str) -> dict[str, Any]:
    """Busca as linhas de até oito eventos, preservando resultados parciais e diagnósticos."""
    events_result = fetch_events(api_key)
    events = events_result.get("events", [])
    if not events:
        return {
            "props": [],
            "error": events_result.get("error") or "nenhum evento retornado pela Odds API",
            "events_total": 0,
            "events_considered": 0,
            "event_errors": [],
        }

    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    horizon = now_utc + timedelta(hours=36)
    valid_events = [
        event for event in events
        if event.get("id") and ((parse_commence_time(event.get("commence_time")) or now_utc) <= horizon)
    ]
    if not valid_events:
        valid_events = [event for event in events if event.get("id")][:6]
    else:
        valid_events = valid_events[:8]

    collected: list[dict[str, Any]] = []
    event_errors: list[str] = []
    workers = min(MAX_ODDS_WORKERS, max(1, len(valid_events)))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(fetch_event_odds, str(event["id"]), api_key): event
            for event in valid_events
        }
        for future in as_completed(future_map):
            event = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"payload": None, "error": f"Falha ao consultar a The Odds API ({type(exc).__name__}).", "status_code": None}
            payload = result.get("payload")
            if result.get("error") and len(event_errors) < 6:
                matchup = f"{event.get('away_team', '?')} @ {event.get('home_team', '?')}"
                event_errors.append(f"{matchup}: {result['error']}")
            collected.extend(extract_props_from_event(event, payload))

    error = None
    if not collected:
        if event_errors:
            error = event_errors[0]
        else:
            error = "nenhuma linha de player props disponível para os jogos consultados"

    return {
        "props": collected,
        "error": error,
        "events_total": len(events),
        "events_considered": len(valid_events),
        "event_errors": event_errors,
    }


def dedupe_props(props: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retém a melhor cotação por lado e as referências sem margem de cada casa."""
    best: dict[tuple[str, str, float, str], dict[str, Any]] = {}
    for prop in props:
        key = (prop["player"], prop["stat"], float(prop["line"]), prop["game"])
        if key not in best:
            best[key] = {
                **prop,
                "bookmaker_over": prop["bookmaker"],
                "bookmaker_under": prop["bookmaker"],
                "prob_book_fair_under": 100.0 - prop["prob_book_fair_over"],
            }
            continue

        chosen = best[key]
        if float(prop["odds_over"]) > float(chosen["odds_over"]):
            chosen["odds_over"] = float(prop["odds_over"])
            chosen["bookmaker_over"] = prop["bookmaker"]
            chosen["prob_book_fair_over"] = prop["prob_book_fair_over"]
        if float(prop["odds_under"]) > float(chosen["odds_under"]):
            chosen["odds_under"] = float(prop["odds_under"])
            chosen["bookmaker_under"] = prop["bookmaker"]
            chosen["prob_book_fair_under"] = 100.0 - prop["prob_book_fair_over"]
    return list(best.values())


@st.cache_data(ttl=21_600, show_spinner=False)
def fetch_player_log(player_id: int, season: str) -> Optional[pd.DataFrame]:
    """Busca o histórico normalizado de um jogador por temporada, com cache de seis horas."""
    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season, timeout=REQUEST_TIMEOUT)
    df = gamelog.get_data_frames()[0]
    if df.empty:
        return None
    df = df[["GAME_DATE", "MATCHUP", "PTS", "REB", "AST"]].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df = df.dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE").reset_index(drop=True)
    for col in ["PTS", "REB", "AST"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


HISTORY_STATS = ["GAME_DATE", "MATCHUP", "PTS", "REB", "AST"]
ARCHIVE_COLUMNS = ["player_id", "game_id", "game_date", "team", "opponent", "home", "pts", "reb", "ast"]


@dataclass
class PlayerHistories:
    """Históricos disponíveis e proveniência por temporada."""
    logs: dict[tuple[int, str], pd.DataFrame | None] = field(default_factory=dict)
    sources: dict[str, tuple[str, date]] = field(default_factory=dict)
    unavailable: list[str] = field(default_factory=list)


def normalize_season_logs(raw: pd.DataFrame) -> pd.DataFrame:
    """Padroniza os campos usados pelo painel e remove partidas duplicadas."""
    games = raw[["PLAYER_ID", "GAME_ID", *HISTORY_STATS]].copy()
    games["PLAYER_ID"] = pd.to_numeric(games["PLAYER_ID"], errors="coerce")
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"], errors="coerce", format="mixed")
    for stat in ("PTS", "REB", "AST"):
        games[stat] = pd.to_numeric(games[stat], errors="coerce")
    games = games.dropna(subset=["PLAYER_ID", "GAME_ID", "GAME_DATE"])
    games = games.dropna(subset=["PTS", "REB", "AST"], how="all")
    if games.empty:
        raise ValueError("Histórico vazio ou sem estatísticas válidas.")
    return games.drop_duplicates(["PLAYER_ID", "GAME_ID"]).sort_values("GAME_DATE", kind="stable")


def fetch_archive_player_logs(season: str) -> pd.DataFrame:
    games = pd.read_parquet(download_archive_season(season, timeout=(4, 10)), columns=ARCHIVE_COLUMNS)
    marker = games["home"].map({True: " vs. ", False: " @ "})
    matchup = games["team"].astype("string") + marker + games["opponent"].astype("string")
    return games.assign(matchup=matchup).rename(columns=str.upper)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_season_player_logs(season: str) -> tuple[pd.DataFrame, str]:
    """Carrega a temporada de uma vez e usa o arquivo público se a NBA API falhar."""
    try:
        response = leaguegamelog.LeagueGameLog(
            player_or_team_abbreviation="P", season=season, season_type_all_star="Regular Season", timeout=8
        )
        return normalize_season_logs(response.get_data_frames()[0]), "NBA API"
    except Exception:
        return normalize_season_logs(fetch_archive_player_logs(season)), "arquivo público"


def load_player_histories(player_ids: Iterable[int], seasons: Iterable[str]) -> PlayerHistories:
    """Consulta cada temporada uma vez, em paralelo, e reaproveita o cache entre filtros."""
    selected = {int(player_id) for player_id in player_ids}
    seasons = tuple(dict.fromkeys(seasons))
    result = PlayerHistories()
    if not selected or not seasons:
        return result

    with ThreadPoolExecutor(max_workers=min(2, len(seasons))) as executor:
        jobs = {executor.submit(fetch_season_player_logs, season): season for season in seasons}
        for job in as_completed(jobs):
            season = jobs[job]
            try:
                games, source = job.result()
            except Exception:
                result.unavailable.append(season)
                continue
            result.sources[season] = (source, games["GAME_DATE"].max().date())
            games = games.loc[games["PLAYER_ID"].isin(selected)]
            for player_id, group in games.groupby("PLAYER_ID", sort=False):
                result.logs[int(player_id), season] = group[HISTORY_STATS].reset_index(drop=True)
    return result


def preload_logs(player_ids: Iterable[int], seasons: Iterable[str]) -> dict[tuple[int, str], pd.DataFrame | None]:
    """Mantém o formato de histórico usado pela camada de dados anterior."""
    selected = tuple(dict.fromkeys(int(player_id) for player_id in player_ids))
    seasons = tuple(dict.fromkeys(seasons))
    loaded = load_player_histories(selected, seasons).logs
    return {(player_id, season): loaded.get((player_id, season)) for player_id in selected for season in seasons}
