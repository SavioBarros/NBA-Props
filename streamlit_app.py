from __future__ import annotations

import hashlib
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from threading import local
from time import perf_counter
from typing import Any, Iterable, Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from requests.adapters import HTTPAdapter, Retry

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players, teams

# ====== opcionais
try:
    from rapidfuzz import fuzz, process

    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False


# =====================================================================
# CONFIG GERAL
# =====================================================================
st.set_page_config(
    page_title="NBA Props Analyzer",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------- CSS
st.markdown(
    """
<style>
:root{
  --bg:#0f1216; --bg-soft:#151922; --card:#10151d; --card-2:#0e141b;
  --stroke:#1e2532; --muted:#9aa4b2; --accent:#7c4dff; --accent-2:#31d0aa;
  --over:#2ecc71; --under:#e74c3c; --warning:#ffb020;
}
html, body, .main { background: var(--bg); color:#e6edf3; }
section[data-testid="stSidebar"] { background: var(--bg-soft); border-right:1px solid var(--stroke); }
.block-container { padding-top:1.2rem; padding-bottom:2rem; }
h1,h2,h3 { letter-spacing:.2px; }

div[data-testid="stMetric"]{
  background: linear-gradient(180deg, var(--card) 0%, var(--card-2) 100%);
  border:1px solid var(--stroke); border-radius:12px; padding:14px 16px;
}
div[data-testid="stMetric"] .metric-label{ color:var(--muted); font-weight:600; }
div[data-testid="stMetric"] .metric-value{ font-weight:800; }

.value-positive{ color: var(--over); font-weight:800; }
.value-negative{ color: var(--under); font-weight:800; }

hr{ border:none; border-top:1px solid var(--stroke); margin:18px 0; }

[data-baseweb="tab-list"]{ border-bottom:1px solid var(--stroke); }
button[kind="secondary"]{ border:1px solid var(--stroke) !important; }

.pick-card{
  background: linear-gradient(180deg, var(--card) 0%, var(--card-2) 100%);
  border:1px solid var(--stroke); border-radius:14px; padding:14px 16px;
}

.small-muted{ color:var(--muted); font-size:12.5px; }
.badge{
  display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px;
  background:#101826; border:1px solid var(--stroke); color:#c7d1db;
}
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# CONSTANTES
# =====================================================================
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


TEMPORADA_ATUAL = current_nba_season()
TEMPORADA_COMP = previous_nba_season(TEMPORADA_ATUAL)


DEFAULT_FILTERS = {
    "num_jogos": 15,
    "w_recent": 0.65,
    "bookmaker_sel": "Todos",
    "stat_sel": "Todas",
    "game_sel": "Todos",
    "side_sel": "Todos",
    "min_stars": 1,
    "odds_min": 1.65,
    "odds_max": 2.40,
    "min_value": 5,
    "min_cover_prob": 0,
    "top_per_game": 10,
    "static_plot": True,
    "player_search": "",
    "sort_by": "Value",
    "main_view": "Cheat Sheet",
}


# =====================================================================
# FUNÇÕES AUXILIARES
# =====================================================================
def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets[name]
        return str(value).strip()
    except Exception:
        return os.getenv(name, default).strip()


API_KEY = get_secret("ODDS_API_KEY")


def _safe_key(*parts: Any) -> str:
    raw = "_".join(str(p) for p in parts)
    return re.sub(r"\W+", "_", raw)


def _norm_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def build_session() -> requests.Session:
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


# Uma sessão por thread evita contenção ao fazer requests em paralelo.
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


def format_api_error(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            for key in ("message", "error", "detail"):
                if payload.get(key):
                    return str(payload[key])
    except Exception:
        pass
    body = (response.text or "").strip()
    return body[:220] if body else f"HTTP {response.status_code}"

def prob_from_decimal(odds: float) -> float:
    return 0.0 if not odds or odds <= 1e-9 else 1.0 / odds


def implied_percent(odds: float) -> float:
    return prob_from_decimal(odds) * 100.0


def no_vig_prob_over(odds_over: float, odds_under: float) -> float:
    po, pu = prob_from_decimal(odds_over), prob_from_decimal(odds_under)
    total = po + pu
    return (po / total * 100.0) if total > 0 else 50.0


def normal_over_probability(line_value: float, mu: float, sd: float) -> float:
    sd = max(1e-9, float(sd))
    z = (line_value - float(mu)) / sd
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return max(0.0, min(1.0, 1.0 - cdf))


def estimate_weighted_distribution(
    recent: pd.Series,
    season: pd.Series,
    w_recent: float = 0.65,
) -> Optional[tuple[float, float]]:
    recent = pd.to_numeric(recent, errors="coerce").dropna()
    season = pd.to_numeric(season, errors="coerce").dropna()

    if recent.empty and season.empty:
        return None
    if recent.empty:
        mu = float(season.mean())
        var = float(season.var(ddof=1)) if len(season) > 1 else 4.0
        return mu, math.sqrt(max(1.0, var))
    if season.empty:
        mu = float(recent.mean())
        var = float(recent.var(ddof=1)) if len(recent) > 1 else 4.0
        return mu, math.sqrt(max(1.0, var))

    mu_recent = float(recent.mean())
    mu_season = float(season.mean())
    var_recent = float(recent.var(ddof=1)) if len(recent) > 1 else 4.0
    var_season = float(season.var(ddof=1)) if len(season) > 1 else 4.0

    mu = w_recent * mu_recent + (1.0 - w_recent) * mu_season
    var_within = w_recent * var_recent + (1.0 - w_recent) * var_season
    var_between = w_recent * (mu_recent - mu) ** 2 + (1.0 - w_recent) * (mu_season - mu) ** 2
    sd = math.sqrt(max(1.0, var_within + var_between))
    return mu, sd


def side_hit_mask(series: pd.Series, line_value: float, side: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric > line_value if side == "Over" else numeric <= line_value


def side_hit_rate(series: pd.Series, line_value: float, side: str) -> Optional[float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(side_hit_mask(numeric, line_value, side).mean() * 100.0)


def fmt_rate(rate: Optional[float]) -> str:
    return "—" if rate is None else f"{rate:.1f}%"


def compute_star_rating(value_pct: float, cover_prob_pct: float) -> int:
    """1–5 estrelas (estilo cheat sheet): combina edge e probabilidade de cover."""
    edge = max(0.0, float(value_pct))
    cover = float(cover_prob_pct)
    score = edge * 1.35 + max(0.0, cover - 52.0) * 0.55
    if score >= 28:
        return 5
    if score >= 22:
        return 4
    if score >= 16:
        return 3
    if score >= 10:
        return 2
    return 1


def format_stars(count: int) -> str:
    count = max(1, min(5, int(count)))
    return "★" * count + "☆" * (5 - count)


def build_rec_id(player: str, stat: str, line_value: float, game: str, side: str) -> str:
    base = f"{player}|{stat}|{line_value}|{game}|{side}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()[:12]


# =====================================================================
# ODDS API
# =====================================================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_events(api_key: str) -> dict[str, Any]:
    url = f"{BASE_URL}/sports/basketball_nba/events"
    try:
        response = get_session().get(url, params={"apiKey": api_key}, timeout=REQUEST_TIMEOUT)
        if not response.ok:
            return {"events": [], "error": format_api_error(response), "status_code": response.status_code}
        data = response.json()
        return {"events": data if isinstance(data, list) else [], "error": None, "status_code": response.status_code}
    except Exception as exc:
        return {"events": [], "error": str(exc), "status_code": None}


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
            return {"payload": None, "error": format_api_error(response), "status_code": response.status_code}
        return {"payload": response.json(), "error": None, "status_code": response.status_code}
    except Exception as exc:
        return {"payload": None, "error": str(exc), "status_code": None}


def extract_props_from_event(event_row: dict[str, Any], odds_payload: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
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
                result = {"payload": None, "error": str(exc), "status_code": None}
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
    best: dict[tuple[str, str, float, str], dict[str, Any]] = {}
    for prop in props:
        key = (prop["player"], prop["stat"], float(prop["line"]), prop["game"])
        if key not in best:
            best[key] = {
                **prop,
                "bookmaker_over": prop["bookmaker"],
                "bookmaker_under": prop["bookmaker"],
            }
            continue

        chosen = best[key]
        if float(prop["odds_over"]) > float(chosen["odds_over"]):
            chosen["odds_over"] = float(prop["odds_over"])
            chosen["bookmaker_over"] = prop["bookmaker"]
        if float(prop["odds_under"]) > float(chosen["odds_under"]):
            chosen["odds_under"] = float(prop["odds_under"])
            chosen["bookmaker_under"] = prop["bookmaker"]
        chosen["prob_book_fair_over"] = no_vig_prob_over(chosen["odds_over"], chosen["odds_under"])
    return list(best.values())


# =====================================================================
# NBA API
# =====================================================================
@st.cache_data(ttl=21_600, show_spinner=False)
def fetch_player_log(player_id: int, season: str) -> Optional[pd.DataFrame]:
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = gamelog.get_data_frames()[0]
        if df.empty:
            return None
        df = df[["GAME_DATE", "MATCHUP", "PTS", "REB", "AST"]].copy()
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        df = df.dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE").reset_index(drop=True)
        for col in ["PTS", "REB", "AST"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return None


def preload_logs(player_ids: Iterable[int], seasons: Iterable[str]) -> dict[tuple[int, str], Optional[pd.DataFrame]]:
    tasks = [(int(player_id), str(season)) for player_id in player_ids for season in seasons]
    if not tasks:
        return {}

    results: dict[tuple[int, str], Optional[pd.DataFrame]] = {}
    workers = min(MAX_STATS_WORKERS, max(1, len(tasks)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(fetch_player_log, player_id, season): (player_id, season)
            for player_id, season in tasks
        }
        for future in as_completed(future_map):
            key = future_map[future]
            try:
                results[key] = future.result()
            except Exception:
                results[key] = None
    return results


# =====================================================================
# ANALYTICS
# =====================================================================
def filter_h2h(df: Optional[pd.DataFrame], away_name: str, home_name: str) -> pd.DataFrame:
    columns = ["GAME_DATE", "MATCHUP", "PTS", "REB", "AST"]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    away = resolve_team_abbr(away_name)
    home = resolve_team_abbr(home_name)
    if not away and not home:
        return pd.DataFrame(columns=columns)

    patterns = [token for token in [away, home] if token]
    regex = re.compile(rf"({'|'.join(re.escape(token) for token in patterns)})", re.IGNORECASE)
    mask = df["MATCHUP"].astype(str).apply(lambda value: bool(regex.search(value)))
    return df.loc[mask].copy()


@st.cache_data(show_spinner=False)
def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False)
def records_to_json(records: list[dict[str, Any]]) -> str:
    return json.dumps(records, ensure_ascii=False, indent=2, default=str)


# =====================================================================
# RENDER DETALHE
# =====================================================================
def render_analysis_window(
    df_sel: Optional[pd.DataFrame],
    stat: str,
    base_line: float,
    detail_key: str,
    window_name: str,
    side: str,
    static_plot: bool,
    w_recent: float,
    df_a: Optional[pd.DataFrame],
    df_b: Optional[pd.DataFrame],
    container: Any,
) -> None:
    with container:
        if df_sel is None or df_sel.empty:
            st.info(f"Sem jogos em {window_name}")
            return

        plot_df = df_sel[["GAME_DATE", "MATCHUP", stat]].copy()
        plot_df[stat] = pd.to_numeric(plot_df[stat], errors="coerce")
        plot_df = plot_df.dropna(subset=[stat]).reset_index(drop=True)
        if plot_df.empty:
            st.info(f"Sem dados para {window_name}")
            return

        series = plot_df[stat]
        base_key = _safe_key(detail_key, stat, window_name)
        span = max(3.0, 0.3 * base_line)
        alt_line = st.slider(
            "Linha alternativa",
            min_value=float(max(0.0, base_line - span)),
            max_value=float(base_line + span),
            value=float(base_line),
            step=0.5,
            key=f"alt_{base_key}",
        )

        over_hits = int((series > alt_line).sum())
        under_hits = int((series <= alt_line).sum())
        total = int(len(series))
        over_pct = (over_hits / total * 100.0) if total else 0.0
        under_pct = 100.0 - over_pct

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(label="📈 Over (acertos)", value=f"{over_hits}/{total}", delta=f"{over_pct:.1f}%")
        with c2:
            st.metric(label="📉 Under (acertos)", value=f"{under_hits}/{total}", delta=f"{under_pct:.1f}%")
        with c3:
            st.metric(label="📊 Média da janela", value=f"{series.mean():.1f}")

        labels: list[str] = []
        for _, row in plot_df.iterrows():
            date_value = row.get("GAME_DATE")
            date_str = pd.to_datetime(date_value).strftime("%m/%d") if pd.notna(date_value) else "--/--"
            matchup = str(row.get("MATCHUP", "")).strip()
            labels.append(f"{matchup}\n{date_str}" if matchup else date_str)

        colors = ["#2ecc71" if value > alt_line else "#e74c3c" for value in series.tolist()]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=labels,
                y=series.tolist(),
                marker_color=colors,
                text=[f"{value:.1f}" for value in series.tolist()],
                textposition="outside",
                hovertemplate="%{x}<br>Valor: %{y}<extra></extra>",
            )
        )
        fig.add_hline(
            y=alt_line,
            line_dash="dash",
            line_width=2,
            line_color="#FFA500",
            annotation_text=f"Linha {alt_line:.1f}",
            annotation_position="top left",
        )
        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=30, b=80),
            xaxis_tickangle=-35,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(
            fig,
            width="stretch",
            config={"displayModeBar": False, "staticPlot": static_plot},
            key=f"plot_{base_key}",
        )

        ser_recent = pd.Series(dtype=float)
        ser_comp = pd.Series(dtype=float)
        if df_a is not None and not df_a.empty and stat in df_a.columns:
            ser_recent = pd.to_numeric(df_a[stat], errors="coerce").dropna().tail(15)
        if df_b is not None and not df_b.empty and stat in df_b.columns:
            ser_comp = pd.to_numeric(df_b[stat], errors="coerce").dropna()

        distribution = estimate_weighted_distribution(ser_recent, ser_comp, w_recent=w_recent)
        if distribution:
            mu, sd = distribution
            p_over_model = normal_over_probability(alt_line, mu, sd) * 100.0
            chosen_prob = p_over_model if side == "Over" else 100.0 - p_over_model
            st.caption(
                f"🎯 Probabilidade do modelo para **{side} {alt_line:.1f}**: "
                f"**{chosen_prob:.1f}%** · média {mu:.1f} · desvio {sd:.1f}"
            )


@st.fragment
def render_selected_detail(detail: dict[str, Any], static_plot: bool, w_recent: float) -> None:
    player_id = int(detail["player_id"])
    df_a = fetch_player_log(player_id, TEMPORADA_ATUAL)
    df_b = fetch_player_log(player_id, TEMPORADA_COMP)

    st.markdown("<hr/>", unsafe_allow_html=True)
    col1, col2 = st.columns([5, 1])
    with col1:
        st.subheader(f"🔎 {detail['Jogador']} — {detail['Stat']} {detail['Lado']} {detail['Linha']}")
        st.caption(
            f"{detail['Jogo']} · Melhor book: {detail['Melhor Book']} · Odds: {detail['Odds']:.2f} · "
            f"Value: {detail['Value (modelo) %']:.1f}%"
        )
    with col2:
        if st.button("Fechar", key=f"close_{detail['rec_id']}", width="stretch"):
            st.session_state["selected_rec_id"] = None
            st.rerun()

    if df_a is None and df_b is None:
        st.warning("Sem histórico suficiente para exibir detalhes desse jogador.")
        return

    tabs = st.tabs(["📅 L5", "📅 L10", "📅 L15", "🆚 H2H", f"📆 {TEMPORADA_ATUAL}", f"📆 {TEMPORADA_COMP}"])
    stat = detail["Stat"]
    line_value = float(detail["Linha"])
    away = detail["away_team"]
    home = detail["home_team"]
    side = detail["Lado"]
    detail_key = detail["rec_id"]

    render_analysis_window(
        df_a.tail(5) if df_a is not None else None,
        stat,
        line_value,
        detail_key,
        "L5",
        side,
        static_plot,
        w_recent,
        df_a,
        df_b,
        tabs[0],
    )
    render_analysis_window(
        df_a.tail(10) if df_a is not None else None,
        stat,
        line_value,
        detail_key,
        "L10",
        side,
        static_plot,
        w_recent,
        df_a,
        df_b,
        tabs[1],
    )
    render_analysis_window(
        df_a.tail(15) if df_a is not None else None,
        stat,
        line_value,
        detail_key,
        "L15",
        side,
        static_plot,
        w_recent,
        df_a,
        df_b,
        tabs[2],
    )

    h2h_frames = []
    if df_a is not None and not df_a.empty:
        h2h_frames.append(filter_h2h(df_a, away, home))
    if df_b is not None and not df_b.empty:
        h2h_frames.append(filter_h2h(df_b, away, home))
    h2h_frames = [frame for frame in h2h_frames if not frame.empty]
    df_h2h = pd.concat(h2h_frames, ignore_index=True) if h2h_frames else pd.DataFrame()
    render_analysis_window(
        df_h2h,
        stat,
        line_value,
        detail_key,
        "H2H",
        side,
        static_plot,
        w_recent,
        df_a,
        df_b,
        tabs[3],
    )

    render_analysis_window(
        df_a,
        stat,
        line_value,
        detail_key,
        TEMPORADA_ATUAL,
        side,
        static_plot,
        w_recent,
        df_a,
        df_b,
        tabs[4],
    )
    render_analysis_window(
        df_b,
        stat,
        line_value,
        detail_key,
        TEMPORADA_COMP,
        side,
        static_plot,
        w_recent,
        df_a,
        df_b,
        tabs[5],
    )


# =====================================================================
# APP
# =====================================================================
slate_date = datetime.now(timezone.utc).strftime("%d/%m/%Y")
st.title("📋 Prop Bet Cheat Sheet — NBA")
st.subheader(f"Melhores props para {slate_date}")
st.caption("Compara linhas das casas com projeções modeladas (forma recente + temporada).")
header_left, header_right = st.columns([5, 1])
with header_left:
    st.caption(
        f"Temporada {TEMPORADA_ATUAL} · base {TEMPORADA_COMP} · "
        "PTS/REB/AST · odds e stats em cache até você clicar em Atualizar."
    )
with header_right:
    if st.button("🔄 Atualizar", width="stretch"):
        st.cache_data.clear()
        st.session_state["selected_rec_id"] = None
        st.rerun()

if not API_KEY:
    st.error(
        "Configure `ODDS_API_KEY` em `.streamlit/secrets.toml` ou na variável de ambiente antes de rodar o app."
    )
    st.stop()

fetch_start = perf_counter()
with st.spinner("🔍 Buscando linhas das casas..."):
    odds_result = get_player_props(API_KEY)
    raw_props = odds_result.get("props", [])
fetch_seconds = perf_counter() - fetch_start

if not raw_props:
    st.error("❌ Não consegui buscar linhas da Odds API agora.")
    if odds_result.get("error"):
        st.caption(f"Motivo provável: {odds_result['error']}")
    if odds_result.get("events_total") is not None:
        st.caption(
            f"Eventos retornados: {odds_result.get('events_total', 0)} · "
            f"eventos consultados: {odds_result.get('events_considered', 0)}"
        )
    if odds_result.get("event_errors"):
        with st.expander("Ver erros da Odds API"):
            for item in odds_result["event_errors"]:
                st.write(f"- {item}")
    st.stop()

bookmaker_options = ["Todos"] + sorted({str(prop["bookmaker"]) for prop in raw_props})
game_options = ["Todos"] + sorted({str(prop["game"]) for prop in raw_props})
for key, value in DEFAULT_FILTERS.items():
    st.session_state.setdefault(key, value)

st.sidebar.title("⚙️ Configurações")
st.sidebar.caption("As alterações só entram quando você clicar em **Aplicar filtros**.")

with st.sidebar.form("filters_form"):
    num_jogos_form = st.slider("📊 Jogos recentes", 5, 30, st.session_state["num_jogos"])
    w_recent_form = st.slider("⚖️ Peso forma recente", 0.0, 1.0, float(st.session_state["w_recent"]), 0.05)

    st.markdown(
        f"**Temporada atual:** `{TEMPORADA_ATUAL}`  \n"
        f"**Temporada comparação:** `{TEMPORADA_COMP}`"
    )
    st.markdown("---")

    bookmaker_sel_form = st.selectbox(
        "🎲 Bookmaker",
        bookmaker_options,
        index=bookmaker_options.index(st.session_state["bookmaker_sel"])
        if st.session_state["bookmaker_sel"] in bookmaker_options
        else 0,
    )
    stat_sel_form = st.selectbox(
        "📊 Estatística",
        ["Todas", "PTS", "REB", "AST"],
        index=["Todas", "PTS", "REB", "AST"].index(st.session_state["stat_sel"]),
    )
    game_sel_form = st.selectbox(
        "🏟️ Jogo",
        game_options,
        index=game_options.index(st.session_state["game_sel"])
        if st.session_state["game_sel"] in game_options
        else 0,
    )
    side_sel_form = st.selectbox(
        "↕️ Lado recomendado",
        ["Todos", "Over", "Under"],
        index=["Todos", "Over", "Under"].index(st.session_state["side_sel"]),
    )
    min_stars_form = st.slider("⭐ Estrelas mínimas", 1, 5, int(st.session_state["min_stars"]))
    player_search_form = st.text_input("🔎 Filtrar jogador", value=st.session_state["player_search"])
    sort_by_form = st.selectbox(
        "↕️ Ordenar por",
        ["Value", "Probabilidade", "Odds"],
        index=["Value", "Probabilidade", "Odds"].index(st.session_state["sort_by"]),
    )

    st.markdown("---")
    odds_min_form = st.number_input("Odds mín", 1.01, 10.0, float(st.session_state["odds_min"]), 0.05)
    odds_max_form = st.number_input("Odds máx", 1.01, 10.0, float(st.session_state["odds_max"]), 0.05)
    min_value_form = st.slider("💰 Value mínimo (%)", 0, 40, int(st.session_state["min_value"]))
    min_cover_prob_form = st.slider(
        "🎯 Cover prob. mínima (modelo) %",
        0,
        90,
        int(st.session_state["min_cover_prob"]),
    )

    st.markdown("---")
    top_per_game_form = st.slider("🧹 Máx picks por jogo", 3, 30, int(st.session_state["top_per_game"]), 1)
    static_plot_form = st.toggle(
        "🌙 Modo gráfico estático",
        value=bool(st.session_state["static_plot"]),
        help="Desabilita interatividade dos gráficos para reduzir CPU no navegador.",
    )

    main_view_form = st.radio(
        "Visão principal",
        ["Cheat Sheet", "Por jogo", "Top picks"],
        index=["Cheat Sheet", "Por jogo", "Top picks"].index(st.session_state["main_view"]),
        horizontal=True,
    )

    submitted = st.form_submit_button("Aplicar filtros", width="stretch")

if submitted:
    st.session_state.update(
        {
            "num_jogos": num_jogos_form,
            "w_recent": w_recent_form,
            "bookmaker_sel": bookmaker_sel_form,
            "stat_sel": stat_sel_form,
            "game_sel": game_sel_form,
            "side_sel": side_sel_form,
            "min_stars": min_stars_form,
            "odds_min": odds_min_form,
            "odds_max": odds_max_form,
            "min_value": min_value_form,
            "min_cover_prob": min_cover_prob_form,
            "top_per_game": top_per_game_form,
            "static_plot": static_plot_form,
            "player_search": player_search_form,
            "sort_by": sort_by_form,
            "main_view": main_view_form,
            "selected_rec_id": None,
        }
    )

cfg = {key: st.session_state[key] for key in DEFAULT_FILTERS}

props = raw_props
if cfg["bookmaker_sel"] != "Todos":
    props = [prop for prop in props if prop["bookmaker"] == cfg["bookmaker_sel"]]
if cfg["stat_sel"] != "Todas":
    props = [prop for prop in props if prop["stat"] == cfg["stat_sel"]]
if cfg["player_search"].strip():
    needle = cfg["player_search"].strip().lower()
    props = [prop for prop in props if needle in str(prop["player"]).lower()]
props = [
    prop
    for prop in props
    if cfg["odds_min"] <= float(prop["odds_over"]) <= cfg["odds_max"]
    and cfg["odds_min"] <= float(prop["odds_under"]) <= cfg["odds_max"]
]
props = dedupe_props(props)

if not props:
    st.warning("⚠️ Nenhuma linha disponível com os filtros atuais.")
    st.stop()

st.success(f"✅ {len(props)} linhas após filtros")

unique_players = sorted({prop["player"] for prop in props})
name_to_id: dict[str, int] = {}
unmatched_players: list[str] = []
for player_name in unique_players:
    player_id = resolve_player_id(player_name)
    if player_id is None:
        unmatched_players.append(player_name)
    else:
        name_to_id[player_name] = player_id

stats_start = perf_counter()
with st.spinner(f"📚 Carregando histórico de {len(name_to_id)} jogadores..."):
    logs = preload_logs(sorted(set(name_to_id.values())), [TEMPORADA_ATUAL, TEMPORADA_COMP])
stats_seconds = perf_counter() - stats_start

analysis_start = perf_counter()
progress = st.progress(0)
status = st.empty()
recs: list[dict[str, Any]] = []

for index, prop in enumerate(props, start=1):
    progress.progress(index / len(props))
    status.text(f"Analisando {prop['player']} ({index}/{len(props)})")

    player_id = name_to_id.get(prop["player"])
    if player_id is None:
        continue

    df_current = logs.get((player_id, TEMPORADA_ATUAL))
    df_compare = logs.get((player_id, TEMPORADA_COMP))
    if df_current is None and df_compare is None:
        continue

    stat = str(prop["stat"])
    line_value = float(prop["line"])
    ser_recent = (
        pd.to_numeric(df_current[stat], errors="coerce").dropna().tail(int(cfg["num_jogos"]))
        if df_current is not None and not df_current.empty
        else pd.Series(dtype=float)
    )
    ser_comp = (
        pd.to_numeric(df_compare[stat], errors="coerce").dropna()
        if df_compare is not None and not df_compare.empty
        else pd.Series(dtype=float)
    )

    distribution = estimate_weighted_distribution(ser_recent, ser_comp, w_recent=float(cfg["w_recent"]))
    if not distribution:
        continue
    mu, sd = distribution

    p_over = normal_over_probability(line_value, mu, sd)
    p_under = 1.0 - p_over

    imp_over = implied_percent(float(prop["odds_over"]))
    imp_under = implied_percent(float(prop["odds_under"]))
    val_over = p_over * 100.0 - imp_over
    val_under = p_under * 100.0 - imp_under

    side, model_value = ("Over", val_over) if val_over >= val_under else ("Under", val_under)
    if model_value < float(cfg["min_value"]):
        continue

    chosen_prob = p_over * 100.0 if side == "Over" else p_under * 100.0
    chosen_odds = float(prop["odds_over"] if side == "Over" else prop["odds_under"])
    chosen_bookmaker = prop["bookmaker_over"] if side == "Over" else prop["bookmaker_under"]
    fair_side_prob = float(prop["prob_book_fair_over"] if side == "Over" else 100.0 - prop["prob_book_fair_over"])

    ser_l5 = (
        pd.to_numeric(df_current[stat], errors="coerce").dropna().tail(5)
        if df_current is not None and not df_current.empty
        else pd.Series(dtype=float)
    )
    ser_l10 = (
        pd.to_numeric(df_current[stat], errors="coerce").dropna().tail(10)
        if df_current is not None and not df_current.empty
        else pd.Series(dtype=float)
    )
    recent_rate = side_hit_rate(ser_recent, line_value, side)
    l5_rate = side_hit_rate(ser_l5, line_value, side)
    l10_rate = side_hit_rate(ser_l10, line_value, side)
    comp_rate = side_hit_rate(ser_comp, line_value, side)

    h2h_frames = []
    if df_current is not None and not df_current.empty:
        h2h_frames.append(filter_h2h(df_current, prop["away_team"], prop["home_team"]))
    if df_compare is not None and not df_compare.empty:
        h2h_frames.append(filter_h2h(df_compare, prop["away_team"], prop["home_team"]))
    h2h_frames = [frame for frame in h2h_frames if not frame.empty]
    df_h2h = pd.concat(h2h_frames, ignore_index=True) if h2h_frames else pd.DataFrame()
    if not df_h2h.empty:
        hits = int(side_hit_mask(df_h2h[stat], line_value, side).sum())
        total_h2h = int(len(df_h2h))
        h2h_txt = f"{hits}/{total_h2h} ({(hits / total_h2h) * 100:.1f}%)"
    else:
        h2h_txt = "—"

    stars = compute_star_rating(model_value, chosen_prob)
    rec_id = build_rec_id(prop["player"], stat, line_value, prop["game"], side)
    recs.append(
        {
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
            "Jogos recentes usados": int(len(ser_recent)),
            "away_team": prop["away_team"],
            "home_team": prop["home_team"],
        }
    )

progress.empty()
status.empty()
analysis_seconds = perf_counter() - analysis_start

if unmatched_players:
    st.caption(
        f"⚠️ {len(unmatched_players)} jogador(es) não foram casados automaticamente na NBA API. "
        f"Ex.: {', '.join(unmatched_players[:5])}"
    )

if not recs:
    st.warning(f"⚠️ Nenhuma aposta com value ≥ {cfg['min_value']}%.")
    st.stop()

recs = [
    rec
    for rec in recs
    if int(rec["Estrelas"]) >= int(cfg["min_stars"])
    and float(rec["Prob (modelo) %"]) >= float(cfg["min_cover_prob"])
    and (cfg["game_sel"] == "Todos" or rec["Jogo"] == cfg["game_sel"])
    and (cfg["side_sel"] == "Todos" or rec["Lado"] == cfg["side_sel"])
]
if not recs:
    st.warning("⚠️ Nenhuma pick passou nos filtros de estrelas, jogo, lado ou cover prob.")
    st.stop()

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric(label="📊 Linhas", value=len(props))
with c2:
    st.metric(label="🎯 Picks com value", value=len(recs))
with c3:
    avg_over = sum(float(prop["odds_over"]) for prop in props) / len(props)
    st.metric(label="💰 Odds média (Over)", value=f"{avg_over:.2f}")
with c4:
    total_games = len({rec["Jogo"] for rec in recs})
    st.metric(label="🏟️ Jogos com picks", value=total_games)
with c5:
    five_star = sum(1 for rec in recs if int(rec["Estrelas"]) >= 5)
    st.metric(label="⭐ Picks 5 estrelas", value=five_star)

st.caption(
    f"⏱️ Odds: {fetch_seconds:.1f}s · Stats: {stats_seconds:.1f}s · Análise: {analysis_seconds:.1f}s"
)

if cfg["sort_by"] == "Probabilidade":
    recs = sorted(recs, key=lambda item: (-item["Prob (modelo) %"], -item["Value (modelo) %"], item["Jogo"]))
elif cfg["sort_by"] == "Odds":
    recs = sorted(recs, key=lambda item: (-item["Odds"], -item["Value (modelo) %"], item["Jogo"]))
else:
    recs = sorted(recs, key=lambda item: (-item["Estrelas"], -item["Value (modelo) %"], -item["Prob (modelo) %"], item["Jogo"]))

CHEET_SHEET_COLUMNS = [
    "Rating",
    "Jogador",
    "Jogo",
    "Stat",
    "Pick",
    "Projeção",
    "Diff vs linha",
    "Prob (modelo) %",
    "Value (modelo) %",
    "L5 %",
    "L10 %",
    "H2H",
    "Odds",
    "Melhor Book",
]

grouped: dict[str, list[dict[str, Any]]] = {}
for rec in recs:
    grouped.setdefault(rec["Jogo"], []).append(rec)

game_order = sorted(
    grouped,
    key=lambda game: max(item["Value (modelo) %"] for item in grouped[game]),
    reverse=True,
)

st.markdown("<hr/>", unsafe_allow_html=True)
main_view = cfg["main_view"]

if main_view == "Cheat Sheet":
    st.header("📋 Cheat Sheet")
    st.caption(
        "Linha vs projeção modelada, cover probability, edge e hit rates — inspirado no fluxo do BettingPros."
    )
    df_sheet = pd.DataFrame(recs)
    sheet_display = df_sheet[[col for col in CHEET_SHEET_COLUMNS if col in df_sheet.columns]]
    st.dataframe(
        sheet_display,
        height=min(720, 48 + len(sheet_display) * 36),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rating": st.column_config.TextColumn("⭐", width="small"),
            "Projeção": st.column_config.NumberColumn(format="%.1f"),
            "Diff vs linha": st.column_config.NumberColumn(format="%+.1f"),
            "Prob (modelo) %": st.column_config.ProgressColumn(
                "Cover %",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "Value (modelo) %": st.column_config.NumberColumn("Edge %", format="+%.1f"),
            "L5 %": st.column_config.NumberColumn(format="%.0f%%"),
            "L10 %": st.column_config.NumberColumn(format="%.0f%%"),
            "Odds": st.column_config.NumberColumn(format="%.2f"),
        },
    )

elif main_view == "Top picks":
    st.header("⭐ Top picks")
    st.caption("Picks com 4–5 estrelas, ordenadas por edge e cover probability.")
    top_recs = [rec for rec in recs if int(rec["Estrelas"]) >= 4][:25]
    if not top_recs:
        st.info("Nenhuma pick com 4+ estrelas nos filtros atuais. Reduza value mínimo ou estrelas mínimas.")
    for idx, rec in enumerate(top_recs, start=1):
        with st.container(border=True):
            col_a, col_b, col_c = st.columns([4, 1, 1])
            with col_a:
                st.markdown(
                    f"### {rec['Rating']} · {rec['Jogador']} — {rec['Stat']} {rec['Pick']}"
                )
                st.markdown(
                    f"<span class='small-muted'>"
                    f"🏟️ {rec['Jogo']} · Projeção **{rec['Projeção']:.1f}** (linha {rec['Linha']}) · "
                    f"Cover **{rec['Prob (modelo) %']:.1f}%** · L5 {fmt_rate(rec.get('L5 %'))} · L10 {fmt_rate(rec.get('L10 %'))}"
                    f"</span>",
                    unsafe_allow_html=True,
                )
            with col_b:
                st.markdown(
                    f"<div class='value-positive' style='font-size:26px;text-align:center;'>"
                    f"+{rec['Value (modelo) %']:.1f}%</div>",
                    unsafe_allow_html=True,
                )
            with col_c:
                if st.button("📊 Detalhes", key=f"top_{rec['rec_id']}", width="stretch"):
                    st.session_state["selected_rec_id"] = rec["rec_id"]
                    st.rerun()
else:
    st.header("🎯 Por jogo")
    st.caption(
        "Agrupado por partida. Detalhes completos sob demanda para não sobrecarregar o navegador."
    )

for game_idx, game in enumerate(game_order):
    if main_view != "Por jogo":
        break
    items = grouped[game]
    render_items = items[: int(cfg["top_per_game"])]
    best_value = max(item["Value (modelo) %"] for item in items)
    expander_label = (
        f"🏟️ {game} — {len(items)} sugestões "
        f"(mostrando {len(render_items)}) · melhor value {best_value:.1f}%"
    )
    with st.expander(expander_label, expanded=(game_idx == 0), width="stretch"):
        for idx, rec in enumerate(render_items, start=1):
            with st.container(border=True):
                col_a, col_b, col_c = st.columns([4, 1, 1])
                with col_a:
                    st.markdown(
                        f"### {idx}. {rec.get('Rating', '')} {rec['Jogador']} — {rec['Stat']} {rec['Pick']}"
                    )
                    st.markdown(
                        f"<span class='small-muted'>"
                        f"🎲 {rec['Melhor Book']} · Odds {rec['Odds']:.2f} · Projeção {rec['Projeção']:.1f} "
                        f"(Δ {rec['Diff vs linha']:+.1f})<br>"
                        f"📊 Cover: {rec['Prob (modelo) %']:.1f}% · Edge vs book: +{rec['Value (modelo) %']:.1f}%<br>"
                        f"📈 L5: {fmt_rate(rec.get('L5 %'))} · L10: {fmt_rate(rec.get('L10 %'))} · "
                        f"{TEMPORADA_COMP}: {fmt_rate(rec.get('Taxa lado comp %'))} · H2H: {rec['H2H']}"
                        f"</span>",
                        unsafe_allow_html=True,
                    )
                with col_b:
                    klass = "value-positive" if rec["Value (modelo) %"] >= 0 else "value-negative"
                    st.markdown(
                        f"<div class='{klass}' style='font-size:26px;text-align:center;'>"
                        f"+{rec['Value (modelo) %']:.1f}%</div>"
                        f"<div class='small-muted' style='text-align:center;'>Value</div>",
                        unsafe_allow_html=True,
                    )
                with col_c:
                    if st.button("📊 Detalhes", key=f"detail_{rec['rec_id']}", width="stretch"):
                        st.session_state["selected_rec_id"] = rec["rec_id"]
                        st.rerun()

        st.markdown("<hr/>", unsafe_allow_html=True)
        df_game = pd.DataFrame(render_items)
        if not df_game.empty:
            df_game["Label"] = df_game["Jogador"].str.split().str[-1] + " " + df_game["Stat"] + " " + df_game["Lado"]
            fig = go.Figure(
                go.Bar(
                    x=df_game["Label"],
                    y=df_game["Value (modelo) %"],
                    text=[f"{value:.1f}%" for value in df_game["Value (modelo) %"]],
                    textposition="outside",
                    hovertemplate="%{x}<br>Value: %{y:.1f}%<extra></extra>",
                )
            )
            fig.update_layout(
                title=f"Value por aposta — {game}",
                xaxis_title="Aposta",
                yaxis_title="Value (%)",
                xaxis_tickangle=-25,
                height=340,
                margin=dict(l=10, r=10, t=50, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(
                fig,
                width="stretch",
                config={"displayModeBar": False, "staticPlot": cfg["static_plot"]},
                key=f"game_chart_{_safe_key(game)}",
            )

selected_rec_id = st.session_state.get("selected_rec_id")
selected_lookup = {rec["rec_id"]: rec for rec in recs}
if selected_rec_id and selected_rec_id in selected_lookup:
    render_selected_detail(
        selected_lookup[selected_rec_id],
        static_plot=bool(cfg["static_plot"]),
        w_recent=float(cfg["w_recent"]),
    )
elif selected_rec_id:
    st.session_state["selected_rec_id"] = None
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.info("A seleção atual saiu do filtro. Escolha outra aposta em **Detalhes**.")
else:
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.info("Selecione uma aposta em **Detalhes** para abrir a análise completa sem pesar a página inteira.")

st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader("📋 Tabela completa")
df_all = pd.DataFrame(recs)
df_display = df_all.drop(columns=["player_id", "away_team", "home_team", "event_id", "rec_id"], errors="ignore")
st.dataframe(df_display, height=420, use_container_width=True, hide_index=True)

export_records = [
    {key: value for key, value in rec.items() if key not in {"player_id", "away_team", "home_team", "event_id", "rec_id"}}
    for rec in recs
]

col_csv, col_json = st.columns(2)
with col_csv:
    st.download_button(
        "📥 Baixar CSV",
        data=dataframe_to_csv_bytes(df_display),
        file_name=f"nba_props_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        width="stretch",
    )
with col_json:
    st.download_button(
        "📥 Baixar JSON",
        data=records_to_json(export_records),
        file_name=f"nba_props_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
        width="stretch",
    )

st.markdown("<hr/>", unsafe_allow_html=True)
st.info(
    """
### Como usar (estilo cheat sheet)
- **Cheat Sheet**: tabela única com estrelas, projeção, cover %, edge e L5/L10 — parecido com o [BettingPros Prop Cheat Sheet](https://www.bettingpros.com/nba/picks/prop-bets/).
- **Top picks**: só 4–5 estrelas; bom para montar um card rápido do dia.
- **Por jogo**: visão antiga agrupada por partida, com gráfico de value.

### Próximos passos possíveis
- Mais mercados (3PT, PRA, steals/blocks) via Odds API.
- Comparar várias casas na mesma linha (não só melhor odd).
- Histórico de acerto das picks salvas no dia.

### Dicas
- Filtre **L5/L10 altos + 4+ estrelas** para achar props “quentes”.
- **Modo gráfico estático** economiza CPU no navegador.
"""
)
