from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nba_props_data import (
    current_nba_season,
    dedupe_props,
    get_player_props,
    head_to_head_games,
    implied_percent,
    preload_logs,
    previous_nba_season,
    resolve_player_id,
)
from nba_props_demo import DEMO_PLAYER_IDS, DEMO_SEASONS, demo_player_logs, demo_player_props
from nba_props_model import (
    estimate_line_probabilities,
    estimate_weighted_distribution,
    line_result_counts,
    side_hit_rate,
)

# =====================================================================
# CONFIG GERAL
# =====================================================================
st.set_page_config(
    page_title="NBA Player Performance Analytics",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------- CSS
st.markdown(
    """
<style>
:root{
  --card: color-mix(in srgb, currentColor 4%, transparent);
  --stroke: color-mix(in srgb, currentColor 15%, transparent);
  --over: #178150;
  --under: #bd3838;
}
section[data-testid="stSidebar"] { border-right: 1px solid var(--stroke); }
.block-container { padding-top:1.2rem; padding-bottom:2rem; }
h1,h2,h3 { letter-spacing:.2px; }

div[data-testid="stMetric"]{
  background: var(--card);
  border:1px solid var(--stroke); border-radius:12px; padding:14px 16px;
}
div[data-testid="stMetric"] .metric-label{ opacity:.8; font-weight:600; }
div[data-testid="stMetric"] .metric-value{ font-weight:800; }

.value-positive{ color: var(--over); font-weight:800; }
.value-negative{ color: var(--under); font-weight:800; }

hr{ border:none; border-top:1px solid var(--stroke); margin:18px 0; }

[data-baseweb="tab-list"]{ border-bottom:1px solid var(--stroke); }
button[kind="secondary"]{ border:1px solid var(--stroke) !important; }

.pick-card{
  background: var(--card);
  border:1px solid var(--stroke); border-radius:14px; padding:14px 16px;
}

.small-muted{ opacity:.8; font-size:12.5px; }
.badge{
  display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px;
  background:var(--card); border:1px solid var(--stroke);
}
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# CONSTANTES
# =====================================================================
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
    "main_view": "Visão geral",
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


# Uma sessão por thread evita contenção ao fazer requests em paralelo.
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
    distribution: tuple[float, float],
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
        span = max(3.0, math.ceil(0.6 * base_line) / 2)
        alt_line = st.slider(
            "Linha alternativa",
            min_value=float(max(0.0, base_line - span)),
            max_value=float(base_line + span),
            value=float(base_line),
            step=0.5,
            key=f"alt_{base_key}",
        )

        over_hits, under_hits, pushes = line_result_counts(series, alt_line)
        decisions = over_hits + under_hits
        over_rate = over_hits / decisions * 100.0 if decisions else None
        under_rate = under_hits / decisions * 100.0 if decisions else None

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                label="Over (acertos)",
                value=f"{over_hits}/{decisions}" if decisions else "-",
                delta=fmt_rate(over_rate) if over_rate is not None else None,
            )
        with c2:
            st.metric(
                label="Under (acertos)",
                value=f"{under_hits}/{decisions}" if decisions else "-",
                delta=fmt_rate(under_rate) if under_rate is not None else None,
            )
        with c3:
            st.metric(label="Média da janela", value=f"{series.mean():.1f}")
        if pushes:
            st.caption(f"Devoluções: {pushes}/{len(series)} jogos. Taxas calculadas sobre jogos sem devolução.")

        labels: list[str] = []
        for _, row in plot_df.iterrows():
            date_value = row.get("GAME_DATE")
            date_str = pd.to_datetime(date_value).strftime("%m/%d") if pd.notna(date_value) else "--/--"
            matchup = str(row.get("MATCHUP", "")).strip()
            labels.append(f"{matchup}\n{date_str}" if matchup else date_str)

        colors = [
            "#2ecc71" if value > alt_line else "#e74c3c" if value < alt_line else "#94a3b8"
            for value in series
        ]
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

        mu, sd = distribution
        prob_over, prob_under, prob_push = estimate_line_probabilities(alt_line, mu, sd)
        chosen_prob = prob_over if side == "Over" else prob_under
        push_text = f" · devolução {prob_push:.1%}" if prob_push else ""
        st.caption(
            f"Probabilidade do modelo para **{side} {alt_line:.1f}**: "
            f"**{chosen_prob:.1%}** · média {mu:.1f} · desvio {sd:.1f}{push_text}"
        )


@st.fragment
def render_selected_detail(
    detail: dict[str, Any],
    df_a: pd.DataFrame | None,
    df_b: pd.DataFrame | None,
    static_plot: bool,
) -> None:

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

    recent_games = df_a.dropna(subset=[stat]) if df_a is not None else None
    windows = [
        (recent_games.tail(5) if recent_games is not None else None, "L5"),
        (recent_games.tail(10) if recent_games is not None else None, "L10"),
        (recent_games.tail(15) if recent_games is not None else None, "L15"),
        (head_to_head_games(df_a, df_b, away, home), "H2H"),
        (df_a, TEMPORADA_ATUAL),
        (df_b, TEMPORADA_COMP),
    ]
    for (games, window_name), tab in zip(windows, tabs):
        render_analysis_window(
            games, stat, line_value, detail_key, window_name, side, static_plot, detail["distribution"], tab
        )


# =====================================================================
# APP
# =====================================================================
st.sidebar.title("Configurações")
data_mode = st.sidebar.radio("Fonte dos dados", ("Demonstração", "Ao vivo"))
demo_mode = data_mode == "Demonstração"
if demo_mode:
    TEMPORADA_ATUAL, TEMPORADA_COMP = DEMO_SEASONS
else:
    TEMPORADA_ATUAL = current_nba_season()
    TEMPORADA_COMP = previous_nba_season(TEMPORADA_ATUAL)

st.title("NBA Player Performance Analytics")
st.write("Explore o histórico de pontos, rebotes e assistências e compare projeções com linhas de referência.")
if demo_mode:
    st.info("Modo demonstração. Jogadores, estatísticas e cotações simulados.")
else:
    slate_date = datetime.now(timezone.utc).strftime("%d/%m/%Y")
    st.caption(f"Dados ao vivo · {slate_date} UTC")
    if not API_KEY:
        st.error("Configure ODDS_API_KEY para usar dados ao vivo. A demonstração continua disponível na barra lateral.")
        st.stop()
    if st.button("Atualizar dados ao vivo", width="content"):
        st.cache_data.clear()
        st.session_state["selected_rec_id"] = None
        st.rerun()

fetch_start = perf_counter()
with st.spinner("Carregando linhas de referência..."):
    odds_result = demo_player_props() if demo_mode else get_player_props(API_KEY)
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
        "↕️ Lado da linha",
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
        ["Visão geral", "Por jogo", "Destaques"],
        index=["Visão geral", "Por jogo", "Destaques"].index(st.session_state["main_view"]),
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
    player_id = DEMO_PLAYER_IDS.get(player_name) if demo_mode else resolve_player_id(player_name)
    if player_id is None:
        unmatched_players.append(player_name)
    else:
        name_to_id[player_name] = player_id

stats_start = perf_counter()
with st.spinner(f"📚 Carregando histórico de {len(name_to_id)} jogadores..."):
    logs = (
        demo_player_logs()
        if demo_mode
        else preload_logs(sorted(set(name_to_id.values())), [TEMPORADA_ATUAL, TEMPORADA_COMP])
    )
stats_seconds = perf_counter() - stats_start

analysis_start = perf_counter()
progress = st.progress(0)
status = st.empty()
recs: list[dict[str, Any]] = []
h2h_by_event_player: dict[tuple[str, int], pd.DataFrame] = {}

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

    p_over, p_under, p_push = estimate_line_probabilities(line_value, mu, sd)

    imp_over = implied_percent(float(prop["odds_over"]))
    imp_under = implied_percent(float(prop["odds_under"]))
    # Devoluções têm retorno líquido zero. Compare as odds com a probabilidade entre jogos decididos.
    p_decision = p_over + p_under
    if not p_decision:
        continue
    val_over = p_over / p_decision * 100.0 - imp_over
    val_under = p_under / p_decision * 100.0 - imp_under

    side, model_value = ("Over", val_over) if val_over >= val_under else ("Under", val_under)
    if model_value < float(cfg["min_value"]):
        continue

    chosen_prob = p_over * 100.0 if side == "Over" else p_under * 100.0
    chosen_odds = float(prop["odds_over"] if side == "Over" else prop["odds_under"])
    chosen_bookmaker = prop["bookmaker_over"] if side == "Over" else prop["bookmaker_under"]
    fair_side_prob = float(prop["prob_book_fair_over"] if side == "Over" else prop["prob_book_fair_under"])

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

    h2h_key = (prop["event_id"], player_id)
    if h2h_key not in h2h_by_event_player:
        h2h_by_event_player[h2h_key] = head_to_head_games(
            df_current, df_compare, prop["away_team"], prop["home_team"]
        )
    df_h2h = h2h_by_event_player[h2h_key]
    h2h_txt = "-"
    if not df_h2h.empty:
        over, under, pushes = line_result_counts(df_h2h[stat], line_value)
        decisions = over + under
        hits = over if side == "Over" else under
        if decisions:
            h2h_txt = f"{hits}/{decisions} ({hits / decisions:.1%})"
        if pushes:
            h2h_txt += f" · {pushes} devol."

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
            "Jogos recentes usados": int(len(ser_recent)),
            "away_team": prop["away_team"],
            "home_team": prop["home_team"],
            "distribution": distribution,
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

if main_view == "Visão geral":
    st.header("📋 Visão geral")
    st.caption(
        "Compare a linha de referência, a projeção e a frequência observada nos jogos recentes."
    )
    df_sheet = pd.DataFrame(recs)
    sheet_display = df_sheet[[col for col in CHEET_SHEET_COLUMNS if col in df_sheet.columns]]
    st.dataframe(
        sheet_display,
        height=min(720, 48 + len(sheet_display) * 36),
        width="stretch",
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

elif main_view == "Destaques":
    st.header("⭐ Destaques")
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
    detail = selected_lookup[selected_rec_id]
    player_id = int(detail["player_id"])
    render_selected_detail(
        detail,
        logs.get((player_id, TEMPORADA_ATUAL)),
        logs.get((player_id, TEMPORADA_COMP)),
        static_plot=bool(cfg["static_plot"]),
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
internal_fields = {"player_id", "away_team", "home_team", "event_id", "rec_id", "distribution"}
export_records = [{key: value for key, value in rec.items() if key not in internal_fields} for rec in recs]
df_display = pd.DataFrame(export_records)
st.dataframe(df_display, height=420, width="stretch", hide_index=True)

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
### Como interpretar
- **Visão geral** compara a projeção com a linha, o histórico recente e o confronto direto.
- **Destaques** agrupa linhas com maior pontuação na regra de classificação.
- **Por jogo** permite abrir gráficos individuais e testar uma linha alternativa.
- **Edge** é a diferença, em pontos percentuais, entre a probabilidade estimada entre resultados decididos e a probabilidade implícita na cotação.
"""
)
