from __future__ import annotations

from random import Random

import pandas as pd

from nba_props_data import extract_props_from_event

DEMO_SEASONS = ("2025-26", "2024-25")
DEMO_EVENTS = (
    {"id": "demo-bos-lal", "away_team": "Boston Celtics", "home_team": "Los Angeles Lakers"},
    {"id": "demo-den-nyk", "away_team": "Denver Nuggets", "home_team": "New York Knicks"},
)
DEMO_PLAYERS = (
    (1, "Alex Morgan", "BOS", "LAL", 0, (27, 8, 5), (23.5, 6.5, 4.5)),
    (2, "Jordan Ellis", "LAL", "BOS", 0, (18, 6, 9), (21.5, 6.5, 7.5)),
    (3, "Taylor Brooks", "DEN", "NYK", 1, (25, 13, 9), (23.5, 11.5, 8.5)),
    (4, "Casey Reed", "NYK", "DEN", 1, (24, 4, 7), (21.5, 5.5, 5.5)),
)
DEMO_PLAYER_IDS = {name: player_id for player_id, name, *_ in DEMO_PLAYERS}


def demo_player_props() -> dict:
    props = []
    for event_index, event in enumerate(DEMO_EVENTS):
        bookmakers = []
        for name, prices in (("Demo A", (2.0, 1.8)), ("Demo B", (1.78, 2.02))):
            markets = []
            for stat_index, stat in enumerate(("points", "rebounds", "assists")):
                outcomes = [
                    {"description": player, "name": side, "point": lines[stat_index], "price": price}
                    for _, player, _, _, player_event, _, lines in DEMO_PLAYERS
                    if player_event == event_index
                    for side, price in zip(("Over", "Under"), prices)
                ]
                markets.append({"key": f"player_{stat}", "outcomes": outcomes})
            bookmakers.append({"title": name, "markets": markets})
        props.extend(extract_props_from_event(event, {"bookmakers": bookmakers}))
    return {"props": props}


def demo_player_logs() -> dict[tuple[int, str], pd.DataFrame]:
    logs = {}
    for player_id, _, team, opponent, _, means, _ in DEMO_PLAYERS:
        for season in DEMO_SEASONS:
            seed = player_id * 10000 + int(season[:4])
            rng = Random(seed)
            dates = pd.date_range(f"{season[:4]}-10-22", periods=32, freq="2D")
            opponents = ("PHI", "OKC", "MIA", opponent, "GSW")
            rows = []
            for index, date in enumerate(dates):
                matchup = f"{team} {'@' if index % 2 else 'vs.'} {opponents[index % len(opponents)]}"
                rows.append(
                    {
                        "GAME_ID": f"demo-{season}-{player_id}-{index:02d}",
                        "GAME_DATE": date,
                        "MATCHUP": matchup,
                        "PTS": max(0, round(rng.gauss(means[0], 5))),
                        "REB": max(0, round(rng.gauss(means[1], 2))),
                        "AST": max(0, round(rng.gauss(means[2], 2))),
                    }
                )
            logs[player_id, season] = pd.DataFrame(rows)
    return logs
