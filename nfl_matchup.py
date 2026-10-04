"""
NFL Matchup Analyzer
--------------------
Compares an offensive player's last N games against what their upcoming
opponent's defense has allowed to that position over its last N games,
and produces a matchup-adjusted projection (plus over/under lean if you
supply a prop line).

Data: nflverse (free, public) weekly player stats + schedule.

Examples:
    python nfl_matchup.py player "Ja'Marr Chase"
    python nfl_matchup.py player "Bijan Robinson" --stat rushing_yards --line 82.5
    python nfl_matchup.py slate --position WR --stat receiving_yards
    python nfl_matchup.py defense KC
    python nfl_matchup.py week              # live DraftKings lines, all games this week
    python nfl_matchup.py week --live 5     # auto-refresh every 5 minutes + HTML dashboard
"""

import argparse
import difflib
import math
import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
SCHEDULE_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_HOURS = 6

POSITIONS = ["QB", "RB", "WR", "TE"]

STATS_BY_POS = {
    "QB": ["attempts", "completions", "passing_yards", "passing_tds",
           "passing_interceptions", "carries", "rushing_yards", "fantasy_points_ppr"],
    "RB": ["carries", "rushing_yards", "rushing_tds", "targets", "receptions",
           "receiving_yards", "fantasy_points_ppr"],
    "WR": ["targets", "receptions", "receiving_yards", "receiving_tds", "fantasy_points_ppr"],
    "TE": ["targets", "receptions", "receiving_yards", "receiving_tds", "fantasy_points_ppr"],
}
ALL_STATS = sorted({s for stats in STATS_BY_POS.values() for s in stats})

# How hard to trust the defense's sample: factor is pulled toward 1.0
# by PRIOR_GAMES worth of "league average" games.
PRIOR_GAMES = 4


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def _cached_csv(url: str, name: str, refresh: bool) -> pd.DataFrame | None:
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / name
    fresh = path.exists() and (time.time() - path.stat().st_mtime) < CACHE_HOURS * 3600
    if fresh and not refresh:
        return pd.read_csv(path, low_memory=False)
    try:
        df = pd.read_csv(url, low_memory=False)
    except Exception as exc:  # network error / season not published yet
        if path.exists():
            print(f"[warn] download failed ({exc}); using stale cache {name}", file=sys.stderr)
            return pd.read_csv(path, low_memory=False)
        return None
    df.to_csv(path, index=False)
    return df


def current_season(today: date) -> int:
    return today.year if today.month >= 8 else today.year - 1


def load_data(refresh: bool = False):
    season = current_season(date.today())
    frames = []
    for s in (season - 1, season):
        df = _cached_csv(STATS_URL.format(season=s), f"stats_{s}.csv", refresh)
        if df is not None:
            frames.append(df)
    if not frames:
        sys.exit("Could not download any player stats. Check your internet connection.")
    stats = pd.concat(frames, ignore_index=True)
    stats = stats[stats["position"].isin(POSITIONS)].copy()
    stats[ALL_STATS] = stats[ALL_STATS].fillna(0)
    stats["game_order"] = stats["season"] * 100 + stats["week"]

    sched = _cached_csv(SCHEDULE_URL, "schedule.csv", refresh)
    if sched is None:
        sys.exit("Could not download the NFL schedule.")
    sched = sched[sched["season"] >= season - 1].copy()
    return stats, sched, season


# ----------------------------------------------------------------------------
# Opponent lookup
# ----------------------------------------------------------------------------

def next_game(sched: pd.DataFrame, team: str, season: int, week: int | None):
    s = sched[(sched["season"] == season) &
              ((sched["home_team"] == team) | (sched["away_team"] == team))]
    if week is not None:
        s = s[s["week"] == week]
    else:
        s = s[s["result"].isna()].sort_values("gameday")
    if s.empty:
        return None
    g = s.iloc[0]
    home = g["home_team"] == team
    opp = g["away_team"] if home else g["home_team"]
    # Vegas implied team total, if lines are available
    implied = None
    if pd.notna(g.get("total_line")) and pd.notna(g.get("spread_line")):
        # nflverse spread_line is from the home team's perspective (positive = home favored)
        home_total = (g["total_line"] + g["spread_line"]) / 2
        implied = home_total if home else g["total_line"] - home_total
    return {"opp": opp, "week": int(g["week"]), "gameday": g["gameday"],
            "home": home, "implied_total": implied, "total_line": g.get("total_line"),
            "spread": g.get("spread_line")}


def week_matchups(sched: pd.DataFrame, season: int, week: int | None):
    s = sched[sched["season"] == season]
    if week is None:
        pending = s[s["result"].isna()]
        if pending.empty:
            sys.exit("No upcoming games found this season.")
        week = int(pending["week"].min())
    games = s[s["week"] == week]
    out = {}
    for _, g in games.iterrows():
        out[g["home_team"]] = g["away_team"]
        out[g["away_team"]] = g["home_team"]
    return week, out


# ----------------------------------------------------------------------------
# Core analytics
# ----------------------------------------------------------------------------

def player_last_n(stats: pd.DataFrame, player_id: str, n: int) -> pd.DataFrame:
    p = stats[stats["player_id"] == player_id].sort_values("game_order")
    return p.tail(n)


def defense_allowed(stats: pd.DataFrame, n: int) -> pd.DataFrame:
    """Per-game stats allowed by each defense to each position, over each
    defense's last n games. Returns rows indexed by (defense, position)."""
    per_game = (stats.groupby(["opponent_team", "game_id", "game_order", "position"])[ALL_STATS]
                .sum().reset_index())
    # Pick each defense's last n games (by any position appearing)
    games = (per_game[["opponent_team", "game_id", "game_order"]].drop_duplicates()
             .sort_values("game_order"))
    last_games = games.groupby("opponent_team").tail(n)
    per_game = per_game.merge(last_games[["opponent_team", "game_id"]],
                              on=["opponent_team", "game_id"])
    n_games = last_games.groupby("opponent_team").size().rename("games")
    totals = per_game.groupby(["opponent_team", "position"])[ALL_STATS].sum()
    totals = totals.join(n_games, on="opponent_team")
    allowed = totals[ALL_STATS].div(totals["games"], axis=0)
    allowed["games"] = totals["games"]
    allowed.index.names = ["defense", "position"]
    return allowed


def matchup_factors(allowed: pd.DataFrame, defense: str, position: str):
    """Returns {stat: (allowed_per_game, league_avg, shrunk_factor, rank)}.
    rank 1 = stingiest defense, 32 = most generous."""
    pos = allowed.xs(position, level="position")
    if defense not in pos.index:
        return {}
    games = pos.loc[defense, "games"]
    w = games / (games + PRIOR_GAMES)
    out = {}
    for stat in STATS_BY_POS[position]:
        lg = pos[stat].mean()
        val = pos.loc[defense, stat]
        raw = val / lg if lg > 0 else 1.0
        factor = 1 + (raw - 1) * w
        rank = int(pos[stat].rank(method="min").loc[defense])
        out[stat] = (val, lg, factor, rank)
    return out


def prob_over(proj: float, std: float, line: float) -> float:
    std = max(std, 0.35 * max(proj, 1.0), 0.5)  # floor: small samples understate variance
    z = (line - proj) / std
    return 0.5 * (1 - math.erf(z / math.sqrt(2)))


def find_player(stats: pd.DataFrame, name: str):
    names = stats.drop_duplicates("player_id").set_index("player_id")["player_display_name"].dropna()
    exact = names[names.str.lower() == name.lower()]
    if exact.empty:
        exact = names[names.str.lower().str.contains(name.lower(), regex=False)]
    if exact.empty:
        close = difflib.get_close_matches(name, names.tolist(), n=5, cutoff=0.6)
        if not close:
            sys.exit(f"No player found matching '{name}'.")
        exact = names[names.isin(close)]
    if len(exact) > 1:
        # prefer the most recently active player
        recent = (stats[stats["player_id"].isin(exact.index)]
                  .sort_values("game_order").drop_duplicates("player_id", keep="last")
                  .sort_values("game_order", ascending=False))
        if len(recent) > 1:
            others = ", ".join(f"{r.player_display_name} ({r.team})" for r in recent.iloc[1:6].itertuples())
            print(f"[note] multiple matches; using most recent. Others: {others}\n")
        return recent.iloc[0]["player_id"]
    return exact.index[0]


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------

def cmd_player(args, stats, sched, season):
    pid = find_player(stats, args.name)
    games = player_last_n(stats, pid, args.games)
    latest = games.iloc[-1]
    name, pos, team = latest["player_display_name"], latest["position"], latest["team"]

    if args.opponent:
        g = {"opp": args.opponent.upper(), "week": args.week, "gameday": "?", "home": None,
             "implied_total": None, "total_line": None, "spread": None}
    else:
        g = next_game(sched, team, season, args.week)
        if g is None:
            sys.exit(f"No upcoming game found for {team}. Use --opponent to set one.")
    opp = g["opp"]

    allowed = defense_allowed(stats, args.games)
    factors = matchup_factors(allowed, opp, pos)
    if not factors:
        sys.exit(f"No defensive data for {opp}.")
    def_games = int(allowed.loc[(opp, pos), "games"])

    print(f"{name} ({pos}, {team})  vs  {opp}"
          + (f"   Week {g['week']}, {g['gameday']}" if g["week"] else ""))
    if g["implied_total"] is not None:
        print(f"Vegas: total {g['total_line']}, {team} implied team total ~{g['implied_total']:.1f}")
    print(f"Sample: player's last {len(games)} games | {opp} defense's last {def_games} games\n")

    rows = []
    for stat in STATS_BY_POS[pos]:
        vals = games[stat]
        avg, med, std = vals.mean(), vals.median(), vals.std(ddof=0)
        allowed_pg, lg, factor, rank = factors[stat]
        rows.append({
            "stat": stat,
            "L10 avg": round(avg, 1),
            "L10 med": round(med, 1),
            "L10 range": f"{vals.min():g}-{vals.max():g}",
            f"{opp} allows/{pos}": round(allowed_pg, 1),
            "lg avg": round(lg, 1),
            "def rank": f"{rank}/32",
            "matchup": f"{(factor - 1) * 100:+.0f}%",
            "projection": round(avg * factor, 1),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n  def rank: 1 = stingiest vs this position, 32 = most generous."
          "\n  matchup %: how much this defense inflates/deflates the stat vs league average"
          f"\n             (regressed toward average by {PRIOR_GAMES} games of prior).")

    print("\nGame log:")
    log_cols = ["season", "week", "opponent_team"] + STATS_BY_POS[pos]
    print(games[log_cols].rename(columns={"opponent_team": "opp"}).to_string(index=False))

    if args.line is not None:
        stat = args.stat or ("passing_yards" if pos == "QB" else
                             "rushing_yards" if pos == "RB" else "receiving_yards")
        if stat not in games.columns:
            sys.exit(f"Unknown stat '{stat}'.")
        vals = games[stat]
        f = factors.get(stat, (0, 0, 1.0, 0))[2]
        proj = vals.mean() * f
        p = prob_over(proj, vals.std(ddof=0), args.line)
        hits = int((vals > args.line).sum())
        print(f"\nPROP: {stat} {args.line}")
        print(f"  Hit rate (over) last {len(vals)}: {hits}/{len(vals)}")
        print(f"  Matchup projection: {proj:.1f}  (edge {proj - args.line:+.1f})")
        print(f"  Est. P(over) ~{p:.0%}   P(under) ~{1 - p:.0%}")
        implied = 0.5238  # break-even at -110
        if p > implied + 0.03:
            lean = "OVER"
        elif (1 - p) > implied + 0.03:
            lean = "UNDER"
        else:
            lean = "PASS (no edge beyond -110 vig)"
        print(f"  Lean: {lean}")


def cmd_slate(args, stats, sched, season):
    pos = args.position.upper()
    stat = args.stat or "fantasy_points_ppr"
    if stat not in STATS_BY_POS[pos]:
        sys.exit(f"'{stat}' isn't tracked for {pos}. Options: {', '.join(STATS_BY_POS[pos])}")
    week, opp_map = week_matchups(sched, season, args.week)
    allowed = defense_allowed(stats, args.games)

    # Players at this position whose most recent team is playing this week
    recent = stats[stats["position"] == pos].sort_values("game_order")
    last_row = recent.drop_duplicates("player_id", keep="last")
    # only players who appeared in their team's current season
    last_row = last_row[(last_row["season"] == season) & last_row["team"].isin(opp_map)]

    rows = []
    for r in last_row.itertuples():
        g = player_last_n(stats, r.player_id, args.games)
        if len(g) < args.min_games:
            continue
        avg = g[stat].mean()
        opp = opp_map[r.team]
        f = matchup_factors(allowed, opp, pos)
        if stat not in f:
            continue
        _, _, factor, rank = f[stat]
        rows.append({"player": r.player_display_name, "team": r.team, "opp": opp,
                     "games": len(g), "L10 avg": round(avg, 1),
                     "def rank": f"{rank}/32", "matchup": round((factor - 1) * 100),
                     "projection": round(avg * factor, 1)})
    if not rows:
        sys.exit("No qualifying players found.")
    df = pd.DataFrame(rows)
    df = df[df["L10 avg"] >= args.min_avg]
    sort_col = "matchup" if args.sort == "matchup" else "projection"
    df = df.sort_values(sort_col, ascending=False).head(args.top)
    df["matchup"] = df["matchup"].map(lambda x: f"{x:+d}%")
    print(f"Week {week} {pos} slate - {stat} (last {args.games} games, matchup-adjusted)\n")
    print(df.to_string(index=False))


def cmd_defense(args, stats, sched, season):
    team = args.team.upper()
    allowed = defense_allowed(stats, args.games)
    if team not in allowed.index.get_level_values("defense"):
        sys.exit(f"Unknown team '{team}'.")
    print(f"{team} defense - per-game production allowed, last {args.games} games")
    print("(rank 1 = stingiest, 32 = most generous)\n")
    for pos in POSITIONS:
        f = matchup_factors(allowed, team, pos)
        rows = [{"stat": s, "allowed": round(v[0], 1), "lg avg": round(v[1], 1),
                 "rank": f"{v[3]}/32", "matchup": f"{(v[2] - 1) * 100:+.0f}%"}
                for s, v in f.items()]
        print(f"vs {pos}")
        print(pd.DataFrame(rows).to_string(index=False))
        print()


# ----------------------------------------------------------------------------
# Weekly report: every game, every relevant QB/RB/WR/TE, yardage props
# ----------------------------------------------------------------------------

PLAYERS_URL = "https://github.com/nflverse/nflverse-data/releases/download/players/players.csv"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
ESPN_PROPS = ("https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{eid}"
              "/competitions/{eid}/odds/100/propBets?lang=en&region=us&limit=1000&page={page}")

# ESPN prop type ids (provider 100 = DraftKings)
ESPN_MAIN = {"8": "passing_yards", "12": "rushing_yards", "13": "receiving_yards"}
ESPN_MILESTONE = {"194": "passing_yards", "196": "rushing_yards", "195": "receiving_yards"}

ODDS_MARKETS = {"player_pass_yds": "passing_yards",
                "player_rush_yds": "rushing_yards",
                "player_reception_yds": "receiving_yards"}

# Fallback alt-line ladders (as "N+" thresholds) when the book has no milestones
FALLBACK_MILESTONES = {
    "passing_yards": [150, 175, 200, 225, 250, 275, 300, 325],
    "rushing_yards": [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 125],
    "receiving_yards": [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 125],
}

# Which yardage props to evaluate per position, with the min L10 avg to bother
PROPS_BY_POS = {
    "QB": [("passing_yards", 120), ("rushing_yards", 20)],
    "RB": [("rushing_yards", 25), ("receiving_yards", 15)],
    "WR": [("receiving_yards", 25)],
    "TE": [("receiving_yards", 20)],
}
STAT_SHORT = {"passing_yards": "pass yds", "rushing_yards": "rush yds", "receiving_yards": "rec yds"}
BREAK_EVEN = 0.5238  # -110


def _norm(name: str) -> str:
    return "".join(c for c in str(name).lower() if c.isalnum())


def _get_json(url: str):
    import json
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def espn_to_gsis(refresh: bool) -> dict:
    p = _cached_csv(PLAYERS_URL, "players.csv", refresh)
    if p is None:
        return {}
    p = p.dropna(subset=["espn_id", "gsis_id"])
    return {str(int(e)): g for e, g in zip(p["espn_id"], p["gsis_id"])}


def fetch_espn_props(season: int, week: int, refresh: bool, finished: bool = False):
    """Live DraftKings player props via ESPN's public odds feed.
    Returns (main_lines, milestones), keyed by (gsis player_id, stat).
    main_lines value: {"line", "open", "updated"}; milestones value: sorted list of N (meaning N+).
    finished=True also reads completed games (their final pre-game lines) for backtests."""
    import re
    from concurrent.futures import ThreadPoolExecutor

    board = _get_json(f"{ESPN_SCOREBOARD}?seasontype=2&week={week}&dates={season}")
    event_ids = [e["id"] for e in board.get("events", [])
                 if finished or e.get("status", {}).get("type", {}).get("state") == "pre"]
    id_map = espn_to_gsis(refresh)

    def pull(eid):
        items, page, pages = [], 1, 1
        while page <= pages:
            try:
                d = _get_json(ESPN_PROPS.format(eid=eid, page=page))
            except Exception as exc:
                print(f"[warn] props fetch failed for event {eid}: {exc}", file=sys.stderr)
                break
            items += d.get("items", [])
            pages = d.get("pageCount", 1)
            page += 1
        return items

    with ThreadPoolExecutor(max_workers=8) as pool:
        all_items = [it for items in pool.map(pull, event_ids) for it in items]

    main, miles = {}, {}
    for it in all_items:
        tid = it.get("type", {}).get("id")
        if tid not in ESPN_MAIN and tid not in ESPN_MILESTONE:
            continue
        m = re.search(r"/athletes/(\d+)", it.get("athlete", {}).get("$ref", ""))
        pid = id_map.get(m.group(1)) if m else None
        cur = it.get("current", {}).get("target", {}).get("value")
        if pid is None or cur is None:
            continue
        if tid in ESPN_MAIN:
            main[(pid, ESPN_MAIN[tid])] = {
                "line": float(cur),
                "open": it.get("open", {}).get("target", {}).get("value"),
                "updated": it.get("lastUpdated"),
            }
        else:
            miles.setdefault((pid, ESPN_MILESTONE[tid]), set()).add(float(cur))
    return main, {k: sorted(v) for k, v in miles.items()}


def fetch_odds_api_lines(api_key: str) -> dict:
    """Consensus lines from The Odds API (the-odds-api.com). {(normalized name, stat): line}"""
    import statistics
    base = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
    lines: dict = {}
    for ev in _get_json(f"{base}/events?apiKey={api_key}"):
        try:
            data = _get_json(f"{base}/events/{ev['id']}/odds?apiKey={api_key}&regions=us"
                             f"&markets={','.join(ODDS_MARKETS)}&oddsFormat=american")
        except Exception as exc:
            print(f"[warn] odds fetch failed: {exc}", file=sys.stderr)
            continue
        points: dict = {}
        for book in data.get("bookmakers", []):
            for market in book.get("markets", []):
                stat = ODDS_MARKETS.get(market["key"])
                for o in market.get("outcomes", []):
                    if stat and o.get("name") == "Over" and "point" in o:
                        points.setdefault((_norm(o.get("description")), stat), []).append(o["point"])
        for k, pts in points.items():
            lines[k] = statistics.median(pts)
    return lines


def load_lines_csv(path: str) -> dict:
    """CSV with columns: player,stat,line   (stat = passing_yards / rushing_yards / receiving_yards)"""
    df = pd.read_csv(path)
    return {(_norm(r.player), r.stat.strip()): float(r.line) for r in df.itertuples()}


def build_week(args, stats, sched, season):
    """Compute every player/prop row for the target week. Returns (week, games, rows, source)."""
    import os

    s = sched[sched["season"] == season]
    if args.week is None:
        pending = s[s["result"].isna()]
        if pending.empty:
            sys.exit("No upcoming games found this season.")
        week = int(pending["week"].min())
    else:
        week = args.week
    games = s[s["week"] == week]
    if not args.include_played:
        games = games[games["result"].isna()]
    games = games.sort_values(["gameday", "gametime"])
    teams = set(games["home_team"]) | set(games["away_team"])

    # Line sources: CSV > Odds API > ESPN/DraftKings (default, free, live)
    main_by_id, miles_by_id, name_lines = {}, {}, {}
    if args.lines:
        name_lines = load_lines_csv(args.lines)
        source = f"CSV ({args.lines})"
    elif os.environ.get("ODDS_API_KEY"):
        name_lines = fetch_odds_api_lines(os.environ["ODDS_API_KEY"])
        source = f"The Odds API ({len(name_lines)} lines)"
    else:
        main_by_id, miles_by_id = fetch_espn_props(season, week, args.refresh,
                                                   finished=getattr(args, "backtest", False))
        source = f"DraftKings via ESPN ({len(main_by_id)} yardage lines)"

    allowed = defense_allowed(stats, args.games)
    ordered = stats.sort_values("game_order")
    last_row = ordered.drop_duplicates("player_id", keep="last")
    team_override = getattr(args, "team_override", None)
    if team_override:  # backtests: rosters as of that week (known before kickoff)
        last_row = last_row.assign(team=last_row["player_id"].map(team_override).fillna(last_row["team"]))
    # live: only players active this season; week-1 backtests fall back to last season
    this_season = (last_row["season"] == season) | bool(getattr(args, "backtest", False))
    last_row = last_row[this_season & last_row["team"].isin(teams)]
    team_last_game = (ordered[ordered["season"] == season]
                      .drop_duplicates("team", keep="last").set_index("team")["game_id"])

    rows = []
    for g in games.itertuples():
        for team, opp in ((g.away_team, g.home_team), (g.home_team, g.away_team)):
            factors_by_pos = {pos: matchup_factors(allowed, opp, pos) for pos in POSITIONS}
            for pos in POSITIONS:
                players = last_row[(last_row["team"] == team) & (last_row["position"] == pos)]
                for pr in players.itertuples():
                    hist = player_last_n(stats, pr.player_id, args.games)
                    if len(hist) < args.min_games:
                        continue
                    for stat, min_avg in PROPS_BY_POS[pos]:
                        vals = hist[stat]
                        avg = vals.mean()
                        main = main_by_id.get((pr.player_id, stat))
                        if main is None and (pr.player_display_name and
                                             (_norm(pr.player_display_name), stat) in name_lines):
                            main = {"line": name_lines[(_norm(pr.player_display_name), stat)],
                                    "open": None, "updated": None}
                        # Book posts a line -> always include; otherwise volume filter
                        if main is None and (avg < min_avg or
                                             (pos == "QB" and hist["attempts"].tail(3).mean() < 15)):
                            continue
                        _, _, factor, rank = factors_by_pos[pos].get(stat, (0, 0, 1.0, 0))
                        proj = avg * factor
                        std = vals.std(ddof=0)
                        row = {"game": f"{g.away_team}@{g.home_team}", "kickoff": f"{g.gameday} {g.gametime}",
                               "team": team, "opp": opp, "player": pr.player_display_name, "player_id": pr.player_id, "pos": pos,
                               "stat": stat, "n": len(vals), "L10_avg": round(avg, 1),
                               "L10_med": round(vals.median(), 1), "def_rank": rank,
                               "matchup_pct": int(round((factor - 1) * 100)), "proj": round(proj, 1),
                               # only meaningful in-season (a week-1 player's last game was last year)
                               "missed_last_game": pr.season == season and pr.game_id != team_last_game.get(team),
                               "last10": " ".join(f"{v:g}" for v in vals)}
                        if main is not None:
                            line = main["line"]
                            p = prob_over(proj, std, line)
                            hr = (vals > line).mean()
                            row.update({"line": line, "open": main["open"],
                                        "line_hits": f"{int((vals > line).sum())}/{len(vals)}",
                                        "line_hit_rate": round(hr, 2), "p_over": round(p, 2),
                                        "edge": round(proj - line, 1),
                                        "lean": ("OVER" if p > BREAK_EVEN + 0.03 else
                                                 "UNDER" if 1 - p > BREAK_EVEN + 0.03 else "")})
                        # Highest milestone (N+) cleared at >= hit rate; prefer the book's real ladder
                        ladder = miles_by_id.get((pr.player_id, stat)) or FALLBACK_MILESTONES[stat]
                        best = None
                        for m in ladder:
                            if (vals >= m).mean() >= args.hit_rate:
                                best = m
                        if best is not None:
                            row.update({"alt": f"{best:g}+",
                                        "alt_book": (pr.player_id, stat) in miles_by_id,
                                        "alt_hits": f"{int((vals >= best).sum())}/{len(vals)}",
                                        "alt_p": round(prob_over(proj, std, best - 0.5), 2)})
                        # for the staking plan: game-to-game spread and the book's alt-line ladder
                        row.update({"sd": round(float(std), 1),
                                    "ladder": list(miles_by_id.get((pr.player_id, stat), []))})
                        usage_col = USAGE_STAT[stat]
                        usage = hist[usage_col]
                        row.update({"usage_name": usage_col,
                                    "usage_l3": round(usage.tail(3).mean(), 1),
                                    "usage_l10": round(usage.mean(), 1)})
                        lines_expected = bool(main_by_id) and avg >= min_avg * 1.5
                        notes, rating = analyze_row(row, list(vals), lines_expected)
                        row["notes"] = notes
                        row.update(rating)
                        rows.append(row)
    return week, games, rows, source


USAGE_STAT = {"passing_yards": "attempts", "rushing_yards": "carries", "receiving_yards": "targets"}
POS_GROUP = {"passing_yards": "QBs", "rushing_yards": "the run", "receiving_yards": None}


def analyze_row(row: dict, vals: list, lines_expected: bool):
    """Scan a player's last-N games for things that stand out, and rate the prop.
    Returns (notes, rating) where notes = [{"t": text, "k": good|bad|info}]
    (written from the OVER bettor's point of view) and rating holds pick/score/grade."""
    notes = []
    n = len(vals)
    avg = row["L10_avg"]
    line = row.get("line")
    stat_word = {"passing_yards": "pass yds", "rushing_yards": "rush yds",
                 "receiving_yards": "rec yds"}[row["stat"]]
    vs = POS_GROUP[row["stat"]] or (f"{row['pos']}s")

    if row["missed_last_game"] and row.get("line") is None:
        notes.append({"t": "Missed team's last game — check injury status", "k": "warn"})
    if line is None and lines_expected:
        notes.append({"t": "No line posted for a regular contributor — possible injury or role change",
                      "k": "warn"})

    # Streaks against the posted line (most recent games last)
    if line is not None:
        streak_over = 0
        for v in reversed(vals):
            if v > line:
                streak_over += 1
            else:
                break
        streak_under = 0
        for v in reversed(vals):
            if v < line:
                streak_under += 1
            else:
                break
        if streak_over >= 3:
            notes.append({"t": f"Over {line:g} in {streak_over} straight games", "k": "good"})
        elif streak_under >= 3:
            notes.append({"t": f"Under {line:g} in {streak_under} straight games", "k": "bad"})

    # Recent form vs. 10-game baseline
    l3 = sum(vals[-3:]) / min(3, n)
    if avg > 0 and l3 - avg >= max(10, 0.25 * avg):
        notes.append({"t": f"Trending up: {l3:.0f} {stat_word} last 3 vs {avg:.0f} L10 avg", "k": "good"})
    elif avg > 0 and avg - l3 >= max(10, 0.25 * avg):
        notes.append({"t": f"Trending down: {l3:.0f} {stat_word} last 3 vs {avg:.0f} L10 avg", "k": "bad"})

    # Usage (targets / carries / attempts)
    u3, u10, uname = row["usage_l3"], row["usage_l10"], row["usage_name"]
    if u10 >= 2 and u3 >= u10 * 1.3 and u3 - u10 >= 2:
        notes.append({"t": f"Bigger role: {u3:g} {uname}/game last 3 vs {u10:g} L10", "k": "good"})
    elif u10 >= 2 and u3 <= u10 * 0.7 and u10 - u3 >= 2:
        notes.append({"t": f"Shrinking role: {u3:g} {uname}/game last 3 vs {u10:g} L10", "k": "bad"})

    # Floor / ceiling / volatility
    lo, hi = min(vals), max(vals)
    if n >= 6 and line is not None and lo > line:
        notes.append({"t": f"Cleared {line:g} in all of the last {n} games (low {lo:g})", "k": "good"})
    elif n >= 6 and avg >= 20 and lo >= 0.5 * avg:
        notes.append({"t": f"Reliable floor: never under {lo:g} in last {n}", "k": "good"})
    if line is not None:
        big = sum(v >= 1.5 * line for v in vals)
        if big >= 3:
            notes.append({"t": f"Ceiling: {big} games of {1.5 * line:.0f}+ (1.5× the line)", "k": "info"})
    if avg >= 15 and n >= 6:
        mean = sum(vals) / n
        sd = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
        if sd / mean > 0.75:
            notes.append({"t": f"Boom-or-bust: ranged {lo:g} to {hi:g}", "k": "info"})

    # Matchup extremes
    if row["def_rank"] >= 29:
        notes.append({"t": f"Soft matchup: {row['opp']} ranks {row['def_rank']}/32 vs {vs} ({stat_word})",
                      "k": "good"})
    elif 0 < row["def_rank"] <= 4:
        notes.append({"t": f"Tough matchup: {row['opp']} ranks {row['def_rank']}/32 vs {vs} ({stat_word})",
                      "k": "bad"})

    # Market signals
    trap = False
    if line is not None:
        if row.get("open") is not None and not pd.isna(row["open"]) and abs(line - row["open"]) >= 4:
            direction = "up" if line > row["open"] else "down"
            notes.append({"t": f"Line moved {direction} {row['open']:g} → {line:g} since open",
                          "k": "info"})
        if abs(line - avg) / max(avg, 10) > 0.45:
            trap = True

    # Rating (only for posted lines)
    rating = {"pick": "", "score": None, "grade": ""}
    if line is not None:
        p = row["p_over"]
        over = p >= 0.5
        conf = p if over else 1 - p
        side_hits = sum((v > line) if over else (v < line) for v in vals) / n
        recent = vals[-3:]
        trend = sum((v > line) if over else (v < line) for v in recent) / len(recent)
        score = 10 * (0.45 * min(max((conf - 0.5) / 0.35, 0), 1) + 0.35 * side_hits + 0.20 * trend)

        # Trap checks: reasons the attractive-looking side may be a setup
        traps = []
        if trap:
            traps.append(f"Line {line:g} vs {avg:.0f} L10 avg — gap this big usually means the book "
                         "knows about an injury or role change")
            score *= 0.45
        if row["missed_last_game"]:
            traps.append("Missed the team's last game — line may assume limited snaps")
            score *= 0.6
        if over and u10 >= 2 and u3 <= u10 * 0.7 and u10 - u3 >= 2:
            traps.append(f"Model likes the over on old usage, but {uname} fell to {u3:g}/game")
            score *= 0.75
        if not over and u10 >= 2 and u3 >= u10 * 1.3 and u3 - u10 >= 2:
            traps.append(f"Model likes the under on old usage, but {uname} rose to {u3:g}/game")
            score *= 0.75
        opened = row.get("open")
        moved = None if opened is None or pd.isna(opened) else line - opened
        if n < 6:
            score *= 0.8

        # Contrarian angles (proxy: no free public-betting % feed, so use line movement and the
        # well-known public lean toward overs on popular players)
        popular = line >= {"passing_yards": 240, "rushing_yards": 60, "receiving_yards": 60}[row["stat"]]
        contrarian = ""
        if moved is not None and moved >= 3 and not over:
            contrarian = (f"Fade the public: line climbed {opened:g} → {line:g} on over money, "
                          f"but the model projects {row['proj']:g}")
        elif moved is not None and moved <= -3 and not over and popular:
            contrarian = (f"Sharp-side under: line dropped {opened:g} → {line:g} on a popular player "
                          "despite the public's over lean, and the model agrees")
        elif moved is not None and moved <= -3 and over and conf >= 0.6:
            contrarian = (f"Buy low: market dropped this {opened:g} → {line:g}, "
                          f"model still projects {row['proj']:g}")
        elif popular and not over and conf >= 0.58:
            contrarian = "Star under: the public rarely bets against popular players — model sides under"
        if traps or score < 4.5:
            contrarian = ""  # only surface contrarian angles that are playable

        grade = "A" if score >= 7.5 else "B" if score >= 6 else "C" if score >= 4.5 else "D"
        rating = {"pick": f"{'OVER' if over else 'UNDER'} {line:g}", "score": round(score, 1),
                  "grade": grade, "traps": traps, "contrarian": contrarian,
                  # a trap is worth flagging when the bad side still looks tempting on the surface
                  "trap_alert": bool(traps) and max(line, avg) >= 15 and (
                      row["line_hit_rate"] >= 0.7 or row["line_hit_rate"] <= 0.3 or conf >= 0.7)}
    return notes, rating


def print_week(week, games, rows, source, args):
    print(f"WEEK {week} MATCHUP REPORT  |  lookback: last {args.games} games  |  lines: {source}")
    print(f"Updated {time.strftime('%Y-%m-%d %H:%M:%S')}   Hit-rate target: {args.hit_rate:.0%}\n")
    df = pd.DataFrame(rows)
    for g in games.itertuples():
        key = f"{g.away_team}@{g.home_team}"
        header = f"{g.away_team} @ {g.home_team}  -  {g.gameday} {g.gametime}"
        if pd.notna(g.total_line):
            home_tt = (g.total_line + g.spread_line) / 2
            header += (f"   | O/U {g.total_line}, implied: {g.away_team} {g.total_line - home_tt:.1f}"
                       f" / {g.home_team} {home_tt:.1f}")
        print("=" * len(header) + "\n" + header + "\n" + "=" * len(header))
        for team, opp in ((g.away_team, g.home_team), (g.home_team, g.away_team)):
            sub = [r for r in rows if r["game"] == key and r["team"] == team]
            if not sub:
                continue
            print(f"\n  {team} offense vs {opp} defense")
            for r in sub:
                flag = "  [missed last game - check injury]" if r["missed_last_game"] else ""
                print(f"   {r['pos']:<2} {r['player']:<24} {STAT_SHORT[r['stat']]:<9} L10 {r['L10_avg']:>6} "
                      f"| {opp} rank {r['def_rank']:>2}/32 ({r['matchup_pct']:+d}%) "
                      f"| proj {r['proj']:>6}{flag}")
                if "line" in r and pd.notna(r.get("line")):
                    move = (f" (open {r['open']:g})" if r.get("open") not in (None, r["line"])
                            and pd.notna(r.get("open")) else "")
                    star = "  <<< 70%+ HIT RATE" if r["line_hit_rate"] >= args.hit_rate else ""
                    print(f"      LINE {r['line']:g}{move}: hit {r['line_hits']}, P(over) {r['p_over']:.0%}"
                          f" -> {r['lean'] or 'no edge'}{star}")
                if r.get("alt"):
                    src = "DK alt" if r["alt_book"] else "alt"
                    print(f"      {src} {r['alt']}: hit {r['alt_hits']}, model P {r['alt_p']:.0%}")
        print()

    if df.empty:
        return
    print("#" * 70)
    print(f"LINES HIT {args.hit_rate:.0%}+ OF LAST {args.games} GAMES")
    print("#" * 70)
    if "line" in df.columns:
        hot = df[df["line_hit_rate"] >= args.hit_rate].sort_values("p_over", ascending=False)
        print(f"\nPosted sportsbook lines ({len(hot)}):")
        if not hot.empty:
            print(hot[["player", "pos", "team", "opp", "stat", "line", "line_hits", "proj",
                       "p_over", "lean", "missed_last_game"]].to_string(index=False))
        leans = df[df["lean"].isin(["OVER", "UNDER"])].copy()
        leans["conf"] = (leans["p_over"] - 0.5).abs()
        leans = leans.sort_values("conf", ascending=False).head(args.top)
        print(f"\nBiggest model edges vs posted lines (top {len(leans)}):")
        if not leans.empty:
            print(leans[["player", "pos", "team", "opp", "stat", "line", "proj", "edge",
                         "line_hits", "p_over", "lean"]].to_string(index=False))
    if "alt" in df.columns:
        alt = df.dropna(subset=["alt"])
        alt = alt[alt["n"] >= 6].sort_values("alt_p", ascending=False).head(args.top)
        print(f"\nHighest alt-line milestones cleared {args.hit_rate:.0%}+ (top {len(alt)}):")
        print(alt[["player", "pos", "team", "opp", "stat", "alt", "alt_hits", "L10_avg", "proj",
                   "alt_p", "missed_last_game"]].to_string(index=False))


def _now_et() -> str:
    """Timestamp in US Eastern (the published page is built on a UTC server)."""
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).strftime("%a %b %d, %I:%M %p ET")
    except Exception:
        return datetime.now().strftime("%a %b %d, %I:%M %p")


ESPN_LOGO_ABBR = {"WAS": "wsh", "LA": "lar"}

PAGE_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">__META__
<title>Tids Takedowns · Week __WEEK__</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root { --bg:#eef2f6; --card:#fff; --ink:#0c1722; --mute:#5b6875; --line:#d9e1e8; --soft:#edf1f5;
  --blue:#0076b6; --blue-dk:#005a8c; --silver:#b0b7bc;
  --good:#11804a; --good-bg:#e1f3e9; --bad:#c0352b; --bad-bg:#fbe4e1; --warn:#a15c00; --warn-bg:#fff0d4;
  --info:#0067a0; --info-bg:#e2eff8; --con:#6b3fa0; --con-bg:#efe6fa;
  --A:#0076b6; --B:#4f7f9e; --C:#a07a12; --D:#8a929a; --accent:#0076b6; }
@media (prefers-color-scheme: dark) { :root { --bg:#07111b; --card:#0f1c29; --ink:#e7edf2; --mute:#93a3b2;
  --line:#1f3244; --soft:#152536; --blue:#1a8fd6; --blue-dk:#0b5e94; --silver:#8f989f;
  --good:#5fd394; --good-bg:#11301f; --bad:#f2877c; --bad-bg:#3a1b18; --warn:#f1b65a; --warn-bg:#352710;
  --info:#7cc0ef; --info-bg:#122a40; --con:#c9a8f5; --con-bg:#2b2040;
  --A:#1a8fd6; --B:#6f9bb8; --C:#e0b84a; --D:#7d868e; --accent:#5fb4ec; } }
* { box-sizing:border-box }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 Inter,system-ui,-apple-system,"Segoe UI",sans-serif }
.cond { font-family:"Barlow Condensed",Inter,system-ui,sans-serif }
.wrap { max-width:1240px; margin:0 auto; padding:0 16px }
.hero { background:linear-gradient(135deg,var(--blue) 0%,var(--blue-dk) 100%); color:#fff;
  border-bottom:4px solid var(--silver) }
.hero .wrap { padding-top:22px; padding-bottom:18px; position:relative; z-index:1 }
.hero { position:relative; overflow:hidden; background-size:200% 200%; animation:heroShift 14s ease-in-out infinite }
@keyframes heroShift { 0%,100% { background-position:0% 50% } 50% { background-position:100% 50% } }
.hero::before { content:""; position:absolute; inset:-40% -10%; pointer-events:none; opacity:.22;
  background:repeating-linear-gradient(100deg, transparent 0 46px, rgba(255,255,255,.55) 46px 49px, transparent 49px 120px);
  animation:streak 2.6s linear infinite }
@keyframes streak { from { transform:translateX(-120px) } to { transform:translateX(0) } }
.hero .twosix { position:absolute; right:-10px; bottom:-34px; font:800 230px/1 "Barlow Condensed",sans-serif;
  color:transparent; -webkit-text-stroke:3px rgba(255,255,255,.35); transform:skewX(-12deg); letter-spacing:-.04em;
  text-shadow:0 0 40px rgba(176,183,188,.35); animation:glow 3.2s ease-in-out infinite; pointer-events:none; z-index:0 }
@keyframes glow { 0%,100% { -webkit-text-stroke-color:rgba(255,255,255,.28) } 50% { -webkit-text-stroke-color:rgba(255,255,255,.6) } }
@media (max-width:600px) { .hero .twosix { font-size:150px; bottom:-20px } }
.live { display:inline-flex; align-items:center; gap:6px; font-weight:700; letter-spacing:.06em; color:#fff }
.live::before { content:""; width:8px; height:8px; border-radius:50%; background:#5dff9d; box-shadow:0 0 0 0 rgba(93,255,157,.7);
  animation:pulse 1.6s infinite }
@keyframes pulse { 70% { box-shadow:0 0 0 9px rgba(93,255,157,0) } 100% { box-shadow:0 0 0 0 rgba(93,255,157,0) } }
.ticker { background:#04121d; color:#e8f3fb; border-bottom:2px solid var(--silver); overflow:hidden; white-space:nowrap;
  font:600 13px/34px Inter,sans-serif; display:flex }
.ticker .tlabel { flex:none; position:relative; z-index:1; padding:0 18px 0 14px; background:var(--blue); color:#fff;
  font:800 16px/34px "Barlow Condensed",sans-serif; letter-spacing:.08em; text-transform:uppercase;
  clip-path:polygon(0 0,100% 0,calc(100% - 10px) 100%,0 100%); box-shadow:6px 0 14px rgba(0,0,0,.45) }
.ticker .tscroll { flex:1; overflow:hidden; min-width:0 }
.ticker .tk { display:inline-block; padding-left:100%; animation:tick 70s linear infinite }
@media (hover: hover) and (pointer: fine) { .ticker:hover .tk { animation-play-state:paused } }
@keyframes tick { to { transform:translateX(-100%) } }
.ticker .it { margin-right:34px } .ticker .gr { display:inline-block; min-width:18px; text-align:center; border-radius:4px;
  background:var(--blue); color:#fff; font-weight:800; margin-right:6px; padding:0 4px }
.ticker .OVER { color:#7dffb1 } .ticker .UNDER { color:#ffa59b }
.card, .tp, .pl-row, .panel { transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease }
.card:hover, .tp:hover, .pl-row:hover { transform:translateY(-3px);
  box-shadow:0 10px 28px -12px color-mix(in srgb,var(--blue) 55%,transparent); border-color:var(--blue) }
.tp { overflow:hidden }
.tp::after { content:""; position:absolute; top:0; left:-60%; width:40%; height:100%; pointer-events:none;
  background:linear-gradient(100deg,transparent,rgba(255,255,255,.35),transparent); transform:skewX(-20deg) }
.tp:hover::after { animation:shine .8s ease }
@keyframes shine { to { left:130% } }
.tp .rank { background:linear-gradient(180deg,var(--blue),var(--silver)); -webkit-background-clip:text; background-clip:text;
  color:transparent; opacity:.9 }
.gA { box-shadow:0 0 0 0 color-mix(in srgb,var(--A) 60%,transparent); animation:aglow 2.4s infinite }
@keyframes aglow { 0%,100% { box-shadow:0 0 0 0 color-mix(in srgb,var(--A) 0%,transparent) }
  50% { box-shadow:0 0 12px 2px color-mix(in srgb,var(--A) 55%,transparent) } }
.gibbs { position:relative; overflow:hidden; border-radius:16px; color:#fff; padding:18px 18px 16px;
  background:linear-gradient(120deg,#021019 0%,#06324f 45%,var(--blue) 100%); border:2px solid var(--silver);
  box-shadow:0 18px 40px -22px rgba(0,118,182,.9) }
.gibbs::before { content:""; position:absolute; inset:-30% -10%; opacity:.18; pointer-events:none;
  background:repeating-linear-gradient(100deg,transparent 0 30px,#fff 30px 32px,transparent 32px 90px); animation:streak 1.4s linear infinite }
.gibbs .num { position:absolute; right:14px; top:-18px; font:800 170px/1 "Barlow Condensed",sans-serif; color:transparent;
  -webkit-text-stroke:2px rgba(255,255,255,.45); transform:skewX(-12deg); pointer-events:none }
.gibbs .in { position:relative; z-index:1 }
.gibbs h3 { margin:0; font:800 34px/1 "Barlow Condensed",sans-serif; letter-spacing:.04em; text-transform:uppercase }
.gibbs .kick { opacity:.8; font-size:12.5px; margin-top:4px }
.gstats { display:flex; flex-wrap:wrap; gap:10px; margin:14px 0 }
.gstat { background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.22); border-radius:12px; padding:8px 14px; min-width:96px }
.gstat .v { font:800 28px/1 "Barlow Condensed",sans-serif } .gstat .l { font-size:11.5px; opacity:.8; text-transform:uppercase; letter-spacing:.05em }
.gprops { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:10px }
.gprop { background:rgba(0,0,0,.28); border:1px solid rgba(255,255,255,.18); border-radius:12px; padding:10px 12px }
.gprop .pk { font:800 24px/1.1 "Barlow Condensed",sans-serif; letter-spacing:.02em }
.gprop .OVER { color:#7dffb1 } .gprop .UNDER { color:#ffa59b }
.gprop .m { font-size:12px; opacity:.85 }
.glog { display:flex; align-items:flex-end; gap:6px; height:92px; margin-top:12px }
.glog .b { flex:1; display:flex; flex-direction:column; justify-content:flex-end; align-items:center; gap:2px; font-size:10.5px; opacity:.95 }
.glog .bar { width:100%; border-radius:4px 4px 0 0; background:linear-gradient(180deg,#9fd8ff,#0076b6);
  transform-origin:bottom; animation:grow .9s cubic-bezier(.2,.8,.2,1) both }
.glog .bar.rec { background:linear-gradient(180deg,#e6ecef,#8f989f) }
@keyframes grow { from { transform:scaleY(0) } }
.glog .td { color:#ffd166; font-weight:800 }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation:none !important; transition:none !important }
  .ticker .tscroll { overflow-x:auto; -webkit-overflow-scrolling:touch } .ticker .tk { padding-left:12px } }
.brand { display:flex; align-items:center; gap:14px }
.mark { width:52px; height:52px; border-radius:50%; border:3px solid var(--silver); display:grid; place-items:center;
  font:800 22px/1 "Barlow Condensed",sans-serif; letter-spacing:.02em; background:rgba(255,255,255,.08); flex:none }
h1 { margin:0; font:800 38px/1 "Barlow Condensed",sans-serif; letter-spacing:.03em; text-transform:uppercase }
.tagline { opacity:.88; margin-top:4px }
.hero .sub { color:rgba(255,255,255,.78); font-size:12.5px; margin-top:12px }
.hero .stat { background:rgba(255,255,255,.1); border-color:rgba(255,255,255,.22); color:#fff }
.hero .stat .l { color:rgba(255,255,255,.75) }
.hero .stat .v.pos { color:#8ff0b6 } .hero .stat .v.neg { color:#ffb3aa }
h2 { font:700 24px/1.1 "Barlow Condensed",sans-serif; letter-spacing:.03em; text-transform:uppercase; margin:30px 0 12px;
  display:flex; align-items:center; gap:10px }
h2::before { content:""; width:6px; height:22px; background:var(--blue); border-radius:2px }
.lead { color:var(--mute); margin:-6px 0 12px; font-size:13px }
.sub { color:var(--mute) }
.legend { display:flex; gap:12px; flex-wrap:wrap; margin-top:10px; color:var(--mute); font-size:12px; align-items:center }
.g { display:inline-grid; place-items:center; width:26px; height:26px; border-radius:7px; color:#fff; font-weight:700; font-size:13px; flex:none }
.gA { background:var(--A) } .gB { background:var(--B) } .gC { background:var(--C) } .gD { background:var(--D) }
.games { display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); gap:14px }
@media (max-width:420px) { .games { grid-template-columns:1fr } }
.card { background:var(--card); border:1px solid var(--line); border-radius:14px; display:flex; flex-direction:column; overflow:hidden }
.ch { padding:14px 14px 10px; border-bottom:1px solid var(--line) }
.teams { display:flex; align-items:center; gap:8px; font-weight:700; font-size:17px }
.teams img { width:30px; height:30px; object-fit:contain }
.at { color:var(--mute); font-weight:400; font-size:14px }
.kick { margin-left:auto; color:var(--mute); font-size:12px; font-weight:500; text-align:right }
.vegas { color:var(--mute); font-size:12px; margin-top:6px }
.plays { padding:6px 14px 4px; flex:1 }
.play { display:flex; gap:10px; padding:9px 0; border-bottom:1px solid var(--soft) }
.play:last-child { border-bottom:0 }
.pl { min-width:0; flex:1 }
.pname { font-weight:600 }
.pmeta { color:var(--mute); font-size:12px }
.pick { font-weight:700; white-space:nowrap }
.OVER { color:var(--good) } .UNDER { color:var(--bad) }
.chips { display:flex; flex-wrap:wrap; gap:4px; margin-top:4px }
.chip { font-size:11.5px; padding:2px 7px; border-radius:99px; line-height:1.35 }
.k-good { background:var(--good-bg); color:var(--good) } .k-bad { background:var(--bad-bg); color:var(--bad) }
.k-warn { background:var(--warn-bg); color:var(--warn) } .k-info { background:var(--info-bg); color:var(--info) }
.cf { padding:8px 14px 12px }
.strip { display:flex; gap:10px; flex-wrap:wrap; margin-top:14px }
.stat { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:10px 14px; min-width:120px }
.stat .v { font-size:20px; font-weight:700; font-variant-numeric:tabular-nums }
.stat .l { color:var(--mute); font-size:12px }
.pos { color:var(--good) } .neg { color:var(--bad) }
.res { font-weight:700; font-size:12px; padding:2px 8px; border-radius:99px }
.res-win { background:var(--good-bg); color:var(--good) } .res-loss { background:var(--bad-bg); color:var(--bad) }
.res-pending { background:var(--info-bg); color:var(--info) } .res-void, .res-push { background:var(--soft); color:var(--mute) }
.tag { font-size:10.5px; font-weight:700; letter-spacing:.04em; padding:1px 6px; border-radius:5px; margin-left:4px; vertical-align:1px }
.tag-trap { background:var(--warn-bg); color:var(--warn) } .tag-con { background:var(--con-bg); color:var(--con) }
.k-con { background:var(--con-bg); color:var(--con) }
.duo { display:grid; grid-template-columns:repeat(auto-fit,minmax(380px,1fr)); gap:14px }
@media (max-width:420px) { .duo { grid-template-columns:1fr } }
.panel { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:4px 14px 8px }
.panel h3 { font-size:15px; margin:12px 0 2px }
.panel .why { color:var(--mute); font-size:12px; margin-bottom:6px }
.item { padding:9px 0; border-bottom:1px solid var(--soft) }
.item:last-child { border-bottom:0 }
.reason { font-size:12.5px; margin-top:3px }
.cf button { background:none; border:1px solid var(--line); color:var(--accent); border-radius:8px; padding:6px 10px; cursor:pointer; font:inherit; font-size:13px }
.empty { color:var(--mute); padding:10px 0 }
.bar { display:flex; gap:8px; flex-wrap:wrap; margin:6px 0 10px; align-items:center }
.bar input[type=search], .bar select { padding:7px 10px; border:1px solid var(--line); border-radius:8px; background:var(--card); color:var(--ink); font:inherit }
.bar label { display:flex; align-items:center; gap:6px; color:var(--mute) }
.tw { background:var(--card); border:1px solid var(--line); border-radius:14px; overflow-x:auto }
table { border-collapse:collapse; width:100%; min-width:980px }
th, td { padding:9px 10px; text-align:left; border-bottom:1px solid var(--line); vertical-align:top }
th { color:var(--mute); font-weight:600; font-size:12px; cursor:pointer; white-space:nowrap; user-select:none }
td.n { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap }
tr:last-child td { border-bottom:0 }
.spark { display:block }
footer { color:var(--mute); font-size:12px; padding:24px 0 40px }
.top5 { display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:12px }
.tp { background:var(--card); border:1px solid var(--line); border-top:4px solid var(--blue); border-radius:12px;
  padding:12px 14px; position:relative }
.tp .rank { position:absolute; top:6px; right:12px; font:800 34px/1 "Barlow Condensed",sans-serif; color:var(--silver); opacity:.6 }
.tp .pick { font:700 22px/1.1 "Barlow Condensed",sans-serif; letter-spacing:.02em; margin:6px 0 2px }
.tp .who { display:flex; align-items:center; gap:8px }
.card.lions { border:2px solid var(--blue); box-shadow:0 0 0 3px color-mix(in srgb,var(--blue) 18%,transparent) }
.lionsbadge { display:inline-block; background:var(--blue); color:#fff; font:700 11px/1 Inter,sans-serif; letter-spacing:.05em;
  padding:4px 7px; border-radius:5px; margin-bottom:6px; text-transform:uppercase }
.ladder { display:grid; gap:10px }
.pl-row { background:var(--card); border:1px solid var(--line); border-radius:12px; display:grid;
  grid-template-columns:96px 1fr 120px; gap:12px; padding:12px 14px; align-items:start }
.pl-n { font:800 26px/1 "Barlow Condensed",sans-serif; color:var(--blue) }
.pl-n small { display:block; font:600 11px/1.4 Inter,sans-serif; color:var(--mute); letter-spacing:.04em; text-transform:uppercase }
.pl-legs div { padding:2px 0 }
.pl-legs .new { font-weight:700 }
.pl-legs .new::after { content:"NEW"; font:700 9.5px/1 Inter,sans-serif; background:var(--blue); color:#fff; padding:2px 5px;
  border-radius:4px; margin-left:6px; vertical-align:2px }
.pl-pay { text-align:right }
.pl-pay .odds { font:800 24px/1 "Barlow Condensed",sans-serif }
@media (max-width:560px) { .pl-row { grid-template-columns:1fr; } .pl-pay { text-align:left } }
</style></head><body>
<header class="hero"><div class="twosix" aria-hidden="true">__JERSEY__</div><div class="wrap">
  <div class="brand"><div class="mark">TT</div>
    <div><h1>Tids Takedowns</h1><div class="tagline">NFL Week __WEEK__ · player prop matchups, ratings &amp; picks ·
      <a href="reports/index.html" style="color:#fff;font-weight:600">Weekly report cards →</a></div></div></div>
  <div class="strip" id="bankstrip"></div>
  <div class="sub"><span class="live">LIVE</span> · Updated __UPDATED__ · Lines: __SOURCE__ · Each player's last __N__ games vs. the opponent defense's last __N__</div>
</div></header>
<div class="ticker" aria-label="Tids Ticker: top rated plays"><span class="tlabel">Tids Ticker</span><div class="tscroll"><div class="tk" id="ticker"></div></div></div>
<div class="wrap">
  <div class="legend"><span><span class="g gA">A</span> strong</span><span><span class="g gB">B</span> good</span>
  <span><span class="g gC">C</span> lean</span><span><span class="g gD">D</span> pass</span>
  <span>Rating = model projection vs line + hit rate + last 3 games, penalized for injury/role red flags.</span></div>

<h2>Tids' Top 5</h2>
<div class="lead">The highest-rated plays on the board right now — no traps, one prop per player.
  <a href="top5/" style="font-weight:700">Share the Top 5 →</a></div>
<section class="top5" id="top5"></section>

<h2>Gibbs Watch</h2>
<section id="gibbs"></section>

<h2>This week's games</h2>
<section class="games" id="games"></section>

<h2>Parlay builder</h2>
<div class="lead">Rebuilt every refresh from A/B-rated, trap-free props. Each rung adds the next-best leg.
Payouts assume -110 per leg; hit chance discounts the model's confidence by about half, because models run hot.</div>
<div class="bar"><label><input type="checkbox" id="onepergame"> One leg per game</label></div>
<section class="ladder" id="ladder"></section>

<h2>Trap alerts &amp; contrarian plays</h2>
<div class="duo">
  <section class="panel"><h3>⚠ Trap alerts</h3>
    <div class="why">Props that look tempting on the surface (high hit rate or big model edge) but carry a red flag.</div>
    <div id="traps"></div></section>
  <section class="panel"><h3>↔ Contrarian plays</h3>
    <div class="why">Going against public money. Free public-betting % isn't available, so this reads line movement
    and the public's well-known lean toward overs on popular players.</div>
    <div id="contra"></div></section>
</div>

<h2 id="bank">Paper bankroll — Claude's picks</h2>
<div class="sub" style="margin-bottom:10px">Fake money, real tracking. Started with $1,000. Each Thursday and Monday night game gets $200 spread
across up to 5 different players. No player gets more than 30% of it. The strongest overs are split between the posted line,
a safer alt line and a plus-money alt line, and a small parlay rides on the top picks. Bets are graded against final box scores;
a player who doesn't play voids the leg. Main lines assume -110. Alt-line odds are estimates, marked "est".</div>
<div class="panel" id="bankpanel"></div>

<h2 id="all">All player props</h2>
<div class="bar">
  <input type="search" id="q" placeholder="Search player or team">
  <select id="game"><option value="">All games</option></select>
  <select id="stat"><option value="">All stats</option><option value="passing_yards">Passing yds</option>
    <option value="rushing_yards">Rushing yds</option><option value="receiving_yards">Receiving yds</option></select>
  <label><input type="checkbox" id="good"> A/B ratings only</label>
  <label><input type="checkbox" id="trapf"> Traps</label>
  <label><input type="checkbox" id="conf"> Contrarian</label>
  <label><input type="checkbox" id="hot"> Hit line __HITPCT__+ of last __N__</label>
</div>
<div class="tw"><table><thead><tr>
  <th data-s="score">Rating</th><th data-s="player">Player / what stands out</th><th data-s="pick">Pick</th>
  <th data-s="proj">Proj</th><th data-s="line_hit_rate">Hit L__N__</th><th data-s="p_over">P(over)</th>
  <th data-s="def_rank">Opp rank</th><th>Last __N__ (bar = line)</th><th data-s="alt">Alt __HITPCT__+</th>
</tr></thead><tbody id="rows"></tbody></table></div>

<footer>Opp rank: 1 = stingiest vs that position, 32 = most generous. Proj = player's L__N__ average × how much this defense
allows vs league average (regressed toward average). P(over) is a model estimate, not a guarantee. Chip colors are relative to the pick
(green helps it, red hurts it, amber = red flag). Lines move — confirm at your sportsbook. Built from free nflverse stats and
DraftKings lines via ESPN.<br><br>Tids Takedowns is a fan project, not affiliated with the NFL or the Detroit Lions. For entertainment and research only — not betting advice. 21+. Gambling problem? Call 1-800-GAMBLER.</footer>
</div>
<script>
const ROWS = __ROWS__;
const GAMES = __GAMES__;
const HIT = __HIT__;
const STAT = {passing_yards:"pass yds", rushing_yards:"rush yds", receiving_yards:"rec yds"};
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const pct = v => v == null ? "" : Math.round(v * 100) + "%";

function kind(note, pick) {
  if (note.k === "good" || note.k === "bad") {
    const under = (pick || "").startsWith("UNDER");
    return under ? (note.k === "good" ? "bad" : "good") : note.k;
  }
  return note.k;
}
function chips(r, max) {
  const order = {warn:0, good:1, bad:2, info:3};
  const ns = (r.notes || []).map(n => ({t:n.t, k:kind(n, r.pick)})).sort((a, b) => order[a.k] - order[b.k]);
  const extra = (r.traps || []).map(t => ({t:"Trap: " + t, k:"warn"}));
  if (r.contrarian) extra.push({t:r.contrarian, k:"con"});
  const all = extra.concat(ns);
  return `<div class="chips">${all.slice(0, max ?? 99).map(n => `<span class="chip k-${n.k}">${esc(n.t)}</span>`).join("")}</div>`;
}
function tags(r) {
  return (r.traps && r.traps.length ? `<span class="tag tag-trap">TRAP</span>` : "")
    + (r.contrarian ? `<span class="tag tag-con">CONTRARIAN</span>` : "");
}
function grade(r) { return r.grade ? `<span class="g g${r.grade}" title="score ${r.score}/10">${r.grade}</span>` : `<span class="g gD" style="opacity:.35">–</span>`; }
function logo(t) { return `<img src="https://a.espncdn.com/i/teamlogos/nfl/500/${(GAMES.logo[t] || t).toLowerCase()}.png" alt="" onerror="this.style.display='none'">`; }
function kick(g) {
  const d = new Date(g.gameday + "T12:00:00");
  const [h, m] = g.gametime.split(":").map(Number);
  const day = d.toLocaleDateString("en-US", {weekday:"short", month:"short", day:"numeric"});
  return `${day}<br>${((h + 11) % 12) + 1}:${String(m).padStart(2, "0")} ${h < 12 ? "AM" : "PM"} ET`;
}
function spark(r) {
  const vals = String(r.last10 || "").split(" ").map(Number).filter(v => !isNaN(v));
  if (!vals.length) return "";
  const W = 120, H = 30, bw = W / 10, max = Math.max(...vals, r.line || 0, 1);
  const bars = vals.map((v, i) => {
    const h = Math.max(1.5, (Math.max(v, 0) / max) * (H - 2));
    const c = r.line == null ? "var(--D)" : v > r.line ? "var(--good)" : "var(--bad)";
    return `<rect x="${i * bw + 1}" y="${H - h}" width="${bw - 2}" height="${h}" rx="1.5" fill="${c}" opacity=".85"><title>${v}</title></rect>`;
  }).join("");
  const ly = r.line == null ? "" : `<line x1="0" x2="${W}" y1="${H - (r.line / max) * (H - 2)}" y2="${H - (r.line / max) * (H - 2)}" stroke="var(--ink)" stroke-dasharray="3 2" stroke-width="1"/>`;
  return `<svg class="spark" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">${bars}${ly}</svg>`;
}

function renderGames() {
  document.getElementById("games").innerHTML = GAMES.list.map(g => {
    const rs = ROWS.filter(r => r.game === g.key && r.score != null && !(r.traps && r.traps.length))
      .sort((a, b) => b.score - a.score);
    const top = rs.slice(0, 4);
    const plays = top.length ? top.map(r => `<div class="play">${grade(r)}<div class="pl">
        <div><span class="pname">${esc(r.player)}</span>${tags(r)} <span class="pmeta">${r.pos} · ${r.team}</span></div>
        <div><span class="pick ${r.pick.split(" ")[0]}">${r.pick}</span> <span class="pmeta">${STAT[r.stat]} · proj ${r.proj} · hit ${r.line_hits}</span></div>
        ${chips(r, 2)}</div></div>`).join("") : `<div class="empty">No player lines posted yet.</div>`;
    const lions = g.away === "DET" || g.home === "DET";
    return `<article class="card${lions ? " lions" : ""}"><div class="ch">${lions ? '<span class="lionsbadge">Lions game</span>' : ""}
        <div class="teams">${logo(g.away)}${g.away} <span class="at">@</span> ${logo(g.home)}${g.home}<span class="kick">${kick(g)}</span></div>
        <div class="vegas">${esc(g.vegas)}</div></div>
      <div class="plays">${plays}</div>
      <div class="cf"><button data-game="${g.key}">All ${ROWS.filter(r => r.game === g.key).length} props in this game →</button></div></article>`;
  }).join("");
  document.querySelectorAll(".cf button").forEach(b => b.onclick = () => {
    document.getElementById("game").value = b.dataset.game; renderRows();
    document.getElementById("all").scrollIntoView({behavior:"smooth"});
  });
}

let sortKey = "score", sortDir = -1;
function renderRows() {
  const q = document.getElementById("q").value.toLowerCase(), gm = document.getElementById("game").value;
  const st = document.getElementById("stat").value, good = document.getElementById("good").checked;
  const hot = document.getElementById("hot").checked;
  const trapf = document.getElementById("trapf").checked, conf = document.getElementById("conf").checked;
  const rs = ROWS.filter(r => (!q || (r.player + " " + r.team + " " + r.opp).toLowerCase().includes(q))
    && (!gm || r.game === gm) && (!st || r.stat === st) && (!good || r.grade === "A" || r.grade === "B")
    && (!hot || (r.line_hit_rate != null && r.line_hit_rate >= HIT))
    && (!trapf || (r.traps && r.traps.length)) && (!conf || r.contrarian));
  rs.sort((a, b) => {
    const x = a[sortKey], y = b[sortKey];
    if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1;
    return (typeof x === "string" ? x.localeCompare(y) : x - y) * sortDir;
  });
  document.getElementById("rows").innerHTML = rs.map(r => {
    const move = r.open != null && r.line != null && r.open !== r.line ? `<div class="pmeta">open ${r.open}</div>` : "";
    return `<tr><td>${grade(r)}</td>
      <td><span class="pname">${esc(r.player)}</span>${tags(r)} <span class="pmeta">${r.pos} · ${r.team} vs ${r.opp} · ${STAT[r.stat]} · L${r.n} avg ${r.L10_avg}</span>${chips(r)}</td>
      <td class="n">${r.pick ? `<span class="pick ${r.pick.split(" ")[0]}">${r.pick}</span>${move}` : `<span class="pmeta">no line</span>`}</td>
      <td class="n"><b>${r.proj}</b></td><td class="n">${r.line_hits || ""}</td><td class="n">${pct(r.p_over)}</td>
      <td class="n">${r.def_rank}/32</td><td>${spark(r)}</td>
      <td class="n">${r.alt ? `${r.alt}<div class="pmeta">${r.alt_hits} · ${pct(r.alt_p)}</div>` : ""}</td></tr>`;
  }).join("") || `<tr><td colspan="9" class="empty">No matches.</td></tr>`;
}

const sel = document.getElementById("game");
GAMES.list.forEach(g => sel.insertAdjacentHTML("beforeend", `<option value="${g.key}">${g.away} @ ${g.home}</option>`));
["q","game","stat","good","hot","trapf","conf"].forEach(id => document.getElementById(id).addEventListener("input", renderRows));
document.querySelectorAll("th[data-s]").forEach(th => th.onclick = () => {
  const k = th.dataset.s; sortDir = k === sortKey ? -sortDir : (k === "player" || k === "def_rank" ? 1 : -1); sortKey = k; renderRows();
});
function renderPanels() {
  const tempting = r => r.line_hit_rate >= 0.5 ? `OVER ${r.line}` : `UNDER ${r.line}`;
  const tr = ROWS.filter(r => r.trap_alert)
    .sort((a, b) => Math.abs(b.line_hit_rate - 0.5) - Math.abs(a.line_hit_rate - 0.5)).slice(0, 12);
  document.getElementById("traps").innerHTML = tr.map(r => `<div class="item">
      <span class="pname">${esc(r.player)}</span> <span class="pmeta">${r.pos} · ${r.team} vs ${r.opp} · ${STAT[r.stat]}</span>
      <div class="reason">Looks like: <b>${tempting(r)}</b> <span class="pmeta">(hit ${r.line_hits}, model ${pct(Math.max(r.p_over, 1 - r.p_over))})</span></div>
      ${r.traps.map(t => `<div class="reason" style="color:var(--warn)">⚠ ${esc(t)}</div>`).join("")}</div>`).join("")
    || `<div class="empty">No trap alerts right now.</div>`;
  const cr = ROWS.filter(r => r.contrarian).sort((a, b) => b.score - a.score).slice(0, 12);
  document.getElementById("contra").innerHTML = cr.map(r => `<div class="item">${grade(r)}
      <span class="pname" style="margin-left:6px">${esc(r.player)}</span> <span class="pmeta">${r.pos} · ${r.team} vs ${r.opp}</span>
      <div class="reason"><span class="pick ${r.pick.split(" ")[0]}">${r.pick}</span> <span class="pmeta">${STAT[r.stat]} · proj ${r.proj} · hit ${r.line_hits}</span></div>
      <div class="reason" style="color:var(--con)">${esc(r.contrarian)}</div>
      ${(r.traps || []).map(t => `<div class="reason" style="color:var(--warn)">⚠ ${esc(t)}</div>`).join("")}</div>`).join("")
    || `<div class="empty">No contrarian spots right now.</div>`;
}
const BANK = __BANK__;
const money = v => (v < 0 ? "-$" : "$") + Math.abs(v).toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 2});
function renderBank() {
  const s = BANK.summary; if (!s) return;
  const pl = s.profit, roi = s.roi == null ? "—" : (s.roi * 100).toFixed(1) + "%";
  // win % of settled bets (voids excluded); green once it beats the -110 break-even of 52.4%
  const winPct = s.wins + s.losses ? s.wins / (s.wins + s.losses) : null;
  const legPct = s.legs_won + s.legs_lost ? s.legs_won / (s.legs_won + s.legs_lost) : null;
  document.getElementById("bankstrip").innerHTML = `
    <a class="stat" href="#bank" style="text-decoration:none;color:inherit"><div class="v">${money(s.balance)}</div><div class="l">Paper bankroll</div></a>
    <div class="stat"><div class="v ${pl > 0 ? "pos" : pl < 0 ? "neg" : ""}">${pl >= 0 ? "+" : ""}${money(pl)}</div><div class="l">Profit / loss</div></div>
    <div class="stat"><div class="v ${winPct == null ? "" : winPct >= 0.524 ? "pos" : "neg"}">${winPct == null ? "—" : (winPct * 100).toFixed(1) + "%"}</div>
      <div class="l">Win % · legs ${s.legs_won}-${s.legs_lost}${legPct == null ? "" : " (" + Math.round(legPct * 100) + "%)"}</div></div>
    <div class="stat"><div class="v">${s.wins}-${s.losses}${s.voids ? "-" + s.voids : ""}</div><div class="l">Record · ROI ${roi}</div></div>
    <div class="stat"><div class="v">${money(s.at_risk)}</div><div class="l">Open bets</div></div>`;
  let chart = "";
  if (s.history.length) {
    const pts = [s.start].concat(s.history.map(h => h.balance)), W = 600, H = 120;
    const lo = Math.min(...pts), hi = Math.max(...pts), span = Math.max(hi - lo, 1);
    const xy = pts.map((v, i) => `${(i / Math.max(pts.length - 1, 1)) * (W - 10) + 5},${H - 8 - ((v - lo) / span) * (H - 16)}`).join(" ");
    const y0 = H - 8 - ((s.start - lo) / span) * (H - 16);
    chart = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:${W}px;height:auto;margin:10px 0">
      <line x1="0" x2="${W}" y1="${y0}" y2="${y0}" stroke="var(--line)" stroke-dasharray="4 3"/>
      <polyline points="${xy}" fill="none" stroke="var(--accent)" stroke-width="2.5" stroke-linejoin="round"/></svg>`;
  }
  const bets = BANK.bets || [];
  const toWin = b => b.stake * (b.odds > 0 ? b.odds / 100 : 100 / -b.odds);
  const rows = bets.map(b => `<tr><td>W${b.week}<div class="pmeta">${b.game}</div></td>
      <td>${b.legs.map(l => `<div><b>${esc(l.player)}</b> ${l.side.toUpperCase()} ${l.line} <span class="pmeta">${STAT[l.stat]}${l.actual != null ? " · actual " + l.actual : ""}</span>
        <span class="res res-${l.result}">${l.result}</span></div>`).join("")}<div class="pmeta">${esc(b.note || "")}</div></td>
      <td>${b.kind}</td><td class="n">${money(b.stake)}</td><td class="n">${b.odds > 0 ? "+" : ""}${b.odds}</td>
      <td><span class="res res-${b.result}">${b.result}</span></td>
      <td class="n ${b.profit > 0 ? "pos" : b.profit < 0 ? "neg" : ""}">${b.result === "pending" ? "to win " + money(toWin(b)) : (b.profit >= 0 ? "+" : "") + money(b.profit)}</td></tr>`).join("");
  document.getElementById("bankpanel").innerHTML = chart + (bets.length
    ? `<div style="overflow-x:auto"><table style="min-width:760px"><thead><tr><th>Week</th><th>Bet</th><th>Type</th><th>Stake</th><th>Odds</th><th>Result</th><th>P/L</th></tr></thead><tbody>${rows}</tbody></table></div>`
    : `<div class="empty">No bets yet. First picks go in before Thursday Night Football; Monday nights too.</div>`);
}
function bestLegs() {
  const seen = new Set();
  return ROWS.filter(r => (r.grade === "A" || r.grade === "B") && !(r.traps && r.traps.length))
    .sort((a, b) => b.score - a.score)
    .filter(r => !seen.has(r.player) && seen.add(r.player));
}
function goodNote(r) {
  const n = (r.notes || []).map(n => ({t: n.t, k: kind(n, r.pick)})).find(n => n.k === "good");
  return n ? `<div class="chips"><span class="chip k-good">${esc(n.t)}</span></div>` : "";
}
function renderTop5() {
  const top = bestLegs().slice(0, 5);
  const gm = Object.fromEntries(GAMES.list.map(g => [g.key, g]));
  document.getElementById("top5").innerHTML = top.map((r, i) => `<div class="tp"><div class="rank">${i + 1}</div>
      <div class="who">${grade(r)}<div><div class="pname">${esc(r.player)}</div>
      <div class="pmeta">${r.pos} · ${r.team} vs ${r.opp}${gm[r.game] ? " · " + kick(gm[r.game]).replace("<br>", " ") : ""}</div></div></div>
      <div class="pick ${r.pick.split(" ")[0]}">${r.pick} <span class="pmeta" style="font:500 13px Inter,sans-serif">${STAT[r.stat]}</span></div>
      <div class="pmeta">proj ${r.proj} · hit ${r.line_hits} · score ${r.score}/10</div>${goodNote(r)}</div>`).join("")
    || `<div class="empty">No A/B-rated plays posted yet — check back when lines open.</div>`;
}
function renderParlays() {
  const one = document.getElementById("onepergame").checked;
  const per = {}, legs = [];
  for (const r of bestLegs()) {
    if ((per[r.game] || 0) >= (one ? 1 : 2)) continue;
    per[r.game] = (per[r.game] || 0) + 1; legs.push(r);
    if (legs.length === 7) break;
  }
  const leg = r => { const c = Math.max(r.p_over, 1 - r.p_over); return 0.5 + (c - 0.5) * 0.45; };
  const amer = d => d >= 2 ? "+" + Math.round((d - 1) * 100) : String(Math.round(-100 / (d - 1)));
  let out = "";
  for (let n = 2; n <= legs.length; n++) {
    const L = legs.slice(0, n), dec = Math.pow(1 + 100 / 110, n), p = L.reduce((a, r) => a * leg(r), 1);
    out += `<div class="pl-row"><div class="pl-n">${n}-LEG<small>parlay</small></div>
      <div class="pl-legs">${L.map((r, i) => `<div class="${i === n - 1 ? "new" : ""}"><span class="pick ${r.pick.split(" ")[0]}">${r.pick}</span>
        ${esc(r.player)} <span class="pmeta">${STAT[r.stat]} · ${r.team} vs ${r.opp} · ${r.grade}</span></div>`).join("")}</div>
      <div class="pl-pay"><div class="odds">${amer(dec)}</div><div class="pmeta">$10 pays $${(10 * dec).toFixed(2)}</div>
        <div class="pmeta">~${Math.round(p * 100)}% est. hit</div></div></div>`;
  }
  document.getElementById("ladder").innerHTML = out || `<div class="empty">Not enough A/B-rated, trap-free legs yet.</div>`;
}
document.getElementById("onepergame").addEventListener("input", renderParlays);
const FEATURE = __FEATURE__;
function renderTicker() {
  const items = bestLegs().slice(0, 14).map(r => `<span class="it"><span class="gr">${r.grade}</span>${esc(r.player)}
    <span class="${r.pick.split(" ")[0]}">${r.pick}</span> ${STAT[r.stat]} · proj ${r.proj} · ${r.team} vs ${r.opp}</span>`);
  document.getElementById("ticker").innerHTML = items.length ? items.join("") + items.join("")
    : `<span class="it">Lines open soon — check back for this week's top plays</span>`;
}
function renderGibbs() {
  const f = FEATURE, el = document.getElementById("gibbs");
  if (!f || !f.name) { el.innerHTML = ""; return; }
  const g = f.props.length ? GAMES.list.find(x => x.key === f.props[0].game) : null;
  const props = f.props.length ? f.props.map(r => `<div class="gprop">
      <div class="m">${STAT[r.stat].toUpperCase()} · vs ${r.opp}</div>
      ${r.pick ? `<div class="pk"><span class="${r.pick.split(" ")[0]}">${r.pick}</span> ${grade(r)}</div>
        <div class="m">proj ${r.proj} · hit ${r.line_hits} · L10 avg ${r.L10_avg}${r.open != null && r.open !== r.line ? ` · opened ${r.open}` : ""}</div>`
        : `<div class="pk">No line yet</div><div class="m">proj ${r.proj} · L10 avg ${r.L10_avg}</div>`}
      ${chips(r, 2)}</div>`).join("")
    : `<div class="gprop"><div class="pk">Lions on bye / no lines yet</div><div class="m">Check back when this week's props post.</div></div>`;
  const mx = Math.max(1, ...f.log.map(x => x.rush + Math.max(x.rec, 0)));
  const log = f.log.map((x, i) => `<div class="b" title="${x.season} wk ${x.week} vs ${x.opp}: ${x.rush} rush, ${x.rec} rec, ${x.td} TD">
      ${x.td ? `<span class="td">${x.td}TD</span>` : ""}
      <div class="bar rec" style="height:${Math.max(x.rec, 0) / mx * 64}px;animation-delay:${i * 60}ms"></div>
      <div class="bar" style="height:${Math.max(x.rush, 0) / mx * 64}px;animation-delay:${i * 60}ms;border-radius:0"></div>
      <span>${x.opp}</span></div>`).join("");
  const stat = (v, l) => `<div class="gstat"><div class="v" data-count="${v}">${v}</div><div class="l">${l}</div></div>`;
  el.innerHTML = `<div class="gibbs"><div class="num" aria-hidden="true">${esc(f.jersey)}</div><div class="in">
    <h3>${esc(f.name)}</h3>
    <div class="kick">${f.team}${f.jersey ? " · #" + esc(f.jersey) : ""}${g ? " · " + kick(g).replace("<br>", " ") + " · " + g.away + " @ " + g.home : ""}</div>
    <div class="gstats">${stat(f.rush_yds, `${f.season} rush yds`)}${stat(f.rush_td, "rush TD")}${stat(f.rec, "catches")}
      ${stat(f.rec_yds, "rec yds")}${stat(f.rec_td, "rec TD")}${stat(f.rush_yds + f.rec_yds, "scrimmage yds")}</div>
    <div class="gprops">${props}</div>
    <div class="glog">${log}</div>
    <div class="m" style="font-size:11.5px;opacity:.75;margin-top:4px">Last ${f.log.length} games · blue = rushing, silver = receiving · gold = touchdowns</div>
  </div></div>`;
}
function countUp() {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  document.querySelectorAll("[data-count]").forEach(el => {
    const end = Number(el.dataset.count); if (!isFinite(end) || end <= 0) return;
    const t0 = performance.now(), dur = 1100;
    const step = t => { const k = Math.min(1, (t - t0) / dur); el.textContent = Math.round(end * (1 - Math.pow(1 - k, 3)));
      if (k < 1) requestAnimationFrame(step); };
    requestAnimationFrame(step);
  });
}
renderBank(); renderTicker(); renderTop5(); renderGibbs(); renderGames(); renderParlays(); renderPanels(); renderRows(); countUp();
</script></body></html>"""


FEATURE_PLAYER = "Jahmyr Gibbs"  # the site's featured player (Lions); jersey comes from roster data


def feature_player(stats: pd.DataFrame, rows: list, season: int, name: str = FEATURE_PLAYER) -> dict:
    """Season totals, recent game log and this week's rated props for the featured player."""
    p = stats[stats["player_display_name"] == name].sort_values("game_order")
    if p.empty:
        return {}
    cur = p[p["season"] == season].fillna(0)
    tot = lambda c: int(cur[c].sum()) if c in cur else 0
    log = [{"season": int(r.season), "week": int(r.week), "opp": r.opponent_team,
            "rush": int(r.rushing_yards), "rec": int(r.receiving_yards),
            "td": int((r.rushing_tds or 0) + (r.receiving_tds or 0))}
           for r in p.tail(10).fillna(0).itertuples()]
    keys = ("stat", "pick", "grade", "score", "proj", "line", "open", "line_hits", "p_over", "opp", "game",
            "kickoff", "notes", "traps", "L10_avg", "contrarian")
    props = [{k: r.get(k) for k in keys} for r in rows if r["player"] == name]
    jersey = ""
    roster = _cached_csv(PLAYERS_URL, "players.csv", False)
    if roster is not None:
        j = roster.loc[roster["gsis_id"] == p.iloc[-1]["player_id"], "jersey_number"].dropna()
        jersey = str(int(j.iloc[0])) if len(j) else ""
    return {"name": name, "team": p.iloc[-1]["team"], "jersey": jersey, "season": season, "games": len(cur),
            "carries": tot("carries"), "rush_yds": tot("rushing_yards"), "rush_td": tot("rushing_tds"),
            "rec": tot("receptions"), "rec_yds": tot("receiving_yards"), "rec_td": tot("receiving_tds"),
            "log": log, "props": props}


def write_html(week, games, rows, source, args, path: Path, refresh_secs: int | None, bank=None, feature=None):
    import html
    import json

    def clean(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        if hasattr(v, "item"):  # numpy scalar
            return v.item()
        return v

    data = [{k: clean(v) for k, v in r.items()} for r in rows]

    def clean_tree(o):
        if isinstance(o, dict):
            return {k: clean_tree(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean_tree(v) for v in o]
        return clean(o)
    game_list = []
    for g in games.itertuples():
        vegas = "No game line yet"
        if pd.notna(g.total_line) and pd.notna(g.spread_line):
            home_tt = (g.total_line + g.spread_line) / 2
            fav, pts = (g.home_team, g.spread_line) if g.spread_line > 0 else (g.away_team, -g.spread_line)
            spread = f"{fav} -{pts:g}" if pts else "Pick'em"
            vegas = (f"{spread} · O/U {g.total_line:g} · Implied: {g.away_team} "
                     f"{g.total_line - home_tt:.1f}, {g.home_team} {home_tt:.1f}")
        game_list.append({"key": f"{g.away_team}@{g.home_team}", "away": g.away_team, "home": g.home_team,
                          "gameday": str(g.gameday), "gametime": str(g.gametime), "vegas": vegas})
    games_json = {"list": game_list, "logo": ESPN_LOGO_ABBR}

    meta = f'\n<meta http-equiv="refresh" content="{refresh_secs}">' if refresh_secs else ""
    page = (PAGE_TEMPLATE
            .replace("__META__", meta)
            .replace("__WEEK__", str(week))
            .replace("__UPDATED__", html.escape(_now_et()))
            .replace("__SOURCE__", html.escape(source))
            .replace("__N__", str(args.games))
            .replace("__HITPCT__", f"{args.hit_rate:.0%}")
            .replace("__HIT__", str(args.hit_rate))
            .replace("__GAMES__", json.dumps(games_json))
            .replace("__BANK__", json.dumps(bank or {}, default=str).replace("</", "<\\/"))
            .replace("__JERSEY__", html.escape(str((feature or {}).get("jersey", ""))))
            .replace("__FEATURE__", json.dumps(clean_tree(feature or {}), default=str).replace("</", "<\\/"))
            .replace("__ROWS__", json.dumps(data, default=str).replace("</", "<\\/")))
    path.write_text(page, encoding="utf-8")


# ----------------------------------------------------------------------------
# Paper bankroll: ledger, auto-grading, and Claude's automatic Thursday & Monday night bets
# ----------------------------------------------------------------------------

LEDGER = Path(__file__).parent / "bets.json"
DEFAULT_ODDS = -110  # ESPN's feed has lines but no prices; assume standard juice


def load_ledger() -> dict:
    import json
    if LEDGER.exists():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {"start_bankroll": 1000, "bets": []}


def save_ledger(ledger: dict):
    import json
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")


def american_to_decimal(odds: int) -> float:
    return 1 + (odds / 100 if odds > 0 else 100 / -odds)


def decimal_to_american(dec: float) -> int:
    return int(round((dec - 1) * 100)) if dec >= 2 else int(round(-100 / (dec - 1)))


def grade_ledger(ledger: dict, stats: pd.DataFrame, sched: pd.DataFrame) -> list:
    """Grade every bet against final box scores. Returns bets with result/profit filled in.
    A leg whose player didn't play is void (as sportsbooks do); a parlay pays on its live legs."""
    finals = sched[sched["result"].notna()]
    graded = []
    for bet in ledger["bets"]:
        legs, dec, states = [], 1.0, []
        for leg in bet["legs"]:
            g = finals[(finals["season"] == bet["season"]) & (finals["week"] == bet["week"]) &
                       ((finals["home_team"] == leg["team"]) | (finals["away_team"] == leg["team"]))]
            leg = dict(leg)
            if g.empty:
                leg["result"], leg["actual"] = "pending", None
            else:
                st = stats[(stats["player_id"] == leg["player_id"]) & (stats["season"] == bet["season"]) &
                           (stats["week"] == bet["week"])]
                if st.empty:
                    leg["result"], leg["actual"] = "void", None
                else:
                    actual = float(st.iloc[0][leg["stat"]])
                    leg["actual"] = actual
                    if actual == leg["line"]:
                        leg["result"] = "push"
                    else:
                        won = actual > leg["line"] if leg["side"] == "over" else actual < leg["line"]
                        leg["result"] = "win" if won else "loss"
            if leg["result"] == "win":
                dec *= american_to_decimal(leg.get("odds", DEFAULT_ODDS))
            states.append(leg["result"])
            legs.append(leg)
        b = dict(bet, legs=legs)
        if "loss" in states:
            b["result"], b["profit"] = "loss", -bet["stake"]
        elif "pending" in states:
            b["result"], b["profit"] = "pending", 0.0
        elif "win" in states:
            b["result"], b["profit"] = "win", round(bet["stake"] * (dec - 1), 2)
        else:  # all legs void/push
            b["result"], b["profit"] = "void", 0.0
        graded.append(b)
    return graded


def bankroll_summary(ledger: dict, graded: list) -> dict:
    graded = [b for b in graded if b["kind"] != "none"]
    settled = [b for b in graded if b["result"] in ("win", "loss", "void")]
    pending = [b for b in graded if b["result"] == "pending"]
    profit = sum(b["profit"] for b in settled)
    risked = sum(b["stake"] for b in settled if b["result"] != "void")
    history, bal = [], ledger["start_bankroll"]
    for b in sorted(settled, key=lambda x: x["placed_at"]):
        bal += b["profit"]
        history.append({"id": b["id"], "balance": round(bal, 2)})
    return {
        "start": ledger["start_bankroll"],
        "balance": round(ledger["start_bankroll"] + profit, 2),
        "at_risk": round(sum(b["stake"] for b in pending), 2),
        "available": round(ledger["start_bankroll"] + profit - sum(b["stake"] for b in pending), 2),
        "wins": sum(b["result"] == "win" for b in settled),
        "losses": sum(b["result"] == "loss" for b in settled),
        "voids": sum(b["result"] == "void" for b in settled),
        "profit": round(profit, 2),
        "roi": round(profit / risked, 4) if risked else None,
        "legs_won": sum(l["result"] == "win" for b in graded for l in b["legs"]),
        "legs_lost": sum(l["result"] == "loss" for b in graded for l in b["legs"]),
        "history": history,
    }


def _bet_odds(legs) -> int:
    dec = 1.0
    for leg in legs:
        dec *= american_to_decimal(leg.get("odds", DEFAULT_ODDS))
    return decimal_to_american(dec)


def _leg_from_row(r: dict, line: float | None = None, odds: int = DEFAULT_ODDS, alt: str = "") -> dict:
    side, main = r["pick"].split()
    leg = {"player": r["player"], "player_id": r["player_id"], "team": r["team"], "opp": r["opp"],
           "stat": r["stat"], "side": side.lower(), "line": float(main) if line is None else line,
           "odds": odds, "grade": r["grade"], "score": r["score"]}
    if alt:
        leg["alt"] = alt  # e.g. "40+" — odds are an estimate (the free feed has no alt prices)
    return leg


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def estimate_alt_odds(main_line: float, alt_threshold: float, sd: float) -> int:
    """Estimate a sportsbook's price for 'alt_threshold+' from its main line.
    Treat the main line as the market's median outcome, spread by the player's game-to-game
    standard deviation, then add a normal book margin (-110 on a 50/50)."""
    sd = max(sd, 0.35 * max(main_line, 1.0), 2.0)
    p_market = 1 - _norm_cdf((alt_threshold - 0.5 - main_line) / sd)
    p_book = min(max(p_market * 1.048, 0.03), 0.95)
    american = decimal_to_american(1 / p_book)
    return int(round(american / 5) * 5)


def _conf(r: dict) -> float:
    return max(r["p_over"], 1 - r["p_over"])


def choose_bets(rows: list, budget: float, max_players: int = 5, max_share: float = 0.30,
                parlay_share: float = 0.15) -> list:
    """Claude's staking plan v2 — spread the money so no single pick controls the game:
    - Up to `max_players` DIFFERENT players, one prop each: A/B first, then C-rated props the model
      still leans >=55% on. Trap-flagged props never qualify.
    - Straight-bet money is weighted by rating score; no player gets more than `max_share` of the
      budget (leftover stays in the bankroll rather than piling onto one pick).
    - A/B overs are split across the main line (50%), a safer lower alt line (30%) and a
      plus-money higher alt line (20%) from DraftKings' milestone ladder. Unders stay on the line.
    - `parlay_share` of the budget goes to a parlay of the top 2-3 players."""
    pool = sorted((r for r in rows if r.get("pick") and not r.get("traps")
                   and (r["grade"] in ("A", "B") or (r["grade"] == "C" and _conf(r) >= 0.55))),
                  key=lambda r: r["score"], reverse=True)
    picks, seen = [], set()
    for r in pool:
        if r["player_id"] not in seen:
            seen.add(r["player_id"])
            picks.append(r)
        if len(picks) == max_players:
            break
    if not picks:
        return []

    parlay_stake = round(budget * parlay_share / 5) * 5 if len(picks) >= 2 else 0
    straight = budget - parlay_stake
    cap = budget * max_share
    alloc = {r["player_id"]: 0.0 for r in picks}
    # water-fill: weight by score, cap each player, hand the excess to the uncapped ones
    open_ids = set(alloc)
    remaining = straight
    while remaining > 1 and open_ids:
        tot = sum(r["score"] for r in picks if r["player_id"] in open_ids)
        spill = 0.0
        for r in picks:
            if r["player_id"] in open_ids:
                want = alloc[r["player_id"]] + remaining * r["score"] / tot
                alloc[r["player_id"]] = min(want, cap)
                spill += want - alloc[r["player_id"]]
                if alloc[r["player_id"]] >= cap:
                    open_ids.discard(r["player_id"])
        remaining = spill

    rnd = lambda x: int(round(x / 5) * 5)
    out = []
    for r in picks:
        amt = alloc[r["player_id"]]
        side, line = r["pick"].split()
        line = float(line)
        ladder = sorted(set(r.get("ladder") or []) | set(FALLBACK_MILESTONES[r["stat"]]))
        lower = [m for m in ladder if m - 0.5 <= line * 0.9 and m >= 1]
        upper = [m for m in ladder if m - 0.5 >= line * 1.15]
        if side == "OVER" and r["grade"] in ("A", "B") and lower and upper and rnd(amt * 0.2) >= 5:
            safe, boom = max(lower), min(upper)
            parts = [(rnd(amt * 0.5), None, DEFAULT_ODDS, ""),
                     (rnd(amt * 0.3), safe - 0.5, estimate_alt_odds(line, safe, r.get("sd", 0)), f"{safe:g}+"),
                     (rnd(amt * 0.2), boom - 0.5, estimate_alt_odds(line, boom, r.get("sd", 0)), f"{boom:g}+")]
        else:
            parts = [(rnd(amt), None, DEFAULT_ODDS, "")]
        for stake, ln, odds, alt in parts:
            if stake >= 5:
                out.append({"kind": "straight", "stake": stake, "legs": [_leg_from_row(r, ln, odds, alt)]})
    if parlay_stake:
        out.append({"kind": "parlay", "stake": parlay_stake,
                    "legs": [_leg_from_row(r) for r in picks[:3 if len(picks) >= 4 else 2]]})
    return out


def choose_bets_v1(rows: list, budget: float) -> list:
    """Original plan (kept for backtest comparison). Only A/B-rated props with no trap flags qualify;
    best 4 at most. ~80% of the budget to straights weighted by rating, ~20% to a 2-leg parlay."""
    picks = [r for r in rows if r.get("grade") in ("A", "B") and not r.get("traps")]
    picks = sorted(picks, key=lambda r: r["score"], reverse=True)[:4]
    if not picks:
        return []
    out = []
    parlay_stake = round(budget * 0.2 / 5) * 5 if len(picks) >= 2 else 0
    straight_budget = budget - parlay_stake if len(picks) >= 2 else budget / 2
    total_score = sum(r["score"] for r in picks)
    for r in picks:
        stake = max(10, round(straight_budget * r["score"] / total_score / 5) * 5)
        leg = _leg_from_row(r)
        out.append({"kind": "straight", "stake": stake, "legs": [leg]})
    if parlay_stake:
        legs = [_leg_from_row(r) for r in picks[:2]]
        out.append({"kind": "parlay", "stake": parlay_stake, "legs": legs})
    return out


def cmd_autobet(args, stats, sched, season):
    """Place Claude's paper bets on today's games in a window before kickoff. Idempotent:
    does nothing outside the window or if this game already has bets."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("America/New_York"))
    days = [d.strip().lower() for d in (args.day or "").split(",") if d.strip()]
    if days and now.strftime("%A").lower() not in days and not args.force:
        print(f"autobet: today is {now:%A}, only betting on {args.day}")
        return
    s = sched[(sched["season"] == season) & sched["result"].isna()]
    today = s[s["gameday"] == now.strftime("%Y-%m-%d")]
    if today.empty:
        print("autobet: no games today")
        return
    ledger = load_ledger()
    for g in today.itertuples():
        key = f"{g.away_team}@{g.home_team}"
        kick = datetime.strptime(f"{g.gameday} {g.gametime}", "%Y-%m-%d %H:%M").replace(
            tzinfo=ZoneInfo("America/New_York"))
        mins = (kick - now).total_seconds() / 60
        if not args.force and not (15 <= mins <= args.window * 60):
            print(f"autobet: {key} kicks off in {mins:.0f} min; outside betting window")
            continue
        if any(b["game"] == key and b["season"] == season and b["week"] == g.week for b in ledger["bets"]):
            print(f"autobet: already bet {key}")
            continue
        wargs = argparse.Namespace(week=int(g.week), include_played=False, lines=None, refresh=True,
                                   games=args.games, min_games=4, hit_rate=0.70)
        _, _, rows, _ = build_week(wargs, stats, sched, season)
        rows = [r for r in rows if r["game"] == key]
        summary = bankroll_summary(ledger, grade_ledger(ledger, stats, sched))
        budget = min(args.budget, max(summary["available"], 0))
        picks = choose_bets(rows, budget)
        if not picks:
            ledger["bets"].append({"id": f"{season}-w{g.week}-{key}-nobet", "placed_at": now.isoformat(),
                                   "season": season, "week": int(g.week), "game": key, "kind": "none",
                                   "stake": 0, "legs": [], "note": "No qualifying trap-free props — passed."})
            print(f"autobet: {key} — nothing met the bar, passing")
        for i, p in enumerate(picks, 1):
            p.update({"id": f"{season}-w{g.week}-{key}-{i}", "placed_at": now.isoformat(), "season": season,
                      "week": int(g.week), "game": key, "odds": _bet_odds(p["legs"]),
                      "note": "Claude auto-pick"})
            ledger["bets"].append(p)
            desc = " + ".join(f"{l['player']} {l['side'].upper()} {l['line']:g} {l['stat']}" for l in p["legs"])
            print(f"autobet: ${p['stake']} {p['kind']} ({p['odds']:+d}): {desc}")
        save_ledger(ledger)


def cmd_bet(args, stats, sched, season):
    """Manually record a paper bet. --leg 'Player Name|stat|over|40.5' (repeat for a parlay)."""
    from datetime import datetime
    ledger = load_ledger()
    legs = []
    for spec in args.leg:
        name, stat, side, line = [x.strip() for x in spec.split("|")]
        pid = find_player(stats, name)
        latest = stats[stats["player_id"] == pid].sort_values("game_order").iloc[-1]
        legs.append({"player": latest["player_display_name"], "player_id": pid, "team": latest["team"],
                     "stat": stat, "side": side.lower(), "line": float(line), "odds": args.odds})
    g = next_game(sched, legs[0]["team"], season, args.week)
    if g is None:
        sys.exit("No upcoming game found for that player.")
    home = legs[0]["team"] if g["home"] else g["opp"]
    away = g["opp"] if g["home"] else legs[0]["team"]
    for leg in legs:
        leg["opp"] = g["opp"] if leg["team"] == legs[0]["team"] else legs[0]["team"]
    bet = {"id": f"{season}-w{g['week']}-manual-{len(ledger['bets']) + 1}",
           "placed_at": datetime.now().astimezone().isoformat(), "season": season, "week": g["week"],
           "game": f"{away}@{home}", "kind": "parlay" if len(legs) > 1 else "straight",
           "stake": args.stake, "legs": legs, "odds": _bet_odds(legs), "note": args.note or "manual"}
    ledger["bets"].append(bet)
    save_ledger(ledger)
    print(f"Recorded ${args.stake} {bet['kind']} at {bet['odds']:+d}")


def cmd_bankroll(args, stats, sched, season):
    ledger = load_ledger()
    graded = grade_ledger(ledger, stats, sched)
    s = bankroll_summary(ledger, graded)
    roi = f"{s['roi']:.1%}" if s["roi"] is not None else "n/a"
    print(f"Balance ${s['balance']:,.2f} (start ${s['start']:,})  |  record {s['wins']}-{s['losses']}"
          f"{'-' + str(s['voids']) + ' void' if s['voids'] else ''}  |  P/L ${s['profit']:+,.2f}  |  ROI {roi}"
          f"  |  at risk ${s['at_risk']:,.2f}")
    for b in graded:
        if b["kind"] == "none":
            continue
        legs = " + ".join(f"{l['player']} {l['side']} {l['line']:g} ({l.get('actual', '-')}: {l['result']})"
                          for l in b["legs"])
        print(f"  W{b['week']} {b['game']:<9} ${b['stake']:>5} {b['kind']:<8} {b['odds']:+5d}  "
              f"{b['result']:<7} {b['profit']:+8.2f}  {legs}")


# ----------------------------------------------------------------------------
# Weekly report cards: snapshot ratings before kickoff, grade them after
# ----------------------------------------------------------------------------

SNAPSHOT_FIELDS = ["game", "kickoff", "team", "opp", "player", "player_id", "pos", "stat", "line",
                   "pick", "score", "grade", "p_over", "proj", "line_hits", "traps"]
TOP_NS = list(range(10, 1, -1))  # 10, 9, ..., 2


def _et_now():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York"))


def _kickoff_et(kickoff: str):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.strptime(kickoff, "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo("America/New_York"))


def update_snapshot(snap_dir: str, season: int, week: int, rows: list) -> Path:
    """Record each rated prop's latest pre-kickoff rating. Props freeze once their game starts,
    so the stored version is (close to) the closing line we rated."""
    import json
    path = Path(snap_dir) / f"{season}-w{week:02d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    snap = (json.loads(path.read_text(encoding="utf-8")) if path.exists()
            else {"season": season, "week": week, "props": {}})
    now = _et_now()
    for r in rows:
        if not r.get("pick") or _kickoff_et(r["kickoff"]) <= now:
            continue
        rec = {k: r.get(k) for k in SNAPSHOT_FIELDS}
        rec["captured_at"] = now.isoformat(timespec="minutes")
        snap["props"][f"{r['player_id']}|{r['stat']}"] = rec
    path.write_text(json.dumps(snap, indent=1, default=str) + "\n", encoding="utf-8")
    return path


def grade_snapshot(snap: dict, stats: pd.DataFrame, sched: pd.DataFrame) -> list:
    season, week = snap["season"], snap["week"]
    wk = sched[(sched["season"] == season) & (sched["week"] == week)]
    final_teams = set(wk[wk["result"].notna()]["home_team"]) | set(wk[wk["result"].notna()]["away_team"])
    st = stats[(stats["season"] == season) & (stats["week"] == week)].set_index("player_id")
    out = []
    for p in snap["props"].values():
        p = dict(p)
        side, line = p["pick"].split()
        line = float(line)
        if p["team"] not in final_teams:
            p["result"], p["actual"] = "pending", None
        elif p["player_id"] not in st.index:
            p["result"], p["actual"] = "void", None
        else:
            actual = float(st.loc[p["player_id"]][p["stat"]])
            p["actual"] = actual
            if actual == line:
                p["result"] = "push"
            else:
                p["result"] = "win" if (actual > line) == (side == "OVER") else "loss"
        out.append(p)
    return out


def summarize_week(graded: list) -> dict:
    """Per game: hit % of the top-N rated props (N = 2..10). Overall: same, pooled across games.
    Ranking = our rating score, trap-flagged props excluded (we never recommended those)."""
    games: dict = {}
    for p in graded:
        if not p.get("traps"):
            games.setdefault(p["game"], []).append(p)
    # each tier = [hit, missed, void/push, picked]; hit % = hit / (hit + missed)
    per_game, overall = {}, {n: [0, 0, 0, 0] for n in TOP_NS}
    for g, props in games.items():
        props.sort(key=lambda p: p["score"] or 0, reverse=True)
        top = props[:10]
        tiers = {}
        for n in TOP_NS:
            picked = top[:n]  # a game with fewer than n rated props contributes what it has
            t = [sum(p["result"] == "win" for p in picked), sum(p["result"] == "loss" for p in picked),
                 sum(p["result"] in ("void", "push") for p in picked), len(picked)]
            if len(top) >= n:
                tiers[n] = t
            for i in range(4):
                overall[n][i] += t[i]
        per_game[g] = {"tiers": tiers, "props": top,
                       "kickoff": top[0]["kickoff"] if top else ""}
    by_grade = {}
    for gr in "ABCD":
        d = [p for p in graded if p["grade"] == gr and not p.get("traps") and p["result"] in ("win", "loss")]
        w = sum(p["result"] == "win" for p in d)
        by_grade[gr] = [w, len(d) - w, 0, len(d)]
    pending = sum(p["result"] == "pending" for p in graded)
    return {"per_game": dict(sorted(per_game.items(), key=lambda kv: kv[1]["kickoff"])),
            "overall": overall, "by_grade": by_grade, "pending": pending, "games": len(per_game)}


def _pct(hit, missed, *_):
    return f"{hit / (hit + missed):.0%}" if hit + missed else "—"


def _tier_text(t) -> str:
    """'4 of 5 hit (80%)' — voids/pushes noted, not counted."""
    hit, missed, vp, _ = t
    extra = f", {vp} void/push" if vp else ""
    return f"{hit} of {hit + missed} hit ({_pct(*t)}){extra}"


def report_markdown(season, week, summary, bank_line=None, label="") -> str:
    L = [f"# Tids Takedowns — Week {week} Report Card{label} ({season})", ""]
    if summary["pending"]:
        L += [f"> {summary['pending']} props still pending (game or stats not final yet).", ""]
    if bank_line:
        L += [f"**Paper bankroll:** {bank_line}", ""]
    L += [f"## Whole week — top N props from each of {summary['games']} games", "",
          "| Top N per game | Props picked | Hit | Missed | Void/push | Hit % |", "|---|---|---|---|---|---|"]
    for n in TOP_NS:
        hit, missed, vp, picked = summary["overall"][n]
        L.append(f"| Top {n} | {picked} | {hit} | {missed} | {vp} | **{_pct(hit, missed)}** |")
    L += ["", "**By grade:** " + " · ".join(f"{g}: {_pct(*t)} ({t[0]}-{t[1]})"
                                           for g, t in summary["by_grade"].items()), "",
          "## Game by game", ""]
    for g, info in summary["per_game"].items():
        L += [f"### {g.replace('@', ' @ ')}", ""]
        if info["tiers"]:
            L += ["| Top N | Result |", "|---|---|"]
            L += [f"| Top {n} | {_tier_text(info['tiers'][n])} |" for n in sorted(info["tiers"])]
        else:
            L.append("_Fewer than 2 rated props._")
        L += ["",
              "| # | Player | Pick | Grade | Actual | Result |", "|---|---|---|---|---|---|"]
        for i, p in enumerate(info["props"], 1):
            mark = {"win": "✅", "loss": "❌", "push": "➖", "void": "void", "pending": "…"}[p["result"]]
            act = "" if p["actual"] is None else f"{p['actual']:g}"
            stat = {"passing_yards": "pass", "rushing_yards": "rush", "receiving_yards": "rec"}[p["stat"]]
            L.append(f"| {i} | {p['player']} ({p['team']}) | {p['pick']} {stat} | {p['grade']} | {act} | {mark} |")
        L.append("")
    L += ["_Ranking uses each prop's last rating before kickoff; trap-flagged props are excluded. "
          "Pushes and voids (player didn't play) don't count. For entertainment only._"]
    return "\n".join(L)


REPORT_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>__TITLE__</title>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@700;800&family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root { --bg:#eef2f6; --card:#fff; --ink:#0c1722; --mute:#5b6875; --line:#d9e1e8; --blue:#0076b6; --blue-dk:#005a8c;
  --silver:#b0b7bc; --good:#11804a; --good-bg:#e1f3e9; --bad:#c0352b; --bad-bg:#fbe4e1; }
@media (prefers-color-scheme: dark) { :root { --bg:#07111b; --card:#0f1c29; --ink:#e7edf2; --mute:#93a3b2; --line:#1f3244;
  --blue:#1a8fd6; --blue-dk:#0b5e94; --silver:#8f989f; --good:#5fd394; --good-bg:#11301f; --bad:#f2877c; --bad-bg:#3a1b18; } }
* { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 Inter,system-ui,sans-serif }
.hero { background:linear-gradient(135deg,var(--blue),var(--blue-dk)); color:#fff; border-bottom:4px solid var(--silver) }
.wrap { max-width:1100px; margin:0 auto; padding:0 16px } .hero .wrap { padding:20px 16px }
h1 { margin:0; font:800 34px/1 "Barlow Condensed",sans-serif; letter-spacing:.03em; text-transform:uppercase }
h2 { font:700 22px/1.1 "Barlow Condensed",sans-serif; letter-spacing:.03em; text-transform:uppercase; margin:26px 0 10px }
a { color:var(--blue) } .hero a { color:#fff }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px 14px; margin:10px 0; overflow-x:auto }
table { border-collapse:collapse; width:100%; } th, td { padding:6px 8px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap }
th { color:var(--mute); font-size:12px } tr:last-child td { border-bottom:0 }
.big { font:800 22px/1 "Barlow Condensed",sans-serif } .win { color:var(--good) } .loss { color:var(--bad) }
.tiers { display:flex; flex-wrap:wrap; gap:6px; margin:6px 0 10px } .tier { background:var(--bg); border-radius:8px; padding:4px 8px; font-size:12.5px }
.bar { height:8px; background:var(--line); border-radius:4px; min-width:120px } .bar > i { display:block; height:8px; border-radius:4px; background:var(--blue) }
.mute { color:var(--mute); font-size:12px }
.games { display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:12px }
</style></head><body>
<header class="hero"><div class="wrap"><h1>__H1__</h1><div>__SUB__</div></div></header>
<div class="wrap">__BODY__
<p class="mute">Ranking uses each prop's last rating before kickoff; trap-flagged props are excluded. Pushes and voids (player
didn't play) don't count. For entertainment only — not betting advice. 21+. 1-800-GAMBLER.</p></div></body></html>"""


def report_html(season, week, summary, bank_line=None, label="") -> str:
    import html as H
    rows = ""
    for n in TOP_NS:
        hit, missed, vp, picked = summary["overall"][n]
        pct = hit / (hit + missed) if hit + missed else 0
        rows += (f"<tr><td><b>Top {n}</b> per game</td><td>{picked}</td><td class='win'><b>{hit}</b></td>"
                 f"<td class='loss'>{missed}</td><td>{vp}</td><td class='big'>{_pct(hit, missed)}</td>"
                 f"<td><div class='bar'><i style='width:{pct * 100:.0f}%'></i></div></td></tr>")
    grades = " · ".join(f"<b>{g}</b> {_pct(*t)} <span class='mute'>({t[0]}-{t[1]})</span>"
                        for g, t in summary["by_grade"].items())
    body = ""
    if summary["pending"]:
        body += f"<p class='mute'>{summary['pending']} props still pending (game or stats not final yet).</p>"
    if bank_line:
        body += f"<p><b>Paper bankroll:</b> {H.escape(bank_line)}</p>"
    body += (f"<h2>Whole week — top N props from each of {summary['games']} games</h2><div class='card'><table>"
             f"<tr><th>Picks</th><th>Props picked</th><th>Hit</th><th>Missed</th><th>Void/push</th><th>Hit %</th><th></th></tr>"
             f"{rows}</table><p>By grade: {grades}</p></div><h2>Game by game</h2><div class='games'>")
    for g, info in summary["per_game"].items():
        tiers = "".join(f"<span class='tier'>Top {n}: <b>{t[0]}/{t[0] + t[1]}</b> ({_pct(*t)})</span>"
                        for n, t in sorted(info["tiers"].items()))
        trs = ""
        for i, p in enumerate(info["props"], 1):
            res = p["result"]
            act = "" if p["actual"] is None else f"{p['actual']:g}"
            stat = {"passing_yards": "pass", "rushing_yards": "rush", "receiving_yards": "rec"}[p["stat"]]
            trs += (f"<tr><td>{i}</td><td>{H.escape(p['player'])} <span class='mute'>{p['team']}</span></td>"
                    f"<td>{p['pick']} {stat}</td><td>{p['grade']}</td><td>{act}</td>"
                    f"<td class='{res}'><b>{res.upper()}</b></td></tr>")
        body += (f"<div class='card'><b class='big'>{g.replace('@', ' @ ')}</b><div class='tiers'>{tiers}</div>"
                 f"<table><tr><th>#</th><th>Player</th><th>Pick</th><th>Gr</th><th>Actual</th><th></th></tr>{trs}</table></div>")
    body += "</div>"
    title = f"Week {week} Report Card{label}"
    return (REPORT_TEMPLATE.replace("__TITLE__", f"Tids Takedowns · {title}")
            .replace("__H1__", f"Tids Takedowns — {title}")
            .replace("__SUB__", f"{season} season · <a href='../index.html'>← back to this week's picks</a> · "
                                f"<a href='index.html'>all report cards</a>")
            .replace("__BODY__", body))


def _bank_line(stats, sched, season, week) -> str:
    ledger = load_ledger()
    graded = [b for b in grade_ledger(ledger, stats, sched) if b["kind"] != "none"]
    wk = [b for b in graded if b["season"] == season and b["week"] == week and b["result"] != "pending"]
    s = bankroll_summary(ledger, graded)
    pl = sum(b["profit"] for b in wk)
    w = sum(b["result"] == "win" for b in wk)
    l = sum(b["result"] == "loss" for b in wk)
    return (f"week {week}: {w}-{l}, {'+' if pl >= 0 else '-'}${abs(pl):,.2f} · "
            f"balance ${s['balance']:,.2f} (started ${s['start']:,})")


def cmd_report(args, stats, sched, season):
    """Build report cards from snapshots. --out writes HTML pages for every week;
    --markdown writes the most recent completed week (or --week) as markdown."""
    import json
    snaps = sorted(Path(args.snapshot_dir).glob("*-w*.json")) if Path(args.snapshot_dir).exists() else []
    weeks = []
    for path in snaps:
        snap = json.loads(path.read_text(encoding="utf-8"))
        if not snap["props"]:
            continue
        summary = summarize_week(grade_snapshot(snap, stats, sched))
        summary["label"] = " (backtest)" if snap.get("backtest") else ""
        weeks.append((snap["season"], snap["week"], summary))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        links = ""
        for s_, w_, summ in sorted(weeks, reverse=True):
            (out / f"{s_}-w{w_:02d}.html").write_text(
                report_html(s_, w_, summ, None if summ["label"] else _bank_line(stats, sched, s_, w_),
                            label=summ["label"]), encoding="utf-8")
            t3 = summ["overall"][3]
            links += (f"<div class='card'><a class='big' href='{s_}-w{w_:02d}.html'>Week {w_}{summ['label']} · {s_}</a>"
                      f"<div class='mute'>Top 10 per game: {_tier_text(summ['overall'][10])} · "
                      f"Top 5: {_pct(*summ['overall'][5])} · Top 3: {_pct(*t3)} · Top 2: {_pct(*summ['overall'][2])}"
                      f"{' · in progress' if summ['pending'] else ''}</div></div>")
        (out / "index.html").write_text(
            REPORT_TEMPLATE.replace("__TITLE__", "Tids Takedowns · Report Cards")
            .replace("__H1__", "Tids Takedowns — Report Cards")
            .replace("__SUB__", "How our top-rated props did each week · <a href='../index.html'>← back to picks</a>")
            .replace("__BODY__", links or "<p>No completed weeks yet.</p>"), encoding="utf-8")
        print(f"Wrote {len(weeks)} report card(s) to {out}")
    if args.markdown:
        done = [(s_, w_, summ) for s_, w_, summ in weeks
                if (args.week is None or w_ == args.week)
                and sched[(sched["season"] == s_) & (sched["week"] == w_)]["result"].notna().all()]
        if not done:
            print("No completed week to report.")
            return
        s_, w_, summ = max(done)
        Path(args.markdown).write_text(report_markdown(s_, w_, summ, _bank_line(stats, sched, s_, w_),
                                                       label=summ["label"]),
                                       encoding="utf-8")
        print(f"Week {w_} report written to {args.markdown}")
        print(f"TITLE=Week {w_} Report Card ({s_})")


def cmd_backtest(args, stats, sched, season):
    """Re-run a finished week as if it hadn't happened: stats frozen before that week, DraftKings'
    final pre-game lines, then grade every rating against what actually happened."""
    import json
    week = args.week
    cutoff = season * 100 + week
    stats_asof = stats[stats["game_order"] < cutoff]
    sched_asof = sched.copy()
    later = (sched_asof["season"] == season) & (sched_asof["week"] >= week)
    sched_asof.loc[later, "result"] = float("nan")

    # Roster as of that week (who was on which team is known before kickoff; performance isn't used)
    wk_rows = stats[(stats["season"] == season) & (stats["week"] == week)]
    rosters = dict(zip(wk_rows["player_id"], wk_rows["team"]))
    wargs = argparse.Namespace(week=week, include_played=True, lines=None, refresh=args.refresh,
                               games=args.games, min_games=4, hit_rate=0.70, backtest=True,
                               team_override=rosters)
    _, games, rows, source = build_week(wargs, stats_asof, sched_asof, season)
    rated = [r for r in rows if r.get("pick")]
    print(f"Week {week} backtest: {len(rated)} rated props across {len(games)} games ({source})")

    snap = {"season": season, "week": week, "backtest": True,
            "props": {f"{r['player_id']}|{r['stat']}": {k: r.get(k) for k in SNAPSHOT_FIELDS} for r in rated}}
    graded = grade_snapshot(snap, stats, sched)
    summary = summarize_week(graded)
    md = report_markdown(season, week, summary, label=" (backtest)")

    # What the $200 primetime plans would have done on that week's Thursday and Monday night games
    prime = games[pd.to_datetime(games["gameday"]).dt.day_name().isin(["Thursday", "Monday"])]
    lines = [""]
    for g in prime.itertuples():
        key = f"{g.away_team}@{g.home_team}"
        day = pd.to_datetime(g.gameday).day_name()
        game_rows = [r for r in rated if r["game"] == key]
        for label, plan in (("old plan", choose_bets_v1), ("new plan", choose_bets)):
            bets = plan(game_rows, 200)
            for b in bets:
                b.update({"season": season, "week": week, "game": key, "placed_at": "", "id": ""})
            settled = grade_ledger({"bets": bets}, stats, sched)
            staked = sum(b["stake"] for b in settled)
            total = sum(b["profit"] for b in settled)
            lines += [f"## {day} night — {g.away_team} @ {g.home_team} · {label}: "
                      f"{'+' if total >= 0 else '-'}${abs(total):.2f} on ${staked:g} staked", ""]
            for b in settled:
                desc = " + ".join(
                    f"{l['player']} {l['side'].upper()} {l['alt'] or format(l['line'], 'g') if l.get('alt') else format(l['line'], 'g')}"
                    f" {'(est ' + format(l['odds'], '+d') + ')' if l.get('alt') else ''}"
                    f"→{'DNP' if l['actual'] is None else format(l['actual'], 'g')}" for l in b["legs"])
                lines.append(f"- ${b['stake']:g} {b['kind']}: {desc} — **{b['result']} "
                             f"{'+' if b['profit'] >= 0 else '-'}${abs(b['profit']):.2f}**")
            if not bets:
                lines.append("- No qualifying props — would have passed.")
            lines.append("")
    md += "\n" + "\n".join(lines)

    if args.snapshot_dir:
        path = Path(args.snapshot_dir) / f"{season}-w{week:02d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snap, indent=1, default=str) + "\n", encoding="utf-8")
        print(f"Saved backtest snapshot to {path}")
    if args.markdown:
        Path(args.markdown).write_text(md, encoding="utf-8")
        print(f"Report written to {args.markdown}")


SITE_URL = "https://dougfoster68-coder.github.io/nfl-matchup-analyzer/"

TOP5_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta http-equiv="refresh" content="300">
<title>Tids' Top 5 · NFL Week __WEEK__</title>
<meta name="description" content="__DESC__">
<meta property="og:type" content="website"><meta property="og:url" content="__URL__">
<meta property="og:title" content="Tids' Top 5 · NFL Week __WEEK__ player props">
<meta property="og:description" content="__DESC__">
<meta name="twitter:card" content="summary"><meta name="twitter:title" content="Tids' Top 5 · NFL Week __WEEK__">
<meta name="twitter:description" content="__DESC__">
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root { --bg:#eef2f6; --card:#fff; --ink:#0c1722; --mute:#5b6875; --line:#d9e1e8; --blue:#0076b6; --blue-dk:#005a8c;
  --silver:#b0b7bc; --good:#11804a; --bad:#c0352b; --A:#0076b6; --B:#4f7f9e; }
@media (prefers-color-scheme: dark) { :root { --bg:#07111b; --card:#0f1c29; --ink:#e7edf2; --mute:#93a3b2; --line:#1f3244;
  --blue:#1a8fd6; --blue-dk:#0b5e94; --silver:#8f989f; --good:#5fd394; --bad:#f2877c; --A:#1a8fd6; --B:#6f9bb8; } }
* { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 Inter,system-ui,sans-serif }
.hero { position:relative; overflow:hidden; color:#fff; border-bottom:4px solid var(--silver);
  background:linear-gradient(135deg,var(--blue),var(--blue-dk)); background-size:200% 200%; animation:shift 14s ease-in-out infinite }
@keyframes shift { 0%,100% { background-position:0% 50% } 50% { background-position:100% 50% } }
.hero::before { content:""; position:absolute; inset:-40% -10%; opacity:.2; pointer-events:none;
  background:repeating-linear-gradient(100deg,transparent 0 46px,rgba(255,255,255,.55) 46px 49px,transparent 49px 120px);
  animation:streak 2.6s linear infinite }
@keyframes streak { from { transform:translateX(-120px) } to { transform:translateX(0) } }
.wrap { max-width:720px; margin:0 auto; padding:0 16px } .hero .wrap { position:relative; padding:24px 16px 20px }
.kicker { font:700 13px/1 Inter,sans-serif; letter-spacing:.12em; text-transform:uppercase; opacity:.85 }
h1 { margin:6px 0 4px; font:800 46px/0.95 "Barlow Condensed",sans-serif; letter-spacing:.03em; text-transform:uppercase }
.sub { opacity:.85; font-size:13px }
.list { display:grid; gap:12px; margin:18px 0 }
.pick { position:relative; overflow:hidden; display:grid; grid-template-columns:64px 1fr; gap:14px; align-items:center;
  background:var(--card); border:1px solid var(--line); border-left:6px solid var(--blue); border-radius:14px; padding:14px 16px;
  animation:rise .6s cubic-bezier(.2,.8,.2,1) both }
@keyframes rise { from { opacity:0; transform:translateY(14px) } }
.rank { font:800 56px/1 "Barlow Condensed",sans-serif; text-align:center;
  background:linear-gradient(180deg,var(--blue),var(--silver)); -webkit-background-clip:text; background-clip:text; color:transparent }
.name { font:700 18px/1.2 Inter,sans-serif } .meta { color:var(--mute); font-size:13px }
.line { font:800 30px/1.05 "Barlow Condensed",sans-serif; letter-spacing:.02em; margin:4px 0 2px }
.OVER { color:var(--good) } .UNDER { color:var(--bad) }
.g { display:inline-grid; place-items:center; width:24px; height:24px; border-radius:6px; color:#fff; font:700 13px Inter,sans-serif;
  vertical-align:3px; margin-left:6px } .gA { background:var(--A) } .gB { background:var(--B) }
.why { margin-top:6px; font-size:12.5px; color:var(--good) }
.actions { display:flex; gap:10px; flex-wrap:wrap; margin:6px 0 18px }
.btn { appearance:none; border:0; border-radius:10px; padding:11px 16px; font:700 14px Inter,sans-serif; cursor:pointer;
  background:var(--blue); color:#fff; text-decoration:none } .btn.alt { background:var(--card); color:var(--blue); border:1px solid var(--line) }
.fine { color:var(--mute); font-size:12px; padding-bottom:30px }
.empty { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:20px; color:var(--mute) }
@media (prefers-reduced-motion: reduce) { *, *::before { animation:none !important } }
</style></head><body>
<header class="hero"><div class="wrap"><div class="kicker">Tids Takedowns · NFL Week __WEEK__</div>
<h1>Tids' Top 5</h1><div class="sub">The highest-rated player props on the board · updated __UPDATED__</div></div></header>
<div class="wrap">
<section class="list">__PICKS__</section>
<div class="actions"><button class="btn" id="share">Share these picks</button>
<a class="btn alt" href="../">See every game &amp; rating →</a></div>
<p class="fine">Ratings compare each player's last 10 games with the opponent defense's last 10 against the posted DraftKings line,
then factor in how often the line has hit and recent form. Trap-flagged props are excluded. Lines move, so confirm at your
sportsbook. For entertainment only, not betting advice. 21+. Gambling problem? Call 1-800-GAMBLER. A fan project, not affiliated
with the NFL.</p></div>
<script>
document.getElementById("share").addEventListener("click", async () => {
  const data = { title: document.title, text: __SHARETEXT__, url: location.href };
  try {
    if (navigator.share) { await navigator.share(data); return; }
    await navigator.clipboard.writeText(data.text + "\n" + data.url);
    const b = document.getElementById("share"); b.textContent = "Link copied!"; setTimeout(() => b.textContent = "Share these picks", 2000);
  } catch (e) {}
});
</script></body></html>"""


def top5_rows(rows: list) -> list:
    """Same rule as the dashboard: A/B-rated, trap-free, best score first, one prop per player."""
    seen, out = set(), []
    for r in sorted((r for r in rows if r.get("grade") in ("A", "B") and not r.get("traps")),
                    key=lambda r: r["score"], reverse=True):
        if r["player_id"] not in seen:
            seen.add(r["player_id"])
            out.append(r)
        if len(out) == 5:
            break
    return out


def write_top5(week, games, rows, path: Path):
    import html as H
    import json
    stat_name = {"passing_yards": "pass yds", "rushing_yards": "rush yds", "receiving_yards": "rec yds"}
    kick = {}
    for g in games.itertuples():
        d = pd.to_datetime(g.gameday)
        h, m = map(int, str(g.gametime).split(":"))
        kick[f"{g.away_team}@{g.home_team}"] = f"{d:%a} {(h + 11) % 12 + 1}:{m:02d} {'AM' if h < 12 else 'PM'} ET"
    top = top5_rows(rows)
    cards = ""
    for i, r in enumerate(top, 1):
        side = r["pick"].split()[0]
        good = next((n["t"] for n in r.get("notes", []) if (n["k"] == "good") == (side == "OVER") and n["k"] in ("good", "bad")), "")
        cards += (f"<article class='pick' style='animation-delay:{i * 90}ms'><div class='rank'>{i}</div><div>"
                  f"<div class='name'>{H.escape(r['player'])}<span class='g g{r['grade']}'>{r['grade']}</span></div>"
                  f"<div class='meta'>{r['pos']} · {r['team']} vs {r['opp']} · {kick.get(r['game'], '')}</div>"
                  f"<div class='line {side}'>{r['pick']} <span style='font-size:18px'>{stat_name[r['stat']]}</span></div>"
                  f"<div class='meta'>Projection {r['proj']:g} · hit {r['line_hits']} of last 10 · rating {r['score']}/10</div>"
                  + (f"<div class='why'>✓ {H.escape(good)}</div>" if good else "") + "</div></article>")
    if not cards:
        cards = "<div class='empty'>No A/B-rated plays posted yet. Lines usually open midweek, so check back soon.</div>"
    desc = " · ".join(f"{i}. {r['player']} {r['pick']} {stat_name[r['stat']]}" for i, r in enumerate(top, 1)) \
        or "This week's top-rated NFL player props."
    share = f"Tids' Top 5, NFL Week {week}: " + desc
    page = (TOP5_TEMPLATE.replace("__WEEK__", str(week)).replace("__DESC__", H.escape(desc))
            .replace("__URL__", SITE_URL + "top5/").replace("__UPDATED__", H.escape(_now_et()))
            .replace("__PICKS__", cards).replace("__SHARETEXT__", json.dumps(share).replace("</", "<\\/")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


def cmd_week(args, stats, sched, season):
    out_dir = Path(__file__).parent
    while True:
        week, games, rows, source = build_week(args, stats, sched, season)
        if args.snapshot_dir:
            update_snapshot(args.snapshot_dir, season, week, rows)
        if args.live:
            os.system("cls" if os.name == "nt" else "clear")
        print_week(week, games, rows, source, args)
        csv_path = out_dir / f"week{week}_report.csv"
        html_path = Path(args.html) if args.html else out_dir / f"week{week}_dashboard.html"
        html_path.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
        # a published page re-polls every 5 min so viewers pick up new deploys
        refresh = int(args.live * 60) if args.live else (300 if args.html else None)
        ledger = load_ledger()
        graded = grade_ledger(ledger, stats, sched)
        bank = {"summary": bankroll_summary(ledger, graded),
                "bets": [b for b in graded if b["kind"] != "none"][::-1]}
        write_html(week, games, rows, source, args, html_path, refresh_secs=refresh, bank=bank,
                   feature=feature_player(stats, rows, season))
        if args.html:  # shareable Top 5 page next to the dashboard
            write_top5(week, games, rows, html_path.parent / "top5" / "index.html")
        print(f"\nSaved {csv_path.name} and {html_path.name} in {out_dir}")
        if not args.live:
            return
        print(f"Live mode: refreshing lines every {args.live:g} min (Ctrl+C to stop)...")
        try:
            time.sleep(args.live * 60)
        except KeyboardInterrupt:
            return
        # player/defense stats refresh on their own 6h cache; re-check so finished games count
        stats, sched, season = load_data(False)


def main():
    ap = argparse.ArgumentParser(description="NFL player vs. defense matchup analyzer")
    ap.add_argument("--games", type=int, default=10, help="lookback window (default 10)")
    ap.add_argument("--refresh", action="store_true", help="force re-download of data")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("player", help="analyze one player vs. upcoming opponent")
    p.add_argument("name")
    p.add_argument("--week", type=int, help="target week (default: next unplayed game)")
    p.add_argument("--opponent", help="override opponent team abbreviation (e.g. KC)")
    p.add_argument("--stat", help="stat for prop analysis (e.g. receiving_yards)")
    p.add_argument("--line", type=float, help="sportsbook prop line, e.g. 64.5")

    s = sub.add_parser("slate", help="rank every player at a position for a week")
    s.add_argument("--position", required=True, choices=POSITIONS + [x.lower() for x in POSITIONS])
    s.add_argument("--stat", help="default fantasy_points_ppr")
    s.add_argument("--week", type=int)
    s.add_argument("--top", type=int, default=25)
    s.add_argument("--min-games", type=int, default=4)
    s.add_argument("--min-avg", type=float, default=0.0, help="hide low-volume players")
    s.add_argument("--sort", choices=["projection", "matchup"], default="projection")

    d = sub.add_parser("defense", help="show what a defense allows by position")
    d.add_argument("team")

    w = sub.add_parser("week", help="full report: every game this week, QB/RB/WR/TE yardage props")
    w.add_argument("--week", type=int)
    w.add_argument("--lines", help="CSV of prop lines (player,stat,line); otherwise uses "
                                   "ODDS_API_KEY env var if set")
    w.add_argument("--hit-rate", type=float, default=0.70, help="hit-rate threshold (default 0.70)")
    w.add_argument("--min-games", type=int, default=4)
    w.add_argument("--top", type=int, default=40)
    w.add_argument("--include-played", action="store_true", help="include games already finished")
    w.add_argument("--live", type=float, metavar="MIN", help="keep running, refresh lines every MIN minutes")
    w.add_argument("--html", help="write the dashboard to this path (e.g. site/index.html)")
    w.add_argument("--snapshot-dir", help="save pre-kickoff ratings here for weekly report cards")

    ab = sub.add_parser("autobet", help="place Claude's paper bets on today's games (pre-kickoff window)")
    ab.add_argument("--budget", type=float, default=200)
    ab.add_argument("--day", default="Thursday,Monday",
                    help="comma-separated weekdays to bet on ('' = any day)")
    ab.add_argument("--window", type=float, default=3.0, help="hours before kickoff to start betting")
    ab.add_argument("--force", action="store_true", help="ignore day/time window (testing)")

    bt = sub.add_parser("bet", help="record a manual paper bet")
    bt.add_argument("--leg", action="append", required=True, help="'Player Name|stat|over|40.5' (repeat for parlay)")
    bt.add_argument("--stake", type=float, required=True)
    bt.add_argument("--odds", type=int, default=DEFAULT_ODDS, help="American odds per leg (default -110)")
    bt.add_argument("--week", type=int)
    bt.add_argument("--note")

    sub.add_parser("bankroll", help="show paper bankroll and bet history")

    rp = sub.add_parser("report", help="weekly report cards: hit %% of our top-N props per game")
    rp.add_argument("--snapshot-dir", default="data/snapshots")
    rp.add_argument("--out", help="write HTML report pages for every week to this folder")
    rp.add_argument("--markdown", help="write the latest completed week (or --week) as markdown")
    rp.add_argument("--week", type=int)

    bk = sub.add_parser("backtest", help="replay a finished week with stats frozen before it, then grade")
    bk.add_argument("--week", type=int, required=True)
    bk.add_argument("--snapshot-dir", help="save the backtest ratings here so report cards include it")
    bk.add_argument("--markdown", help="write the backtest report as markdown")

    args = ap.parse_args()
    stats, sched, season = load_data(args.refresh)
    {"player": cmd_player, "slate": cmd_slate, "defense": cmd_defense,
     "week": cmd_week, "autobet": cmd_autobet, "bet": cmd_bet,
     "bankroll": cmd_bankroll, "report": cmd_report,
     "backtest": cmd_backtest}[args.cmd](args, stats, sched, season)


if __name__ == "__main__":
    main()
