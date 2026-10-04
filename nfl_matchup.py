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


def fetch_espn_props(season: int, week: int, refresh: bool):
    """Live DraftKings player props via ESPN's public odds feed.
    Returns (main_lines, milestones), keyed by (gsis player_id, stat).
    main_lines value: {"line", "open", "updated"}; milestones value: sorted list of N (meaning N+)."""
    import re
    from concurrent.futures import ThreadPoolExecutor

    board = _get_json(f"{ESPN_SCOREBOARD}?seasontype=2&week={week}&dates={season}")
    event_ids = [e["id"] for e in board.get("events", [])
                 if e.get("status", {}).get("type", {}).get("state") == "pre"]
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
        main_by_id, miles_by_id = fetch_espn_props(season, week, args.refresh)
        source = f"DraftKings via ESPN ({len(main_by_id)} yardage lines)"

    allowed = defense_allowed(stats, args.games)
    ordered = stats.sort_values("game_order")
    last_row = ordered.drop_duplicates("player_id", keep="last")
    last_row = last_row[(last_row["season"] == season) & last_row["team"].isin(teams)]
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
                               "team": team, "opp": opp, "player": pr.player_display_name, "pos": pos,
                               "stat": stat, "n": len(vals), "L10_avg": round(avg, 1),
                               "L10_med": round(vals.median(), 1), "def_rank": rank,
                               "matchup_pct": int(round((factor - 1) * 100)), "proj": round(proj, 1),
                               "missed_last_game": pr.game_id != team_last_game.get(team),
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
                        rows.append(row)
    return week, games, rows, source


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


def write_html(week, games, rows, source, args, path: Path, refresh_secs: int | None):
    import html
    import json

    data = json.dumps([{k: (None if isinstance(v, float) and math.isnan(v) else v)
                        for k, v in r.items()} for r in rows], default=str)
    games_info = []
    for g in games.itertuples():
        info = f"{g.away_team} @ {g.home_team} · {g.gameday} {g.gametime}"
        if pd.notna(g.total_line):
            home_tt = (g.total_line + g.spread_line) / 2
            info += (f" · O/U {g.total_line} · implied {g.away_team} {g.total_line - home_tt:.1f}"
                     f" / {g.home_team} {home_tt:.1f}")
        games_info.append({"key": f"{g.away_team}@{g.home_team}", "label": info})
    meta = f'<meta http-equiv="refresh" content="{refresh_secs}">' if refresh_secs else ""
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">{meta}
<title>Week {week} Props</title>
<style>
:root {{ --bg:#f7f7f5; --card:#fff; --ink:#1d1d1b; --mute:#6b6b66; --line:#e3e2dd;
        --good:#1a7f45; --good-bg:#e3f4ea; --bad:#b4342b; --bad-bg:#fbe6e4; --hot:#9a6400; --hot-bg:#fff3d6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141413; --card:#1e1e1c; --ink:#ecebe6; --mute:#9c9b94;
        --line:#33332f; --good:#5fd08f; --good-bg:#16301f; --bad:#f0857b; --bad-bg:#3a1c19; --hot:#f2c35c; --hot-bg:#352a10; }} }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif }}
header {{ padding:20px 16px 8px; max-width:1200px; margin:auto }}
h1 {{ margin:0 0 4px; font-size:22px }}
.sub {{ color:var(--mute) }}
.bar {{ display:flex; gap:8px; flex-wrap:wrap; max-width:1200px; margin:8px auto; padding:0 16px }}
.bar input, .bar select {{ padding:7px 10px; border:1px solid var(--line); border-radius:8px; background:var(--card); color:var(--ink) }}
.bar label {{ display:flex; align-items:center; gap:6px; color:var(--mute) }}
main {{ max-width:1200px; margin:auto; padding:0 16px 40px }}
.game {{ background:var(--card); border:1px solid var(--line); border-radius:12px; margin:14px 0; overflow:hidden }}
.game h2 {{ font-size:15px; margin:0; padding:12px 14px; border-bottom:1px solid var(--line) }}
.wrap {{ overflow-x:auto }}
table {{ border-collapse:collapse; width:100%; min-width:860px }}
th, td {{ padding:7px 10px; text-align:left; border-bottom:1px solid var(--line); white-space:nowrap }}
th {{ color:var(--mute); font-weight:600; font-size:12px; cursor:pointer }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums }}
.pill {{ padding:2px 8px; border-radius:99px; font-weight:600; font-size:12px }}
.OVER {{ background:var(--good-bg); color:var(--good) }} .UNDER {{ background:var(--bad-bg); color:var(--bad) }}
.hot {{ background:var(--hot-bg); color:var(--hot) }}
.inj {{ color:var(--bad); font-size:12px }}
.l10 {{ color:var(--mute); font-size:12px }}
footer {{ color:var(--mute); font-size:12px; max-width:1200px; margin:auto; padding:0 16px 30px }}
</style></head><body>
<header><h1>Week {week} NFL yardage props</h1>
<div class="sub">Lines: {html.escape(source)} · last {args.games} games vs opponent defense ·
updated {_now_et()}{f' · auto-refresh every {refresh_secs}s' if refresh_secs else ''}</div></header>
<div class="bar">
 <input id="q" placeholder="Search player or team">
 <select id="stat"><option value="">All stats</option><option>passing_yards</option><option>rushing_yards</option><option>receiving_yards</option></select>
 <label><input type="checkbox" id="hot"> 70%+ hit rate only</label>
 <label><input type="checkbox" id="lean"> Leans only</label>
</div>
<main id="out"></main>
<footer>Def rank: 1 = stingiest vs that position, 32 = most generous. Matchup %: defense's allowed rate vs league avg (regressed).
P(over) is a model estimate; lean shown only when it beats -110 break-even (52.4%) by 3+ points. Alt = highest milestone (N+) cleared {args.hit_rate:.0%}+ of last {args.games}.
Check injury reports — red flags mark players who missed their team's last game.<br><br>
For entertainment and research only — not betting advice. Lines move; confirm at your sportsbook. 21+. Gambling problem? Call 1-800-GAMBLER.</footer>
<script>
const ROWS = {data}; const GAMES = {json.dumps(games_info)}; const HIT = {args.hit_rate};
const short = {{passing_yards:"pass yds", rushing_yards:"rush yds", receiving_yards:"rec yds"}};
const pct = v => v == null ? "" : Math.round(v*100) + "%";
function render() {{
  const q = document.getElementById("q").value.toLowerCase(), st = document.getElementById("stat").value;
  const hot = document.getElementById("hot").checked, lean = document.getElementById("lean").checked;
  let out = "";
  for (const g of GAMES) {{
    const rs = ROWS.filter(r => r.game === g.key
      && (!q || (r.player + " " + r.team).toLowerCase().includes(q)) && (!st || r.stat === st)
      && (!hot || (r.line_hit_rate != null && r.line_hit_rate >= HIT)) && (!lean || r.lean));
    if (!rs.length) continue;
    out += `<section class="game"><h2>${{g.label}}</h2><div class="wrap"><table><tr>
      <th>Player</th><th>Team</th><th>Stat</th><th>L10 avg</th><th>Opp rank</th><th>Matchup</th><th>Proj</th>
      <th>Line</th><th>Hit L10</th><th>P(over)</th><th>Lean</th><th>Alt 70%+</th><th>Last 10</th></tr>`;
    for (const r of rs) {{
      const isHot = r.line_hit_rate != null && r.line_hit_rate >= HIT;
      const move = r.open != null && r.line != null && r.open !== r.line ? ` <span class="l10">(open ${{r.open}})</span>` : "";
      out += `<tr><td>${{r.pos}} <b>${{r.player}}</b>${{r.missed_last_game ? ' <span class="inj">missed last game</span>' : ''}}</td>
        <td>${{r.team}} vs ${{r.opp}}</td><td>${{short[r.stat]}}</td><td class="num">${{r.L10_avg}}</td>
        <td class="num">${{r.def_rank}}/32</td><td class="num">${{r.matchup_pct > 0 ? "+" : ""}}${{r.matchup_pct}}%</td>
        <td class="num"><b>${{r.proj}}</b></td><td class="num">${{r.line ?? "—"}}${{move}}</td>
        <td class="num">${{r.line_hits ? `<span class="pill ${{isHot ? 'hot' : ''}}">${{r.line_hits}}</span>` : ""}}</td>
        <td class="num">${{pct(r.p_over)}}</td><td>${{r.lean ? `<span class="pill ${{r.lean}}">${{r.lean}}</span>` : ""}}</td>
        <td>${{r.alt ? `${{r.alt}} <span class="l10">${{r.alt_hits}} · ${{pct(r.alt_p)}}</span>` : ""}}</td>
        <td class="l10">${{r.last10}}</td></tr>`;
    }}
    out += "</table></div></section>";
  }}
  document.getElementById("out").innerHTML = out || "<p>No matches.</p>";
}}
["q","stat","hot","lean"].forEach(id => document.getElementById(id).addEventListener("input", render));
render();
</script></body></html>"""
    path.write_text(page, encoding="utf-8")


def cmd_week(args, stats, sched, season):
    out_dir = Path(__file__).parent
    while True:
        week, games, rows, source = build_week(args, stats, sched, season)
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
        write_html(week, games, rows, source, args, html_path, refresh_secs=refresh)
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

    args = ap.parse_args()
    stats, sched, season = load_data(args.refresh)
    {"player": cmd_player, "slate": cmd_slate, "defense": cmd_defense,
     "week": cmd_week}[args.cmd](args, stats, sched, season)


if __name__ == "__main__":
    main()
