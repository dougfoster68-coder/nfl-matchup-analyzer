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

    if row["missed_last_game"]:
        notes.append({"t": "Missed team's last game â€” check injury status", "k": "warn"})
    if line is None and lines_expected:
        notes.append({"t": "No line posted for a regular contributor â€” possible injury or role change",
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
            notes.append({"t": f"Ceiling: {big} games of {1.5 * line:.0f}+ (1.5Ã— the line)", "k": "info"})
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
            notes.append({"t": f"Line moved {direction} {row['open']:g} â†’ {line:g} since open",
                          "k": "info"})
        if abs(line - avg) / max(avg, 10) > 0.45:
            trap = True
            notes.append({"t": f"Line ({line:g}) is far from the {avg:.0f} L10 avg â€” book likely "
                               "knows about a role/injury change", "k": "warn"})

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
        if trap:
            score *= 0.45
        if row["missed_last_game"]:
            score *= 0.6
        if n < 6:
            score *= 0.8
        grade = "A" if score >= 7.5 else "B" if score >= 6 else "C" if score >= 4.5 else "D"
        rating = {"pick": f"{'OVER' if over else 'UNDER'} {line:g}", "score": round(score, 1),
                  "grade": grade}
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
<title>NFL Week __WEEK__ Matchups</title>
<style>
:root { --bg:#f6f5f1; --card:#fff; --ink:#1c1c1a; --mute:#6b6a64; --line:#e4e2db; --soft:#efeee8;
  --good:#17803f; --good-bg:#e2f3e8; --bad:#b4342b; --bad-bg:#fbe5e2; --warn:#a15c00; --warn-bg:#fff0d4;
  --info:#3a5a8c; --info-bg:#e6edf8; --A:#17803f; --B:#2f7a8a; --C:#a07a12; --D:#8a8983; --accent:#1f3a5f; }
@media (prefers-color-scheme: dark) { :root { --bg:#131312; --card:#1d1d1b; --ink:#ecebe5; --mute:#9d9c95;
  --line:#34332f; --soft:#262623; --good:#62d290; --good-bg:#15301f; --bad:#f08a80; --bad-bg:#3a1d1a;
  --warn:#f1b65a; --warn-bg:#382810; --info:#9dbcf0; --info-bg:#1b2638; --A:#4fc47f; --B:#5fbfd1;
  --C:#e0b84a; --D:#8f8e88; --accent:#9dbcf0; } }
* { box-sizing:border-box }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif }
.wrap { max-width:1240px; margin:0 auto; padding:0 16px }
header { padding:24px 0 6px }
h1 { margin:0; font-size:26px; letter-spacing:-.01em }
h2 { font-size:18px; margin:28px 0 10px }
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
</style></head><body>
<div class="wrap">
<header>
  <h1>NFL Week __WEEK__ Matchups &amp; Player Ratings</h1>
  <div class="sub">Updated __UPDATED__ Â· Lines: __SOURCE__ Â· Each player's last __N__ games vs. the opponent defense's last __N__</div>
  <div class="legend"><span><span class="g gA">A</span> strong</span><span><span class="g gB">B</span> good</span>
  <span><span class="g gC">C</span> lean</span><span><span class="g gD">D</span> pass</span>
  <span>Rating = model projection vs line + hit rate + last 3 games, penalized for injury/role red flags.</span></div>
</header>

<h2>This week's games</h2>
<section class="games" id="games"></section>

<h2 id="all">All player props</h2>
<div class="bar">
  <input type="search" id="q" placeholder="Search player or team">
  <select id="game"><option value="">All games</option></select>
  <select id="stat"><option value="">All stats</option><option value="passing_yards">Passing yds</option>
    <option value="rushing_yards">Rushing yds</option><option value="receiving_yards">Receiving yds</option></select>
  <label><input type="checkbox" id="good"> A/B ratings only</label>
  <label><input type="checkbox" id="hot"> Hit line __HITPCT__+ of last __N__</label>
</div>
<div class="tw"><table><thead><tr>
  <th data-s="score">Rating</th><th data-s="player">Player / what stands out</th><th data-s="pick">Pick</th>
  <th data-s="proj">Proj</th><th data-s="line_hit_rate">Hit L__N__</th><th data-s="p_over">P(over)</th>
  <th data-s="def_rank">Opp rank</th><th>Last __N__ (bar = line)</th><th data-s="alt">Alt __HITPCT__+</th>
</tr></thead><tbody id="rows"></tbody></table></div>

<footer>Opp rank: 1 = stingiest vs that position, 32 = most generous. Proj = player's L__N__ average Ã— how much this defense
allows vs league average (regressed toward average). P(over) is a model estimate, not a guarantee. Chip colors are relative to the pick
(green helps it, red hurts it, amber = red flag). Lines move â€” confirm at your sportsbook. Built from free nflverse stats and
DraftKings lines via ESPN.<br><br>For entertainment and research only â€” not betting advice. 21+. Gambling problem? Call 1-800-GAMBLER.</footer>
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
  return `<div class="chips">${ns.slice(0, max ?? 99).map(n => `<span class="chip k-${n.k}">${esc(n.t)}</span>`).join("")}</div>`;
}
function grade(r) { return r.grade ? `<span class="g g${r.grade}" title="score ${r.score}/10">${r.grade}</span>` : `<span class="g gD" style="opacity:.35">â€“</span>`; }
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
    const rs = ROWS.filter(r => r.game === g.key && r.score != null).sort((a, b) => b.score - a.score);
    const top = rs.slice(0, 4);
    const plays = top.length ? top.map(r => `<div class="play">${grade(r)}<div class="pl">
        <div><span class="pname">${esc(r.player)}</span> <span class="pmeta">${r.pos} Â· ${r.team}</span></div>
        <div><span class="pick ${r.pick.split(" ")[0]}">${r.pick}</span> <span class="pmeta">${STAT[r.stat]} Â· proj ${r.proj} Â· hit ${r.line_hits}</span></div>
        ${chips(r, 2)}</div></div>`).join("") : `<div class="empty">No player lines posted yet.</div>`;
    return `<article class="card"><div class="ch">
        <div class="teams">${logo(g.away)}${g.away} <span class="at">@</span> ${logo(g.home)}${g.home}<span class="kick">${kick(g)}</span></div>
        <div class="vegas">${esc(g.vegas)}</div></div>
      <div class="plays">${plays}</div>
      <div class="cf"><button data-game="${g.key}">All ${ROWS.filter(r => r.game === g.key).length} props in this game â†’</button></div></article>`;
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
  const rs = ROWS.filter(r => (!q || (r.player + " " + r.team + " " + r.opp).toLowerCase().includes(q))
    && (!gm || r.game === gm) && (!st || r.stat === st) && (!good || r.grade === "A" || r.grade === "B")
    && (!hot || (r.line_hit_rate != null && r.line_hit_rate >= HIT)));
  rs.sort((a, b) => {
    const x = a[sortKey], y = b[sortKey];
    if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1;
    return (typeof x === "string" ? x.localeCompare(y) : x - y) * sortDir;
  });
  document.getElementById("rows").innerHTML = rs.map(r => {
    const move = r.open != null && r.line != null && r.open !== r.line ? `<div class="pmeta">open ${r.open}</div>` : "";
    return `<tr><td>${grade(r)}</td>
      <td><span class="pname">${esc(r.player)}</span> <span class="pmeta">${r.pos} Â· ${r.team} vs ${r.opp} Â· ${STAT[r.stat]} Â· L${r.n} avg ${r.L10_avg}</span>${chips(r)}</td>
      <td class="n">${r.pick ? `<span class="pick ${r.pick.split(" ")[0]}">${r.pick}</span>${move}` : `<span class="pmeta">no line</span>`}</td>
      <td class="n"><b>${r.proj}</b></td><td class="n">${r.line_hits || ""}</td><td class="n">${pct(r.p_over)}</td>
      <td class="n">${r.def_rank}/32</td><td>${spark(r)}</td>
      <td class="n">${r.alt ? `${r.alt}<div class="pmeta">${r.alt_hits} Â· ${pct(r.alt_p)}</div>` : ""}</td></tr>`;
  }).join("") || `<tr><td colspan="9" class="empty">No matches.</td></tr>`;
}

const sel = document.getElementById("game");
GAMES.list.forEach(g => sel.insertAdjacentHTML("beforeend", `<option value="${g.key}">${g.away} @ ${g.home}</option>`));
["q","game","stat","good","hot"].forEach(id => document.getElementById(id).addEventListener("input", renderRows));
document.querySelectorAll("th[data-s]").forEach(th => th.onclick = () => {
  const k = th.dataset.s; sortDir = k === sortKey ? -sortDir : (k === "player" || k === "def_rank" ? 1 : -1); sortKey = k; renderRows();
});
renderGames(); renderRows();
</script></body></html>"""


def write_html(week, games, rows, source, args, path: Path, refresh_secs: int | None):
    import html
    import json

    def clean(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        if hasattr(v, "item"):  # numpy scalar
            return v.item()
        return v

    data = [{k: clean(v) for k, v in r.items()} for r in rows]
    game_list = []
    for g in games.itertuples():
        vegas = "No game line yet"
        if pd.notna(g.total_line) and pd.notna(g.spread_line):
            home_tt = (g.total_line + g.spread_line) / 2
            fav, pts = (g.home_team, g.spread_line) if g.spread_line > 0 else (g.away_team, -g.spread_line)
            spread = f"{fav} -{pts:g}" if pts else "Pick'em"
            vegas = (f"{spread} Â· O/U {g.total_line:g} Â· Implied: {g.away_team} "
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
            .replace("__ROWS__", json.dumps(data, default=str).replace("</", "<\\/")))
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
