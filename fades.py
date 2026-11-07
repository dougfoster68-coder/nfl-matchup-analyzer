"""Fade the Public: weekly picks and a live season tracker (site/fades/).

Two rules, both found in the 2026 Weeks 1-8 backtest:
  * Fade 70%+     - the public (majority of spread tickets) has 70% or more of the bets -> take the other side.
  * Stats agree 3+ - our stats model is on the same side as the public by 3+ points vs the line -> fade anyway.
A game that hits both is a "double".

The public side and line come from the last ScoresAndOdds capture before kickoff (data/splits, saved
every 15 minutes by the publish workflow). Finished games with no capture are backfilled once from
ScoresAndOdds' archive page for that week. The stats model is opponent-adjusted EPA per play using only
plays before each week, so past weeks are graded exactly as they would have been picked.

    python fades.py --splits-dir data/splits --out site/fades/index.html
"""
from __future__ import annotations

import argparse
import html as H
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import nfl_matchup as nm

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
PBP_COLS = ["season", "week", "season_type", "posteam", "defteam", "epa", "play_type"]
PUBLIC_MIN = 70      # Fade 70%+ rule
AGREE_MIN = 3.0      # Stats agree 3+ rule (points vs the line)
PLAYS = 60           # plays per team per game, to turn EPA/play into points
HFA = 1.5            # home field, points
PRIOR_W = 0.35       # weight on last season's plays
RIDGE = 300.0        # shrink ratings toward average (in plays)
STAKE = 100.0
WIN = STAKE * 100 / 110  # -110


# ---------------------------------------------------------------------------- stats model

def load_pbp(season: int, refresh: bool = False) -> pd.DataFrame:
    frames = []
    for s in (season - 1, season):
        path = nm.CACHE_DIR / f"pbp_{s}.csv.gz"
        stale = not path.exists() or (s == season and time.time() - path.stat().st_mtime > nm.CACHE_HOURS * 3600)
        if refresh or stale:
            try:
                req = urllib.request.Request(PBP_URL.format(season=s), headers={"User-Agent": "Mozilla/5.0"})
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(urllib.request.urlopen(req, timeout=120).read())
            except Exception as exc:
                print(f"[warn] play-by-play {s} download failed: {exc}", file=sys.stderr)
        if path.exists():
            frames.append(pd.read_csv(path, usecols=PBP_COLS, low_memory=False))
    p = pd.concat(frames, ignore_index=True)
    p = p[(p.season_type == "REG") & p.play_type.isin(["pass", "run"]) & p.epa.notna() & p.posteam.notna()].copy()
    p["epa"] = p.epa.clip(-4.5, 4.5)  # one huge play shouldn't swing a team
    return p


def team_ratings(pbp: pd.DataFrame, season: int, week: int) -> dict:
    """Net EPA/play (offense minus defense allowed), adjusted for opponents, from plays before `week`."""
    x = pbp[(pbp.season == season - 1) | ((pbp.season == season) & (pbp.week < week))]
    teams = sorted(set(x.posteam) | set(x.defteam))
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    w = np.where(x.season == season - 1, PRIOR_W, 1.0)
    o, d, y = x.posteam.map(ix).values, x.defteam.map(ix).values, x.epa.values
    A, b = np.zeros((2 * n, 2 * n)), np.zeros(2 * n)
    np.add.at(A, (o, o), w); np.add.at(A, (d + n, d + n), w)
    np.add.at(A, (o, d + n), w); np.add.at(A, (d + n, o), w)
    r = (y - np.average(y, weights=w)) * w
    np.add.at(b, o, r); np.add.at(b, d + n, r)
    sol = np.linalg.solve(A + RIDGE * np.eye(2 * n), b)
    return {t: sol[ix[t]] - sol[ix[t] + n] for t in teams}


def model_home_margin(R: dict, away: str, home: str) -> float:
    return PLAYS * (R.get(home, 0.0) - R.get(away, 0.0)) + HFA


# ---------------------------------------------------------------------------- splits

def fetch_archive(date: str) -> dict:
    """Spread splits from ScoresAndOdds' page for a past date: {game: capture}."""
    req = urllib.request.Request(f"{nm.SPLITS_URL}?date={date}", headers={"User-Agent": "Mozilla/5.0"})
    page = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
    out = {}
    for card in re.split(r'<div class="trend-card consensus ', page)[1:]:
        m = re.match(r"consensus-table-(\w+)--", card)
        teams = re.findall(r'class="team-flag" (\w+)""', card)
        if not m or m.group(1) != "spread" or len(teams) < 2:
            continue
        p = [int(v) for v in re.findall(r'class="percentage-[ab]"[^>]*>(\d+)%', card)]
        if len(p) < 4:
            continue
        sides = re.search(r'trend-graph-sides">\s*<strong>([\s\S]*?)</strong>\s*<span>% of Bets</span>', card)
        ln = re.search(r"\(([+-]?[\d.]+)\)", sides.group(1)) if sides else None
        away, home = (nm.SPLITS_TEAM_FIX.get(t, t) for t in teams[:2])
        out[f"{away}@{home}"] = {"line": float(ln.group(1)) if ln else None,
                                 "bets_a": p[0], "bets_b": p[1], "money_a": p[2], "money_b": p[3]}
    return out


def week_splits(splits_dir: Path, season: int, week: int, games: pd.DataFrame, live: bool) -> dict:
    """Spread capture per game: the last one before kickoff. Backfills finished games that were never
    captured (written back to the week file so it only happens once)."""
    path = splits_dir / f"{season}-w{week:02d}.json"
    snap = (json.loads(path.read_text(encoding="utf-8")) if path.exists()
            else {"season": season, "week": week, "source": nm.SPLITS_URL, "games": {}})
    out = {}
    for game, g in snap["games"].items():
        caps = g.get("spread") or []
        kick = g.get("kickoff_utc", "")[:16]
        pre = [c for c in caps if c.get("at") == "backfill" or not kick or c["at"][:16] <= kick] or caps
        if pre:
            out[game] = pre[-1]
    finished = games[games.home_score.notna()]
    missing = [k for k in finished.key if k not in out]
    if missing:
        sunday = games.gameday.mode().iloc[0]
        try:
            arch = fetch_archive(sunday)
        except Exception as exc:
            print(f"[warn] archive splits for week {week} failed: {exc}", file=sys.stderr)
            arch = {}
        added = 0
        for k in missing:
            if k in arch:
                out[k] = {"at": "backfill", **arch[k]}
                kick = finished.loc[finished.key == k, "kickoff_utc"].iloc[0]
                snap["games"].setdefault(k, {"kickoff_utc": kick})["spread"] = [out[k]]
                added += 1
        if added:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(snap, indent=1) + "\n", encoding="utf-8")
            print(f"Week {week}: backfilled {added} games from the ScoresAndOdds archive")
    if live:  # upcoming games with no capture yet (local runs)
        need = [k for k in games[games.home_score.isna()].key if k not in out]
        if need:
            try:
                for r in nm.fetch_splits():
                    if r["market"] == "spread" and r["game"] in need:
                        out[r["game"]] = {k: r[k] for k in ("line", "bets_a", "bets_b", "money_a", "money_b")}
            except Exception as exc:
                print(f"[warn] live splits failed: {exc}", file=sys.stderr)
    return out


def espn_scores(season: int, week: int) -> dict:
    """Live/final scores for the current week from ESPN: {game: {state, detail, away_score, home_score}}."""
    try:
        board = nm._get_json(f"{nm.ESPN_SCOREBOARD}?{nm.espn_week_query(week)}&dates={season}")
    except Exception as exc:
        print(f"[warn] ESPN scoreboard failed: {exc}", file=sys.stderr)
        return {}
    out = {}
    for e in board.get("events", []):
        st = e.get("status", {}).get("type", {})
        comp = e["competitions"][0]
        side = {c["homeAway"]: c for c in comp["competitors"]}
        ab = lambda c: nm.ESPN_TEAM_FIX.get(c["team"]["abbreviation"], c["team"]["abbreviation"])
        out[f"{ab(side['away'])}@{ab(side['home'])}"] = {
            "state": st.get("state"), "detail": st.get("shortDetail", ""),
            "away_score": float(side["away"].get("score") or 0), "home_score": float(side["home"].get("score") or 0)}
    return out


# ---------------------------------------------------------------------------- picks

def build(season: int, sched: pd.DataFrame, splits_dir: Path, live: bool, refresh: bool) -> tuple[list, int]:
    g = sched[(sched.season == season) & (sched.game_type == "REG")].copy()
    g["key"] = g.away_team + "@" + g.home_team
    g["kickoff_utc"] = (pd.to_datetime(g.gameday + " " + g.gametime).dt.tz_localize("America/New_York")
                        .dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    open_weeks = g[g.home_score.isna()].week
    cur = int(open_weeks.min()) if len(open_weeks) else int(g.week.max())
    pbp = load_pbp(season, refresh)
    live_scores = espn_scores(season, cur)
    rows = []
    for week in range(1, cur + 1):
        wg = g[g.week == week]
        sp = week_splits(splits_dir, season, week, wg, live and week == cur)
        R = team_ratings(pbp, season, week)
        for r in wg.itertuples():
            s = sp.get(r.key)
            if not s or s.get("line") is None or s["bets_a"] == s["bets_b"]:
                continue
            away_line = float(s["line"])                 # points the away team gets (+) or gives (-)
            pub_away = s["bets_a"] > s["bets_b"]
            pub_pct = s["bets_a"] if pub_away else s["bets_b"]
            pub_money = s["money_a"] if pub_away else s["money_b"]
            home_line = away_line                        # home favored by this many
            edge = model_home_margin(R, r.away_team, r.home_team) - home_line  # + = stats like home
            stats_away = edge < 0
            agree = (stats_away == pub_away) and abs(edge) >= AGREE_MIN
            big = pub_pct >= PUBLIC_MIN
            if not (big or agree):
                continue
            fade_team = r.home_team if pub_away else r.away_team
            fade_line = -away_line if pub_away else away_line
            pub_team = r.away_team if pub_away else r.home_team
            # result: final from nflverse, else live/final from ESPN for the current week
            state, detail, a_sc, h_sc = "pre", "", None, None
            if pd.notna(r.home_score):
                state, a_sc, h_sc, detail = "post", float(r.away_score), float(r.home_score), "Final"
            elif r.key in live_scores and live_scores[r.key]["state"] != "pre":
                ls = live_scores[r.key]
                state, detail, a_sc, h_sc = ls["state"], ls["detail"], ls["away_score"], ls["home_score"]
            ml = r.home_moneyline if pub_away else r.away_moneyline   # fade side's moneyline
            ml = None if pd.isna(ml) else int(ml)
            fade_dog = fade_line > 0
            result, cover, su = None, None, None
            if a_sc is not None:
                margin = a_sc - h_sc + away_line                 # > 0: away covered
                cover = margin if not pub_away else -margin       # from the fade side's view
                if state == "post":
                    result = "P" if margin == 0 else ("W" if cover > 0 else "L")
                    diff = (h_sc - a_sc) if pub_away else (a_sc - h_sc)   # fade side's straight-up margin
                    su = "P" if diff == 0 else ("W" if diff > 0 else "L")
            rows.append({
                "week": week, "game": r.key, "kickoff": r.kickoff_utc, "away": r.away_team, "home": r.home_team,
                "public": f"{pub_team} {(-away_line if not pub_away else away_line):+g}",
                "public_pct": pub_pct, "public_money": pub_money,
                "fade": f"{fade_team} {fade_line:+g}", "stats_edge": round(abs(edge), 1),
                "stats_side": "public" if stats_away == pub_away else "fade",
                "rules": [k for k, on in (("70", big), ("agree", agree)) if on],
                "state": state, "detail": detail,
                "score": None if a_sc is None else f"{r.away_team} {a_sc:g} – {r.home_team} {h_sc:g}",
                "cover": None if cover is None else round(cover, 1), "result": result,
                "dog": fade_dog, "ml": ml, "su": su,
                "source": "archive" if s.get("at") == "backfill" else "capture"})
    return rows, cur


def record(rows: list) -> dict:
    w = sum(r["result"] == "W" for r in rows); l = sum(r["result"] == "L" for r in rows)
    p = sum(r["result"] == "P" for r in rows)
    return {"w": w, "l": l, "p": p, "pct": round(100 * w / (w + l), 1) if w + l else None,
            "units": round(w * WIN - l * STAKE, 2)}


def ml_profit(ml: int) -> float:
    return STAKE * ml / 100 if ml > 0 else STAKE * 100 / -ml


def ml_record(rows: list) -> dict:
    """Straight-up record and $100 moneyline result for fade-side underdogs."""
    d = [r for r in rows if r["dog"] and r["su"] and r["ml"]]
    w = [r for r in d if r["su"] == "W"]; l = [r for r in d if r["su"] == "L"]
    units = sum(ml_profit(r["ml"]) for r in w) - STAKE * len(l)
    return {"w": len(w), "l": len(l), "p": sum(r["su"] == "P" for r in d),
            "pct": round(100 * len(w) / (len(w) + len(l)), 1) if w or l else None, "units": round(units, 2),
            "avg_ml": round(sum(r["ml"] for r in d) / len(d)) if d else None}


def summary(rows: list, cur: int) -> dict:
    done = [r for r in rows if r["result"]]
    rules = {"70": [r for r in done if "70" in r["rules"]], "agree": [r for r in done if "agree" in r["rules"]],
             "both": [r for r in done if len(r["rules"]) == 2]}
    weekly = []
    run = {"70": 0.0, "agree": 0.0}
    for wk in range(1, cur + 1):
        row = {"week": wk}
        for k in run:
            rec = record([r for r in rules[k] if r["week"] == wk])
            run[k] += rec["units"]
            row[k] = round(run[k], 2); row[k + "_rec"] = f"{rec['w']}-{rec['l']}" + (f"-{rec['p']}" if rec["p"] else "")
            row[k + "_n"] = rec["w"] + rec["l"] + rec["p"]
        if any(row[k + "_n"] for k in run):
            weekly.append(row)
    return ({k: record(v) for k, v in rules.items()} | {"ml_" + k: ml_record(v) for k, v in rules.items()}
            | {"weekly": weekly})


def agree_buckets(season: int, sched: pd.DataFrame, splits_dir: Path, cur: int) -> list:
    """Fade record for every finished game where stats sided with the public, by how strongly.
    This is the breakdown behind the 'Stats agree 3+' rule."""
    g = sched[(sched.season == season) & (sched.game_type == "REG") & sched.home_score.notna()].copy()
    g["key"] = g.away_team + "@" + g.home_team
    g["kickoff_utc"] = ""
    pbp = load_pbp(season)
    edges = [(0, 1.5, "Under 1.5"), (1.5, 3, "1.5 – 3"), (3, 99, "3+")]
    tally = {lab: [0, 0, 0] for *_, lab in edges}
    for week in sorted(g.week.unique()):
        wg = g[g.week == week]
        sp = week_splits(splits_dir, season, int(week), wg, False)
        R = team_ratings(pbp, season, int(week))
        for r in wg.itertuples():
            s = sp.get(r.key)
            if not s or s.get("line") is None or s["bets_a"] == s["bets_b"]:
                continue
            pub_away = s["bets_a"] > s["bets_b"]
            edge = model_home_margin(R, r.away_team, r.home_team) - float(s["line"])
            if (edge < 0) != pub_away:
                continue
            margin = r.away_score - r.home_score + float(s["line"])
            res = 2 if margin == 0 else (0 if (margin > 0) != pub_away else 1)  # 0 = fade win
            for lo, hi, lab in edges:
                if lo <= abs(edge) < hi:
                    tally[lab][res] += 1
    return [{"bucket": lab, "w": v[0], "l": v[1], "p": v[2],
             "pct": round(100 * v[0] / (v[0] + v[1]), 1) if v[0] + v[1] else None} for lab, v in tally.items()]


# ---------------------------------------------------------------------------- prop unders

# Each trend: which stat, and which props qualify. "proj" is our model's projection from the
# pre-kickoff snapshot (data/snapshots); props with no snapshot have no projection.
PROP_TRENDS = [
    {"key": "qbr", "title": "QB rushing unders", "pill": "QB rush", "stat": "rushing_yards", "unit": "rush yds",
     "keep": lambda r: r.get("pos") == "QB",
     "desc": "the under on a quarterback's rushing yards. The public loves the over on running QBs, and most games come in under."},
    # Receiving unders (model projects under) went 278-227 thru Wk 8 but were taken off the page 11/6
    # (thin edge, ~break-even at -120). Re-add the entry here to bring the list back.
]
ESPN_STAT_TYPE = {"rushing_yards": "12", "receiving_yards": "13", "passing_yards": "8"}


def _players() -> pd.DataFrame:
    p = pd.read_csv(nm.CACHE_DIR / "players.csv", usecols=["gsis_id", "display_name", "position", "latest_team", "espn_id"],
                    low_memory=False)
    return p.dropna(subset=["gsis_id"])


def _utc(k) -> str:
    """Snapshot kickoffs are Eastern time like '2026-11-08 16:25'; return ISO UTC."""
    try:
        t = pd.Timestamp(str(k))
        t = t.tz_localize("America/New_York") if t.tzinfo is None else t
        return t.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ""


def espn_prop_board(season: int, week: int) -> dict:
    """DraftKings main yardage props (via ESPN) for games not yet started: {(gsis_id, stat): {...}}.
    ESPN lists a prop twice when DraftKings posts both the over and the under; listed once means a side
    is missing, so 'both' = under available."""
    from collections import defaultdict
    board = nm._get_json(f"{nm.ESPN_SCOREBOARD}?{nm.espn_week_query(week)}&dates={season}")
    id_map = nm.espn_to_gsis(False)
    pl = _players().set_index("gsis_id")
    types = {v: k for k, v in ESPN_STAT_TYPE.items()}
    out = {}
    for e in board.get("events", []):
        if e.get("status", {}).get("type", {}).get("state") != "pre":
            continue
        comp = e["competitions"][0]
        side = {c["homeAway"]: nm.ESPN_TEAM_FIX.get(c["team"]["abbreviation"], c["team"]["abbreviation"])
                for c in comp["competitors"]}
        seen = defaultdict(list)
        page, pages = 1, 1
        while page <= pages:
            try:
                d = nm._get_json(nm.ESPN_PROPS.format(eid=e["id"], page=page))
            except Exception as exc:
                print(f"[warn] props {e['id']}: {exc}", file=sys.stderr)
                break
            for it in d.get("items", []):
                stat = types.get(it.get("type", {}).get("id"))
                m = re.search(r"/athletes/(\d+)", it.get("athlete", {}).get("$ref", ""))
                cur = it.get("current", {}).get("target", {}).get("value")
                if stat and m and cur is not None:
                    seen[(m.group(1), stat)].append((float(cur), it.get("open", {}).get("target", {}).get("value")))
            pages = d.get("pageCount", 1)
            page += 1
        for (eid, stat), entries in seen.items():
            gid = id_map.get(eid)
            if not gid or gid not in pl.index:
                continue
            line = entries[-1][0]
            team = pl.loc[gid, "latest_team"]
            out[(gid, stat)] = {"player": pl.loc[gid, "display_name"], "player_id": gid, "pos": pl.loc[gid, "position"],
                                "team": team, "opp": side["home"] if team == side["away"] else side["away"],
                                "game": f"{side['away']}@{side['home']}", "kickoff": e.get("date", ""), "stat": stat,
                                "line": line, "open": entries[-1][1],
                                "both": sum(1 for x in entries if x[0] == line) >= 2}
    return out


def _snap_props(snap_dir: Path, season: int, wk: int) -> list:
    path = snap_dir / f"{season}-w{wk:02d}.json"
    if not path.exists():
        return []
    props = json.loads(path.read_text(encoding="utf-8")).get("props", {})
    return [p for p in (props.values() if isinstance(props, dict) else props) if isinstance(p, dict)]


def prop_trends(season: int, cur: int, snap_dir: Path, stats: pd.DataFrame) -> list:
    st = stats[stats.season == season]
    try:
        board = espn_prop_board(season, cur)
    except Exception as exc:
        print(f"[warn] DraftKings prop board failed: {exc}", file=sys.stderr)
        board = {}
    out = []
    for T in PROP_TRENDS:
        stat = T["stat"]
        graded, skipped = [], set()
        actual = {(r.player_id, r.week): getattr(r, stat) for r in st[["player_id", "week", stat]].itertuples()}
        for wk in range(1, cur + 1):
            rows = {}
            for p in _snap_props(snap_dir, season, wk):
                if p.get("stat") != stat:
                    continue
                r = {"week": wk, "player": p["player"], "player_id": p["player_id"], "pos": p.get("pos"), "team": p["team"],
                     "opp": p["opp"], "game": p["game"], "line": float(p["line"]), "proj": p.get("proj"),
                     "open": None, "kickoff": _utc(p.get("kickoff", ""))}
                if wk == cur and (p["player_id"], stat) in board:   # game not started: use the live line
                    b = board[(p["player_id"], stat)]
                    if not b["both"]:
                        if T["keep"]({**r, "line": b["line"]}):
                            skipped.add(p["player"])
                        continue
                    r.update(line=b["line"], open=b["open"])
                if T["keep"](r):
                    rows[p["player_id"]] = r
            if wk == cur:   # live props with no snapshot yet (only trends that don't need a projection)
                for (gid, s), b in board.items():
                    if s == stat and gid not in rows:
                        r = {**b, "week": wk, "proj": None}
                        if T["keep"](r):
                            if b["both"]:
                                rows[gid] = r
                            else:
                                skipped.add(b["player"])
            for r in rows.values():
                a = actual.get((r["player_id"], wk))
                act = float(a) if a is not None and pd.notna(a) else None
                r["actual"] = act
                r["result"] = None if act is None else ("P" if act == r["line"] else ("W" if act < r["line"] else "L"))
                graded.append(r)
        done = [g for g in graded if g["result"]]
        weekly, run = [], 0.0
        for wk in range(1, cur + 1):
            rw = record([g for g in done if g["week"] == wk])
            run += rw["units"]
            if rw["w"] + rw["l"] + rw["p"]:
                weekly.append({"week": wk, "rec": f"{rw['w']}-{rw['l']}" + (f"-{rw['p']}" if rw["p"] else ""),
                               "units": rw["units"], "cum": round(run, 2)})
        week_rows = [{k: v for k, v in g.items() if k != "player_id"} for g in graded if g["week"] == cur]
        out.append({"key": T["key"], "title": T["title"], "pill": T["pill"], "unit": T["unit"], "desc": T["desc"],
                    "week": week_rows, "summary": record(done), "weekly": weekly, "skipped": sorted(skipped)})
    return out


# ---------------------------------------------------------------------------- page

def write_page(path: Path, rows: list, cur: int, summ: dict, buckets: list, season: int,
               league: str = "NFL", url: str | None = None, other: tuple = ("College football", "college/"),
               cfb: dict | None = None, props: list | None = None):
    data = {"season": season, "week": cur, "updated": nm._now_et(), "rows": rows, "summary": summ,
            "buckets": buckets, "rules": {"public_min": PUBLIC_MIN, "agree_min": AGREE_MIN},
            "league": league, "other": {"label": other[0], "href": other[1]}}
    if cfb:
        data["cfb"] = cfb
    if props:
        data["props"] = props
    page = (FADES_TEMPLATE.replace("__WEEK__", str(cur)).replace("__URL__", url or nm.SITE_URL + "fades/")
            .replace("__LEAGUE__", league)
            .replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    print(f"Fade the Public ({league}): week {cur}, {len([r for r in rows if r['week'] == cur])} picks -> {path}")


FADES_TEMPLATE = (Path(__file__).parent / "fades_template.html").read_text(encoding="utf-8") \
    if (Path(__file__).parent / "fades_template.html").exists() else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--out", default="site/fades/index.html")
    ap.add_argument("--json", help="also write the picks as JSON here")
    ap.add_argument("--no-live", action="store_true", help="don't read live splits for uncaptured games")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--snapshot-dir", default="data/snapshots", help="pre-kickoff prop snapshots (QB rushing unders)")
    ap.add_argument("--cfb-json", help="college data written by fades_cfb.py --data-out (adds the College tab)")
    a = ap.parse_args()
    stats, sched, season = nm.load_data(a.refresh)
    splits_dir = Path(a.splits_dir)
    rows, cur = build(season, sched, splits_dir, not a.no_live, a.refresh)
    summ = summary(rows, cur)
    buckets = agree_buckets(season, sched, splits_dir, cur)
    cfb = None
    if a.cfb_json and Path(a.cfb_json).exists():
        cfb = json.loads(Path(a.cfb_json).read_text(encoding="utf-8"))
    props = None
    try:
        props = prop_trends(season, cur, Path(a.snapshot_dir), stats)
        for t in props:
            q = t["summary"]
            print(f"  {t['title']}: {q['w']}-{q['l']}-{q['p']}  ${q['units']:+,.2f} at $100; this week {len(t['week'])} listed, {len(t['skipped'])} skipped (no under)")
    except Exception as exc:
        print(f"[warn] prop unders failed: {exc}", file=sys.stderr)
    write_page(Path(a.out), rows, cur, summ, buckets, season, cfb=cfb, props=props)
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": rows, "summary": summ, "buckets": buckets}, indent=1), encoding="utf-8")
    for k, lab in (("70", "Fade 70%+"), ("agree", "Stats agree 3+"), ("both", "Both")):
        s = summ[k]
        print(f"  {lab:15} {s['w']}-{s['l']}-{s['p']}  ${s['units']:+,.2f} at $100")
    for k, lab in (("70", "Fade 70%+"), ("agree", "Stats agree 3+"), ("both", "Both")):
        s = summ["ml_" + k]
        print(f"  ML dogs {lab:15} {s['w']}-{s['l']} SU  ${s['units']:+,.2f} at $100 (avg ML {s['avg_ml']})")
    for b in buckets:
        print(f"  agree {b['bucket']:9} fade {b['w']}-{b['l']}-{b['p']}")


if __name__ == "__main__":
    main()
