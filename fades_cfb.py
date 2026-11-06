"""Fade the Public, college football data (the College tab on site/fades/).

Same two rules as the NFL page (fades.py):
  * 80%+     - 80% or more of spread tickets on one side -> take the other side (NFL uses 70%).
  * Stats 3+ - our stats model agrees with the public side by 3+ points vs the line -> fade anyway.

College differences:
  * Scores and schedule come from ESPN (FBS games, groups=80).
  * The stats model rates teams from opponent-adjusted point margins (capped at 28, home field 2.5,
    neutral sites 0), using only games played before each week plus a discounted 2025 season.
    College play-by-play isn't readily available, and margin ratings are the standard college approach.
  * Splits come from ScoresAndOdds' NCAAF consensus page. Each run saves the current week's numbers to
    data/splits_cfb (only changes); finished games with no capture are backfilled once from the
    archive page for that week.

    python fades_cfb.py --splits-dir data/splits_cfb --data-out cache/cfb_fades.json
    python fades.py ... --cfb-json cache/cfb_fades.json      (adds the College tab to site/fades/)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import fades
import nfl_matchup as nm

SB = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
SPLITS_URL = "https://www.scoresandodds.com/ncaaf/consensus-picks"
HFA = 2.5
CAP = 28
PRIOR_W = 0.35
RIDGE = 3.0
PUBLIC_MIN = 80   # college: public rule fires at 80%+ (70% was too weak in the 2026 backtest)
# ScoresAndOdds name -> ESPN location name
NAME_FIX = {"Appalachian State": "App State", "Grambling State": "Grambling", "LIU": "Long Island University",
            "Louisiana-Monroe": "UL Monroe", "Miami (FL)": "Miami", "Nicholls State": "Nicholls",
            "Southeastern Louisiana": "SE Louisiana", "Tennessee-Martin": "UT Martin", "UMass": "Massachusetts",
            "University at Albany": "UAlbany"}


def norm(x) -> str:
    return unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().strip()


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=40).read()


# ---------------------------------------------------------------------------- games

def _week_games(year: int, week: int) -> list:
    d = json.loads(_get(f"{SB}?week={week}&seasontype=2&groups=80&limit=400&dates={year}"))
    out = []
    for ev in d.get("events", []):
        c = ev["competitions"][0]
        side = {x["homeAway"]: x for x in c["competitors"]}
        st = ev["status"]["type"]
        ml = {}
        for o in c.get("odds") or []:
            for k in ("home", "away"):
                v = (o.get(f"{k}TeamOdds") or {}).get("moneyLine")
                if v is not None:
                    ml[k] = int(v)
            break
        live = st.get("state") != "pre"
        out.append({"season": year, "week": week, "date": ev["date"], "state": st.get("state"), "detail": st.get("shortDetail", ""),
                    "neutral": bool(c.get("neutralSite")), "home": norm(side["home"]["team"]["location"]),
                    "away": norm(side["away"]["team"]["location"]),
                    "home_score": float(side["home"].get("score") or 0) if live else None,
                    "away_score": float(side["away"].get("score") or 0) if live else None,
                    "home_ml": ml.get("home"), "away_ml": ml.get("away")})
    return out


def load_games(year: int, refresh: bool) -> tuple[pd.DataFrame, dict]:
    """All regular-season games for a season, cached; the current and previous week are always re-read."""
    cal = json.loads(_get(f"{SB}?dates={year}&groups=80&limit=1"))["leagues"][0]["calendar"]
    reg = next(c for c in cal if c.get("value") == "2")["entries"]
    weeks = {int(e["value"]): (e["startDate"][:10], e["endDate"][:10]) for e in reg}
    path = nm.CACHE_DIR / f"cfb_games_{year}.csv"
    old = pd.read_csv(path) if path.exists() and not refresh else None
    today = dt.date.today().isoformat()
    rows = []
    for wk, (start, end) in weeks.items():
        cached = old[old.week == wk] if old is not None else None
        have = cached is not None and len(cached) > 0
        settled = have and (cached.state == "post").all() and \
            (dt.date.fromisoformat(end) + dt.timedelta(days=8)).isoformat() < today   # past week, all final
        future_fresh = have and start > today and time.time() - path.stat().st_mtime < 6 * 3600
        if settled or future_fresh:
            rows += cached.to_dict("records")
        else:
            try:
                rows += _week_games(year, wk)
            except Exception as exc:
                print(f"[warn] ESPN week {wk} failed: {exc}", file=sys.stderr)
                if cached is not None:
                    rows += cached.to_dict("records")
    g = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(path, index=False)
    return g, weeks


# ---------------------------------------------------------------------------- ratings

def ratings(G: pd.DataFrame, season: int, week: int) -> dict:
    x = G[(G.state == "post") & ((G.season == season - 1) | ((G.season == season) & (G.week < week)))]
    teams = sorted(set(x.home) | set(x.away))
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    w = np.where(x.season == season - 1, PRIOR_W, 1.0)
    y = np.clip(x.home_score - x.away_score, -CAP, CAP) - np.where(x.neutral, 0, HFA)
    A, b = np.zeros((n, n)), np.zeros(n)
    h, a = x.home.map(ix).values, x.away.map(ix).values
    np.add.at(A, (h, h), w); np.add.at(A, (a, a), w); np.add.at(A, (h, a), -w); np.add.at(A, (a, h), -w)
    np.add.at(b, h, w * y); np.add.at(b, a, -w * y)
    return dict(zip(teams, np.linalg.solve(A + RIDGE * np.eye(n), b)))


# ---------------------------------------------------------------------------- splits

def fetch_splits(date: str | None = None) -> dict:
    page = _get(SPLITS_URL + (f"?date={date}" if date else "")).decode("utf-8", "ignore")
    out = {}
    for c in re.split(r'<div class="trend-card consensus ', page)[1:]:
        if not c.startswith("consensus-table-spread"):
            continue
        names = re.findall(r'<span class="team-name">\s*<span>([^<]+)</span>', c)
        p = [int(v) for v in re.findall(r'class="percentage-[ab]"[^>]*>(\d+)%', c)]
        s = re.search(r'trend-graph-sides">\s*<strong>([\s\S]*?)</strong>\s*<span>% of Bets</span>', c)
        ln = re.search(r"\(([+-]?[\d.]+)\)", s.group(1)) if s else None
        kick = re.search(r'data-value="([^"]+)"', c)
        if len(names) < 2 or len(p) < 4:
            continue
        away, home = (norm(NAME_FIX.get(v.strip(), v.strip())) for v in names[:2])
        out[f"{away}@{home}"] = {"kickoff_utc": kick.group(1) if kick else "", "line": float(ln.group(1)) if ln else None,
                                 "bets_a": p[0], "bets_b": p[1], "money_a": p[2], "money_b": p[3]}
    return out


def _saturday(start: str) -> str:
    s = dt.date.fromisoformat(start)
    return (s + dt.timedelta((5 - s.weekday()) % 7)).isoformat()


def week_splits(splits_dir: Path, season: int, week: int, wg: pd.DataFrame, live: dict | None, start: str) -> dict:
    path = splits_dir / f"{season}-w{week:02d}.json"
    snap = (json.loads(path.read_text(encoding="utf-8")) if path.exists()
            else {"season": season, "week": week, "source": SPLITS_URL, "games": {}})
    keys = set(wg.away + "@" + wg.home)
    changed = False
    if live:  # save the current numbers for games that haven't kicked off
        stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes")
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M")
        for k, r in live.items():
            if k not in keys or not r["kickoff_utc"] or r["kickoff_utc"][:16] <= now:
                continue
            g = snap["games"].setdefault(k, {"kickoff_utc": r["kickoff_utc"]})
            caps = g.setdefault("spread", [])
            cap = {x: r[x] for x in ("line", "bets_a", "bets_b", "money_a", "money_b")}
            if not caps or {x: caps[-1][x] for x in cap} != cap:
                caps.append({"at": stamp, **cap}); changed = True
    out = {}
    for k, g in snap["games"].items():
        caps = g.get("spread") or []
        kick = g.get("kickoff_utc", "")[:16]
        pre = [c for c in caps if c.get("at") == "backfill" or not kick or c["at"][:16] <= kick] or caps
        if pre:
            out[k] = pre[-1]
    missing = [k for k in (wg[wg.state == "post"].away + "@" + wg[wg.state == "post"].home) if k not in out]
    if missing:
        try:
            arch = fetch_splits(_saturday(start))
        except Exception as exc:
            print(f"[warn] archive splits week {week} failed: {exc}", file=sys.stderr)
            arch = {}
        for k in missing:
            if k in arch:
                r = arch[k]
                out[k] = {"at": "backfill", **{x: r[x] for x in ("line", "bets_a", "bets_b", "money_a", "money_b")}}
                snap["games"].setdefault(k, {"kickoff_utc": r["kickoff_utc"]})["spread"] = [out[k]]
                changed = True
    if live:  # upcoming games: newest numbers
        for k, r in live.items():
            if k in keys and k not in out:
                out[k] = r
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snap, indent=1) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------------------- picks

def build(season: int, splits_dir: Path, refresh: bool) -> tuple[list, int]:
    cur_g, weeks = load_games(season, refresh)
    prev_path = nm.CACHE_DIR / f"cfb_games_{season - 1}.csv"
    prev = pd.read_csv(prev_path) if prev_path.exists() else load_games(season - 1, False)[0]
    G = pd.concat([prev, cur_g], ignore_index=True)
    open_weeks = cur_g[cur_g.state != "post"].week
    cur = int(open_weeks.min()) if len(open_weeks) else int(cur_g.week.max())
    try:
        live = fetch_splits()
    except Exception as exc:
        print(f"[warn] live splits failed: {exc}", file=sys.stderr)
        live = {}
    rows = []
    for week in sorted(w for w in weeks if w <= cur):
        wg = cur_g[cur_g.week == week]
        sp = week_splits(splits_dir, season, week, wg, live if week == cur else None, weeks[week][0])
        R = ratings(G, season, week)
        for r in wg.itertuples():
            key = f"{r.away}@{r.home}"
            s = sp.get(key)
            if not s or s.get("line") is None or s["bets_a"] == s["bets_b"]:
                continue
            away_line = float(s["line"])
            pub_away = s["bets_a"] > s["bets_b"]
            pub_pct = s["bets_a"] if pub_away else s["bets_b"]
            edge = R.get(r.home, 0.0) - R.get(r.away, 0.0) + (0 if r.neutral else HFA) - away_line
            stats_away = edge < 0
            agree = (stats_away == pub_away) and abs(edge) >= fades.AGREE_MIN
            big = pub_pct >= PUBLIC_MIN
            if not (big or agree):
                continue
            fade_team, pub_team = (r.home, r.away) if pub_away else (r.away, r.home)
            fade_line = -away_line if pub_away else away_line
            ml = r.home_ml if pub_away else r.away_ml
            ml = None if pd.isna(ml) else int(ml)
            state = r.state
            a_sc = None if pd.isna(r.away_score) or state == "pre" else float(r.away_score)
            h_sc = None if a_sc is None else float(r.home_score)
            result, cover = None, None
            if a_sc is not None:
                margin = a_sc - h_sc + away_line
                cover = -margin if pub_away else margin
                if state == "post":
                    result = "P" if margin == 0 else ("W" if cover > 0 else "L")
            rows.append({
                "week": week, "game": key, "kickoff": pd.Timestamp(r.date).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "away": r.away, "home": r.home,
                "public": f"{pub_team} {(away_line if pub_away else -away_line):+g}",
                "public_pct": pub_pct, "public_money": s["money_a"] if pub_away else s["money_b"],
                "fade": f"{fade_team} {fade_line:+g}", "stats_edge": round(abs(edge), 1),
                "stats_side": "public" if stats_away == pub_away else "fade",
                "rules": [k for k, on in (("70", big), ("agree", agree)) if on],
                "state": state, "detail": "Final" if state == "post" else r.detail,
                "score": None if a_sc is None else f"{r.away} {a_sc:g} – {r.home} {h_sc:g}",
                "cover": None if cover is None else round(cover, 1), "result": result,
                "dog": fade_line > 0, "ml": ml, "su": None, "neutral": bool(r.neutral)})
    return rows, cur


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-dir", default="data/splits_cfb")
    ap.add_argument("--data-out", default="cache/cfb_fades.json", help="college data for fades.py --cfb-json")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    season = nm.current_season(dt.date.today())
    rows, cur = build(season, Path(a.splits_dir), a.refresh)
    summ = fades.summary(rows, cur)
    out = Path(a.data_out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"season": season, "week": cur, "rows": rows, "summary": summ,
                               "rules": {"public_min": PUBLIC_MIN, "agree_min": fades.AGREE_MIN}}), encoding="utf-8")
    print(f"College: week {cur}, {len([r for r in rows if r['week'] == cur])} picks -> {out}")
    for k, lab in (("70", f"{PUBLIC_MIN}%+"), ("agree", "Stats 3+"), ("both", "Both")):
        s = summ[k]
        print(f"  {lab:9} {s['w']}-{s['l']}-{s['p']}  ${s['units']:+,.2f} at $100")


if __name__ == "__main__":
    main()
