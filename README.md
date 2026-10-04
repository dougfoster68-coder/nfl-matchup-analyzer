# NFL Matchup Analyzer

**Live dashboard:** https://dougfoster68-coder.github.io/nfl-matchup-analyzer/
A GitHub Action rebuilds it with fresh DraftKings lines about every 15 minutes.

Compares each offensive player's last 10 games against what their opponent's
defense has allowed to that position over its own last 10 games, then gives
matchup-adjusted yardage projections and over/under probabilities.

Data: nflverse (free). Downloads are cached in `cache/` for 6 hours; use `--refresh` to force.

## Commands

```
python nfl_matchup.py week                                  # every game this week, QB/RB/WR/TE yardage
python nfl_matchup.py week --lines lines_template.csv       # score your sportsbook lines
python nfl_matchup.py player "Ja'Marr Chase" --line 84.5    # one player, full game log
python nfl_matchup.py slate --position WR --stat receiving_yards
python nfl_matchup.py defense KC                            # what a defense allows by position
```

Options: `--games 10` (lookback), `--hit-rate 0.7`, `--week N`, `--include-played`.

## Live sportsbook lines

By default `week` pulls live DraftKings lines from ESPN's public odds feed. That includes
main yardage lines, opening lines (for line movement), and DK's alt "N+" milestone ladders.
No API key is needed.

```
python nfl_matchup.py week --live 5
```

Live mode refreshes every 5 minutes. Each refresh reprints the report and rewrites
`week<N>_dashboard.html`, which reloads itself on the same interval. Keep the dashboard
open in your browser and it stays current.

Other line sources: `--lines lines_template.csv` uses your own lines, and setting
`$env:ODDS_API_KEY` uses consensus lines from https://the-odds-api.com.

## Paper bankroll

A $1,000 fake-money account tracked in `bets.json` and shown on the dashboard.

- **Automatic picks** (`autobet`): on Thursdays, in the 3 hours before kickoff, the GitHub Action
  places $200 of paper bets on that night's game. It's run once per game.
  - Only A/B-rated props with no trap flags qualify, at most 4 picks.
  - About 80% of the $200 goes to straight bets weighted by rating, and 20% to a 2-leg parlay of the top two.
  - If nothing qualifies, it passes and bets nothing.
- **Grading**: every refresh grades bets against final box scores. A player who doesn't play voids the leg.
- **Manual bets**:
  ```
  python nfl_matchup.py bet --stake 50 --leg "Juwan Johnson|receiving_yards|over|40.5"
  python nfl_matchup.py bankroll
  ```
  Repeat `--leg` to make a parlay. Run `git pull` first, because the Action commits bets too.

## Weekly report cards

Every refresh saves each rated prop's latest pre-kickoff rating, pick and line to the `data` branch
(`snapshots/<season>-w<week>.json`). Once a prop's game kicks off, its entry stops changing.
After games end, the props are graded against box scores:

- **Game by game:** of our top 2, 3 … 10 rated props in each game, how many hit.
- **Whole week:** the same tiers combined across every game. For example, the top 10 from each of
  16 games is 160 props, and the report shows how many of those hit.
- **By grade:** the hit % for A, B, C and D props.

Trap-flagged props are left out. Voids (the player didn't play) and pushes are shown but don't count toward hit %.

Report pages are at `/reports/` on the site. Every **Tuesday at 11 AM ET** the "Weekly report card"
workflow also posts the completed week as a GitHub issue, which emails the repo owner. To rerun it, use
Actions → Weekly report card → Run workflow, optionally with a week number.

```
python nfl_matchup.py report --snapshot-dir data/snapshots --markdown report.md
```

## How the math works

- **Defense rank**: 1 = allows the fewest yards to that position, 32 = allows the most.
- **Matchup %**: the defense's yards allowed per game divided by the league average, pulled
  toward average by 4 games so a small sample doesn't swing it too far.
- **Projection**: the player's L10 average × the matchup factor.
- **P(over)**: a normal distribution built from the projection and the player's game-to-game variance.
- **Lean**: shown only when the probability clears the -110 break-even of 52.4% by 3 points or more.

Not included: injuries, weather, snap-share changes, coaching changes. Check injury
reports before betting, especially for players flagged "missed last game".
