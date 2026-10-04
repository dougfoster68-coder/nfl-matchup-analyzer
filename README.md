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

- **Automatic picks** (`autobet`): on Thursday and Monday nights, in the 3 hours before kickoff,
  the GitHub Action places $200 of paper bets on each primetime game. It's run once per game.
  The plan spreads the money so no single pick controls it:
  - **Up to 5 different players**, one prop each. A/B-rated props come first, then C-rated props
    the model still leans at least 55% on. Trap-flagged props never qualify. Small lines (0.5, 4.5 …) are allowed.
  - **Weighted by rating**, with no player getting more than 30% of the $200. Anything over the cap stays in the bankroll.
  - **Alt-line split** on A/B overs: 50% on the posted line, 30% on a safer lower alt line,
    and 20% on a plus-money higher alt line, using DraftKings' milestone ladder. Unders stay on the main line.
  - **15% on a parlay** of the top 2–3 players.
  - The free feed has no alt-line prices, so alt odds are estimated. The estimate treats the posted line as the market's
    middle outcome, uses the player's game-to-game spread, and adds a normal sportsbook margin.
  - `python nfl_matchup.py backtest --week N` compares this plan with the original one on that week's
    Thursday and Monday night games.
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

## Betting splits (% of bets vs % of money)

Every refresh also saves the consensus betting splits for each game's spread, total and moneyline
(from ScoresAndOdds, pooled across sportsbooks) to the `data` branch as `splits/<season>-w<week>.json`.
A new capture is stored only when the numbers change, and a game stops updating at kickoff, so the last
entry is the closing split.

When a side's share of the money is well above its share of the bets, fewer people are betting bigger
amounts on it, which is the classic sign of confident or sharp money. The report shows:

- **Gap:** money % minus bets % on the big-bet side.
- **Ratio:** the average bet size on that side divided by the average bet size on the other side.
  For example, 30% of bets with 60% of money works out to 3.5x.
- **Sharp lean:** a gap of 10+ points with a ratio of 2x or more.

After games finish, `--grade` checks whether the big-bet side won, broken down by gap size, by market,
and by reverse line movement (the line moved toward the big-bet side while most tickets were on the other side).

```
python nfl_matchup.py splits                       # show current splits and save a capture
python nfl_matchup.py splits --grade               # record of the big-bet side, all captured weeks
python nfl_matchup.py splits --grade --week 4
```

On the dashboard, each upcoming game's card shows the big-bet side for the spread, total and moneyline (% of bets vs % of money),
tagged **SHARP** or **RLM** when it qualifies. The "Sharp money vs the public" panel lists every upcoming game line with a 10+ point
money gap. Player props get a note when big money is on that game's total.

## Live results

While games are on, ESPN's live box scores grade everything on the fly:

- **Our ratings:** the "Results as games finish" section and the started games' cards show each rated prop's actual yards vs the
  line, marked Hit, Miss or Live, plus a running hit %.
- **Paper bets:** open paper bets show at the top of the page, leg by leg, and the header shows **Live P/L**. The bankroll itself
  still settles officially from the next-day box scores.

## How the math works

- **Defense rank**: 1 = allows the fewest yards to that position, 32 = allows the most.
- **Matchup %**: the defense's yards allowed per game divided by the league average, pulled
  toward average by 4 games so a small sample doesn't swing it too far.
- **Projection**: the player's L10 average × the matchup factor.
- **P(over)**: a normal distribution built from the projection and the player's game-to-game variance.
- **Lean**: shown only when the probability clears the -110 break-even of 52.4% by 3 points or more.

Not included: injuries, weather, snap-share changes, coaching changes. Check injury
reports before betting, especially for players flagged "missed last game".
