# nflelo — unit-level Elo power ratings and weekly NFL spreads

Free data only (nflverse play-by-play + nfldata schedules, optional Pro Football
Reference pressure scrape). Python CLI, CSV output. No market lines in the
output by design — this produces your own number, independent of the book's.

```bash
pip install pandas pyarrow lxml           # lxml only needed for --pfr
python -m nflelo ratings                  # power ratings + unit splits -> CSV
python -m nflelo spreads                  # next unplayed week -> CSV
python -m nflelo spreads --week 5 --pfr   # specific week, with charted pressure
python -m nflelo backtest --from 2018     # how it has held up
python -m nflelo fit --holdout 2024       # re-tune (read the warning it prints)
```

First run downloads ~6 seasons of play-by-play (a few hundred MB) and caches it
in `~/.nflelo_cache`. Subsequent runs re-pull only the current season, so the
weekly refresh takes a few seconds.

## How the rating works

Four ratings per team, all in **EPA per play** — which is already points, so no
arbitrary scale conversion is needed until display time:

| rating | meaning |
|---|---|
| `pass_off`, `rush_off` | EPA/play generated above league average |
| `pass_def`, `rush_def` | EPA/play **allowed** above average (positive = bad) |

Each game, for each phase, the model predicts the offense's EPA/play as
`league_mean + offense_rating + defense_rating`, then pushes the residual into
both ratings with an Elo-style K-weighted update. That is Elo's mechanic —
expectation, surprise, weighted correction — but updated on opponent-adjusted
per-play efficiency instead of final score. A conventional single-number Elo
cannot give you the four unit splits, because one game result carries only one
number of information; this does.

Converting to a spread is then just arithmetic, because the ratings are already
in points:

```
margin = Σ_phase plays × (home_off + away_def)  −  Σ_phase plays × (away_off + home_def)
       + home field + rest + pressure + down-leverage + QB
```

Elo-style display numbers (`1500 + 25 × power_rating`) are in the ratings CSV
for readability, but the points column is the one that does the work.

## What happened to your metric list

Every requested metric is implemented. Most of them then got shrunk to almost
nothing by the fitting process, which is itself the finding:

| metric | where it lives | fitted weight | verdict |
|---|---|---|---|
| Pressures on defense | `press_def` EWMA, sacks + QB hits per dropback | 0.10 pts per pp | **kept** — improves out-of-sample RMSE |
| QB yards per attempt | replaced by QB EPA/dropback in `QBBook` | 0.25 of full delta | **kept** — biggest single lever |
| Sack percentage | inside pass-offense EPA + carried with the QB | — | absorbed |
| Third-down success | `down_leverage` = late-down EPA − early-down EPA | 0.20 | marginal |
| Turnover percentage | fumble-recovery luck stripped before the update | 0.50 | marginal |
| RB yards per rush | replaced by rush EPA/play | — | absorbed |
| Head coaching record | 4th-down go rate + pace (not W-L) | **0.0** | **off** — did not generalize |

Three things to know about that table:

- **Raw YPA and YPC are not in the model, on purpose.** YPA ignores sacks and
  interceptions; YPC is the noisiest number in football (it's O-line, box count
  and game script far more than the back). EPA/play on dropbacks and on rushes
  measures the same thing without the blind spots.
- **Coaching record is a confound, not a signal.** It's a proxy for having had
  a good quarterback. The parts of coaching that are actually measurable —
  fourth-down aggression, pace — are implemented and tested, and they made the
  holdout *worse*. `W_COACH = 0.0`. Set it to 0.3 in `config.py` if you want to
  keep experimenting.
- **Turnovers are mostly luck laundering.** Fumble *recovery* is near a coin
  flip, but the EPA of a lost fumble is charged in full. The model estimates
  expected fumbles lost at 50% of fumbles and adds the difference back before
  updating, so you're not chasing last week's bounces.

Free data has no hurries, so the default pressure input is sacks + QB hits per
dropback. `--pfr` swaps in Pro Football Reference's charted pressure totals
(cached, once a week). If that fetch fails, the model silently falls back.

## Backtest — read this before betting anything

Fitted on 2018–2023, scored on **2024–2026 held out**:

| | model | closing line |
|---|---|---|
| RMSE vs actual margin | **13.07** | 12.46 |
| MAE vs actual margin | 10.27 | — |
| Correlation with closing line | 0.82 | — |
| Mean absolute gap vs closing line | 2.70 | — |
| ATS record, all games | 48.8% | — |
| ATS record, 3+ point disagreements | **47.0%** (n=202) | — |

The model is good. It is not better than the market, and on the holdout it did
not beat the spread. Breakeven at −110 is 52.4%. Treat this as a reference
number that tells you *where you disagree and why* — the CSV breaks every
projection into base / HFA / QB / pressure / rest so a disagreement is
auditable — not as a bet signal.

Where a number like this earns money, if it ever does, is the early week: your
Sunday-night number against a Monday opening line, before the market has
digested injuries. Chasing the closing number on Sunday morning is a losing
game and the table above is what that looks like.

Ablation on the holdout (RMSE, lower is better):

```
full model                    13.066
no pressure term              13.109
no QB adjustment              13.104
no home field                 13.194
rushing merged into one       13.161
no rest adjustment            13.077
no coach term                 13.062   <- why it ships off
```

## New data sources (added after the initial build)

Four nflverse sources were added on top of the original engine. Two were
tested the same rigorous way as everything else and turned off; two are live.

| source | what it adds | status |
|---|---|---|
| FTN charting | QB-fault-sack rate, blitz rate | **off by default** — fit on 2018-2023, scored on 2024-2026 held out, neither moved RMSE. `W_QB_FAULT_SACK` / `W_BLITZ` in config.py, both 0.0. Raise them if you want to keep testing on a longer sample; the columns already print in spreads output at 0.0 so you can see the plumbing is there. |
| Stadium roof | shrinks projected margin slightly for dome/closed games | **off by default**, same reason — improvement on holdout was inside noise (13.0624 → 13.0622 RMSE). `DOME_MARGIN_SHRINK = 0.0`. |
| Weekly injury reports + snap counts | subtracts points from a team's own projection for Out/Doubtful/Questionable non-QB players, scaled by snap share | **on by default** in `spreads`, off in `ratings` (it's a game-week thing, not a team quality). Disable with `--no-injuries`. |
| Depth charts | — | not yet wired in; see caveat below |

**A word about FTN charting, since I initially described it wrong when
proposing it:** it does not contain an explicit "pressure" flag. It has
blitz rate, number of pass rushers, and whether a sack was the QB's fault
or the O-line's — useful data, but not a pressure feed. The pressure input
in the ratings table (`pressure_gen` / `pressure_allowed`) is still the
original play-by-play proxy (sacks + QB hits per dropback), same as before
this update. If you want charted pressure specifically, that's what `--pfr`
is for (Pro Football Reference), not FTN.

**The injury weights are a prior, not a fit.** There isn't enough
per-position injury sample in the data to safely fit ~18 position weights —
that's a fitting exercise that will overfit badly on the available games.
`INJURY_POSITION_VALUE` in config.py is a reasonable starting table (a
starting WR or CB is worth more than a backup linebacker, etc.), scaled by
each player's own recent snap share and by report severity (Out counts full,
Questionable counts less). Treat the `injuries` column in spreads output as
"a plausible adjustment," not a validated one, until you've watched it for a
season. It is **not included in the backtest numbers** below — aligning
historical injury reports to every past game is a bigger job than this pass,
so the injury term only ever fires for the upcoming week from the CLI, never
retroactively.

**Depth charts were not wired in.** The original plan was to use them to
catch a QB change the schedule file hasn't updated yet. In practice the
schedule file's `home_qb_id` / `away_qb_id` are already reasonably current,
and cross-referencing depth charts added complexity without a clear win in
testing. Worth revisiting if you notice the QB adjustment lagging real news
during the season.

## Things that will bite you

- **Home field is 1.5 points**, fitted, not the folk-wisdom 3. If you have a
  number that says 3, it's from 2005.
- **The QB adjustment only fires on a change.** It compares the listed starter
  to whoever the team's rating was actually built on. A big `qb` column in the
  spreads CSV means a starter change — and a stale or provisional listing in
  nfldata will produce a phantom edge. Eyeball any game where `qb` exceeds ~1.5
  points before trusting it.
- **Week 1–3 ratings are mostly last year.** `CARRYOVER_PASS = 0.62` means 38%
  of each passing rating is wiped at the season break. Early-season numbers are
  a prior with a rumor attached.
- **`fit` overfits if you let it.** Fifteen parameters on ~2,000 games is a lot
  of rope. The command prints a holdout comparison for exactly this reason.
- **No injuries, weather, or travel distance.** Rest days are in; a missing All-Pro
  left tackle is not. That's the largest remaining gap and the most likely place
  a manual override is worth more than another metric.

## Layout

```
config.py    all tunables, fitted values, one place to edit
data.py      nflverse pulls, caching, team-game aggregation
ratings.py   the engine: TeamState, QBBook, updates, prediction
backtest.py  scoring + coordinate-descent fitting
pfr.py       optional charted pressure scrape
cli.py       ratings / spreads / backtest / fit
```
