# NFL Analytics

Three Python projects for NFL modeling, built entirely on free public data:

| Project | Folder | What it does |
|---|---|---|
| Player props value finder | `parlays/` | Models anytime-TD, yardage, and receptions props, compares them to sportsbook prices, and builds ranked parlays |
| Survivor pool helper | `survivor/` | Recommends a weekly survivor pick by balancing this week's win chance against the cost of using a team you'll need later |
| Elo ratings model | `elo/` | Unit-level team ratings (pass/rush, offense/defense) that generate weekly point spreads |

The emphasis throughout is on **validating against real outcomes**: every model was backtested on held-out games, and several features that seemed intuitive were dropped because the data didn't support them.

---

## Data sources

| Data | Source |
|---|---|
| Play-by-play, schedules, rosters, weekly player stats, injuries, snap counts | [nflverse](https://github.com/nflverse/nflverse-data) GitHub releases (Parquet) |
| Sportsbook odds and player props | [The Odds API](https://the-odds-api.com) (free tier) |
| Game-day weather | [Open-Meteo](https://open-meteo.com) (free, no key) |

The survivor helper and Elo model need no API key. The props finder needs an Odds API key (see Setup).

---

## 1. Player props value finder (`parlays/`)

### How it works

1. **Team expectation.** Expected team touchdowns and volume are derived from the game's spread and total, so the model starts from the market's view of the game script.
2. **Player share.** Each player gets a share of team opportunities (targets, carries, red-zone chances, snap share), weighted toward recent games and blended with prior-season data to avoid small-sample blowups.
3. **Adjustments.**
   - *Injury redistribution*: when a player is listed Out or Doubtful, his opportunities are reallocated to same-position teammates in proportion to their snap share.
   - *Opponent vs. position*: defensive strength against each position (used for passing only; see results).
4. **Probabilities.** Anytime-TD probability from expected TDs; yardage and receptions milestones from an opportunity × shrunk-efficiency model with a gamma distribution.
5. **Market comparison.** Odds are de-vigged across several books and blended with the model (`MARKET_WEIGHT`). Edge is defined as expected profit at the betting book's actual price.
6. **Parlay builder.** Ranks parlays with one leg per game and per player, filters by payout window, and suggests stake sizing.

### Backtest results (2025 season)

Graded on weeks 12–18, which were never used for fitting. "Skill" is Brier skill score: percentage improvement over a baseline that guesses the average hit rate.

| Market | Skill vs. baseline |
|---|---|
| Rushing yards | ~17% |
| Receiving yards | ~16–17% |
| Receptions | ~15–16% |
| Anytime TD | ~7–10% |
| Passing yards | ~5–9% |

What the calibration found:

- **Injury redistribution was a clear win** for both TD and yardage models.
- **Opponent-vs-position helped passing only** (skill rose from 4.9% to 9.1%) and added nothing for other markets.
- **Recent-form weighting added nothing** in 2025, so equal weighting is used. To be revisited with 2026 data.
- **A logistic correction** improved TD calibration but hurt yardage, so it is applied to TD only.

Settings are saved in `calibration.json` and `calibration_yards.json` and are produced by `calibrate.py` and `calibrate_yards.py`.

### Data bugs caught and fixed

Most of the real work was catching output that didn't match reality:

- **Small-sample blowups.** Backups with one fluky red-zone touch were rated like starters, producing absurd parlay payouts. Fixed by blending in prior-season data.
- **Name collisions.** Abbreviated names such as "K. Williams" matched several different players, attaching one player's odds to another. Fixed by matching on full roster names and verifying the player's team against the game's two teams.
- **Silent fallback matching.** A loose fallback match pasted odds onto the wrong players, and a later dedupe step preferred the wrong price. Removed the fallback and added a hard team check.
- **Combinatorial hang.** About 200 qualifying legs produced billions of parlay combinations. Capped candidate legs and legs per game.
- **Leakage in the injury backtest.** The boost never fired because game-day rosters already excluded injured players. Fixed by using the prior week's roster.

### Key files

| File | Purpose |
|---|---|
| `props_pipeline.py` | Main entry point (all markets) |
| `share_model.py` | Anytime-TD model |
| `yards_model.py` | Yardage and receptions model |
| `features.py` | Recency weights, snap shares, opponent factors, injury redistribution |
| `parlay_builder.py` | Parlay ranking and sizing |
| `weekly_pipeline.py` | Shared helpers (matchups, team names, odds URLs) |
| `calibrate.py`, `calibrate_yards.py` | Backtesting and calibration |
| `name_match.py`, `devig_props.py` | Player name matching and de-vigging |

---

## 2. Survivor pool helper (`survivor/`)

### How it works

1. **This week's win chance** comes from de-vigged moneylines.
2. **Future cost.** For the next 8 weeks, market power ratings (ridge regression on posted spreads) estimate each team's win chance. The Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) finds the best path through the remaining unused teams. Using a team now that the plan needs later carries a cost.
3. **Score** = log(win chance) − `FUTURE_WEIGHT` × future cost.
4. **Context report** for the top options: records, points for/against, defensive yards allowed, QB rating, coaching records, weather, rest days, travel distance and time zones, injured starters, and division flags.

### Factor testing

`factor_test.py` checked, on roughly 5,000 games from 2006–2025, whether common survivor heuristics predict winners **beyond what the betting line already says**:

- Back-to-back road games, long travel, West Coast teams in early East Coast games, warm-weather teams in the cold, cold-weather teams in the heat, rest differential: **all noise** once the line is accounted for.
- Division games: heavy favorites (70%+) won 79.7% in division games vs. 78.7% outside the division, so **no penalty**.

**Conclusion:** the market already prices these factors. The script shows them as context but does not adjust win probability for them.

### Validation

The power ratings used for future weeks score a Brier of about 0.22–0.24 for games 2–8 weeks out, vs. about 0.20 for actual closing lines: reasonable for planning ahead, and the closing line still wins for the current week. Spread-to-win-probability conversion uses σ = 11.8, fit to 2015–2025 moneylines.

### Key files

| File | Purpose |
|---|---|
| `survivor.py` | Main entry point |
| `ratings.py` | Market power ratings |
| `context.py` | Weather, travel, rest, injuries, coaching |
| `team_info.py` | Stadium locations, time zones, climate |
| `factor_test.py` | Historical factor testing |
| `validate_ratings.py` | Power-rating validation |

---

## 3. Elo ratings model (`elo/`)

Unit-level Elo ratings that track passing and rushing strength separately for offense and defense, then combine them into weekly point spreads. Uses free nflverse data only and outputs a table or CSV from the command line.

---

## Setup

```bash
pip install -r parlays/requirements.txt
pip install -r survivor/requirements.txt
```

**Odds API key (props finder only):** put your key in `parlays/key.txt`. This file is listed in `.gitignore` and must never be committed.

Each main script has a settings block at the top (season, week, and so on). Edit it, then run:

```bash
python parlays/props_pipeline.py
python survivor/survivor.py
```

The props finder caches odds per week, so re-runs don't use API quota unless `REFRESH_ODDS = True`.

---

## Roadmap

- [ ] Shared ingest layer loading nflverse and odds data into Postgres on a schedule (GitHub Actions)
- [ ] dbt models for team stats, ratings, and win probabilities
- [ ] Switch all three projects to read from the shared database
- [ ] Automatic logging and grading of picks to measure real-world performance
- [ ] Streamlit front end for the survivor helper
- [ ] Re-calibrate on combined 2025 + 2026 data

---

## A note on betting

Parlays don't create an edge; they multiply whatever edge or house margin exists in each leg. In backtesting, the model tended to rate heavy favorites lower than the books did, which suggests the market is more accurate on obvious picks. This project is about modeling and measurement, not a promise of profit. Bet responsibly — if gambling stops being fun, call 1-800-GAMBLER.
