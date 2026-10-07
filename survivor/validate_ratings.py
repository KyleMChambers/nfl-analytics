"""How well do market power ratings predict games 2-8 weeks out? Tested on 2023-2025."""
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from ratings import load_schedule, market_ratings, game_prob, moneyline_prob, spread_to_prob
import ratings as R

s = load_schedule()
h = s[s.season.between(2015, 2025) & s.home_moneyline.notna() & s.spread_line.notna()]
mlp = np.array([moneyline_prob(a, b) for a, b in zip(h.home_moneyline, h.away_moneyline)])
fit = minimize_scalar(lambda sg: np.mean((R.norm.cdf(h.spread_line / sg) - mlp) ** 2), bounds=(8, 20), method="bounded")
print(f"Spread->win-chance randomness that matches the moneylines: sigma = {fit.x:.1f} points")
R.SPREAD_SIGMA = round(fit.x, 1)

res = []
for season in (2023, 2024, 2025):
    for W in range(4, 15):
        rt, hfa = market_ratings(s, season, W)
        for ahead in range(2, 9):
            wk = s[(s.season == season) & (s.week == W + ahead) & s.result.notna() & (s.result != 0)]
            for r in wk.itertuples():
                p_proj, _, _ = game_prob(r, rt, hfa, use_market=False)
                p_line = moneyline_prob(r.home_moneyline, r.away_moneyline)
                res.append((ahead, p_proj, p_line, int(r.result > 0)))
d = pd.DataFrame(res, columns=["ahead", "proj", "line", "win"])
b = lambda p: np.mean((d.loc[p.index, p.name] - d.loc[p.index, "win"]) ** 2) if False else None
print("\nAccuracy (Brier score, lower = better) for games N weeks out:")
print("  weeks out | our projection | actual closing line | coin flip")
for a, x in d.groupby("ahead"):
    print(f"      {a}     |     {np.mean((x.proj-x.win)**2):.4f}     |       {np.mean((x.line-x.win)**2):.4f}        |  0.2500")
print(f"\nSurvivor-relevant: projected 75%+ favorites won {d[d.proj.ge(.75)|d.proj.le(.25)].pipe(lambda x: np.where(x.proj>.5,x.win,1-x.win).mean()):.1%} "
      f"(projection said {d[d.proj.ge(.75)|d.proj.le(.25)].pipe(lambda x: np.maximum(x.proj,1-x.proj).mean()):.1%})")

# Fit extra uncertainty per week out so projections aren't overconfident
from scipy.stats import norm
rows = []
for season in (2023, 2024, 2025):
    for W in range(4, 15):
        rt, hfa = market_ratings(s, season, W)
        for ahead in range(2, 9):
            for r in s[(s.season == season) & (s.week == W + ahead) & s.result.notna() & (s.result != 0)].itertuples():
                sp = rt.get(r.home_team, 0) - rt.get(r.away_team, 0) + (hfa if r.location == "Home" else 0)
                rows.append((ahead, sp, int(r.result > 0)))
e = pd.DataFrame(rows, columns=["ahead", "spread", "win"])
best = min(((np.mean((norm.cdf(e.spread / (R.SPREAD_SIGMA * (1 + k * e.ahead))) - e.win) ** 2), k) for k in np.arange(0, 0.21, 0.01)))
k = best[1]
e["p"] = norm.cdf(e.spread / (R.SPREAD_SIGMA * (1 + k * e.ahead)))
big = e[(e.p >= .75) | (e.p <= .25)]
print(f"\nExtra uncertainty: sigma grows {k:.0%} per week out.  Brier now {best[0]:.4f}")
print(f"After fix: projected 75%+ favorites won {np.where(big.p>.5,big.win,1-big.win).mean():.1%} (projection said {np.maximum(big.p,1-big.p).mean():.1%}, n={len(big)})")
