"""Do these factors predict winners BEYOND what the betting line already knew? 2006-2025."""
import numpy as np, pandas as pd
from team_info import TEAM_INFO, miles_between

g = pd.read_parquet("https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet")
g = g[(g.game_type == "REG") & g.season.between(2006, 2025)].copy()

def ml_prob(ml):
    return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))

# Previous game location for each team (back-to-back road)
long = pd.concat([g[["season","week","home_team"]].rename(columns={"home_team":"team"}).assign(away=0),
                  g[["season","week","away_team"]].rename(columns={"away_team":"team"}).assign(away=1)])
long = long.sort_values(["team","season","week"])
long["prev_away"] = long.groupby(["team","season"])["away"].shift(1)
prev = long.set_index(["team","season","week"])["prev_away"]
g["away_b2b_road"] = [prev.get((t,s,w), np.nan) for t,s,w in zip(g.away_team,g.season,g.week)]
g["home_b2b_road"] = 0  # home team is home this week by definition

d = g[g.home_moneyline.notna() & g.away_moneyline.notna() & g.result.notna() & (g.result != 0) & (g.location == "Home")].copy()
ph, pa = ml_prob(d.home_moneyline.values), ml_prob(d.away_moneyline.values)
d["p_mkt"] = ph / (ph + pa)
d["home_win"] = (d.result > 0).astype(int)
d = d[d.home_team.isin(TEAM_INFO) & d.away_team.isin(TEAM_INFO)]

d["tz_diff"] = [abs(TEAM_INFO[h][2] - TEAM_INFO[a][2]) for h,a in zip(d.home_team,d.away_team)]
d["miles"] = [miles_between(a,h) for h,a in zip(d.home_team,d.away_team)]
d["long_trip"] = ((d.tz_diff >= 2) | (d.miles > 1500)).astype(int)
d["early_west_to_east"] = [int(TEAM_INFO[a][2] <= -7 and TEAM_INFO[h][2] == -5 and str(t)[:2] in ("12","13"))
                           for h,a,t in zip(d.home_team,d.away_team,d.gametime)]
cold_game = (d.roof == "outdoors") & (d.temp < 40)
away_soft = np.array([TEAM_INFO[a][3] == "warm" or TEAM_INFO[a][4] for a in d.away_team])
d["warm_team_in_cold"] = (cold_game & away_soft).astype(int)
hot_game = (d.roof == "outdoors") & (d.temp >= 85)
d["cold_team_in_heat"] = (hot_game & np.array([TEAM_INFO[a][3] == "cold" for a in d.away_team])).astype(int)
d["away_b2b_road"] = d["away_b2b_road"].fillna(0).astype(int)
d["rest_diff"] = (d.home_rest - d.away_rest).clip(-7, 7)
d["div"] = d.div_game.astype(int)

def logit_fit(X, y, offset):
    X = np.column_stack([np.ones(len(y)), X]); w = np.zeros(X.shape[1])
    for _ in range(50):
        mu = 1/(1+np.exp(-(offset + X@w))); H = (X.T*(mu*(1-mu)))@X
        step = np.linalg.solve(H, X.T@(y-mu)); w += step
        if np.abs(step).max() < 1e-9: break
    se = np.sqrt(np.diag(np.linalg.inv(H))); return w, se

off = np.log(d.p_mkt/(1-d.p_mkt)).values
y = d.home_win.values
print(f"{len(d)} games, 2006-2025. Market favorite won {( (d.p_mkt>0.5)==(d.home_win==1)).mean():.1%} of the time.\n")
print("Effect beyond the betting line (in percentage points of HOME win chance, for a 50/50 game):")
for name in ["away_b2b_road","long_trip","early_west_to_east","warm_team_in_cold","cold_team_in_heat","rest_diff"]:
    x = d[name].values.astype(float)
    w, se = logit_fit(x[:,None], y, off)
    eff, lo, hi = 25*w[1], 25*(w[1]-1.96*se[1]), 25*(w[1]+1.96*se[1])
    n = int((x != 0).sum())
    sig = "REAL" if (lo > 0 or hi < 0) else "noise"
    unit = " per day of rest" if name=="rest_diff" else ""
    print(f"  {name:20s} games={n:5d}  effect={eff:+5.1f} pts{unit}  (95% range {lo:+.1f} to {hi:+.1f})  -> {sig}")

# Division games: are favorites less safe than the line says?
for label, sub in [("division", d[d['div']==1]), ("non-division", d[d['div']==0])]:
    fav_p = np.maximum(sub.p_mkt, 1-sub.p_mkt); fav_won = np.where(sub.p_mkt>0.5, sub.home_win, 1-sub.home_win)
    big = fav_p >= 0.70
    print(f"\n  {label:12s}: favorites of 70%+ -> line said {fav_p[big].mean():.1%}, actually won {fav_won[big].mean():.1%}  (n={big.sum()})")

print("\nWithout the betting line (raw effect on home win %):")
for name in ["away_b2b_road","long_trip","warm_team_in_cold","rest_diff"]:
    x = d[name].values.astype(float)
    w, se = logit_fit(x[:,None], y, np.zeros(len(y)))
    eff, lo, hi = 25*w[1], 25*(w[1]-1.96*se[1]), 25*(w[1]+1.96*se[1])
    print(f"  {name:20s} effect={eff:+5.1f} pts  (95% range {lo:+.1f} to {hi:+.1f})  -> {'REAL' if (lo>0 or hi<0) else 'noise'}")
fav = lambda s: np.where(s.p_mkt>0.5, s.home_win, 1-s.home_win).mean()
print(f"  favorites won {fav(d[d['div']==1]):.1%} in division games vs {fav(d[d['div']==0]):.1%} in non-division "
      f"(but division favorites are usually smaller favorites: avg line {np.maximum(d[d['div']==1].p_mkt,1-d[d['div']==1].p_mkt).mean():.1%} vs "
      f"{np.maximum(d[d['div']==0].p_mkt,1-d[d['div']==0].p_mkt).mean():.1%})")
