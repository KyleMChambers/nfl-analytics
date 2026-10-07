"""Backtesting and parameter fitting.

Scoring rule: RMSE of the PREGAME projected margin against the actual result.
Win/loss accuracy is a bad objective -- it throws away the magnitude that a
spread is made of. The closing line is shown alongside purely as a yardstick;
it is the hardest benchmark in sports and beating its RMSE outright is not the
goal. Landing close to it, with independent inputs, is.
"""

from __future__ import annotations

import copy
import numpy as np
import pandas as pd

from . import config as C
from . import ratings


class Cfg:
    """Mutable copy of config for search."""
    def __init__(self, **over):
        for k in dir(C):
            if k.isupper():
                setattr(self, k, copy.deepcopy(getattr(C, k)))
        for k, v in over.items():
            setattr(self, k, v)

    def copy(self, **over):
        new = copy.deepcopy(self)
        for k, v in over.items():
            setattr(new, k, v)
        return new


def score(units, games, cfg, eval_from=2018, eval_to=9999):
    _, hist = ratings.run(units, games, cfg=cfg)
    h = hist[(hist.season >= eval_from) & (hist.season <= eval_to)].dropna(subset=["actual_margin"])
    err = h.pred_margin - h.actual_margin
    out = {
        "n": len(h),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(err.abs().mean()),
        "bias": float(err.mean()),
    }
    m = h.dropna(subset=["market_spread"])
    if len(m):
        # nfldata spread_line is home-favored-by, same sign as our margin
        out["market_rmse"] = float(np.sqrt(((m.market_spread - m.actual_margin) ** 2).mean()))
        out["vs_market_mae"] = float((m.pred_margin - m.market_spread).abs().mean())
        out["corr_market"] = float(m.pred_margin.corr(m.market_spread))
        ats = np.sign(m.pred_margin - m.market_spread) == np.sign(m.actual_margin - m.market_spread)
        push = m.actual_margin == m.market_spread
        out["ats_win_pct"] = float(ats[~push].mean())
        big = m[(m.pred_margin - m.market_spread).abs() >= 3]
        if len(big):
            a = np.sign(big.pred_margin - big.market_spread) == np.sign(big.actual_margin - big.market_spread)
            p = big.actual_margin == big.market_spread
            out["ats_3pt_disagree"] = float(a[~p].mean())
            out["n_3pt"] = int((~p).sum())
    return out, hist


def fit(units, games, params, base=None, passes=2, eval_from=2018, eval_to=9999, verbose=True):
    """Coordinate descent over a dict of {param: [candidate values]}."""
    cfg = base or Cfg()
    best, _ = score(units, games, cfg, eval_from, eval_to)
    if verbose:
        print(f"start rmse={best['rmse']:.4f}")
    for p in range(passes):
        for name, values in params.items():
            cur = getattr(cfg, name)
            trials = []
            for v in values:
                s, _ = score(units, games, cfg.copy(**{name: v}), eval_from, eval_to)
                trials.append((s["rmse"], v))
            trials.sort()
            if trials[0][0] < best["rmse"] - 1e-6:
                best_rmse, v = trials[0]
                setattr(cfg, name, v)
                best, _ = score(units, games, cfg, eval_from, eval_to)
                if verbose:
                    print(f"  pass{p} {name}: {cur} -> {v}  rmse={best['rmse']:.4f}")
    return cfg, best
