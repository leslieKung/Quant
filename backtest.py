"""A股回测:T+1、涨停买不进、跌停卖不出、印花税/佣金/滑点。非重叠持有期等权。"""
import numpy as np
import pandas as pd

import config
from data import ratio_series


def run_backtest(score: pd.DataFrame, p, stocks, valid, H=None, top_n=None, cost=None, max_per_ind=None):
    H = H or config.HORIZON
    top_n = top_n or config.TOP_N
    cost = cost or config.COST
    max_per_ind = max_per_ind or config.MAX_PER_INDUSTRY
    o, c = p["open"], p["close"]
    pc = c.shift(1)
    ratio = ratio_series(stocks).reindex(c.columns).values[None, :]
    open_limit_up = ((o / pc - 1) >= ratio - 0.0015).values       # 开盘即涨停,买不进
    close_limit_dn = ((c / pc - 1) <= -ratio + 0.0015).values     # 收盘跌停,卖不出
    ind = stocks["industry"].reindex(c.columns).fillna("未知").values
    codes = c.columns
    D = len(c)

    first = score.notna().any(axis=1).values.argmax()
    rows, i = [], first
    while i + H + 1 < D:
        s = score.iloc[i].copy()
        s[~valid[i]] = np.nan
        s[open_limit_up[i + 1]] = np.nan
        s[o.iloc[i + 1].isna().values] = np.nan
        s = s.dropna().sort_values(ascending=False)
        picks, cnt = [], {}
        for code in s.index:
            k = ind[codes.get_loc(code)]
            if cnt.get(k, 0) >= max_per_ind:
                continue
            picks.append(code)
            cnt[k] = cnt.get(k, 0) + 1
            if len(picks) >= top_n:
                break
        if len(picks) < max(3, top_n // 2):
            i += H
            continue
        ex = i + H
        pj = [codes.get_loc(x) for x in picks]
        entry = o.iloc[i + 1].values[pj]
        exit_idx = np.where(close_limit_dn[ex, pj] & (ex + 1 < D), ex + 1, ex)  # 跌停顺延一天
        exit_ = np.array([c.values[e, k] for e, k in zip(exit_idx, pj)])
        ok = ~np.isnan(entry) & ~np.isnan(exit_)
        buy = entry[ok] * (1 + cost["slippage"]) * (1 + cost["commission"])
        sell = exit_[ok] * (1 - cost["slippage"]) * (1 - cost["commission"] - cost["stamp"])
        net = (sell / buy - 1).mean()
        gross = (exit_[ok] / entry[ok] - 1).mean()
        u = valid[i]
        bench = (c.iloc[ex].values[u] / o.iloc[i + 1].values[u] - 1)
        bench = np.nanmean(bench)
        rows.append((c.index[i], c.index[ex], gross, net, bench))
        i += H

    r = pd.DataFrame(rows, columns=["signal_date", "exit_date", "gross", "net", "bench"]).set_index("signal_date")
    if r.empty:
        return r, {}
    eq = (1 + r["net"]).cumprod()
    be = (1 + r["bench"]).cumprod()
    n = len(r)
    ppy = 252 / H
    dd = (eq / eq.cummax() - 1).min()
    stat = dict(
        periods=n,
        annual_return=eq.iloc[-1] ** (ppy / n) - 1,
        bench_annual=be.iloc[-1] ** (ppy / n) - 1,
        sharpe=r["net"].mean() / (r["net"].std() + 1e-12) * np.sqrt(ppy),
        max_drawdown=dd,
        win_rate=(r["net"] > 0).mean(),
        excess_win_rate=(r["net"] > r["bench"]).mean(),
        avg_period_net=r["net"].mean(),
        avg_cost_drag=(r["gross"] - r["net"]).mean(),
    )
    r["equity"], r["bench_equity"] = eq, be
    return r, stat


def plot_equity(r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(r.index, r["equity"], label="Strategy (net)")
    ax.plot(r.index, r["bench_equity"], label="Equal-weight universe")
    ax.legend()
    ax.grid(alpha=.3)
    ax.set_title("Backtest equity (non-overlapping holding periods)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
