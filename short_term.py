"""短线策略库(日线级别):首板接力 / 强势回踩 / 放量突破,附真实规则回测与当日信号扫描。
这些规则是本程序自定义的通用短线思路,不是任何第三方"战法"的复刻;阈值均未优化,请先用自己的数据回测。
回测规则:T 日收盘出信号 -> T+1 开盘买(涨停开盘买不进、高开超过上限不追)-> 最早 T+2 才能卖(T+1 制度)
        -> 止损只从 T+2 起判断;到期日收盘跌停则顺延一天;已扣佣金/印花税/滑点。"""
import numpy as np
import pandas as pd

import config
from data import ratio_series
from factors import tradable_mask

EPS = 0.0015


def prep(p, stocks):
    """一次性算好各策略要用的横截面特征与市场情绪。"""
    c, o, h, l, v, a, t = (p[k] for k in ["close", "open", "high", "low", "volume", "amount", "turnover"])
    pc = c.shift(1)
    pct = c / pc - 1
    ratio = ratio_series(stocks).reindex(c.columns)
    lu = pct.ge(ratio - EPS, axis=1) & pct.notna()
    ld = pct.le(-ratio + EPS, axis=1) & pct.notna()
    open_lu = (o / pc - 1).ge(ratio - EPS, axis=1) & pc.notna()
    arr, s = lu.values, np.zeros(lu.shape, dtype=int)
    for k in range(len(arr)):
        s[k] = (s[k - 1] + 1 if k else 1) * arr[k]
    streak = pd.DataFrame(s, index=c.index, columns=c.columns)
    ind = stocks["industry"].reindex(c.columns).fillna("未知").values
    ind_lu = lu.astype(int).T.groupby(ind).transform("sum").T
    prev_lu = lu.shift(1, fill_value=False)
    return dict(c=c, o=o, h=h, l=l, v=v, a=a, turnover=t, pct=pct, lu=lu, ld=ld, open_lu=open_lu, streak=streak,
                ind_lu=ind_lu, valid=tradable_mask(p, stocks),
                prev_perf=pct.where(prev_lu).mean(axis=1),             # 昨日涨停股今日平均表现 = 赚钱效应
                ld_ratio=ld.sum(axis=1) / pct.notna().sum(axis=1),
                lu_cnt=lu.sum(axis=1), ld_cnt=ld.sum(axis=1))


def _between(df, lo, hi):
    return (df >= lo) & (df <= hi)


def mood_ok(X, ld_max=0.015):
    return (X["prev_perf"] > 0) & (X["ld_ratio"] <= ld_max)


# ------------------------------------------------------------------ 信号
def sig_first_board(X, ld_max=0.015, min_ind_lu=3, turn=(3, 30)):
    """首板接力:今日首次涨停、非一字、同行业涨停>=3 家(有板块共振)、换手适中,且市场赚钱效应为正、跌停不多。"""
    s = X["lu"] & (X["streak"] == 1) & ~X["open_lu"] & X["valid"] & (X["ind_lu"] >= min_ind_lu) \
        & _between(X["turnover"], *turn)
    s = s & mood_ok(X, ld_max).values[:, None]
    return s, X["a"]


def sig_pullback(X):
    """强势回踩:近期有涨停或20日涨超25%,均线多头向上,缩量回踩10日线附近后收阳。"""
    c, o, l, v = X["c"], X["o"], X["l"], X["v"]
    ma5, ma10, ma20 = (c.rolling(n).mean() for n in (5, 10, 20))
    strong = (X["lu"].astype(int).rolling(20).sum() >= 1) | (c / c.shift(20) - 1 > 0.25)
    trend = (ma5 > ma10) & (ma10 > ma20) & (ma20 > ma20.shift(5))
    near = (l <= ma10 * 1.01) & (c >= ma10 * 0.98)
    shrink = v < v.rolling(5).mean()
    s = strong & trend & near & shrink & (c > o) & ~X["lu"] & X["valid"]
    return s, c / c.shift(20) - 1


def sig_breakout(X):
    """放量突破:收盘创20日新高、量比1.5~4、涨幅3%~8%(非涨停)、站上MA60且MA20向上。"""
    c, v = X["c"], X["v"]
    volr = v / v.rolling(5).mean().shift(1)
    ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
    s = (c > c.rolling(20).max().shift(1)) & _between(volr, 1.5, 4) & _between(X["pct"], 0.03, 0.08) \
        & (ma20 > ma20.shift(5)) & (c > ma60) & ~X["lu"] & X["valid"]
    return s, volr


STRATS = {
    "first_board": dict(fn=sig_first_board, label="首板接力", hold=2, gap=(-0.01, 0.06), stop=0.05, slip=0.002),
    "pullback": dict(fn=sig_pullback, label="强势回踩", hold=3, gap=(-0.02, 0.03), stop=0.05, slip=0.001),
    "breakout": dict(fn=sig_breakout, label="放量突破", hold=3, gap=(-0.01, 0.04), stop=0.05, slip=0.001),
}


# ------------------------------------------------------------------ 回测
def backtest(X, stocks, sig, score, hold=2, gap=(-0.01, 0.06), stop=0.05, top=5, slip=None):
    assert hold >= 2, "T+1:买入当日不能卖,hold 至少为 2"
    cst = dict(config.COST)
    if slip is not None:
        cst["slippage"] = slip
    dates, codes = X["c"].index, X["c"].columns
    Oa, La, Ca = X["o"].values, X["l"].values, X["c"].values
    ratio = ratio_series(stocks).reindex(codes).values
    sg, sc, valid = sig.values, score.values, X["valid"].values
    mood = mood_ok(X).values
    last = len(dates) - 1
    bench_cache, rows = {}, []
    for i in range(last):
        E = i + 1
        xi = E + hold - 1
        if xi > last:
            break
        idx = np.where(sg[i])[0]
        if len(idx) == 0:
            continue
        idx = idx[np.argsort(-np.nan_to_num(sc[i, idx]))][:top]
        for j in idx:
            en, pc = Oa[E, j], Ca[i, j]
            if not (np.isfinite(en) and np.isfinite(pc)):
                continue
            g = en / pc - 1
            if g < gap[0] or g >= gap[1] or g >= ratio[j] - EPS:     # 高开过多不追;涨停开盘买不进
                continue
            stp = en * (1 - stop) if stop else None
            ex, reason = None, "到期"
            for d in range(E + 1, xi + 1):                           # T+1 买入,止损从 T+2 起才能执行
                if stp is not None:
                    if Oa[d, j] <= stp:
                        ex, reason = Oa[d, j], "止损(跳空)"
                        break
                    if La[d, j] <= stp:
                        ex, reason = stp, "止损"
                        break
            if ex is None:
                x = xi
                if x < last and Ca[x - 1, j] > 0 and Ca[x, j] / Ca[x - 1, j] - 1 <= -ratio[j] + EPS:
                    x, reason = x + 1, "到期(跌停顺延)"
                ex = Ca[x, j]
            if not np.isfinite(ex):
                continue
            net = (ex * (1 - cst["slippage"]) * (1 - cst["commission"] - cst["stamp"])) / \
                  (en * (1 + cst["slippage"]) * (1 + cst["commission"])) - 1
            key = (E, xi)
            if key not in bench_cache:
                u = valid[i] & np.isfinite(Oa[E]) & np.isfinite(Ca[xi])
                bench_cache[key] = float(np.mean(Ca[xi][u] / Oa[E][u] - 1)) if u.any() else np.nan
            rows.append(dict(signal_date=dates[i], code=codes[j], gap=g, gross=ex / en - 1, net=net,
                             bench=bench_cache[key], mood=bool(mood[i]), reason=reason))
    return pd.DataFrame(rows)


def summarize(tr):
    if tr is None or tr.empty:
        return {}
    w, l = tr.loc[tr["net"] > 0, "net"], tr.loc[tr["net"] <= 0, "net"]
    day = tr.groupby("signal_date")["net"].mean()
    return {"笔数": len(tr), "信号日数": tr["signal_date"].nunique(), "胜率": (tr["net"] > 0).mean(),
            "平均净收益": tr["net"].mean(), "中位数": tr["net"].median(),
            "平均盈利": w.mean() if len(w) else np.nan, "平均亏损": l.mean() if len(l) else np.nan,
            "盈亏比": (w.mean() / abs(l.mean())) if len(w) and len(l) else np.nan,
            "平均超额(对等权)": (tr["net"] - tr["bench"]).mean(), "最差单笔": tr["net"].min(),
            "最好单笔": tr["net"].max(), "止损触发率": tr["reason"].str.startswith("止损").mean(),
            "逐日等权累计": (1 + day).prod() - 1}


def split_by_mood(tr):
    if tr is None or tr.empty:
        return pd.DataFrame()
    g = tr.groupby("mood")
    out = pd.DataFrame({"笔数": g.size(), "胜率": g["net"].apply(lambda s: (s > 0).mean()),
                        "平均净收益": g["net"].mean(), "平均超额": g.apply(lambda d: (d["net"] - d["bench"]).mean(), include_groups=False)})
    out.index = out.index.map({True: "市场情绪满足(赚钱效应>0且跌停少)", False: "情绪不满足"})
    return out


# ------------------------------------------------------------------ 当日信号
def scan_today(X, stocks, sig, score, top=10):
    row = sig.iloc[-1]
    codes = row[row].index
    if len(codes) == 0:
        return pd.DataFrame()
    df = pd.DataFrame({"code": codes, "name": stocks["name"].reindex(codes).values,
                       "industry": stocks["industry"].reindex(codes).values,
                       "score": score.iloc[-1].reindex(codes).values, "close": X["c"].iloc[-1].reindex(codes).values,
                       "pct": X["pct"].iloc[-1].reindex(codes).values, "turnover": X["turnover"].iloc[-1].reindex(codes).values,
                       "ind_lu": X["ind_lu"].iloc[-1].reindex(codes).values, "streak": X["streak"].iloc[-1].reindex(codes).values})
    return df.sort_values("score", ascending=False).head(top).reset_index(drop=True)
