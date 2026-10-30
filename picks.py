"""由模型得分生成次日关注清单:硬过滤 + 行业分散 + 关键价位 + 信号标签 + 失效条件。"""
import numpy as np
import pandas as pd

import config
from data import ratio_series
from factors import atr14


def make_picks(score: pd.Series, p, stocks, valid_last, top_n=None, max_per_ind=None):
    top_n = top_n or config.TOP_N
    max_per_ind = max_per_ind or config.MAX_PER_INDUSTRY
    c, o, h, l, v, t = (p[k] for k in ["close", "open", "high", "low", "volume", "turnover"])
    ratio = ratio_series(stocks).reindex(c.columns)
    pc = c.shift(1)
    pct = (c / pc - 1).iloc[-1]
    at_limit_up = pct >= ratio - 0.0015                       # 今日涨停收盘 -> 次日大概率买不到
    atr = atr14(p).iloc[-1]
    ma20, ma5 = c.rolling(20).mean().iloc[-1], c.rolling(5).mean().iloc[-1]
    hi20_prev = c.rolling(20).max().shift(1).iloc[-1]
    low20, hi60 = l.rolling(20).min().iloc[-1], h.rolling(60).max().iloc[-1]
    vr = (v / v.rolling(5).mean().shift(1)).iloc[-1]
    shrink = (v / v.rolling(20).mean()).iloc[-1]
    mom20 = c.pct_change(20, fill_method=None).iloc[-1]
    rsi = 100 - 100 / (1 + (c.diff().clip(lower=0).rolling(14).mean() /
                           (-c.diff().clip(upper=0)).rolling(14).mean().replace(0, np.nan))).iloc[-1]

    s = score.copy()
    s[~valid_last] = np.nan
    watch_limit = s[at_limit_up.reindex(s.index).fillna(False)].dropna().nlargest(3)   # 涨停股单列观察
    s[at_limit_up.reindex(s.index).fillna(False)] = np.nan
    s = s.dropna().sort_values(ascending=False)

    rows, cnt = [], {}
    for code in s.index:
        ind = stocks.loc[code, "industry"]
        if cnt.get(ind, 0) >= max_per_ind:
            continue
        cnt[ind] = cnt.get(ind, 0) + 1
        tags = []
        if vr[code] > 1.5 and c[code].iloc[-1] > hi20_prev[code]:
            tags.append("放量突破20日高")
        if abs(c[code].iloc[-1] / ma20[code] - 1) < 0.03 and shrink[code] < 0.7 and mom20[code] > 0:
            tags.append("缩量回踩MA20")
        if rsi[code] < 30:
            tags.append("RSI超卖")
        if vr[code] > 1.3 and pct[code] > 0:
            tags.append("量价齐升")
        close = c[code].iloc[-1]
        rows.append(dict(
            code=code, name=stocks.loc[code, "name"], industry=ind, score=s[code], close=close,
            pct=pct[code], vol_ratio=vr[code], turnover=t[code].iloc[-1],
            support=low20[code], resistance=hi60[code],
            stop=close - 1.5 * atr[code], target=close + 2.5 * atr[code],
            tags="、".join(tags) or "多因子综合领先",
            invalid=f"收盘跌破{close - 1.5 * atr[code]:.2f}(1.5ATR)或跌破20日低点{low20[code]:.2f};所属板块转弱",
        ))
        if len(rows) >= top_n:
            break
    picks = pd.DataFrame(rows)
    watch = pd.DataFrame([dict(code=cd, name=stocks.loc[cd, "name"], industry=stocks.loc[cd, "industry"], score=v_)
                          for cd, v_ in watch_limit.items()])
    return picks, watch
