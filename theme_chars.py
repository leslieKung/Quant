"""字辈/名称概念炒作挖掘:按名字里的字(或拼音)给股票分组,检验该组涨停是否异常,
并区分"名字驱动"(涨停股跨很多行业)与"行业/题材驱动"(涨停股集中在少数行业)。"""
import re

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

import config
from data import ratio_series

EPS = 0.0015
_py_cache = {}


def name_chars(name):
    """名称中的汉字(去重保序),自动忽略 ST/*ST/A/B/数字等。"""
    return list(dict.fromkeys(re.findall(r"[\u4e00-\u9fff]", str(name))))


def to_pinyin(ch):
    """无声调拼音;没装 pypinyin 返回 None。pip install pypinyin"""
    if ch not in _py_cache:
        try:
            from pypinyin import lazy_pinyin
            _py_cache[ch] = lazy_pinyin(ch)[0]
        except Exception:
            _py_cache[ch] = None
    return _py_cache[ch]


def limit_streak(p, stocks):
    """涨停标记与连板数(DataFrame, 日期 x 股票)。"""
    c = p["close"]
    pct = c / c.shift(1) - 1
    ratio = ratio_series(stocks).reindex(c.columns)
    lu = pct.ge(ratio - EPS, axis=1) & pct.notna()
    arr, s = lu.values, np.zeros(lu.shape, dtype=int)
    for t in range(len(arr)):
        s[t] = (s[t - 1] + 1 if t else 1) * arr[t]
    return pct, lu, pd.DataFrame(s, index=lu.index, columns=lu.columns)


def snapshot(p, stocks, date, cache=None):
    """某日的横截面:code,name,pct,lu,streak,industry,amount。cache 为 limit_streak 的结果,批量扫描时复用。"""
    pct, lu, streak = cache if cache is not None else limit_streak(p, stocks)
    d = pd.Timestamp(date)
    df = pd.DataFrame({"code": pct.columns, "name": stocks["name"].reindex(pct.columns).values,
                       "industry": stocks["industry"].reindex(pct.columns).values,
                       "pct": pct.loc[d].values, "lu": lu.loc[d].values, "streak": streak.loc[d].values,
                       "amount": p["amount"].loc[d].values})
    return df.dropna(subset=["pct", "name"]).reset_index(drop=True)


def _bh(pvals):
    """Benjamini-Hochberg FDR 校正。"""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    q = np.empty_like(p)
    m = len(p)
    prev = 1.0
    for rank, i in zip(range(m, 0, -1), order[::-1]):
        prev = min(prev, p[i] * m / rank)
        q[i] = prev
    return q


def scan(snap, by="char", min_n=3, max_n=80, min_lu=2, fdr=0.05, spread_cut=0.6):
    """返回按显著性排序的候选"字辈"表。by='char' 按汉字;by='pinyin' 按拼音(羊洋阳杨→yang)。"""
    N, K = len(snap), int(snap["lu"].sum())
    mean_all, sd_all = snap["pct"].mean(), snap["pct"].std()
    has_ind = snap["industry"].notna().any() and (snap["industry"] != "未知").mean() > 0.5
    groups = {}
    for i, nm in enumerate(snap["name"].values):
        keys = set(name_chars(nm)) if by == "char" else {to_pinyin(ch) for ch in name_chars(nm)} - {None}
        for k in keys:
            groups.setdefault(k, []).append(i)

    rows = []
    for k, idx in groups.items():
        n = len(idx)
        if n < min_n or n > max_n:
            continue
        sub = snap.iloc[idx]
        lu = int(sub["lu"].sum())
        pval = float(hypergeom.sf(lu - 1, N, K, n)) if lu > 0 else 1.0        # P(涨停数 >= 观察值)
        lead = sub.sort_values(["streak", "amount"], ascending=False).iloc[0]
        spread = np.nan
        if has_ind and lu >= 2:
            inds = sub.loc[sub["lu"], "industry"]
            spread = inds.nunique() / len(inds)                                 # 1=每只涨停股行业都不同
        rows.append(dict(key=k, n=n, limit_up=lu, lu_ratio=lu / n, mean_pct=sub["pct"].mean(),
                         z_ret=(sub["pct"].mean() - mean_all) / (sd_all / np.sqrt(n) + 1e-12),
                         pval=pval, industry_spread=spread,
                         leader=f"{lead['name']}({int(lead['streak'])}板)" if lead["streak"] > 0 else "-",
                         members=sub.sort_values("pct", ascending=False)["name"].head(10).tolist()))
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["qval"] = _bh(res["pval"].values)
    base = K / max(N, 1)

    def label(r):
        if r["qval"] >= fdr or r["limit_up"] < min_lu:
            return "—"
        if np.isnan(r["industry_spread"]):
            return "显著异常(行业信息缺失,需人工判断)"
        return "名字驱动嫌疑(跨行业)" if r["industry_spread"] >= spread_cut else "更像行业/题材驱动(行业集中)"

    res["label"] = res.apply(label, axis=1)
    res["base_lu_ratio"] = base
    return res.sort_values(["qval", "limit_up"], ascending=[True, False]).reset_index(drop=True)


def scan_history(p, stocks, days=40, **kw):
    """逐日回放:每天最显著的候选字辈(只保留显著的),看一个字辈从哪天被检测到。"""
    cache = limit_streak(p, stocks)
    rows = []
    for d in p["close"].index[-days:]:
        res = scan(snapshot(p, stocks, d, cache), **kw)
        sig = res[res["label"] != "—"] if len(res) else res
        if len(sig):
            r = sig.iloc[0]
            rows.append(dict(date=d, key=r["key"], n=r["n"], limit_up=r["limit_up"], qval=r["qval"],
                             spread=r["industry_spread"], leader=r["leader"], label=r["label"]))
    return pd.DataFrame(rows)


def event_study(p, stocks, events, H=None, by="char"):
    """检测到字辈后的真实后续表现:T+1 开盘买(涨停开盘买不进),T+H 收盘卖,扣成本。
    分组:龙头(最高连板)/ 跟风涨停股 / 同名未涨停股(补涨)。对比全体样本同期平均。"""
    H = H or config.HORIZON
    cache = limit_streak(p, stocks)
    pct, lu, streak = cache
    dates, cols = p["close"].index, list(p["close"].columns)
    O, C = p["open"].values, p["close"].values
    ratio = ratio_series(stocks).reindex(p["close"].columns).values
    cst = config.COST
    names = stocks["name"].reindex(p["close"].columns).values
    rows = []
    for ev in events.itertuples():
        i = dates.get_loc(ev.date)
        e, x = i + 1, i + H
        if x >= len(dates):
            continue                                                  # 还没到期
        mem = [j for j, nm in enumerate(names) if isinstance(nm, str) and
               (ev.key in name_chars(nm) if by == "char" else ev.key in [to_pinyin(c_) for c_ in name_chars(nm)])]
        lead = max(mem, key=lambda j: (streak.values[i, j], p["amount"].values[i, j]))
        allr = np.nanmean(C[x] / O[e] - 1)
        for j in mem:
            if not (np.isfinite(O[e, j]) and np.isfinite(C[x, j])):
                continue
            if O[e, j] / C[i, j] - 1 >= ratio[j] - EPS:
                grp = "开盘涨停(买不进)"
            elif j == lead and lu.values[i, j]:
                grp = "龙头"
            elif lu.values[i, j]:
                grp = "跟风涨停股"
            else:
                grp = "同名未涨停股"
            net = (C[x, j] * (1 - cst["slippage"]) * (1 - cst["commission"] - cst["stamp"])) / \
                  (O[e, j] * (1 + cst["slippage"]) * (1 + cst["commission"])) - 1
            rows.append(dict(date=ev.date, key=ev.key, group=grp, net=net, excess=net - allr))
    r = pd.DataFrame(rows)
    if r.empty:
        return r, r
    summ = r.groupby("group").agg(样本=("net", "count"), 平均净收益=("net", "mean"), 中位数=("net", "median"),
                                  胜率=("net", lambda s: (s > 0).mean()), 平均超额=("excess", "mean"),
                                  最差=("net", "min"), 最好=("net", "max"))
    return r, summ
