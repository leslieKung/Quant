"""大盘复盘:涨跌家数/涨跌停/炸板/赚钱效应/情绪温度/仓位建议;板块强弱、扩散度、拥挤度、原因归因。"""
import numpy as np
import pandas as pd

import config
from data import ratio_series
from factors import clv_money

EPS = 0.0015


def market_review(p, stocks, idx):
    c, h, a = p["close"], p["high"], p["amount"]
    pc = c.shift(1)
    pct = c / pc - 1
    ratio = ratio_series(stocks).reindex(c.columns)
    day = c.index[-1]
    pt = pct.iloc[-1].dropna()
    r = ratio.reindex(pt.index)

    lu = int((pt >= r - EPS).sum())
    ld = int((pt <= -r + EPS).sum())
    hit = (h.iloc[-1] / pc.iloc[-1] - 1).reindex(pt.index) >= r - EPS
    broken = int((hit & (pt < r - EPS)).sum())
    seal = lu / (lu + broken) if (lu + broken) else np.nan
    up, down = int((pt > 0).sum()), int((pt < 0).sum())

    prev_lu = (pct.iloc[-2] >= ratio - EPS)
    prev_lu_perf = pct.iloc[-1][prev_lu].mean() if prev_lu.sum() else np.nan   # 昨日涨停今日表现=赚钱效应

    amt = a.sum(axis=1)
    amt_pct = float((amt.iloc[-60:] <= amt.iloc[-1]).mean())

    rows = []
    for code, name in config.INDEXES.items():
        s = idx[idx["code"] == code].set_index("date")["close"].dropna()
        if len(s) < 21:
            continue
        rows.append(dict(code=code, name=name, close=s.iloc[-1], pct=s.iloc[-1] / s.iloc[-2] - 1,
                         ret5=s.iloc[-1] / s.iloc[-6] - 1, ma5=s.iloc[-5:].mean(),
                         ma20=s.iloc[-20:].mean(), low5=s.iloc[-5:].min(), high20=s.iloc[-20:].max()))
    index_df = pd.DataFrame(rows)

    bench = index_df[index_df["code"] == config.BENCH]
    if len(bench):
        b = bench.iloc[0]
        trend = float(np.clip((b["close"] / b["ma20"] - 1) * 20 + 0.5, 0, 1) * 100)
    else:
        trend = 50.0

    parts = {
        "涨跌家数(上涨占比)": up / max(up + down, 1) * 100,
        "涨停/跌停比": lu / max(lu + ld, 1) * 100,
        "封板率": (seal * 100) if seal == seal else 50.0,
        "成交额分位(60日)": amt_pct * 100,
        "指数趋势(HS300 vs MA20)": trend,
        "赚钱效应(昨涨停今表现)": float(np.clip((prev_lu_perf + 0.03) / 0.08, 0, 1) * 100) if prev_lu_perf == prev_lu_perf else 50.0,
    }
    w = [0.20, 0.15, 0.15, 0.15, 0.20, 0.15]
    score = float(sum(v * x for v, x in zip(parts.values(), w)))
    if score >= 70:
        label, pos = "强势/亢奋", "70%~80%(注意高位分歧,不追涨停)"
    elif score >= 55:
        label, pos = "偏暖", "50%~60%"
    elif score >= 40:
        label, pos = "震荡/中性", "30%~40%"
    elif score >= 25:
        label, pos = "偏冷", "10%~20%"
    else:
        label, pos = "冰点/退潮", "0%~10%(空仓观望为主)"

    return dict(date=day, up=up, down=down, flat=len(pt) - up - down, limit_up=lu, limit_down=ld,
                broken=broken, seal_rate=seal, amount=float(amt.iloc[-1]), amount_pct=amt_pct,
                prev_lu_perf=prev_lu_perf, index_df=index_df, sentiment=score, parts=parts,
                label=label, position=pos, sample_n=len(pt))


def fetch_news(limit=60):
    """可选:抓取财经快讯用于事件归因。失败返回空列表,不影响主流程。"""
    try:
        import akshare as ak
        df = ak.stock_info_global_em().head(limit)
        title = df.columns[0]
        out = []
        for _, r in df.iterrows():
            text = " ".join(str(x) for x in r.values[:2])
            out.append(text)
        return out
    except Exception:
        return []


def sector_review(p, stocks, news=None, topk_leaders=3):
    c, a, h, l = p["close"], p["amount"], p["high"], p["low"]
    pct = c / c.shift(1) - 1
    ratio = ratio_series(stocks).reindex(c.columns)
    ind = stocks["industry"].reindex(c.columns).fillna("未知").values
    names = stocks["name"].reindex(c.columns)

    ind_ret = pct.T.groupby(ind).mean().T                       # 行业等权收益序列
    ind_amt = a.T.groupby(ind).sum().T
    ind_money = clv_money(p).T.groupby(ind).sum().T
    idx_curve = (1 + ind_ret.fillna(0)).cumprod()
    cnt = pd.Series(ind).value_counts()

    today = pct.iloc[-1]
    lu_today = (today >= ratio - EPS)
    g = pd.DataFrame({"pct": today, "ind": ind, "lu": lu_today, "up": today > 0, "amt": a.iloc[-1]})
    agg = g.groupby("ind").agg(pct_eq=("pct", "mean"), breadth=("up", "mean"), limit_up=("lu", "sum"))
    wavg = g.assign(w=g["pct"] * g["amt"]).groupby("ind").apply(lambda x: x["w"].sum() / max(x["amt"].sum(), 1), include_groups=False)
    agg["pct_w"] = wavg

    agg["ret5"] = idx_curve.iloc[-1] / idx_curve.iloc[-6] - 1
    agg["ret20"] = idx_curve.iloc[-1] / idx_curve.iloc[-21] - 1
    agg["amt_ratio"] = ind_amt.iloc[-1] / ind_amt.iloc[-21:-1].mean()
    share = ind_amt.div(ind_amt.sum(axis=1), axis=0)
    agg["crowding"] = (share.iloc[-120:].le(share.iloc[-1])).mean()
    agg["flow5"] = ind_money.iloc[-5:].sum() / ind_amt.iloc[-5:].sum()
    agg["n"] = cnt
    agg = agg[agg["n"] >= 3].copy()
    agg = agg.drop(index="未知", errors="ignore")

    rk = lambda s: s.rank(pct=True)
    agg["score"] = (0.25 * rk(agg["ret5"]) + 0.20 * rk(agg["pct_eq"]) + 0.20 * rk(agg["flow5"])
                    + 0.20 * rk(agg["breadth"]) + 0.15 * rk(agg["ret20"])) * 100
    agg.loc[agg["crowding"] > 0.9, "score"] -= 10                    # 拥挤度惩罚
    agg = agg.sort_values("score", ascending=False)

    leaders, reasons = {}, {}
    for k in agg.index:
        m = g[(g["ind"] == k)].copy()
        m["contrib"] = m["pct"] * m["amt"]
        top = m.sort_values("contrib", ascending=False if agg.loc[k, "pct_eq"] >= 0 else True).head(topk_leaders)
        leaders[k] = [f"{names.get(cd, cd)}({v:+.1%})" for cd, v in zip(top.index, top["pct"])]
        reasons[k] = _explain(k, agg.loc[k], leaders[k], news)
    agg["leaders"] = pd.Series(leaders)
    agg["reason"] = pd.Series(reasons)
    return agg


def _explain(name, r, leaders, news):
    parts = []
    d = "上涨" if r["pct_eq"] > 0 else "下跌"
    parts.append(f"板块等权{d}{abs(r['pct_eq']):.2%}")
    if r["amt_ratio"] > 1.3:
        parts.append(f"放量(成交额为20日均值{r['amt_ratio']:.1f}倍)")
    elif r["amt_ratio"] < 0.8:
        parts.append(f"缩量({r['amt_ratio']:.1f}倍)")
    if r["breadth"] >= 0.75:
        parts.append(f"普涨(上涨占比{r['breadth']:.0%})")
    elif r["breadth"] <= 0.25:
        parts.append(f"普跌(上涨占比仅{r['breadth']:.0%})")
    elif r["pct_w"] - r["pct_eq"] > 0.01:
        parts.append("权重/大成交股领涨,扩散度一般")
    if r["limit_up"] >= 3:
        parts.append(f"涨停{int(r['limit_up'])}家")
    if r["flow5"] > 0.05:
        parts.append("5日资金代理持续净流入")
    elif r["flow5"] < -0.05:
        parts.append("5日资金代理净流出")
    if r["crowding"] > 0.9:
        parts.append("成交占比处于近120日高位,拥挤度高,警惕分歧")
    s = ";".join(parts) + f"。{'领涨' if r['pct_eq'] >= 0 else '领跌'}:{'、'.join(leaders)}"
    if news:
        key = name.replace("行业", "")
        hits = [x[:60] for x in news if key and key in x][:2]
        if hits:
            s += ";相关快讯:" + " / ".join(hits)
    return s
