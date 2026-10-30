"""推荐跟踪:每日清单入库 -> 之后用真实行情复盘。
规则与回测一致:信号日收盘后推荐,次日开盘买入(T+1),持有 HORIZON 个交易日后收盘卖出;
开盘涨停买不进;到期日收盘跌停顺延一天;止损/目标只统计买入日"之后"的最高最低价(T+1 当日买入不能卖);
收益已扣佣金/印花税/滑点,基准为同期样本股等权与沪深300。"""
import numpy as np
import pandas as pd

import config
import data
from data import ratio_series
from report import md_table, _pct

LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS picks_log(
  signal_date TEXT, code TEXT, name TEXT, industry TEXT, rk INTEGER,
  score REAL, close REAL, stop REAL, target REAL, PRIMARY KEY(signal_date, code));
"""
LOG_COLS = ["signal_date", "code", "name", "industry", "rk", "score", "close", "stop", "target"]


def save_picks(c, signal_date, picks):
    c.executescript(LOG_SCHEMA)
    if picks is None or len(picks) == 0:
        return
    d = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
    df = picks.reset_index(drop=True).copy()
    df["signal_date"] = d
    df["rk"] = np.arange(1, len(df) + 1)
    c.execute("DELETE FROM picks_log WHERE signal_date=?", (d,))      # 同日重跑覆盖
    c.commit()
    data.upsert(c, "picks_log", df, LOG_COLS)


def _finite(a):
    return a[np.isfinite(a)]


def evaluate(c, p, stocks, idx, H=None):
    """对 picks_log 中所有已有后续交易日的推荐做复盘,返回明细 DataFrame(无记录返回 None)。"""
    H = H or config.HORIZON
    c.executescript(LOG_SCHEMA)
    log = pd.read_sql("SELECT * FROM picks_log ORDER BY signal_date, rk", c, parse_dates=["signal_date"])
    if log.empty:
        return None
    dates = p["close"].index
    last = len(dates) - 1
    Oa, Ha, La, Ca = (p[k].values for k in ["open", "high", "low", "close"])
    col = {cd: i for i, cd in enumerate(p["close"].columns)}
    ratio = ratio_series(stocks).reindex(p["close"].columns).values
    cst = config.COST

    b = None
    if idx is not None and len(idx):
        b = idx[idx["code"] == config.BENCH].drop_duplicates("date").set_index("date").sort_index()

    def idx_ret(d0, d1):
        try:
            return b.loc[d1, "close"] / b.loc[d0, "open"] - 1
        except Exception:
            return np.nan

    rows = []
    for sd, grp in log.groupby("signal_date"):
        i = dates.searchsorted(sd)
        if i >= len(dates) or dates[i] != sd:
            continue
        e = i + 1
        if e > last:
            continue                                   # 还没有后续交易日
        end = min(i + H, last)
        matured = (i + H) <= last
        u = np.isfinite(Oa[e]) & np.isfinite(Ca[end])
        bench_ew = float(np.mean(Ca[end][u] / Oa[e][u] - 1)) if u.any() else np.nan
        bench_idx = idx_ret(dates[e], dates[end])

        for r in grp.itertuples():
            k = col.get(r.code)
            if k is None:
                continue
            entry, pc = Oa[e, k], Ca[i, k]
            rec = dict(signal_date=sd, entry_date=dates[e], code=r.code, name=r.name, industry=r.industry,
                       rk=r.rk, score=r.score, rec_close=r.close, stop=r.stop, target=r.target,
                       matured=matured, bench_ew=bench_ew, bench_idx=bench_idx)
            gap = entry / pc - 1 if np.isfinite(entry) and np.isfinite(pc) else np.nan
            rec["gap"] = gap
            if not np.isfinite(entry):
                rec.update(buyable=False, note="次日停牌", exit_date=dates[end], held_days=0)
                rows.append(rec)
                continue
            if gap >= ratio[k] - 0.0015:
                rec.update(buyable=False, note="开盘涨停,买不进")
            else:
                rec.update(buyable=True, note="")
            x = end
            if matured and x + 1 <= last and np.isfinite(Ca[x - 1, k]) and Ca[x, k] / Ca[x - 1, k] - 1 <= -ratio[k] + 0.0015:
                x += 1                                  # 到期日跌停卖不出,顺延
            cl = _finite(Ca[e:x + 1, k])
            ex = cl[-1] if len(cl) else np.nan          # 停牌则取最后有效收盘
            lows, highs = La[e:x + 1, k], Ha[e:x + 1, k]
            lows2, highs2 = _finite(La[e + 1:x + 1, k]), _finite(Ha[e + 1:x + 1, k])
            net = (ex * (1 - cst["slippage"]) * (1 - cst["commission"] - cst["stamp"])) / \
                  (entry * (1 + cst["slippage"]) * (1 + cst["commission"])) - 1
            rec.update(
                entry=entry, day1=Ca[e, k] / entry - 1, exit_date=dates[x], exit_px=ex, held_days=x - e + 1,
                gross=ex / entry - 1, net=net,
                max_gain=np.nanmax(highs) / entry - 1 if np.isfinite(highs).any() else np.nan,
                max_dd=np.nanmin(lows) / entry - 1 if np.isfinite(lows).any() else np.nan,
                stop_hit=bool(len(lows2) and lows2.min() <= r.stop),
                target_hit=bool(len(highs2) and highs2.max() >= r.target),
                day1_below_stop=bool(Ca[e, k] <= r.stop))
            rows.append(rec)

    res = pd.DataFrame(rows)
    if res.empty:
        return None
    res["excess_ew"] = res["net"] - res["bench_ew"]
    res["excess_idx"] = res["net"] - res["bench_idx"]
    return res


def to_markdown(res, last_date, H=None):
    H = H or config.HORIZON
    L = ["## 〇、昨日推荐复盘(真实行情回看)", ""]
    if res is None or res.empty:
        L += ["暂无可复盘的历史推荐。从今天起程序会自动记录每日清单,下一个交易日收盘后即可看到复盘。", ""]
        return "\n".join(L)

    y = res[res["entry_date"] == last_date]
    if y.empty:
        L += ["昨日没有推荐记录(可能昨日未运行日报),下面仅展示历史累计跟踪。", ""]
    else:
        sd = y["signal_date"].iloc[0]
        L += [f"信号日 {sd:%Y-%m-%d} 收盘后推荐,{last_date:%Y-%m-%d} 开盘买入。"
              f"注意 T+1:买入当日不能卖出,止损只能从下一交易日起执行。", ""]
        rows = []
        for r in y.sort_values("rk").itertuples():
            if not r.buyable:
                st = r.note
            elif r.day1_below_stop:
                st = "收盘低于止损参考"
            else:
                st = "盈利" if r.net > 0 else "亏损"
            rows.append(dict(
                排名=r.rk, 代码=r.code, 名称=r.name, 行业=r.industry, 推荐价=f"{r.rec_close:.2f}",
                跳空=_pct(r.gap), 买入价=f"{r.entry:.2f}" if np.isfinite(getattr(r, "entry", np.nan)) else "-",
                当日=_pct(getattr(r, "day1", np.nan)), 净收益=_pct(getattr(r, "net", np.nan)) if r.buyable else "-",
                最高=_pct(getattr(r, "max_gain", np.nan)), 最低=_pct(getattr(r, "max_dd", np.nan)), 状态=st))
        L += [md_table(pd.DataFrame(rows)), ""]
        t = y[y["buyable"]]
        if len(t):
            best, worst = t.loc[t["net"].idxmax()], t.loc[t["net"].idxmin()]
            L += [f"- 可成交 **{len(t)}/{len(y)}** 只;平均净收益 **{_pct(t['net'].mean())}**,胜率 {(t['net'] > 0).mean():.0%}",
                  f"- 同期基准:样本等权 {_pct(t['bench_ew'].iloc[0])},沪深300 {_pct(t['bench_idx'].iloc[0])}"
                  f" → 相对等权超额 **{_pct(t['excess_ew'].mean())}**",
                  f"- 最好:{best['code']} {best['name']} {_pct(best['net'])};最差:{worst['code']} {worst['name']} {_pct(worst['net'])}",
                  "- 单日结果噪音很大,模型按 " + f"{H} 日持有期设计,请看下面的累计跟踪。", ""]
        else:
            L += ["昨日清单全部无法成交。", ""]

    # 累计跟踪
    t = res[res["buyable"]]
    if len(t):
        g = t.groupby("signal_date").agg(n=("code", "count"), net=("net", "mean"), bench=("bench_ew", "first"),
                                         win=("net", lambda s: (s > 0).mean()), matured=("matured", "first"),
                                         held=("held_days", "max")).tail(10)
        tab = pd.DataFrame({
            "信号日": [f"{d:%m-%d}" for d in g.index], "只数": g["n"].values,
            "状态": [("已到期" if m else f"持有{int(h)}/{H}天") for m, h in zip(g["matured"], g["held"])],
            "平均净收益": g["net"].map(_pct).values, "等权基准": g["bench"].map(_pct).values,
            "超额": (g["net"] - g["bench"]).map(_pct).values, "胜率": g["win"].map("{:.0%}".format).values})
        L += ["### 历史推荐累计跟踪(最近10个信号日)", "", md_table(tab), ""]
        m = t[t["matured"]]
        if len(m):
            L += [f"已到期 **{m['signal_date'].nunique()}** 个信号日、**{len(m)}** 只推荐:"
                  f"平均净收益 **{_pct(m['net'].mean())}**(中位数 {_pct(m['net'].median())}),胜率 {(m['net'] > 0).mean():.0%};"
                  f"平均超额(对等权) **{_pct(m['excess_ew'].mean())}**,跑赢等权概率 {(m['excess_ew'] > 0).mean():.0%};"
                  f"期间触及止损参考 {m['stop_hit'].mean():.0%},触及目标参考 {m['target_hit'].mean():.0%}。",
                  "样本少于 30 只时统计意义很弱,至少积累 3 个月再判断模型是否有效。", ""]
        else:
            L += [f"还没有到期({H}个交易日)的推荐,暂无累计统计。", ""]
    return "\n".join(L)


def save_csv(res):
    if res is None or res.empty:
        return
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(config.REPORT_DIR / "tracking.csv", index=False, encoding="utf-8-sig")