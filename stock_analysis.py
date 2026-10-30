"""单只股票分析:模型打分/因子归因 + 技术面 + 量价 + 板块与大盘环境 + 该股历史同类信号统计 -> 操作参考。
所有结论来自规则化计算,评分与加减分项全部透明展示;不是预测,不保证盈利。"""
import datetime as dt

import numpy as np
import pandas as pd

import config
import data
import market
from report import md_table, _pct

FACTOR_CN = {
    "mom5": "5日动量", "mom20": "20日动量", "mom60": "60日动量(跳过近5日)", "rev1": "1日反转",
    "vol20": "20日波动率", "maxret20": "20日最大单日涨幅", "atr_pct": "ATR波幅", "amihud": "Amihud非流动性",
    "turn20": "20日换手", "ln_amt20": "成交额规模", "vol_ratio": "量比", "turn_ratio": "换手放大",
    "obv_slope": "OBV斜率", "vp_corr": "量价相关", "ma_bias20": "MA20乖离", "pos60": "60日区间位置",
    "rsi14": "RSI14", "gap": "开盘跳空", "upper_shadow": "上影线", "close_pos": "收盘位置", "flow5": "5日资金代理"}


# ============================================================ 数据
def _fetch_fresh(code):
    today = dt.date.today()
    start = (today - dt.timedelta(days=365 * config.HIST_YEARS)).strftime("%Y%m%d")
    try:
        df = data.fetch_hist(code, start, today.strftime("%Y%m%d"))
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    df.index = pd.to_datetime(df["date"])
    return df.drop(columns=["date", "code"]).sort_index()


def load_series(code, p, fresh=True):
    cols = ["open", "high", "low", "close", "volume", "amount", "turnover"]
    in_panel = code in p["close"].columns
    fdf = _fetch_fresh(code) if fresh else None
    if fdf is not None and len(fdf) >= 80:
        return fdf, "实时拉取(新浪)", in_panel
    if in_panel:
        df = pd.DataFrame({k: p[k][code] for k in cols}).dropna(subset=["close"])
        return df, "本地库", in_panel
    raise RuntimeError(f"{code} 不在本地股票池,且实时拉取失败。请检查网络/代码,或把它加入股票池后再分析")


def _name(code, stocks):
    if code in stocks.index and isinstance(stocks.loc[code, "name"], str):
        return stocks.loc[code, "name"]
    try:
        import akshare as ak
        lst = ak.stock_info_a_code_name()
        m = lst[lst["code"] == code]
        if len(m):
            return str(m["name"].iloc[0])
    except Exception:
        pass
    return code


# ============================================================ 指标
def indicators(df):
    c, h, l, o, v, a, t = (df[k] for k in ["close", "high", "low", "open", "volume", "amount", "turnover"])
    I = pd.DataFrame(index=df.index)
    for n in (5, 10, 20, 60):
        I[f"ma{n}"] = c.rolling(n).mean()
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    I["dif"] = ema12 - ema26
    I["dea"] = I["dif"].ewm(span=9, adjust=False).mean()
    I["hist"] = (I["dif"] - I["dea"]) * 2
    rsv = (c - l.rolling(9).min()) / (h.rolling(9).max() - l.rolling(9).min()).replace(0, np.nan) * 100
    I["k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    I["d"] = I["k"].ewm(alpha=1 / 3, adjust=False).mean()
    I["j"] = 3 * I["k"] - 2 * I["d"]
    d = c.diff()
    up, dn = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(), (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    I["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    std = c.rolling(20).std()
    I["bb_up"], I["bb_lo"] = I["ma20"] + 2 * std, I["ma20"] - 2 * std
    I["pctb"] = (c - I["bb_lo"]) / (I["bb_up"] - I["bb_lo"]).replace(0, np.nan)
    I["bw"] = (I["bb_up"] - I["bb_lo"]) / I["ma20"]
    I["bw_pct"] = I["bw"].rolling(120, min_periods=60).rank(pct=True)
    pc = c.shift(1)
    tr = np.maximum(np.maximum(h - l, (h - pc).abs()), (l - pc).abs())
    I["atr"] = tr.rolling(14).mean()
    I["volr"] = v / v.rolling(5).mean().shift(1)
    I["vol_vs20"] = v / v.rolling(20).mean()
    I["amt_vs20"] = a / a.rolling(20).mean()
    I["turn_pct"] = t.rolling(120, min_periods=60).rank(pct=True)
    I["obv"] = (np.sign(d.fillna(0)) * v).cumsum()
    I["obv_rank"] = I["obv"].rolling(20).rank(pct=True)
    I["px_rank"] = c.rolling(20).rank(pct=True)
    clv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    money = clv.fillna(0) * a
    I["flow5"] = money.rolling(5).sum() / a.rolling(5).sum().replace(0, np.nan)
    I["flow20"] = money.rolling(20).sum() / a.rolling(20).sum().replace(0, np.nan)
    I["vwap20"] = (c * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    I["bias20"] = c / I["ma20"] - 1
    return I


def pivots(df, look=60, w=3):
    h, l = df["high"].values[-look:], df["low"].values[-look:]
    ph = [h[i] for i in range(w, len(h) - w) if h[i] == h[i - w:i + w + 1].max()]
    pl = [l[i] for i in range(w, len(l) - w) if l[i] == l[i - w:i + w + 1].min()]
    return ph, pl


def snapshot(df, I, code, name):
    c, o, h, l, v = (df[k] for k in ["close", "open", "high", "low", "volume"])
    s = {k: float(I[k].iloc[-1]) if np.isfinite(I[k].iloc[-1]) else np.nan for k in I.columns}
    s.update(close=float(c.iloc[-1]), prev=float(c.iloc[-2]), open=float(o.iloc[-1]), high=float(h.iloc[-1]),
             low=float(l.iloc[-1]), date=df.index[-1])
    s["pct"] = s["close"] / s["prev"] - 1
    ratio = data.limit_ratio(code, "ST" in name)
    s["ratio"] = ratio
    s["limit_up"] = s["pct"] >= ratio - 0.0015
    s["limit_dn"] = s["pct"] <= -ratio + 0.0015
    s["ma20_slope"] = I["ma20"].iloc[-1] / I["ma20"].iloc[-6] - 1
    s["bull"] = s["ma5"] > s["ma10"] > s["ma20"]
    s["bear"] = s["ma5"] < s["ma10"] < s["ma20"]
    cross = (I["dif"] > I["dea"]) & (I["dif"].shift() <= I["dea"].shift())
    dead = (I["dif"] < I["dea"]) & (I["dif"].shift() >= I["dea"].shift())
    s["macd_gold"], s["macd_dead"] = bool(cross.iloc[-3:].any()), bool(dead.iloc[-3:].any())
    s["kdj_gold"] = bool(((I["k"] > I["d"]) & (I["k"].shift() <= I["d"].shift())).iloc[-2:].any())
    s["kdj_dead"] = bool(((I["k"] < I["d"]) & (I["k"].shift() >= I["d"].shift())).iloc[-2:].any())
    # K线形态
    rng = max(s["high"] - s["low"], 1e-9)
    body = abs(s["close"] - s["open"])
    upper, lower = s["high"] - max(s["open"], s["close"]), min(s["open"], s["close"]) - s["low"]
    pat = []
    if s["pct"] >= 0.05 and (s["high"] - s["close"]) / rng < 0.2:
        pat.append("大阳线(收近高位)")
    if s["pct"] <= -0.05 and (s["close"] - s["low"]) / rng < 0.2:
        pat.append("大阴线(收近低位)")
    if rng / s["prev"] > 0.03 and upper / rng > 0.5:
        pat.append("长上影(上方抛压)")
    if rng / s["prev"] > 0.03 and lower / rng > 0.5:
        pat.append("长下影(下方承接)")
    if body / rng < 0.1:
        pat.append("十字星(多空僵持)")
    if s["low"] > float(h.iloc[-2]):
        pat.append("向上跳空缺口")
    if s["high"] < float(l.iloc[-2]):
        pat.append("向下跳空缺口")
    s["patterns"] = pat
    # 量价
    r10 = c.pct_change().iloc[-10:]
    v10 = v.iloc[-10:]
    um, dm = v10[r10 > 0].mean(), v10[r10 < 0].mean()
    s["updown_vol"] = float(um / dm) if np.isfinite(um) and np.isfinite(dm) and dm > 0 else np.nan
    s["vwap_dev"] = s["close"] / s["vwap20"] - 1
    s["h20_prev"] = float(h.iloc[-21:-1].max())
    s["l20_prev"] = float(l.iloc[-21:-1].min())
    s["breakout"] = bool(s["volr"] > 1.5 and s["close"] > s["h20_prev"])
    s["breakdown"] = bool(s["volr"] > 1.5 and s["close"] < s["l20_prev"])
    s["pullback"] = bool(abs(s["bias20"]) < 0.03 and s["vol_vs20"] < 0.7 and s["ma20_slope"] > 0)
    s["huge_vol"] = bool(v.iloc[-1] >= v.iloc[-60:].max())
    s["tiny_vol"] = bool(v.iloc[-1] <= v.iloc[-60:].min())
    s["upper_ratio"] = upper / rng
    s["obv_top_div"] = bool(I["px_rank"].iloc[-1] >= 0.9 and I["obv_rank"].iloc[-1] < 0.5)
    s["obv_bot_div"] = bool(I["px_rank"].iloc[-1] <= 0.1 and I["obv_rank"].iloc[-1] > 0.5)
    s["ret20"] = float(c.iloc[-1] / c.iloc[-21] - 1)
    s["ret60"] = float(c.iloc[-1] / c.iloc[-61] - 1)
    return s


def levels(df, s):
    close, atr = s["close"], s["atr"]
    ph, pl = pivots(df)
    h20, h60 = float(df["high"].iloc[-20:].max()), float(df["high"].iloc[-60:].max())
    res_c = sorted(x for x in ph + [h20, h60] if x > close * 1.003)
    sup_c = sorted((x for x in pl + [s["ma20"], s["ma60"], float(df["low"].iloc[-20:].min())]
                    if np.isfinite(x) and x < close * 0.997), reverse=True)
    res1 = res_c[0] if res_c else None
    res2 = next((x for x in res_c if res1 and x > res1 * 1.01), None)
    sup1 = sup_c[0] if sup_c else None
    stop_atr = close - 1.5 * atr
    stop = max(stop_atr, sup1 * 0.99) if sup1 else stop_atr
    stop = min(stop, close - 1.0 * atr)                     # 止损至少留 1 个 ATR,避免被正常波动洗出
    t1 = res1 if (res1 and res1 > close + 1.0 * atr) else close + 2.0 * atr
    t2 = max(res2 or 0, close + 4.0 * atr)
    risk = max(close - stop, 1e-9)
    pull_lo = min(s["ma10"], s["ma20"]) * 0.995
    pull_hi = max(s["ma10"], s["ma20"]) * 1.005
    return dict(res1=res1, res2=res2, sup1=sup1, stop=stop, t1=t1, t2=t2, rr=(t1 - close) / risk,
                stop_pct=risk / close, pull=(pull_lo, pull_hi) if s["close"] > s["ma20"] else None,
                h20=h20, h60=h60)


# ============================================================ 历史同类信号统计
def signal_stats(df, I, H=None):
    H = H or config.HORIZON
    c, o, v = df["close"], df["open"], df["volume"]
    fwd = c.shift(-H) / o.shift(-1) - 1
    sigs = {
        "MACD金叉": (I["dif"] > I["dea"]) & (I["dif"].shift() <= I["dea"].shift()),
        "放量突破20日高": (I["volr"] > 1.5) & (c > df["high"].rolling(20).max().shift(1)),
        "缩量回踩MA20": (I["bias20"].abs() < 0.03) & (I["vol_vs20"] < 0.7) & (I["ma20"] > I["ma20"].shift(5)),
        "RSI超卖(<30)": I["rsi14"] < 30,
        "布林下轨反弹": (c > I["bb_lo"]) & (c.shift() <= I["bb_lo"].shift()),
        "均线刚形成多头排列": ((I["ma5"] > I["ma10"]) & (I["ma10"] > I["ma20"])) &
                         ~((I["ma5"].shift() > I["ma10"].shift()) & (I["ma10"].shift() > I["ma20"].shift())),
        "放量跌破20日低": (I["volr"] > 1.5) & (c < df["low"].rolling(20).min().shift(1)),
    }
    base = fwd.mean()
    rows = []
    for name, sg in sigs.items():
        sg = sg.fillna(False)
        pos, last = [], -10 ** 9
        for k in np.where(sg.values & fwd.notna().values)[0]:     # 去聚集:H 日内重复触发只算一次
            if k - last >= H:
                pos.append(k)
                last = k
        r = fwd.iloc[pos]
        rows.append(dict(信号=name, 当前触发="是" if bool(sg.iloc[-1]) else "否", 次数=len(r),
                         平均=_pct(r.mean()) if len(r) else "-", 中位数=_pct(r.median()) if len(r) else "-",
                         胜率=f"{(r > 0).mean():.0%}" if len(r) else "-",
                         样本="不足(<10)" if len(r) < 10 else "可参考"))
    return pd.DataFrame(rows), base


# ============================================================ 评分
def score_trend(s):
    pts = []
    pts.append(("收盘>MA20", 10) if s["close"] > s["ma20"] else ("收盘<MA20", -10))
    pts.append(("收盘>MA60", 10) if s["close"] > s["ma60"] else ("收盘<MA60", -10))
    pts.append(("MA20向上", 8) if s["ma20_slope"] > 0 else ("MA20向下", -8))
    if s["bull"]:
        pts.append(("MA5>MA10>MA20多头排列", 10))
    if s["bear"]:
        pts.append(("MA5<MA10<MA20空头排列", -10))
    pts.append(("DIF>DEA", 6) if s["dif"] > s["dea"] else ("DIF<DEA", -6))
    pts.append(("DIF在零轴上", 4) if s["dif"] > 0 else ("DIF在零轴下", -4))
    if s["macd_gold"]:
        pts.append(("近3日MACD金叉", 5))
    if s["macd_dead"]:
        pts.append(("近3日MACD死叉", -5))
    if s["rsi14"] > 80:
        pts.append(("RSI>80过热", -8))
    elif s["rsi14"] < 20:
        pts.append(("RSI<20超卖", 4))
    if s["bias20"] > 0.15:
        pts.append(("乖离MA20>15%", -8))
    return float(np.clip(50 + sum(x for _, x in pts), 0, 100)), pts


def score_volprice(s):
    pts = []
    up, volr = s["pct"] > 0, s["volr"]
    if up and volr > 1.2:
        pts.append(("价涨量增", 12))
    elif up and volr < 0.8:
        pts.append(("价涨量缩", -3))
    elif (not up) and volr > 1.3:
        pts.append(("价跌量增(抛压)", -12))
    elif (not up) and volr < 0.8:
        pts.append(("价跌量缩", 3))
    if np.isfinite(s["updown_vol"]):
        if s["updown_vol"] > 1.3:
            pts.append(("近10日上涨日量能占优", 10))
        elif s["updown_vol"] < 0.8:
            pts.append(("近10日下跌日量能占优", -10))
    if s["obv_bot_div"]:
        pts.append(("OBV底背离", 8))
    if s["obv_top_div"]:
        pts.append(("OBV顶背离", -10))
    if s["flow5"] > 0.1:
        pts.append(("5日资金代理净流入", 6))
    elif s["flow5"] < -0.1:
        pts.append(("5日资金代理净流出", -6))
    if s["breakout"]:
        pts.append(("放量突破20日高", 10))
    if s["breakdown"]:
        pts.append(("放量跌破20日低", -12))
    if s["huge_vol"] and s["upper_ratio"] > 0.5:
        pts.append(("天量+长上影(见顶风险)", -8))
    return float(np.clip(50 + sum(x for _, x in pts), 0, 100)), pts


def score_env(s, mkt, sector_row, idx_ret20):
    pts = []
    if sector_row is not None:
        pts.append((f"所属板块强度{sector_row['score']:.0f}", round((sector_row["score"] - 50) * 0.5, 1)))
        if sector_row["crowding"] > 0.9:
            pts.append(("板块拥挤度>90%", -5))
    pts.append((f"市场情绪{mkt['sentiment']:.0f}", round((mkt["sentiment"] - 50) * 0.4, 1)))
    if np.isfinite(idx_ret20):
        rs = s["ret20"] - idx_ret20
        pts.append((f"20日相对沪深300 {rs:+.1%}", round(float(np.clip(rs * 100 * 0.5, -10, 10)), 1)))
    return float(np.clip(50 + sum(x for _, x in pts), 0, 100)), pts


def decide(model_pct, trend, vp, env, s, mkt, lv):
    w = dict(model=0.35, trend=0.25, vp=0.20, env=0.20)
    parts = dict(trend=trend, vp=vp, env=env)
    if np.isfinite(model_pct):
        parts["model"] = model_pct * 100
    else:
        w.pop("model")
    tot_w = sum(w.values())
    total = sum(parts[k] * w[k] for k in w) / tot_w
    notes = []
    if mkt["sentiment"] < 25:
        total = min(total, 55)
        notes.append("市场情绪处于冰点,综合分已封顶 55")
    if np.isfinite(s["ma60"]) and s["close"] < s["ma60"] and I_down(s):
        total = min(total, 52)
        notes.append("价格在MA60下方且MA20向下,综合分封顶 52(下降趋势中不抄底)")
    stance = "偏多" if total >= 70 else "中性偏多" if total >= 55 else "中性" if total >= 40 else "偏空"
    return total, stance, notes


def I_down(s):
    return s["ma20_slope"] < 0


# ============================================================ 建议
def advice(s, lv, stance, total, mkt, notes):
    close = s["close"]
    nh, hd = [], []         # 空仓者 / 持仓者
    pull = lv["pull"]
    pull_txt = f"{pull[0]:.2f}~{pull[1]:.2f}" if pull else None
    brk_txt = f"{lv['res1']:.2f}" if lv["res1"] else f"{s['h20_prev']:.2f}(阶段新高,上方无明显压力)"
    flags = []
    if s["limit_up"]:
        flags.append("今日涨停:次日开盘大概率高开,难以按理想价买入,不追;已持仓者看次日是否放量炸板")
    if s["limit_dn"]:
        flags.append("今日跌停:不抄底;已持仓者次日若反抽无力,以控制风险为先")
    if s["bias20"] > 0.15 or s["rsi14"] > 80:
        flags.append(f"短线过热(乖离MA20 {s['bias20']:+.1%},RSI {s['rsi14']:.0f}):不追高,等回踩")
    if s["obv_top_div"]:
        flags.append("OBV 顶背离:价格新高但量能没跟上,警惕假突破")

    if stance == "偏多":
        nh.append(f"可分批介入:① 回踩 {pull_txt} 缩量企稳时买第一笔(不超过计划仓位一半);" if pull_txt else
                  "可分批介入:等小幅回落、量能萎缩企稳时买第一笔(不超过计划仓位一半);")
        nh.append(f"② 或放量(量比>1.3)有效站上 {brk_txt} 后追加;未放量的突破不跟。")
        hd.append("继续持有;止损位上移到下方给出的止损价,不随意下调。")
        hd.append(f"到第一目标 {lv['t1']:.2f} 先减 1/3~1/2,剩余用 MA10 或「最高价-2×ATR」做移动止盈。")
    elif stance == "中性偏多":
        nh.append("以观察/试探为主:至多用小仓位(见下方仓位上限)参与,且必须等到下面任一条件:")
        nh.append(f"回踩 {pull_txt} 不破并缩量;" if pull_txt else "回落企稳并缩量;")
        nh.append(f"或放量站上 {brk_txt}。条件没出现就不动。")
        hd.append("可持有但不加仓;收盘跌破 MA20 且放量,或触及止损价,先减仓一半。")
        hd.append(f"反弹到 {lv['t1']:.2f} 附近若放量滞涨(上影线变长),逢高减仓。")
    elif stance == "中性":
        nh.append("观望,等方向选择:")
        nh.append(f"向上:放量站上 {brk_txt} 再考虑;向下:跌破 {lv['stop']:.2f} 则回避。")
        hd.append("不加仓。跌破止损价无条件执行;反弹到压力位附近(" + (f"{lv['res1']:.2f}" if lv["res1"] else f"{lv['t1']:.2f}") + ")可减仓降风险。")
    else:
        nh.append(f"回避,不抄底。等重新站上 MA20({s['ma20']:.2f})并放量后再评估。")
        hd.append(f"反弹至 MA5/MA10({s['ma5']:.2f}/{s['ma10']:.2f})或压力位({lv['res1']:.2f})" if lv["res1"] else
                  f"反弹至 MA5/MA10({s['ma5']:.2f}/{s['ma10']:.2f})")
        hd[-1] += "时减仓/清仓;止损价不可下移。"

    if s["limit_up"]:
        nh = [f"今日涨停,次日开盘价不可控,不追。若次日高开回落、缩量企稳在 {pull_txt or 'MA10 附近'} 再评估;连续一字板则根本买不到。"]
        hd.append(f"次日若高开后放量炸板/冲高回落,先减 1/3~1/2;封单稳固可继续持有,止损上移到今日最低价 {s['low']:.2f} 附近。")
    elif s["limit_dn"]:
        nh = ["今日跌停,不抄底。至少等跌停打开后企稳、缩量止跌,并重新站上 MA5 再评估。"]
    elif s["bias20"] > 0.15 or s["rsi14"] > 80:
        nh = [f"短线过热,不追高:等回踩 {pull_txt or 'MA10/MA20'} 缩量企稳后再考虑,此时的突破信号不跟。"]
        hd.append("乖离/RSI 偏高,可分批止盈一部分,剩余用 MA10 做移动止盈。")

    cap = {"偏多": 0.25, "中性偏多": 0.15, "中性": 0.08, "偏空": 0.0}[stance]
    risk_pos = 0.01 / lv["stop_pct"]
    env_f = float(np.clip(mkt["sentiment"] / 60, 0.25, 1.0))
    pos = min(cap, risk_pos) * env_f
    pos_txt = (f"单票仓位上限约 **{pos:.0%}** 总资金"
               f"(= min(档位上限 {cap:.0%}, 单笔风险1%÷止损幅度{lv['stop_pct']:.1%}={risk_pos:.0%}) × 市场情绪系数 {env_f:.2f})")
    if s["limit_up"] or s["limit_dn"]:
        pos_txt += "。今日涨/跌停,新建仓位建议为 0,上面数字仅供后续参考"
    invalid = [f"收盘跌破止损价 {lv['stop']:.2f}",
               f"放量(量比>1.5)跌破20日低点 {s['l20_prev']:.2f}",
               "所属板块转弱(强度分掉出前50%)且个股缩量冲高回落",
               "出现公告类利空/业绩预亏/减持/解禁等基本面事件(本程序未覆盖,需自行核对)"]
    return nh, hd, pos_txt, invalid, flags


# ============================================================ 图
def plot(df, I, path, name, code):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d, Ii = df.iloc[-120:], I.iloc[-120:]
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1.3]})
    ax[0].plot(d.index, d["close"], color="black", lw=1.2, label="Close")
    for n, col in [(5, "#e8a33d"), (10, "#3d8be8"), (20, "#c43de8"), (60, "#888888")]:
        ax[0].plot(Ii.index, Ii[f"ma{n}"], lw=0.9, color=col, label=f"MA{n}")
    ax[0].fill_between(Ii.index, Ii["bb_lo"], Ii["bb_up"], color="#cccccc", alpha=0.3, label="Bollinger")
    ax[0].legend(loc="upper left", ncol=6, fontsize=7)
    ax[0].set_title(f"{code}  (A-share colors: red=up, green=down)")
    up = d["close"] >= d["open"]
    ax[1].bar(d.index, d["volume"], color=np.where(up, "#d9363e", "#2fa05a"), width=0.8)
    ax[1].set_ylabel("Volume")
    ax[2].bar(Ii.index, Ii["hist"], color=np.where(Ii["hist"] >= 0, "#d9363e", "#2fa05a"), width=0.8)
    ax[2].plot(Ii.index, Ii["dif"], color="#3d8be8", lw=0.9)
    ax[2].plot(Ii.index, Ii["dea"], color="#e8a33d", lw=0.9)
    ax[2].set_ylabel("MACD")
    for a_ in ax:
        a_.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ============================================================ 解读文字
def _tech_rows(s):
    def trend_txt():
        if s["bull"]:
            return "多头排列(短期强)"
        if s["bear"]:
            return "空头排列(短期弱)"
        return "均线交织(方向不明)"
    macd = ("零轴上" if s["dif"] > 0 else "零轴下") + ("、近3日金叉" if s["macd_gold"] else "、近3日死叉" if s["macd_dead"] else
                                                   ("、DIF在DEA上方" if s["dif"] > s["dea"] else "、DIF在DEA下方"))
    kdj = "超买(J>100)" if s["j"] > 100 else "超卖(J<0)" if s["j"] < 0 else ("金叉" if s["kdj_gold"] else "死叉" if s["kdj_dead"] else "中性")
    rsi = "过热(>80)" if s["rsi14"] > 80 else "偏强(70~80)" if s["rsi14"] > 70 else "超卖(<20)" if s["rsi14"] < 20 else "偏弱(20~30)" if s["rsi14"] < 30 else "中性"
    bb = ("贴近上轨" if s["pctb"] > 0.9 else "贴近下轨" if s["pctb"] < 0.1 else "带内运行") + \
         ("、带宽收窄(蓄势,常有变盘)" if s["bw_pct"] < 0.2 else "、带宽扩张(波动放大)" if s["bw_pct"] > 0.8 else "")
    return [
        ("均线 MA5/10/20/60", f"{s['ma5']:.2f} / {s['ma10']:.2f} / {s['ma20']:.2f} / {s['ma60']:.2f}", trend_txt()),
        ("MA20 斜率(5日)", f"{s['ma20_slope']:+.2%}", "向上" if s["ma20_slope"] > 0 else "向下"),
        ("收盘 vs MA20 乖离", f"{s['bias20']:+.1%}", "乖离过大(>15%),回落风险" if s["bias20"] > 0.15 else "正常区间"),
        ("MACD DIF/DEA/柱", f"{s['dif']:.3f} / {s['dea']:.3f} / {s['hist']:.3f}", macd),
        ("KDJ K/D/J", f"{s['k']:.0f} / {s['d']:.0f} / {s['j']:.0f}", kdj),
        ("RSI14", f"{s['rsi14']:.0f}", rsi),
        ("布林 %B / 带宽分位", f"{s['pctb']:.2f} / {s['bw_pct']:.0%}", bb),
        ("ATR14", f"{s['atr']:.2f}({s['atr'] / s['close']:.1%})", "日均波幅,用于止损/目标"),
        ("今日K线", f"{s['pct']:+.2%}", "、".join(s["patterns"]) or "无特殊形态"),
    ]


def _vp_rows(s):
    rel = ("价涨量增(健康)" if s["pct"] > 0 and s["volr"] > 1.2 else "价涨量缩(动能不足)" if s["pct"] > 0 and s["volr"] < 0.8 else
           "价跌量增(抛压)" if s["pct"] < 0 and s["volr"] > 1.3 else "价跌量缩(惜售/缩量回调)" if s["pct"] < 0 and s["volr"] < 0.8 else "量价关系平淡")
    obv = "顶背离(价格高位、量能未跟)" if s["obv_top_div"] else "底背离(价格低位、量能转强)" if s["obv_bot_div"] else "无明显背离"
    special = [x for x, f in [("放量突破20日高", s["breakout"]), ("放量跌破20日低", s["breakdown"]), ("缩量回踩MA20", s["pullback"]),
                              ("60日天量", s["huge_vol"]), ("60日地量", s["tiny_vol"])] if f]
    return [
        ("量比(今/前5日均)", f"{s['volr']:.2f}", rel),
        ("成交额 / 20日均", f"{s['amt_vs20']:.2f}x", "放量" if s["amt_vs20"] > 1.5 else "缩量" if s["amt_vs20"] < 0.7 else "正常"),
        ("换手率120日分位", f"{s['turn_pct']:.0%}", "交投活跃" if s["turn_pct"] > 0.8 else "交投清淡" if s["turn_pct"] < 0.2 else "正常"),
        ("近10日 上涨日/下跌日均量", f"{s['updown_vol']:.2f}" if np.isfinite(s["updown_vol"]) else "-",
         "上涨放量占优" if s["updown_vol"] > 1.3 else "下跌放量占优" if s["updown_vol"] < 0.8 else "均衡"),
        ("OBV 与价格(20日)", f"价格分位{s['px_rank']:.0%} / OBV分位{s['obv_rank']:.0%}", obv),
        ("资金代理 5日/20日", f"{s['flow5']:+.2f} / {s['flow20']:+.2f}", "偏流入" if s["flow5"] > 0.1 else "偏流出" if s["flow5"] < -0.1 else "中性(CLV估算,非真实主力资金)"),
        ("收盘 vs 20日量加权均价", f"{s['vwap_dev']:+.1%}", "在成本区上方" if s["vwap_dev"] > 0 else "在成本区下方"),
        ("特殊量价信号", "、".join(special) or "无", ""),
    ]


# ============================================================ 主流程
def analyze(code, p, stocks, idx, use_model=True, fresh=True, with_plot=True):
    from factors import build_dataset
    import model

    df, src, in_panel = load_series(code, p, fresh)
    if len(df) < 80:
        raise RuntimeError(f"{code} 历史数据不足 80 个交易日,无法分析")
    name = _name(code, stocks)
    I = indicators(df)
    s = snapshot(df, I, code, name)

    mkt = market.market_review(p, stocks, idx)
    industry = stocks.loc[code, "industry"] if code in stocks.index else "未知"
    sec = None
    sector_row = None
    try:
        sec = market.sector_review(p, stocks)
        if industry in sec.index:
            sector_row = sec.loc[industry]
    except Exception:
        pass
    b = idx[idx["code"] == config.BENCH].set_index("date")["close"].dropna() if len(idx) else pd.Series(dtype=float)
    idx_r20 = float(b.iloc[-1] / b.iloc[-21] - 1) if len(b) > 21 else np.nan
    idx_r60 = float(b.iloc[-1] / b.iloc[-61] - 1) if len(b) > 61 else np.nan

    # ---- 模型
    mexp, model_note = None, ""
    if use_model:
        try:
            pp, st = p, stocks
            if not in_panel:   # 不在股票池:临时并入横截面(仅内存,不写库)
                pp = {k: v.copy() for k, v in p.items()}
                for k in pp:
                    pp[k][code] = df[k].reindex(pp[k].index)
                st = stocks.copy()
                st.loc[code, ["name", "industry", "is_st", "in_universe"]] = [name, "未知", int("ST" in name), 1]
            ds = build_dataset(pp, st)
            mexp = model.explain_latest(ds, code)
            if mexp and df.index[-1] > ds.dates[-1]:
                model_note = f"(注意:模型基于本地库截至 {ds.dates[-1]:%Y-%m-%d} 的数据,较新的行情请先运行 python main.py update)"
        except Exception as e:
            model_note = f"(模型打分失败:{type(e).__name__}: {e})"
    model_pct = mexp["score"] if mexp and np.isfinite(mexp.get("score", np.nan)) else np.nan
    why = ""
    if mexp is not None and not mexp["valid"]:
        rs = []
        a20 = float(df["amount"].iloc[-20:].mean())
        if a20 < config.MIN_AMOUNT20:
            rs.append(f"近20日日均成交额 {a20 / 1e8:.2f} 亿 < 门槛 {config.MIN_AMOUNT20 / 1e8:.1f} 亿")
        if len(df) < config.MIN_LIST_DAYS:
            rs.append(f"上市交易日 {len(df)} < {config.MIN_LIST_DAYS}(次新)")
        if "ST" in name:
            rs.append("ST 股")
        why = ";".join(rs) or "关键数据缺失"

    # ---- 评分、建议
    trend, tp = score_trend(s)
    vp, vpp = score_volprice(s)
    env, ep = score_env(s, mkt, sector_row, idx_r20)
    lv = levels(df, s)
    total, stance, notes = decide(model_pct, trend, vp, env, s, mkt, lv)
    nh, hd, pos_txt, invalid, flags = advice(s, lv, stance, total, mkt, notes)
    stats, base = signal_stats(df, I)

    # ---- 图
    chart = ""
    if with_plot:
        try:
            config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
            chart = f"stock_{code}_{s['date']:%Y-%m-%d}.png"
            plot(df, I, config.REPORT_DIR / chart, name, code)
        except Exception:
            chart = ""

    # ---- 报告
    fmt = lambda pts: "、".join(f"{k}({v:+g})" for k, v in pts) or "无"
    L = [f"# 个股分析:{name}({code})  {s['date']:%Y-%m-%d}", "",
         "> 规则化分析,仅供个人研究参考,不构成投资建议;不覆盖基本面/公告/解禁/减持等事件,操作前请自行核对。", "",
         f"数据来源:{src};行业:{industry};最新价 **{s['close']:.2f}**({s['pct']:+.2%})", ""]
    if chart:
        L += [f"![chart]({chart})", ""]
    L += ["## 结论速览", "",
          f"**综合评分 {total:.0f}/100 → 倾向「{stance}」**", "",
          md_table(pd.DataFrame([
              ["模型(全市场分位)", f"{model_pct * 100:.0f}" if np.isfinite(model_pct) else "无", "35%", f"第{mexp['rank']}/{mexp['n_valid']}名" if mexp and "rank" in mexp else why or model_note or "模型未运行"],
              ["趋势(均线/MACD/RSI)", f"{trend:.0f}", "25%", fmt(tp)],
              ["量价", f"{vp:.0f}", "20%", fmt(vpp)],
              ["环境(板块/大盘/相对强弱)", f"{env:.0f}", "20%", fmt(ep)]],
              columns=["维度", "得分", "权重", "加减分项/说明"])), ""]
    for n in notes + flags:
        L.append(f"- ⚠ {n}")
    L.append("")

    L += ["## 一、模型视角", ""]
    if mexp and mexp["valid"]:
        L += [f"- 模型分位 **{model_pct:.0%}**(LightGBM {mexp['lgb_pct']:.0%} / ICIR线性 {mexp['lin_pct']:.0%}),"
              f"在 {mexp['n_valid']} 只可交易股票中排第 **{mexp['rank']}**。目标是预测未来 {config.HORIZON} 日的截面相对强弱,"
              f"不是预测涨跌绝对值。{model_note}", ""]
        feats = mexp["feats"]
        key = mexp["shap"] if mexp["shap"] is not None else mexp["lin_contrib"]
        label = "LightGBM SHAP 贡献" if mexp["shap"] is not None else "ICIR线性贡献"
        order = np.argsort(key)
        def rows(ids):
            return pd.DataFrame([[FACTOR_CN.get(feats[i], feats[i]), f"{mexp['z'][i]:+.2f}", f"{key[i]:+.4f}"] for i in ids],
                                columns=["因子", "该股标准分(z)", label])
        L += ["推高得分的因子 Top5:", "", md_table(rows(order[::-1][:5])), "", "拉低得分的因子 Top5:", "", md_table(rows(order[:5])), "",
              "说明:z 为该因子在全市场(行业中性后)的标准分;SHAP 为该因子对模型打分的边际贡献,正值=推高。", ""]
    else:
        L += [f"- 模型未对该股打分:{why or '模型未运行(使用了 --no-model 或模型报错)'}。{model_note}",
              "- 综合评分已自动把模型权重分摊给其余三个维度。", ""]

    L += ["## 二、技术面", "", md_table(pd.DataFrame(_tech_rows(s), columns=["指标", "数值", "解读"])), ""]
    L += ["## 三、量价分析", "", md_table(pd.DataFrame(_vp_rows(s), columns=["指标", "数值", "解读"])), ""]

    L += ["## 四、相对强弱与环境", ""]
    rows = [["该股", _pct(s["ret20"]), _pct(s["ret60"])]]
    if np.isfinite(idx_r20):
        rows.append(["沪深300", _pct(idx_r20), _pct(idx_r60) if np.isfinite(idx_r60) else "-"])
    if sector_row is not None:
        rows.append([f"所属板块({industry},等权)", _pct(sector_row["ret20"]), "-"])
    L += [md_table(pd.DataFrame(rows, columns=["", "20日", "60日"])), ""]
    if sector_row is not None:
        rank = int(sec.index.get_loc(industry) + 1)
        L += [f"- 板块强度分 {sector_row['score']:.0f},排第 {rank}/{len(sec)};今日 {_pct(sector_row['pct_eq'])},5日 {_pct(sector_row['ret5'])},"
              f"拥挤度 {sector_row['crowding']:.0%},上涨占比 {sector_row['breadth']:.0%}", f"- 板块归因:{sector_row['reason']}"]
    else:
        L += ["- 板块信息不可用(不在股票池或行业未知)"]
    L += [f"- 大盘:情绪温度 {mkt['sentiment']:.0f}({mkt['label']});整体仓位参考 {mkt['position']}", ""]

    L += [f"## 五、该股历史同类信号表现(近{config.HIST_YEARS}年,持有{config.HORIZON}日,T+1开盘买入)", "",
          md_table(stats), "",
          f"该股无条件持有{config.HORIZON}日平均收益 {_pct(base)}。次数<10 的统计基本不可靠;信号过去有效不代表以后有效,仅作参考。", ""]

    L += ["## 六、操作参考", "", f"**当前倾向:{stance}(综合 {total:.0f})**", "",
          "### 价位(基于 ATR 与近期结构)", "",
          md_table(pd.DataFrame([
              ["最新价", f"{s['close']:.2f}"],
              ["下方支撑", f"{lv['sup1']:.2f}" if lv["sup1"] else "-"],
              ["上方压力 1 / 2", f"{lv['res1']:.2f}" if lv["res1"] else "无(阶段新高)" ] if not lv["res2"] else ["上方压力 1 / 2", f"{lv['res1']:.2f} / {lv['res2']:.2f}"],
              ["建议止损价", f"{lv['stop']:.2f}(距现价 {lv['stop_pct']:.1%})"],
              ["目标 1 / 目标 2", f"{lv['t1']:.2f} / {lv['t2']:.2f}"],
              ["盈亏比(到目标1)", f"{lv['rr']:.1f}"],
              ["回踩介入区", f"{lv['pull'][0]:.2f}~{lv['pull'][1]:.2f}" if lv["pull"] else "-"]],
              columns=["项目", "数值"])), ""]
    if lv["rr"] < 1.5:
        L += [f"- ⚠ 到目标1的盈亏比仅 {lv['rr']:.1f}(<1.5),性价比一般,宁可等更好的位置。", ""]
    L += ["### 空仓者", ""] + [f"- {x}" for x in nh] + ["", "### 已持仓者", ""] + [f"- {x}" for x in hd]
    L += ["", f"### 仓位", "", f"- {pos_txt}", "- 分批建仓,首笔不超过上限的一半;亏损时不加仓摊平。", ""]
    L += ["### 失效条件(出现任一条,以上判断作废)", ""] + [f"- {x}" for x in invalid] + [""]
    L += ["### 交易规则提醒", "",
          f"- T+1:今日买入明日才能卖;涨跌停幅度 ±{s['ratio']:.0%};止损位只能在下一交易日起执行,跳空时成交价会更差。",
          "- 本分析不含基本面、业绩预告、公告、解禁/减持、融资融券与龙虎榜,请在下单前自行核对。", ""]
    return "\n".join(L), s["date"]
