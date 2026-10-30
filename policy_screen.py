"""政策主题股票池 + "低位放量"筛选。
股票池来自公开报道/自媒体的受益方向梳理,不是推荐名单,代码与名称请自行核对;筛选只看价量结构,不含基本面。"""
import numpy as np
import pandas as pd

import config

POOL = {
    "电网-特高压/变压器": {"600406": "国电南瑞", "000400": "许继电气", "600312": "平高电气", "601179": "中国西电",
                         "600089": "特变电工", "002028": "思源电气"},
    "电网-电缆": {"600487": "亨通光电", "600522": "中天科技", "603606": "东方电缆", "600973": "宝胜股份",
                "605222": "起帆电缆", "002498": "汉缆股份"},
    "电网-配网/电表": {"601222": "林洋能源", "300360": "炬华科技", "601567": "三星医疗", "603556": "海兴电力"},
    "管网-管材": {"002457": "青龙管业", "603856": "东宏股份", "002372": "伟星新材", "002641": "公元股份",
                "300599": "雄塑科技", "000778": "新兴铸管"},
    "管网-检测修复/防水": {"002671": "龙泉股份", "603700": "宁水集团", "002271": "东方雨虹", "300198": "纳川股份",
                       "603903": "中持股份"},
    "管网-智慧管网": {"300007": "汉威科技", "300349": "金卡智能", "300075": "数字政通", "300137": "先河环保"},
    "管网-工程总包": {"600820": "隧道股份", "601668": "中国建筑", "601800": "中国交建", "601390": "中国中铁",
                    "601186": "中国铁建", "600170": "上海建工", "600629": "华建集团"},
    "管网-水务/燃气": {"600008": "首创环保", "000598": "兴蓉环境", "603393": "新天然气", "600903": "贵州燃气"},
}
PRIORITY = {"低位放量上涨(重点观察)": 0, "中低位温和放量": 1, "低位放量但短期已涨多(追高风险)": 2,
            "低位放量下跌(警惕)": 3, "低位缩量(资金未关注)": 4, "高位放量(可能已被炒作)": 5, "—": 6}


def _series(code, p, fresh):
    cols = ["open", "high", "low", "close", "volume", "amount"]
    if code in p["close"].columns:
        df = pd.DataFrame({k: p[k][code] for k in cols}).dropna(subset=["close"])
        if len(df) >= 120:
            return df, "本地库"
    if fresh:
        import stock_analysis as sa
        df = sa._fetch_fresh(code)
        if df is not None and len(df) >= 120:
            return df[cols], "实时拉取"
    return None, ""


def metrics(df):
    c, v, h, l, a = df["close"], df["volume"], df["high"], df["low"], df["amount"]
    w = c.iloc[-250:]
    hi, lo = w.max(), w.min()
    ma60, ma120 = c.rolling(60).mean().iloc[-1], c.rolling(120).mean().iloc[-1]
    r = c.pct_change()
    r10, v10 = r.iloc[-10:], v.iloc[-10:]
    up, dn = v10[r10 > 0].mean(), v10[r10 < 0].mean()
    clv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    return dict(close=c.iloc[-1], pos250=(c.iloc[-1] - lo) / (hi - lo) if hi > lo else np.nan,
                dd=c.iloc[-1] / hi - 1, ret5=c.iloc[-1] / c.iloc[-6] - 1, ret20=c.iloc[-1] / c.iloc[-21] - 1,
                ret60=c.iloc[-1] / c.iloc[-61] - 1,
                volr=v.iloc[-1] / v.iloc[-6:-1].mean(), v5_60=v.iloc[-5:].mean() / v.iloc[-60:].mean(),
                amt_vs20=a.iloc[-1] / a.iloc[-21:-1].mean(),
                big_vol_days=int((v / v.rolling(20).mean().shift(1) > 1.5).iloc[-5:].sum()),
                updown=up / dn if np.isfinite(up) and np.isfinite(dn) and dn > 0 else np.nan,
                flow5=(clv.fillna(0) * a).iloc[-5:].sum() / a.iloc[-5:].sum(),
                above_ma60=bool(c.iloc[-1] > ma60), above_ma120=bool(c.iloc[-1] > ma120), date=df.index[-1])


def classify(m, low=0.35, vol=1.3):
    pos, vv = m["pos250"], m["v5_60"]
    if pos <= low and vv >= vol:
        if m["ret5"] > 0:
            return "低位放量上涨(重点观察)" if m["ret20"] < 0.25 else "低位放量但短期已涨多(追高风险)"
        return "低位放量下跌(警惕)"
    if pos <= 0.5 and 1.1 <= vv < vol:
        return "中低位温和放量"
    if pos > 0.7 and vv >= vol:
        return "高位放量(可能已被炒作)"
    if pos <= low and vv < 0.9:
        return "低位缩量(资金未关注)"
    return "—"


def score(m):
    low = 1 - np.clip(m["pos250"], 0, 1)
    volp = float(np.clip((m["v5_60"] - 1) / 1.0, 0, 1))
    conf = np.mean([m["ret5"] > 0, m["above_ma60"], (m["updown"] > 1) if np.isfinite(m["updown"]) else False,
                    m["flow5"] > 0])
    return 0.4 * low + 0.35 * volp + 0.25 * conf


def run(p, stocks, themes=None, fresh=True, low=0.35, vol=1.3):
    rows, skipped = [], []
    for theme, d in POOL.items():
        if themes and not any(t in theme for t in themes):
            continue
        for code, nm in d.items():
            df, src = _series(code, p, fresh)
            if df is None:
                skipped.append(f"{code}{nm}")
                continue
            real = stocks["name"].get(code) if code in stocks.index else None
            warn = "" if (real is None or real == nm) else f"名称不符:库内为{real}"
            m = metrics(df)
            rows.append(dict(主题=theme, 代码=code, 名称=nm, 来源=src, 备注=warn, 判断=classify(m, low, vol),
                             综合分=score(m), **{k: m[k] for k in ("close", "pos250", "dd", "ret5", "ret20", "ret60",
                                                                  "volr", "v5_60", "amt_vs20", "big_vol_days", "updown", "flow5", "date")}))
    res = pd.DataFrame(rows)
    if len(res):
        res["_o"] = res["判断"].map(PRIORITY)
        res = res.sort_values(["_o", "综合分"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)
    return res, skipped
