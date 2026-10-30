"""因子库 + 预处理(MAD去极值 / z-score / 行业中性化)+ 标签 + 数据集构建。"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config


def _rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def true_range(p):
    h, l, pc = p["high"], p["low"], p["close"].shift(1)
    return np.maximum(np.maximum(h - l, (h - pc).abs()), (l - pc).abs())


def atr14(p):
    return true_range(p).rolling(14).mean()


def clv_money(p):
    """CLV资金代理:amount * ((C-L)-(H-C))/(H-L)。免费数据拿不到真实主力资金,这里用日内收盘位置估算。"""
    h, l, c, a = p["high"], p["low"], p["close"], p["amount"]
    clv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    return clv.fillna(0) * a


def compute_factors(p):
    c, o, h, l, v, a, t = (p[k] for k in ["close", "open", "high", "low", "volume", "amount", "turnover"])
    pc = c.shift(1)
    ret = c.pct_change(fill_method=None)
    F = {}
    # 动量 / 反转
    F["mom5"] = c.pct_change(5, fill_method=None)
    F["mom20"] = c.pct_change(20, fill_method=None)
    F["mom60"] = c.shift(5) / c.shift(60) - 1            # 跳过最近5日,避免与短期反转重叠
    F["rev1"] = -ret
    # 波动
    F["vol20"] = ret.rolling(20).std()
    F["maxret20"] = ret.rolling(20).max()
    F["atr_pct"] = atr14(p) / c
    # 流动性
    F["amihud"] = (ret.abs() / a.replace(0, np.nan)).rolling(20).mean() * 1e8
    F["turn20"] = t.rolling(20).mean()
    F["ln_amt20"] = np.log(a.rolling(20).mean().replace(0, np.nan))
    # 量价
    F["vol_ratio"] = v / v.rolling(5).mean().shift(1)                       # 量比
    F["turn_ratio"] = t.rolling(5).mean() / t.rolling(20).mean()            # 换手放大
    obv = (np.sign(ret.fillna(0)) * v).cumsum()
    F["obv_slope"] = (obv - obv.shift(20)) / v.rolling(20).sum().replace(0, np.nan)
    F["vp_corr"] = ret.rolling(20).corr(v.pct_change(fill_method=None))     # 量价相关
    # 形态
    F["ma_bias20"] = c / c.rolling(20).mean() - 1
    hh, ll = c.rolling(60).max(), c.rolling(60).min()
    F["pos60"] = (c - ll) / (hh - ll).replace(0, np.nan)
    F["rsi14"] = _rsi(c)
    F["gap"] = o / pc - 1
    F["upper_shadow"] = (h - np.maximum(o, c)) / c
    F["close_pos"] = (c - l) / (h - l).replace(0, np.nan)
    # 资金代理
    m = clv_money(p)
    F["flow5"] = m.rolling(5).sum() / a.rolling(5).sum().replace(0, np.nan)
    return F


def tradable_mask(p, stocks):
    a, c = p["amount"], p["close"]
    ok = (a.rolling(20).mean() >= config.MIN_AMOUNT20) & c.notna() & (a > 0)
    ok &= c.notna().cumsum() >= config.MIN_LIST_DAYS
    ok &= ~stocks["is_st"].astype(bool).reindex(c.columns).fillna(False).values[None, :]
    return ok


def mad_clip(df, n=5.0):
    med = df.median(axis=1)
    mad = df.sub(med, axis=0).abs().median(axis=1) * 1.4826
    return df.clip(lower=med - n * mad, upper=med + n * mad, axis=0)


def zscore(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


@dataclass
class Dataset:
    X: np.ndarray        # (D, n, F)
    y: np.ndarray        # (D, n) 未来收益截面排名(0~1)
    fwd: np.ndarray      # (D, n) 未来H日收益(T+1开盘买,T+H收盘卖)
    valid: np.ndarray    # (D, n) 可交易掩码
    dates: pd.DatetimeIndex
    codes: pd.Index
    feats: list


def build_dataset(p, stocks, H=None):
    H = H or config.HORIZON
    assert H >= 2, "T+1 制度下持有期必须>=2"
    F = compute_factors(p)
    codes, dates = p["close"].columns, p["close"].index
    D, n = len(dates), len(codes)
    ind = stocks["industry"].reindex(codes).fillna("未知").values

    cols = []
    for k, df in F.items():
        df = df.replace([np.inf, -np.inf], np.nan).reindex(index=dates, columns=codes)
        df = zscore(mad_clip(df))
        if k != "ln_amt20":  # 规模因子保留原始暴露,其余做行业中性
            ind_mean = df.T.groupby(ind).transform("mean").T
            df = zscore(df - ind_mean)
        cols.append(df.values)
    X = np.stack(cols, axis=2).astype(np.float32)

    fwd = (p["close"].shift(-H) / p["open"].shift(-1) - 1)
    y = fwd.rank(axis=1, pct=True)
    valid = tradable_mask(p, stocks).values
    return Dataset(X, y.values.astype(np.float32), fwd.values, valid, dates, codes, list(F.keys()))
