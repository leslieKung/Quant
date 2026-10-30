"""生成模拟行情写入 data/demo.db,用于无网络/无数据时验证整条流水线(数据为随机生成,不代表真实市场)。"""
import numpy as np
import pandas as pd

import config
import data

INDUSTRIES = ["半导体", "白酒", "银行", "光伏设备", "汽车整车", "医疗器械", "证券", "煤炭行业", "软件开发", "军工电子",
              "化学制药", "房地产", "电力", "有色金属", "通信设备", "游戏", "食品加工", "家电", "钢铁", "航运港口"]


def make_demo(n_stocks=300, n_days=620, seed=7):
    config.DB_PATH = config.DATA_DIR / "demo.db"
    if config.DB_PATH.exists():
        config.DB_PATH.unlink()
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    codes = [f"{600000 + i:06d}" if i % 3 else f"{300000 + i:06d}" for i in range(n_stocks)]
    ind_id = rng.integers(0, len(INDUSTRIES), n_stocks)
    lim = np.array([data.limit_ratio(c) for c in codes])

    mkt = rng.normal(0.0003, 0.011, n_days)
    ind = rng.normal(0, 0.007, (n_days, len(INDUSTRIES)))
    for t in range(1, n_days):                                  # 行业动量
        ind[t] += 0.1 * ind[t - 1]
    beta = rng.uniform(0.7, 1.3, n_stocks)
    size = rng.normal(0, 1, n_stocks)
    ret = np.zeros((n_days, n_stocks))
    for t in range(n_days):
        idio = rng.normal(0, 0.016, n_stocks)
        ret[t] = beta * mkt[t] + ind[t, ind_id] + idio + (-0.04 * ret[t - 1] if t > 0 else 0)  # 仅含极弱短期反转
    ret = np.clip(ret, -lim, lim)
    close = 10 * np.exp(np.cumsum(np.log1p(ret), axis=0)) * np.exp(size)[None, :]
    prev = np.vstack([close[0], close[:-1]])
    open_ = prev * (1 + rng.normal(0, 0.004, close.shape))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, close.shape)))
    amount = np.exp(rng.normal(19.5, 0.5, n_stocks))[None, :] * np.exp(rng.normal(0, 0.35, close.shape)) * (1 + 5 * np.abs(ret))
    volume = amount / close / 100
    turnover = amount / (np.exp(21.0 + size))[None, :] * 100

    c = data.get_conn()
    stocks = pd.DataFrame({"code": codes, "name": [f"模拟{i:03d}" for i in range(n_stocks)],
                           "industry": [INDUSTRIES[i] for i in ind_id], "is_st": 0, "in_universe": 1})
    data.upsert(c, "stocks", stocks, ["code", "name", "industry", "is_st", "in_universe"])
    long = pd.DataFrame({
        "date": np.repeat(dates.strftime("%Y-%m-%d"), n_stocks), "code": np.tile(codes, n_days),
        "open": open_.ravel(), "high": high.ravel(), "low": low.ravel(), "close": close.ravel(),
        "volume": volume.ravel(), "amount": amount.ravel(), "turnover": turnover.ravel()})
    data.upsert(c, "daily", long, data.DAILY_COLS)

    ew = np.cumprod(1 + ret.mean(axis=1)) * 4000
    for code, k in zip(config.INDEXES, [0.8, 1.2, 1.5, 1.0]):
        px = np.cumprod(1 + (ret.mean(axis=1) * k + rng.normal(0, 0.002, n_days))) * 3000
        df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "code": code, "open": px, "high": px * 1.005,
                           "low": px * 0.995, "close": px, "volume": 1e9})
        data.upsert(c, "index_daily", df, ["date", "code", "open", "high", "low", "close", "volume"])
    print(f"[demo] 已生成模拟数据: {config.DB_PATH}")
