"""数据层:AKShare 拉取 -> SQLite 存储 -> 宽表面板。
行情默认走新浪(东财在部分网络下会断连),可在 config.SOURCE_ORDER 调整。"""
import sqlite3
import time
import threading
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily(
  date TEXT, code TEXT, open REAL, high REAL, low REAL, close REAL,
  volume REAL, amount REAL, turnover REAL, PRIMARY KEY(date, code));
CREATE TABLE IF NOT EXISTS stocks(
  code TEXT PRIMARY KEY, name TEXT, industry TEXT,
  is_st INTEGER DEFAULT 0, in_universe INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS index_daily(
  date TEXT, code TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
  PRIMARY KEY(date, code));
"""

DAILY_COLS = ["date", "code", "open", "high", "low", "close", "volume", "amount", "turnover"]
COLMAP = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
          "最低": "low", "成交量": "volume", "成交额": "amount", "换手率": "turnover"}
_sina_lock = threading.Lock()   # V8(py_mini_racer)不能多线程同时初始化,新浪接口必须串行

def get_conn():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(config.DB_PATH)
    c.executescript(SCHEMA)
    return c


def upsert(c, table, df, cols):
    if df is None or len(df) == 0:
        return
    sub = df[cols].astype(object)
    sub = sub.where(sub.notna(), None)
    q = f"INSERT OR REPLACE INTO {table}({','.join(cols)}) VALUES({','.join(['?'] * len(cols))})"
    c.executemany(q, sub.values.tolist())
    c.commit()


def limit_ratio(code, is_st=False):
    """涨跌停幅度。"""
    if code.startswith(("688", "689", "300", "301")):
        return 0.20
    if code[0] in "489":
        return 0.30
    return 0.05 if is_st else 0.10


# ------------------------------------------------------------------ 股票池
def _clean_cand(df, code_col, name_col):
    out = pd.DataFrame({"code": df[code_col].astype(str).str.extract(r"(\d{6})")[0],
                        "name": df[name_col].astype(str)}).dropna(subset=["code"])
    out = out[out["code"].str.startswith(("60", "00", "30", "68"))]
    out = out[~out["name"].str.contains("退", na=False)]
    out["is_st"] = out["name"].str.contains("ST", na=False).astype(int)
    return out[out["is_st"] == 0].drop_duplicates("code")


def build_universe():
    import akshare as ak
    cand, src = None, ""

    # 1) 东方财富:按成交额取前 N
    if "em" in getattr(config, "SOURCE_ORDER", ["em"]):
        try:
            spot = ak.stock_zh_a_spot_em()
            spot["amt"] = pd.to_numeric(spot["成交额"], errors="coerce").fillna(0)
            c1 = _clean_cand(spot, "代码", "名称").merge(spot[["代码", "amt"]], left_on="code", right_on="代码")
            cand = c1.nlargest(config.UNIVERSE_SIZE, "amt")[["code", "name", "is_st"]]
            src = "东方财富(成交额前N)"
        except Exception as e:
            print(f"[universe] 东方财富不可用: {type(e).__name__}")

    # 2) 中证指数成分股(按 config.UNIVERSE_INDEXES 合并去重,最多取 UNIVERSE_SIZE 只)
    if cand is None or cand.empty:
        parts, used = [], []
        for sym in getattr(config, "UNIVERSE_INDEXES", ["000906"]):
            try:
                cons = ak.index_stock_cons_csindex(symbol=sym)
                code_col = next(c for c in cons.columns if "代码" in c and "指数" not in c)
                name_col = next(c for c in cons.columns if "名称" in c and "指数" not in c and "英文" not in c)
                one = _clean_cand(cons, code_col, name_col)[["code", "name", "is_st"]]
                print(f"[universe] 指数 {sym}: {len(one)} 只")
                parts.append(one)
                used.append(sym)
            except Exception as e:
                print(f"[universe] 指数 {sym} 失败: {type(e).__name__}: {e}")
        if parts:
            cand = pd.concat(parts).drop_duplicates("code").head(config.UNIVERSE_SIZE)
            src = "中证指数成分股 " + "+".join(used)

    # 3) 交易所股票列表(无流动性排序,仅兜底)
    if cand is None or cand.empty:
        lst = ak.stock_info_a_code_name()
        cand = _clean_cand(lst, "code", "name")[["code", "name", "is_st"]]
        if len(cand) > config.UNIVERSE_SIZE:
            cand = cand.sample(config.UNIVERSE_SIZE, random_state=0)
        src = "交易所列表(随机抽样,未按流动性筛选)"

    if cand is None or cand.empty:
        raise RuntimeError("三个数据源都没拿到股票列表,请检查网络后重试")

    c = get_conn()
    old = dict(c.execute("SELECT code, industry FROM stocks").fetchall())
    c.execute("UPDATE stocks SET in_universe=0")
    c.commit()
    out = cand.copy()
    out["industry"] = out["code"].map(old).fillna("未知")
    out["in_universe"] = 1
    upsert(c, "stocks", out, ["code", "name", "industry", "is_st", "in_universe"])
    print(f"[universe] 来源: {src},共 {len(out)} 只")


def update_industry():
    """行业映射:东财优先(若在 SOURCE_ORDER 中),失败改用新浪行业。"""
    import akshare as ak
    mapping = {}
    if "em" in getattr(config, "SOURCE_ORDER", ["em"]):
        try:
            names = ak.stock_board_industry_name_em()["板块名称"].tolist()
            for i, nm in enumerate(names):
                for k in range(3):
                    try:
                        cons = ak.stock_board_industry_cons_em(symbol=nm)
                        for code in cons["代码"]:
                            mapping[str(code)] = nm
                        break
                    except Exception:
                        time.sleep(1 + k)
                time.sleep(0.3)
                if i % 20 == 0:
                    print(f"[industry] {i}/{len(names)}")
        except Exception as e:
            print(f"[industry] 东方财富不可用({type(e).__name__}),改用新浪行业分类")

    if len(mapping) < 100:
        try:
            sec = ak.stock_sector_spot(indicator="新浪行业")
            for label, name in zip(sec["label"], sec["板块"]):
                for k in range(3):
                    try:
                        d = ak.stock_sector_detail(sector=label)
                        for code in d["code"]:
                            mapping[str(code).zfill(6)] = name
                        break
                    except Exception:
                        time.sleep(1 + k)
                time.sleep(0.5)
        except Exception as e:
            print(f"[industry] 新浪行业也失败: {type(e).__name__};行业将标为「未知」,板块复盘不可用")

    c = get_conn()
    c.executemany("UPDATE stocks SET industry=? WHERE code=?", [(v, k) for k, v in mapping.items()])
    c.commit()
    print(f"[industry] 映射 {len(mapping)} 只")


# ------------------------------------------------------------------ 行情
_errors = []
_printed = {"n": 0}


def _log_err(msg):
    _errors.append(msg)
    if _printed["n"] < 3:                       # 只打印前几条,避免刷屏
        _printed["n"] += 1
        print("[daily][错误]", msg)


def _fetch_sina(code, start, end):
    import akshare as ak
    sym = ("sh" if code.startswith("6") else "sz") + code
    with _sina_lock:
        df = ak.stock_zh_a_daily(symbol=sym, start_date=start, end_date=end, adjust="qfq")
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=DAILY_COLS)
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"),
        "code": code,
        "open": pd.to_numeric(df["open"], errors="coerce"),
        "high": pd.to_numeric(df["high"], errors="coerce"),
        "low": pd.to_numeric(df["low"], errors="coerce"),
        "close": pd.to_numeric(df["close"], errors="coerce"),
        "volume": pd.to_numeric(df["volume"], errors="coerce") / 100,       # 股 -> 手
        "amount": pd.to_numeric(df["amount"], errors="coerce"),
        "turnover": pd.to_numeric(df["turnover"], errors="coerce") * 100,   # 比例 -> %
    })
    return out[DAILY_COLS]


def _fetch_em(code, start, end):
    import akshare as ak
    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
    if df is None or df.empty:
        return pd.DataFrame(columns=DAILY_COLS)
    df = df.rename(columns=COLMAP)[list(COLMAP.values())]
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["code"] = code
    return df[DAILY_COLS]


def fetch_hist(code, start, end):
    """按 config.SOURCE_ORDER 依次尝试数据源;任何异常都会记录,不再静默。"""
    for src in getattr(config, "SOURCE_ORDER", ["sina"]):
        fn = _fetch_sina if src == "sina" else _fetch_em
        for k in range(3):
            try:
                df = fn(code, start, end)
                if len(df):
                    return df
                _log_err(f"{src} {code}: 返回空表")
            except Exception as e:
                _log_err(f"{src} {code}: {type(e).__name__}: {str(e)[:150]}")
            time.sleep(0.5 + k)
    return None


def update_daily(full=False):
    c = get_conn()
    stats = {"ok": 0, "fail": 0}
    codes = [r[0] for r in c.execute("SELECT code FROM stocks WHERE in_universe=1")]
    if not codes:
        raise RuntimeError("股票池为空,请先运行: python main.py init")
    last = dict(c.execute("SELECT code, MAX(date) FROM daily GROUP BY code").fetchall())
    today = dt.date.today()
    end = today.strftime("%Y%m%d")
    full_start = (today - dt.timedelta(days=365 * config.HIST_YEARS)).strftime("%Y%m%d")

    def task(code, force_full):
        if code in last and not force_full:
            start = (pd.Timestamp(last[code]) - pd.Timedelta(days=10)).strftime("%Y%m%d")
        else:
            start = full_start
        return code, fetch_hist(code, start, end)

    def run(code_list, force_full):
        refetch, done = [], 0
        with ThreadPoolExecutor(config.WORKERS) as ex:
            futs = [ex.submit(task, cd, force_full) for cd in code_list]
            for f in as_completed(futs):
                code, df = f.result()
                done += 1
                if done % 100 == 0:
                    print(f"[daily] {done}/{len(code_list)}")
                if df is None or df.empty:
                    stats["fail"] += 1
                    continue
                stats["ok"] += 1
                # 前复权检测:除权后历史价格会整体变化,重叠区间对不上就整只重拉
                if code in last and not force_full:
                    old = pd.read_sql("SELECT date, close FROM daily WHERE code=? AND date>=?",
                                      c, params=(code, df["date"].min()))
                    m = old.merge(df[["date", "close"]], on="date", suffixes=("_o", "_n"))
                    if len(m) and (m["close_o"] / m["close_n"] - 1).abs().max() > 0.003:
                        refetch.append(code)
                        continue
                upsert(c, "daily", df, DAILY_COLS)
        return refetch

    refetch = run(codes, full)
    if refetch:
        print(f"[daily] 检测到除权,重拉 {len(refetch)} 只")
        c.executemany("DELETE FROM daily WHERE code=?", [(x,) for x in refetch])
        c.commit()
        last.clear()
        run(refetch, True)

    print(f"[daily] 成功 {stats['ok']} / 失败 {stats['fail']}")
    if stats["ok"] == 0:
        print("[daily] 最近的错误样例:")
        for e in _errors[-5:]:
            print("   ", e)
        raise RuntimeError("所有股票行情拉取均失败,请根据上面的错误信息排查网络/数据源")


def update_index():
    import akshare as ak
    c = get_conn()
    for code in config.INDEXES:
        df = None
        for k in range(3):
            try:
                df = ak.stock_zh_index_daily(symbol=code)
                break
            except Exception:
                time.sleep(1 + k)
        if df is None or df.empty:
            print(f"[index] {code} 拉取失败")
            continue
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["code"] = code
        df = df[df["date"] >= (dt.date.today() - dt.timedelta(days=365 * config.HIST_YEARS)).isoformat()]
        upsert(c, "index_daily", df, ["date", "code", "open", "high", "low", "close", "volume"])
    print("[index] 完成")


# ------------------------------------------------------------------ 读取
def load_panel():
    c = get_conn()
    stocks = pd.read_sql("SELECT * FROM stocks WHERE in_universe=1", c).set_index("code")
    df = pd.read_sql("SELECT * FROM daily", c, parse_dates=["date"])
    df = df[df["code"].isin(stocks.index)]
    if df.empty:
        raise RuntimeError("没有行情数据,请先运行: python main.py init")
    panel = {f: df.pivot(index="date", columns="code", values=f).sort_index()
             for f in ["open", "high", "low", "close", "volume", "amount", "turnover"]}
    codes = panel["close"].columns
    stocks = stocks.reindex(codes)
    stocks["industry"] = stocks["industry"].fillna("未知")
    idx = pd.read_sql("SELECT * FROM index_daily", c, parse_dates=["date"]).sort_values("date")
    return panel, stocks, idx


def ratio_series(stocks):
    return pd.Series({cd: limit_ratio(cd, bool(st)) for cd, st in zip(stocks.index, stocks["is_st"])})