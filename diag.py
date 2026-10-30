import config, traceback, pandas as pd
import data

c = data.get_conn()
print("DB路径:", config.DB_PATH)
print("股票池:", c.execute("select in_universe,count(*) from stocks group by 1").fetchall())
print("daily行数:", c.execute("select count(*) from daily").fetchone())
print("index行数:", c.execute("select count(*) from index_daily").fetchone())
sample = [r[0] for r in c.execute("select code from stocks where in_universe=1 limit 3")]
print("样例代码:", sample)

import akshare as ak
print("akshare版本:", ak.__version__)
code = sample[0] if sample else "600519"
tests = {
    "东财": lambda: ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20250101", end_date="20260101", adjust="qfq"),
    "新浪": lambda: ak.stock_zh_a_daily(symbol=("sh" if code.startswith("6") else "sz") + code, start_date="20250101", end_date="20260101", adjust="qfq"),
}
for name, fn in tests.items():
    try:
        df = fn()
        print(f"\n[{name}] 成功 shape={df.shape}\n列名={list(df.columns)}\n{df.tail(2)}")
    except Exception:
        print(f"\n[{name}] 失败:")
        traceback.print_exc(limit=3)

try:
    df = data.fetch_hist(code, "20250101", "20260101")
    print("\nfetch_hist返回:", None if df is None else df.shape)
    if df is not None and len(df):
        data.upsert(c, "daily", df, data.DAILY_COLS)
        print("写入后daily行数:", c.execute("select count(*) from daily").fetchone())
except Exception:
    print("\nfetch_hist/upsert 异常:")
    traceback.print_exc()