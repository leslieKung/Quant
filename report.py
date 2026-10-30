"""生成 Markdown + HTML 复盘报告。所有数字来自量化计算,文字仅做模板表述。"""
import datetime as dt

import numpy as np
import pandas as pd

import config


def _pct(x, d=2):
    return "-" if x is None or x != x else f"{x:+.{d}%}"


def md_table(df, headers=None):
    headers = headers or list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(x) for x in r.values) + " |")
    return "\n".join(lines)


def build_report(mkt, sectors, picks, watch, oos_stat, factor_df, bt_stat=None, tracking_md=""):
    d = mkt["date"].strftime("%Y-%m-%d")
    L = [f"# A股每日复盘与次日关注清单({d})", "",
         "> 本报告由量化程序自动生成,仅供个人研究,不构成投资建议。股市有风险,任何模型都会失效。", ""]
    if tracking_md:
        L += [tracking_md, ""]

    # 1 大盘
    L += ["## 一、大盘复盘", ""]
    idx = mkt["index_df"].copy()
    if len(idx):
        t = pd.DataFrame({"指数": idx["name"], "收盘": idx["close"].map("{:.2f}".format),
                          "涨跌": idx["pct"].map(_pct), "5日": idx["ret5"].map(_pct),
                          "MA5": idx["ma5"].map("{:.0f}".format), "MA20": idx["ma20"].map("{:.0f}".format)})
        L += [md_table(t), ""]
    L += [f"- 样本内({mkt['sample_n']}只高流动性个股)上涨 **{mkt['up']}** / 下跌 **{mkt['down']}** / 平 {mkt['flat']}",
          f"- 涨停 **{mkt['limit_up']}** / 跌停 **{mkt['limit_down']}** / 炸板 {mkt['broken']}(封板率 "
          f"{mkt['seal_rate']:.0%})" if mkt['seal_rate'] == mkt['seal_rate'] else
          f"- 涨停 {mkt['limit_up']} / 跌停 {mkt['limit_down']}",
          f"- 样本成交额 {mkt['amount'] / 1e8:.0f} 亿,处于近60日 **{mkt['amount_pct']:.0%}** 分位",
          f"- 赚钱效应(昨日涨停股今日平均表现):{_pct(mkt['prev_lu_perf'])}",
          f"- **情绪温度 {mkt['sentiment']:.0f}/100 —— {mkt['label']}**,次日建议总仓位参考:**{mkt['position']}**", ""]
    L += ["情绪温度构成:", "",
          md_table(pd.DataFrame({"分项": list(mkt["parts"].keys()),
                                 "得分": [f"{v:.0f}" for v in mkt["parts"].values()]})), ""]

    # 2 板块
    L += ["## 二、板块复盘", ""]
    cols = {"pct_eq": "涨跌", "ret5": "5日", "ret20": "20日", "breadth": "上涨占比", "limit_up": "涨停",
            "amt_ratio": "量比", "crowding": "拥挤度", "score": "强度分"}

    def fmt(df):
        t = pd.DataFrame(index=df.index)
        t["板块"] = df.index
        t["涨跌"] = df["pct_eq"].map(_pct)
        t["5日"] = df["ret5"].map(_pct)
        t["20日"] = df["ret20"].map(_pct)
        t["上涨占比"] = df["breadth"].map("{:.0%}".format)
        t["涨停"] = df["limit_up"].astype(int)
        t["量比"] = df["amt_ratio"].map("{:.2f}".format)
        t["拥挤度"] = df["crowding"].map("{:.0%}".format)
        t["强度分"] = df["score"].map("{:.0f}".format)
        return t

    top = sectors.sort_values("pct_eq", ascending=False)
    L += ["### 今日涨幅前8", "", md_table(fmt(top.head(8))), "", "### 今日跌幅前8", "",
          md_table(fmt(top.tail(8).iloc[::-1])), ""]
    L += ["### 原因归因(统计 + 快讯匹配)", ""]
    for k in list(top.head(5).index) + list(top.tail(3).index):
        L.append(f"- **{k}**:{sectors.loc[k, 'reason']}")
    L += ["", "### 综合强度 Top5(动量+资金代理+扩散度,已扣拥挤度)", "",
          md_table(fmt(sectors.head(5))), ""]
    L += ["> 注:资金流为 CLV 代理指标(收盘在日内区间的位置 × 成交额),并非交易所主力资金口径。", ""]

    # 3 情景
    L += ["## 三、次日情景推演", ""]
    bi = mkt["index_df"][mkt["index_df"]["code"] == config.BENCH]
    if len(bi):
        b = bi.iloc[0]
        L += [f"- **基准**:{'沿5日线上方震荡偏强' if b['close'] > b['ma5'] else '5日线下方整理,先看能否收复'}"
              f"(沪深300 MA5 {b['ma5']:.0f}),情绪「{mkt['label']}」。",
              f"- **乐观**:放量站上近20日高点 {b['high20']:.0f},且涨停数 > 今日 {mkt['limit_up']}、炸板率下降 → 可按上限仓位执行。",
              f"- **悲观**:跌破近5日低点 {b['low5']:.0f} 或跌停数明显增加 → 降到仓位区间下沿,清单标的触发失效条件立即止损。", ""]

    # 4 清单
    L += ["## 四、次日关注清单(量化模型 Top 列表)", "",
          f"持有期假设 {config.HORIZON} 个交易日(T+1 开盘介入),已剔除 ST/次新/低流动性/今日涨停(难以买入),单行业最多 {config.MAX_PER_INDUSTRY} 只。", ""]
    if len(picks):
        t = pd.DataFrame({
            "代码": picks["code"], "名称": picks["name"], "行业": picks["industry"],
            "模型分": picks["score"].map("{:.2f}".format), "收盘": picks["close"].map("{:.2f}".format),
            "今日": picks["pct"].map(_pct), "量比": picks["vol_ratio"].map("{:.2f}".format),
            "支撑": picks["support"].map("{:.2f}".format), "压力": picks["resistance"].map("{:.2f}".format),
            "止损参考": picks["stop"].map("{:.2f}".format), "目标参考": picks["target"].map("{:.2f}".format),
            "信号": picks["tags"]})
        L += [md_table(t), "", "**失效条件**:", ""]
        for _, r in picks.iterrows():
            L.append(f"- {r['code']} {r['name']}:{r['invalid']}")
        L.append("")
    if len(watch):
        L += ["今日涨停但模型分靠前(次日难买入,仅观察,不追):" +
              "、".join(f"{r['code']}{r['name']}" for _, r in watch.iterrows()), ""]

    # 5 模型表现
    L += ["## 五、模型自检(样本外)", ""]
    if oos_stat:
        L += [f"- 近 {oos_stat['n_days']} 个交易日样本外:RankIC 均值 **{oos_stat['rank_ic_mean']:.3f}**,"
              f"ICIR {oos_stat['icir']:.2f},IC>0 占比 {oos_stat['ic_pos_ratio']:.0%}",
              f"- Top{config.TOP_N} 相对全样本平均超额(每{config.HORIZON}日):{_pct(oos_stat['top_excess_mean'])},"
              f"跑赢概率 {oos_stat['top_win_rate']:.0%}",
              "- 若 RankIC 均值长期 < 0.02 或 ICIR < 0.3,说明模型已失效,应降低仓位或重训/停用。", ""]
    if factor_df is not None and len(factor_df):
        f = factor_df.head(8).reset_index()
        f.columns = ["因子", "近250日IC均值", "ICIR"]
        f["近250日IC均值"] = f["近250日IC均值"].map("{:+.4f}".format)
        f["ICIR"] = f["ICIR"].map("{:+.2f}".format)
        L += ["因子 IC 监控(按 |ICIR| 排序,用于观察因子衰减):", "", md_table(f), ""]
    if bt_stat:
        L += [f"回测摘要:年化 {_pct(bt_stat['annual_return'])},基准 {_pct(bt_stat['bench_annual'])},"
              f"夏普 {bt_stat['sharpe']:.2f},最大回撤 {_pct(bt_stat['max_drawdown'])}", ""]
    L += ["---", f"生成时间:{dt.datetime.now():%Y-%m-%d %H:%M}。数据:AKShare(免费源,可能存在缺失/延迟,请自行核对)。"]
    return "\n".join(L)


def save_report(md_text, date_str):
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = config.REPORT_DIR / f"{date_str}.md"
    md_path.write_text(md_text, encoding="utf-8")
    try:
        import markdown
        body = markdown.markdown(md_text, extensions=["tables"])
        html = ("<html><head><meta charset='utf-8'><style>body{font-family:-apple-system,'PingFang SC',"
                "'Microsoft YaHei',sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;line-height:1.6}"
                "table{border-collapse:collapse;font-size:13px;margin:8px 0}td,th{border:1px solid #ccc;"
                "padding:4px 8px}th{background:#f3f3f3}blockquote{color:#a33;border-left:3px solid #a33;"
                "margin:8px 0;padding-left:10px}</style></head><body>" + body + "</body></html>")
        (config.REPORT_DIR / f"{date_str}.html").write_text(html, encoding="utf-8")
    except ImportError:
        pass
    return md_path
