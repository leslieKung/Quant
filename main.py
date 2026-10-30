"""A股量化复盘与选股工具(个人版)

  python main.py init            首次:建股票池 + 行业映射 + 拉取历史行情(耗时较长,约10~40分钟)
  python main.py update          增量更新行情/指数
  python main.py report          更新数据 -> 因子 -> 模型 -> 大盘/板块复盘 -> 次日关注清单 -> 报告
  python main.py report --no-update
  python main.py backtest        全量 walk-forward 样本外回测
  python main.py demo            用模拟数据跑通全流程(无需网络)
"""
import argparse
import sys
import time

import pandas as pd

import config
import data
from factors import build_dataset
from model import walk_forward, predict_latest, eval_oos, factor_report
from backtest import run_backtest, plot_equity
from market import market_review, sector_review, fetch_news
from picks import make_picks
import tracker
from report import build_report, save_report

pd.set_option("display.width", 200)


def cmd_init(_):
    data.get_conn()
    data.build_universe()
    data.update_industry()
    data.update_daily(full=True)
    data.update_index()


def cmd_update(_):
    data.update_daily()
    data.update_index()


def cmd_backtest(args):
    p, stocks, _ = data.load_panel()
    ds = build_dataset(p, stocks)
    t0 = time.time()
    print(f"[backtest] 数据集 {ds.X.shape},开始 walk-forward ...")
    score = walk_forward(ds)
    print(f"[backtest] 完成 {time.time() - t0:.0f}s")
    r_ic, st_ic = eval_oos(score, ds)
    r, st = run_backtest(score, p, stocks, ds.valid)
    print("\n样本外信号质量:", {k: round(float(v), 4) for k, v in st_ic.items()})
    print("回测结果:", {k: round(float(v), 4) for k, v in st.items()})
    config.REPORT_DIR.mkdir(exist_ok=True, parents=True)
    if len(r):
        plot_equity(r, config.REPORT_DIR / "backtest.png")
        r.to_csv(config.REPORT_DIR / "backtest_periods.csv")
        print(f"净值曲线: {config.REPORT_DIR / 'backtest.png'}")


def cmd_report(args):
    if not args.no_update and not args.demo:
        cmd_update(args)
    p, stocks, idx = data.load_panel()
    print(f"[report] 面板 {p['close'].shape},最新交易日 {p['close'].index[-1].date()}")
    ds = build_dataset(p, stocks)

    print("[report] 近 120 日样本外评估 ...")
    score_oos = walk_forward(ds, eval_days=120)
    _, oos_stat = eval_oos(score_oos, ds)
    fac = factor_report(ds)

    print("[report] 训练最新模型并打分 ...")
    score = predict_latest(ds)

    mkt = market_review(p, stocks, idx)
    news = [] if args.demo or args.no_news else fetch_news()
    sec = sector_review(p, stocks, news)
    picks, watch = make_picks(score, p, stocks, pd.Series(ds.valid[-1], index=ds.codes))

    conn = data.get_conn()
    tr = tracker.evaluate(conn, p, stocks, idx)  # 先复盘昨日及历史推荐(今日清单尚未入库,不会参与)
    tracking_md = tracker.to_markdown(tr, p["close"].index[-1])
    tracker.save_csv(tr)
    tracker.save_picks(conn, mkt["date"], picks)  # 再记录今日推荐,供下一交易日复盘

    md = build_report(mkt, sec, picks, watch, oos_stat, fac, tracking_md=tracking_md)
    path = save_report(md, mkt["date"].strftime("%Y-%m-%d"))
    print(f"[report] 已生成: {path}")
    print("\n" + md[:2500] + "\n...")

def cmd_review(args):
    """只复盘历史推荐(不训练模型、不生成新清单),很快。"""
    if not args.no_update:
        cmd_update(args)
    p, stocks, idx = data.load_panel()
    conn = data.get_conn()
    tr = tracker.evaluate(conn, p, stocks, idx)
    md = tracker.to_markdown(tr, p["close"].index[-1])
    tracker.save_csv(tr)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORT_DIR / f"{p['close'].index[-1]:%Y-%m-%d}_review.md"
    path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n已保存: {path}")

def cmd_stock(args):
    """单只股票分析:模型 + 技术面 + 量价 + 环境 -> 操作参考。"""
    import stock_analysis
    code = args.code.strip().zfill(6)
    p, stocks, idx = data.load_panel()
    md, d = stock_analysis.analyze(code, p, stocks, idx, use_model=not args.no_model,
                                   fresh=not args.no_fresh, with_plot=not args.no_plot)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORT_DIR / f"stock_{code}_{d:%Y-%m-%d}.md"
    path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n已保存: {path}")

def cmd_chars(args):
    """字辈/名称概念炒作扫描。"""
    import theme_chars as tc
    p, stocks, idx = data.load_panel()
    cache = tc.limit_streak(p, stocks)
    date = pd.Timestamp(args.date) if args.date else p["close"].index[-1]
    by = "pinyin" if args.pinyin else "char"
    res = tc.scan(tc.snapshot(p, stocks, date, cache), by=by)
    print(f"\n=== {date:%Y-%m-%d} 字辈扫描(按{'拼音' if args.pinyin else '汉字'};样本 {len(stocks)} 只)===")
    if res.empty:
        print("没有满足样本量条件的分组")
    else:
        cols = ["key", "n", "limit_up", "mean_pct", "qval", "industry_spread", "leader", "label"]
        print(res.head(args.top)[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        hit = res[res["label"] != "—"]
        for r in hit.head(3).itertuples():
            print(f"\n[{r.key}] 成员(涨幅前10): {'、'.join(r.members)}")
        if hit.empty:
            print("\n今日没有统计上显著的字辈异动(已做多重检验校正)。")
    if args.history or args.study:
        hist = tc.scan_history(p, stocks, days=args.history or 60, by=by)
        print("\n=== 历史回放:每日最显著的字辈 ===")
        print(hist.to_string(index=False, float_format=lambda x: f"{x:.3f}") if len(hist) else "区间内没有显著事件")
        if args.study and len(hist):
            ev = hist[hist["label"].str.startswith("名字驱动")]
            _, summ = tc.event_study(p, stocks, ev, by=by)
            print(f"\n=== 事件研究:{len(ev)} 个名字驱动事件,T+1开盘买,持有{config.HORIZON}日(已扣成本)===")
            print(summ.to_string(float_format=lambda x: f"{x:.3f}") if len(summ) else "没有已到期的事件")

def cmd_short(args):
    """短线策略:回测 + 当日信号扫描。"""
    import short_term as st
    p, stocks, idx = data.load_panel()
    X = st.prep(p, stocks)
    dts = p["close"].index
    print(f"\n=== 短线策略扫描 {dts[-1]:%Y-%m-%d} | 样本 {len(stocks)} 只 ===")
    print(f"市场情绪:涨停 {int(X['lu_cnt'].iloc[-1])} / 跌停 {int(X['ld_cnt'].iloc[-1])};"
          f"昨日涨停股今日平均 {X['prev_perf'].iloc[-1]:+.2%};首板接力的情绪条件:"
          f"{'满足' if st.mood_ok(X).iloc[-1] else '不满足'}")
    names = list(st.STRATS) if args.strategy == "all" else [args.strategy]
    for nm in names:
        cfg = dict(st.STRATS[nm])
        hold = args.hold or cfg["hold"]
        stop = cfg["stop"] if args.stop is None else args.stop
        sig, score = cfg["fn"](X)
        print(f"\n##### {cfg['label']}  持有{hold}日 | 止损{stop:.0%} | 买入高开区间 {cfg['gap'][0]:+.0%}~{cfg['gap'][1]:+.0%}")
        if not args.no_backtest:
            tr = st.backtest(X, stocks, sig, score, hold=hold, gap=cfg["gap"], stop=stop, top=args.top, slip=cfg["slip"])
            if args.recent and len(tr):
                tr = tr[tr["signal_date"] >= dts[-args.recent]]
            sm = st.summarize(tr)
            if sm:
                rng_txt = f"最近{args.recent}个交易日" if args.recent else "全部历史"
                print(f"回测({rng_txt}):", " | ".join(
                    f"{k} {int(v)}" if k in ("笔数", "信号日数") else f"{k} {v:.2f}" if k == "盈亏比" else f"{k} {v:.2%}"
                    for k, v in sm.items()))
                sp_ = st.split_by_mood(tr)
                if len(sp_):
                    print(sp_.to_string(float_format=lambda x: f"{x:.3f}"))
                if sm["笔数"] < 30:
                    print("⚠ 成交笔数<30,统计意义很弱,不要据此下结论")
            else:
                print("回测:没有满足条件的成交")
        t = st.scan_today(X, stocks, sig, score, top=args.top)
        if len(t):
            print(f"今日信号 {len(t)} 只(次日买入条件见上面的高开区间;一字涨停买不进;最早T+2才能卖):")
            print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        else:
            print("今日无信号")

def cmd_policy(args):
    """政策主题股票池:低位放量筛选(只看价量结构,不含基本面)。"""
    import policy_screen as ps
    p, stocks, idx = data.load_panel()
    themes = None if args.theme == "all" else [args.theme]
    res, skipped = ps.run(p, stocks, themes=themes, fresh=not args.no_fresh, low=args.low, vol=args.vol)
    if res.empty:
        print("没有可分析的股票:本地库没有这些股票且未联网拉取。去掉 --no-fresh,或扩大股票池后重试")
        return
    show = res if args.all else res[res["判断"] != "—"]
    cols = ["主题", "代码", "名称", "判断", "pos250", "dd", "ret5", "ret20", "ret60", "volr", "v5_60", "综合分", "来源", "备注"]
    print(f"\n=== 政策主题低位放量筛选 | 数据截至 {res['date'].max():%Y-%m-%d} | 分析 {len(res)} 只(跳过 {len(skipped)} 只)===")
    print("pos250=近250日价格分位(0=最低);dd=距250日高点;volr=今日量比;v5_60=5日均量/60日均量")
    print(show[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}") if len(show) else "没有满足条件的股票")
    if skipped:
        print("\n跳过(取不到数据):", "、".join(skipped[:15]), "..." if len(skipped) > 15 else "")
    print("\n⚠ 股票池来自公开梳理,仅作线索;低位放量只是价量结构,不代表政策兑现或业绩改善。")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("init").set_defaults(f=cmd_init)
    sp.add_parser("update").set_defaults(f=cmd_update)
    sp.add_parser("backtest").set_defaults(f=cmd_backtest)
    r = sp.add_parser("report")
    r.add_argument("--no-update", action="store_true")
    r.add_argument("--no-news", action="store_true")
    r.set_defaults(f=cmd_report, demo=False)
    rv = sp.add_parser("review")
    rv.add_argument("--no-update", action="store_true")
    rv.set_defaults(f=cmd_review)
    st = sp.add_parser("stock")
    st.add_argument("code", help="6位股票代码,如 600519")
    st.add_argument("--no-model", action="store_true", help="跳过模型打分(更快)")
    st.add_argument("--no-fresh", action="store_true", help="不实时拉取,只用本地库数据")
    st.add_argument("--no-plot", action="store_true")
    st.set_defaults(f=cmd_stock)
    ch = sp.add_parser("chars")
    ch.add_argument("--date", help="扫描日期 YYYY-MM-DD,默认最新交易日")
    ch.add_argument("--pinyin", action="store_true", help="按拼音归并(羊洋阳杨→yang),需 pip install pypinyin")
    ch.add_argument("--history", type=int, default=0, help="历史回放最近N个交易日")
    ch.add_argument("--study", action="store_true", help="事件研究:检测后的真实后续表现")
    ch.add_argument("--top", type=int, default=10)
    ch.set_defaults(f=cmd_chars)
    sh = sp.add_parser("short")
    sh.add_argument("--strategy", default="all", choices=["all", "first_board", "pullback", "breakout"])
    sh.add_argument("--hold", type=int, default=0, help="持有天数(含买入日,>=2),默认用各策略自带值")
    sh.add_argument("--stop", type=float, default=None, help="止损比例,如 0.05;0 表示不止损")
    sh.add_argument("--top", type=int, default=5, help="每日最多买几只 / 今日信号显示几只")
    sh.add_argument("--recent", type=int, default=0, help="只统计最近N个交易日的信号,观察近期市场适应性")
    sh.add_argument("--no-backtest", action="store_true")
    sh.set_defaults(f=cmd_short)
    po = sp.add_parser("policy")
    po.add_argument("--theme", default="all", help="主题关键字:电网 / 管网 / 电缆 / 管材 / 智慧 / 工程 ...")
    po.add_argument("--no-fresh", action="store_true", help="不联网,只用本地库里有的股票")
    po.add_argument("--low", type=float, default=0.35, help="低位阈值(250日价格分位),默认0.35")
    po.add_argument("--vol", type=float, default=1.3, help="放量阈值(5日均量/60日均量),默认1.3")
    po.add_argument("--all", action="store_true", help="显示全部,含未命中任何条件的")
    po.set_defaults(f=cmd_policy)
    sp.add_parser("demo").set_defaults(f=None)
    args = ap.parse_args()

    if args.cmd == "demo":
        from demo import make_demo
        make_demo()
        a = argparse.Namespace(no_update=True, no_news=True, demo=True)
        cmd_backtest(a)
        cmd_report(a)
        return
    args.f(args)


if __name__ == "__main__":
    main()
