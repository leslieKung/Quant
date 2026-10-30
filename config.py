"""全局配置。个人使用,按需修改。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
DB_PATH = Path(os.environ.get("QUANT_DB", DATA_DIR / "quant.db"))

# ---------- 股票池 ----------
UNIVERSE_SIZE = 5000     # 按成交额取前N只(沪深主板/创业板/科创板,剔除ST/退市/北交所)。想覆盖全市场可改成5000,首次拉取会很慢
UNIVERSE_INDEXES = ["000985"]
#UNIVERSE_INDEXES = ["000906", "000852"]   # 中证800 + 中证1000 ≈ 1800 只(含大中小盘)
                                          # 想覆盖全市场改成 ["000985"](中证全指,约 5000 只)
HIST_YEARS = 3             # 首次拉取历史年数
WORKERS = 3                # 并发拉取线程数(过大易被封IP)

# ---------- 选股 ----------
HORIZON = 2                # 持有期(交易日),买入=T+1开盘,卖出=T+H收盘。必须>=2(T+1制度)
TOP_N = 10                 # 推荐数量
MAX_PER_INDUSTRY = 2       # 单行业最多入选数
MIN_AMOUNT20 = 1e8         # 近20日日均成交额下限(元)
MIN_LIST_DAYS = 120        # 上市交易日数下限(过滤次新)
BLEND_LGB = 0.6            # 最终得分 = 0.6*LightGBM + 0.4*ICIR加权线性

# ---------- 交易成本 ----------
COST = dict(commission=0.00025, stamp=0.0005, slippage=0.001)  # 佣金双边 / 印花税卖出 / 滑点单边

# ---------- 指数 ----------
BENCH = "sh000300"
INDEXES = {"sh000001": "上证指数", "sz399001": "深证成指",
           "sz399006": "创业板指", "sh000300": "沪深300"}

# ---------- 网络:国内数据源直连,绕过系统代理 ----------
BYPASS_PROXY = True   # 如果你必须通过代理才能上网,改成 False

if BYPASS_PROXY:
    import os
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        os.environ.pop(k, None)
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "*"
    import requests
    if not getattr(requests.Session, "_patched_no_proxy", False):
        _orig_init = requests.Session.__init__

        def _init(self, *a, **kw):
            _orig_init(self, *a, **kw)
            self.trust_env = False          # 不读环境变量/Windows注册表里的系统代理

        requests.Session.__init__ = _init
        requests.Session._patched_no_proxy = True

SOURCE_ORDER = ["sina"]      # 东财可用时可改为 ["em", "sina"]
