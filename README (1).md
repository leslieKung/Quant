# A股量化复盘与选股工具(个人版)

单机运行:免费行情(AKShare/新浪)→ SQLite → 21 个因子 → LightGBM + ICIR 线性合成 → 大盘情绪/板块复盘 → 次日关注清单 → **昨日推荐复盘** → 单股深度分析 → Markdown/HTML 报告。

> 仅供个人学习研究,不构成投资建议。量化模型会失效,股市有风险。若要向他人提供买卖建议,涉及证券投资咨询资质,请先咨询合规。

---

## 一、功能一览

| 命令 | 作用 | 耗时 |
|---|---|---|
| `python3 main.py demo` | 用模拟数据跑通全流程(无需网络),验证环境 | 1~2 分钟 |
| `python3 main.py init` | 首次:建股票池 + 行业映射 + 拉取 3 年历史行情 | 10~40 分钟 |
| `python3 main.py update` | 增量更新行情和指数 | 几分钟 |
| `python3 main.py report` | **每日主流程**:更新数据 → 复盘昨日推荐 → 大盘/板块复盘 → 模型选股 → 生成报告并记录今日清单 | 几分钟 |
| `python3 main.py review` | 只复盘历史推荐,不训练模型 | 很快 |
| `python3 main.py stock 600519` | 单只股票分析:模型 + 技术面 + 量价 + 环境 + 操作参考 | 10~60 秒 |
| `python3 main.py backtest` | 全量 walk-forward 样本外回测 | 数分钟 |
| `python3 check_data.py` | 数据体检(只读,不改数据) | 几秒 |

常用参数:

```bash
python3 -u main.py report --no-update      # 不更新数据,直接用库里的
python3 -u main.py report --no-news        # 不抓新闻快讯
python3 -u main.py review --no-update
python3 -u main.py stock 600519 --no-model # 跳过模型,更快
python3 -u main.py stock 600519 --no-fresh # 不联网,只用本地库
python3 -u main.py stock 600519 --no-plot  # 不生成图
python3 check_data.py --db data/demo.db    # 体检指定数据库
```

> 建议加 `-u` 运行(不缓冲输出),这样拉取进度能实时看到。

---

## 二、安装

```bash
cd ashare_quant
python3 -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` 必须包含 `scikit-learn`(LightGBM 的依赖),完整清单:

```
akshare>=1.14
pandas>=2.0
numpy>=1.24
scipy>=1.10
lightgbm>=4.0
scikit-learn>=1.3
matplotlib>=3.7
markdown>=3.4
```

macOS 若 LightGBM 报 `libomp` 相关错误:`brew install libomp`。

---

## 三、推荐配置(`config.py`)

下面这几项与网络环境直接相关,请确认存在:

```python
UNIVERSE_SIZE = 800          # 股票池大小(按成交额取前N;东财不可用时退化为中证800成分股)
HIST_YEARS = 3               # 首次拉取历史年数
WORKERS = 3                  # 并发线程数。新浪限流较严,3 比较稳;不稳就改 1
SOURCE_ORDER = ["sina"]      # 行情数据源顺序;东财可用时可改为 ["em", "sina"]

HORIZON = 5                  # 持有期(交易日):T+1 开盘买入,T+H 收盘卖出,必须 >= 2
TOP_N = 10                   # 每日推荐数量
MAX_PER_INDUSTRY = 2         # 单行业最多入选
MIN_AMOUNT20 = 1e8           # 近20日日均成交额下限(元)
MIN_LIST_DAYS = 120          # 上市交易日下限(过滤次新)
```

文件末尾建议加上(国内数据源直连,绕过系统代理;并给网络请求设超时):

```python
BYPASS_PROXY = True          # 必须通过代理上网时改为 False
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
            self.trust_env = False
        requests.Session.__init__ = _init
        requests.Session._patched_no_proxy = True

import socket
socket.setdefaulttimeout(20)
```

---

## 四、每日使用流程

```
收盘后 17:30 以后  →  python3 -u main.py report
```

17:30 以后再跑,是因为免费源的当日成交额、换手率在收盘后一段时间内还可能修正。

`report` 一次完成:

1. 更新行情(失败的股票会打印真实错误,不再静默)
2. 复盘**昨日及历史推荐**(真实行情回看)
3. 大盘复盘:涨跌家数、涨跌停、炸板率、赚钱效应、情绪温度、仓位区间
4. 板块复盘:强度、扩散度、拥挤度、资金代理、原因归因
5. LightGBM + 线性模型打分,生成次日关注清单
6. **把今日清单存入数据库**,供下一个交易日复盘

> 每个交易日都要运行一次。记录发生在 `report` 里,漏跑一天就少一天记录。同日重复运行会覆盖当天清单。

定时运行(macOS/Linux crontab,周一到周五 17:30):

```
30 17 * * 1-5  cd /path/to/ashare_quant && venv/bin/python -u main.py report >> run.log 2>&1
```

macOS 的 cron 需要给终端完整磁盘访问权限,更稳妥的做法是用 launchd。

---

## 五、输出文件

| 文件 | 说明 |
|---|---|
| `reports/日期.md` / `.html` | 每日复盘报告 |
| `reports/日期_review.md` | `review` 命令输出 |
| `reports/tracking.csv` | 历史推荐的逐只复盘明细(可用 Excel 分析) |
| `reports/stock_代码_日期.md` / `.png` | 单股分析报告和图 |
| `reports/backtest.png` / `backtest_periods.csv` | 回测净值曲线和分期明细 |
| `data/quant.db` | SQLite 数据库 |

数据库表:`daily`(日线,前复权)、`stocks`(股票池/行业/ST)、`index_daily`(指数)、`picks_log`(每日推荐记录)。

---

## 六、核心功能说明

### 1. 每日报告结构

| 章节 | 内容 |
|---|---|
| 〇 昨日推荐复盘 | 昨日清单的真实表现 + 历史累计跟踪 |
| 一 大盘复盘 | 指数、涨跌家数、涨跌停、封板率、赚钱效应、情绪温度(0~100)、仓位区间 |
| 二 板块复盘 | 涨跌榜、综合强度 Top5、原因归因(统计特征 + 快讯关键词匹配) |
| 三 次日情景 | 基准/乐观/悲观三情景和触发条件 |
| 四 关注清单 | 模型 Top 列表,附支撑压力、ATR 止损/目标、信号标签、失效条件 |
| 五 模型自检 | 近 120 日样本外 RankIC/ICIR、因子 IC 监控 |

### 2. 昨日推荐复盘(`tracker.py`)

规则与回测完全一致:

- 信号日收盘后推荐,**次日开盘买入**,持有 `HORIZON` 个交易日后收盘卖出
- **开盘涨停算买不进**;到期日收盘跌停顺延一天卖出
- 收益已扣佣金、印花税、滑点;基准为同期样本股等权和沪深300
- T+1:买入当日不能卖,**止损/目标只统计买入日之后的最高最低价**
- 累计统计只算已到期(满 `HORIZON` 天)的推荐,持有中的单独标注

注意:历史推荐无法补录,从第一次运行 `report` 才开始记录;单日 10 只样本噪音很大,至少积累 3 个月(约 60 个信号日)再判断模型是否有效。

### 3. 单股分析(`stock_analysis.py`)

```bash
python3 -u main.py stock 600519
```

| 部分 | 内容 |
|---|---|
| 结论速览 | 综合评分(模型 35% / 趋势 25% / 量价 20% / 环境 20%)和倾向,加减分项全部展示 |
| 模型视角 | 全市场分位与排名;LightGBM SHAP 推高/拉低因子 Top5 |
| 技术面 | 均线排列、MACD、KDJ、RSI、布林带、ATR、乖离率、K线形态 |
| 量价 | 量比、量价关系、OBV 背离、涨跌日量能对比、资金代理、量加权均价偏离、天量地量 |
| 环境 | 相对沪深300/板块强弱、板块强度和拥挤度、大盘情绪 |
| 历史同类信号 | 该股近 3 年各类信号出现后持有 5 日的收益和胜率(去重叠;样本 <10 标注不可靠) |
| 操作参考 | 支撑压力、止损、两档目标、盈亏比、回踩区;空仓者/持仓者建议;仓位上限;失效条件 |

特殊情形会覆盖正常建议:今日涨停不追、今日跌停不抄底、乖离率 >15% 或 RSI >80 不追高、价格在 MA60 下方且 MA20 向下综合分封顶 52、市场情绪冰点封顶 55。股票不在股票池时会实时拉取并临时并入横截面打分(不写库);成交额不足/次新/ST 时模型不打分,报告会写明原因。

### 4. 因子与模型

- **因子(21 个)**:动量/反转、波动、流动性、量价、形态、CLV 资金代理;经 MAD 去极值、z-score、行业中性化
- **模型**:LightGBM(原生 `lgb.train`,不依赖 sklearn 接口)预测未来收益的截面排名,与 ICIR 加权线性模型按 6:4 合成
- **验证**:Walk-forward,训练集截止到预测日前 `H + 2` 天(embargo),防止标签泄漏

### 5. 数据体检(`check_data.py`)

只读检查:数据新鲜度、OHLC 逻辑、异常涨跌幅、成交量/额单位、缺失交易日、某日股票数骤降、等权收益与沪深300相关性。拉取数据后建议跑一次,有异常再排查。

---

## 七、项目结构

```
ashare_quant/
├── main.py             命令行入口
├── config.py           参数与网络设置
├── data.py             数据层:拉取(新浪/东财)→ SQLite → 面板;股票池与行业映射
├── factors.py          因子库、预处理、标签、数据集
├── model.py            LightGBM + ICIR线性、walk-forward、单股归因(explain_latest)
├── backtest.py         A股回测(T+1/涨停买不进/跌停卖不出/成本)
├── market.py           大盘情绪、板块强度与归因
├── picks.py            选股清单:过滤、行业分散、价位、信号标签
├── tracker.py          推荐入库与复盘
├── stock_analysis.py   单股分析
├── report.py           日报生成
├── check_data.py       数据体检
├── demo.py             模拟数据生成
└── requirements.txt
```

---

## 八、数据源说明

- **行情默认走新浪**(`stock_zh_a_daily`),因为东方财富在部分网络下会断连。东财可用时可把 `SOURCE_ORDER` 改为 `["em", "sina"]`
- 新浪接口依赖 `py_mini_racer`(V8 引擎)做解密,**不能多线程同时初始化**,代码里已对该调用加锁串行
- 股票池:东财可用时按成交额取前 N;否则退化为**中证800成分股**;再不行才用交易所列表随机抽样(此时模型质量会打折扣)
- 行业:东财行业板块优先,失败改用新浪行业(分类更粗,约 49 类)
- 不要在同一只股票上混用不同数据源(换手率口径不同)。换源后建议整只重拉
- 资金流是 **CLV 代理**(收盘位置 × 成交额),不是真实主力资金;基本面数据未接入

### 提高数据准确性的建议

1. **股票池有未来函数**:当前用"今天的成交额排名"选池再回测过去,且不含退市股,回测会偏乐观。更严谨的做法是候选池放宽到全市场、按当日滚动成交额筛选
2. **涨跌停判断是近似的**:用前复权价估算,除权日会有偏差;ST 状态只记录当前
3. **前复权价会被改写**:每次除权历史价格整体变化,代码用重叠区间检测并整只重拉。更好的做法是存未复权价 + 复权因子
4. 主库建议换用 Baostock 或 Tushare Pro,新浪只做当日补数据与交叉校验
5. 想要真实主力资金需要 Level-2 付费数据

---

## 九、常见问题

**`ProxyError` / 连接被断开**
系统代理(Clash/V2Ray 等)拦截了国内数据源。确认 `config.py` 里 `BYPASS_PROXY = True`,或在代理软件里给 `sina.com.cn`、`eastmoney.com` 设直连。

**`股票池为空` / `三个数据源都没拿到股票列表`**
先重新运行 `python3 main.py init`,看终端里 `[universe]` 开头的日志。若提示中证800返回列名不符,把日志发出来调整列映射。

**`没有行情数据,请先运行 init`(但拉取"完成"了)**
说明每只股票拉取都失败且被吞掉了。新版会打印 `[daily][错误] ...` 和 `成功 x / 失败 y`。先用下面命令验证数据源:

```bash
python3 -c "import data; df=data._fetch_sina('000001','20250101','20260101'); print(df.shape); print(df.tail(2))"
```

**程序崩溃,报告里有 `libmini_racer` / `partition_alloc`**
V8 多线程初始化崩溃。确认 `data.py` 里 `_fetch_sina` 的 akshare 调用被 `_sina_lock` 包住;仍崩溃就 `pip install -U py-mini-racer`,并把 `WORKERS` 改为 1。

**运行后没有任何输出**
多半是在跑但没打印。加 `-u` 重跑;另开终端查看数据库行数是否在增长:

```bash
python3 -c "import data;c=data.get_conn();print(c.execute('select count(*),count(distinct code) from daily').fetchone())"
```

**`scikit-learn is required for lightgbm.sklearn`**
`pip install scikit-learn`。新版代码已改用原生接口,不再需要它,但装上更保险。

**拉取中途报 `RemoteDisconnected` / 大量失败**
被限流。把 `WORKERS` 降到 1~2,稍后重跑;已入库的数据不会丢,会接着拉。

**单股分析提示"不在本地股票池,且实时拉取失败"**
检查代码是否正确、网络是否通;或先把它纳入股票池(调大 `UNIVERSE_SIZE` 后重跑 `init`)。

**复盘表里"昨日没有推荐记录"**
昨日没有运行 `report`,或这是第一次运行。从今天起会自动记录。

---

## 十、重要提醒

1. **demo 的结果没有意义**:数据是随机生成的,回测和复盘都只是验证代码能跑
2. 免费数据源可能改接口、缺数据、限流。AKShare 的函数名和列名会变,报错先 `pip install -U akshare`
3. 默认股票池是成交额前 800 只,涨停/跌停/涨跌家数统计只是**样本口径**,不是全市场
4. 板块"原因"是统计特征加关键词匹配,**是相关性描述,不是因果结论**
5. 评分、仓位、止损/目标都是规则化的参考值,**不是预测,不保证盈利**;盘中跳空时实际成交价会比止损价更差
6. 基本面、业绩预告、公告、解禁/减持、融资融券、龙虎榜均**未覆盖**,下单前请自行核对
7. 上线实盘前,至少用 `report` 积累 3 个月以上的真实前向跟踪,再评估模型是否有效;持续偏低的 RankIC 和跑输等权基准意味着模型已失效,应降低仓位或停用

*仅供学习研究,不构成投资建议。投资有风险。*
