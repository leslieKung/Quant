# A股量化复盘与选股工具(个人版)

单机运行:AKShare(免费数据) → SQLite → 21个因子 → LightGBM + ICIR线性合成 → 大盘情绪/板块复盘 → 次日关注清单 → Markdown/HTML 报告。

## 安装
```bash
python3 -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip3 install -r requirements.txt
```

## 使用
```bash
python3 main.py demo            # 先用模拟数据验证环境(无需网络,数据随机,结果无意义)
python3 main.py init            # 首次:股票池 + 行业映射 + 3年历史(10~40分钟,需联网)
python3 main.py report          # 每日收盘后(建议 16:30 以后)运行:更新数据 + 生成报告
python3 main.py backtest        # 全量 walk-forward 样本外回测,输出 reports/backtest.png
```
报告输出在 `reports/日期.md` 和 `.html`。定时运行(Linux/Mac):
```
30 16 * * 1-5  cd /path/to/ashare_quant && venv/bin/python main.py report >> run.log 2>&1
```

## 模块
| 文件 | 作用 |
|---|---|
| config.py | 股票池大小、持有期、成本、过滤阈值等参数 |
| data.py | 拉取/存储/前复权漂移检测与重拉/面板读取 |
| factors.py | 动量反转、波动、流动性、量价、形态、CLV资金代理;MAD去极值、z-score、行业中性化 |
| model.py | LightGBM(截面排名标签)+ ICIR加权线性;带 embargo 的 walk-forward;IC监控 |
| backtest.py | T+1、涨停买不进、跌停卖不出、佣金/印花税/滑点 |
| market.py | 涨跌家数、涨跌停、炸板、赚钱效应、情绪温度、仓位区间;板块强度/扩散度/拥挤度/原因归因 |
| picks.py | 硬过滤 + 行业分散 + 支撑压力 + ATR止损/目标 + 信号标签 + 失效条件 |
| report.py | 报告生成 |

## 重要说明
1. **demo 的回测结果无意义**:数据是随机数,只用于验证代码能跑。真实效果必须用真实数据的 `backtest` 判断,并以 3 个月以上模拟盘前向验证为准。
2. 默认股票池是成交额前800只(为了拉取速度)。涨停/跌停/涨跌家数统计仅限样本,不是全市场口径。想要全市场,把 `config.UNIVERSE_SIZE` 调到 5000(首次拉取很慢)。
3. 免费数据源可能改接口/缺数据/限流。AKShare 函数名与列名会变,若报错先升级:`pip install -U akshare`,再检查 data.py 中的列映射。
4. 资金流为 **CLV 代理**,不是真实主力资金;财务基本面因子未接入(免费数据难以做到 point-in-time),如需加入请务必按公告日对齐,避免未来函数。
5. 板块"原因"由统计特征 + 可选快讯关键词匹配生成,是相关性描述,不是因果结论。
6. 个人使用即可;若向他人提供买卖建议,涉及证券投资咨询资质,请先咨询合规。
7. 模型会失效。关注报告第五节的样本外 RankIC / ICIR,持续偏低就降低仓位或停用。

*仅供学习研究,不构成投资建议。*
