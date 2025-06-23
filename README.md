# EPU实验项目

## 项目简介

本项目是一个基于DeepSeek大模型的经济政策不确定性(EPU)指数构建与金融市场波动率预测研究的完整实验框架。项目通过从维基百科和人民日报收集新闻数据，使用DeepSeek API生成5种不同类型的EPU指数，并结合市场波动率数据进行机器学习建模和策略回测。

## 项目特色

- 🤖 **创新的EPU构建方法**: 使用DeepSeek大模型替代传统关键词匹配方法，更智能地识别和量化经济政策不确定性
- 📊 **多维度EPU分析**: 生成向好EPU、向坏EPU、短期高频EPU、长期低频EPU、综合EPU等5种指数
- 🚀 **高效并行处理**: 支持10个API密钥并行调用，大幅提升数据处理效率
- 📈 **完整建模流程**: 包含OLS、LASSO、随机森林、XGBoost、LSTM等多种机器学习模型
- 💼 **实用回测框架**: 基于EPU预测结果设计交易策略并进行回测验证
- 📊 **实时进度追踪**: 全流程详细进度条显示，包括每个细微步骤和完成百分比
- 🔧 **模块化架构**: 高度模块化设计，支持独立运行各个组件

## 目录结构

```
epu_experiments/
├── README.md                           ← 项目说明文档
├── requirements.txt                    ← Python依赖包
├── main.py                            ← 统一入口程序
├── config/                            ← 配置文件目录
│   ├── config.py                      ← 总配置文件
│   ├── deepseek_api_keys.py          ← API密钥配置
│   └── prompts/                       ← EPU提示词模板
│       ├── good_epu_prompt.py         ← 向好EPU提示词
│       ├── bad_epu_prompt.py          ← 向坏EPU提示词
│       ├── short_freq_epu_prompt.py   ← 短期高频EPU提示词
│       ├── long_freq_epu_prompt.py    ← 长期低频EPU提示词
│       └── total_epu_prompt.py        ← 综合EPU提示词
├── data/                              ← 数据存储目录
│   ├── raw_news/                      ← 原始新闻数据
│   ├── raw_epu_results/              ← EPU原始结果
│   ├── daily_epu_table/              ← 日度EPU表格
│   ├── monthly_epu_summary/          ← 月度EPU汇总
│   ├── volatility_data/              ← 波动率数据
│   └── final_table/                  ← 最终合并数据
├── experiments/                       ← 实验结果目录
│   └── YYYYMMDD_HHMMSS_epu_experiment/  ← 单次实验文件夹
├── logs/                             ← 日志文件
├── src/                              ← 源代码目录
│   ├── collect/                      ← 数据收集模块
│   │   ├── collect_wikipedia.py      ← 维基百科数据收集
│   │   ├── collect_people_daily.py   ← 人民日报数据收集
│   │   └── clean_news.py            ← 新闻数据清洗
│   ├── epu_generator/               ← EPU生成模块
│   │   ├── run_deepseek_parallel.py ← DeepSeek并行调用
│   │   ├── format_prompt.py         ← 提示词格式化
│   │   └── aggregate_epu.py         ← EPU数据聚合
│   ├── merge/                       ← 数据合并模块
│   │   ├── merge_epu_volatility.py  ← EPU-波动率数据合并
│   │   └── monthly_aggregation.py   ← 月度数据聚合
│   ├── modeling/                    ← 建模分析模块
│   │   ├── linear_models.py         ← 线性模型(OLS/LASSO)
│   │   ├── random_forest_model.py   ← 随机森林模型
│   │   ├── xgboost_model.py         ← XGBoost模型
│   │   ├── lstm_model.py            ← LSTM模型
│   │   └── model_utils.py           ← 建模工具函数
│   ├── backtest/                    ← 回测分析模块
│   │   ├── strategy_lstm_volatility.py ← LSTM波动率策略
│   │   └── backtest_utils.py        ← 回测工具函数
│   └── utils/                       ← 工具模块
│       └── progress_utils.py        ← 进度追踪工具
└── notebooks/                       ← Jupyter分析笔记本
    ├── explore_epu.ipynb           ← EPU数据探索
    ├── volatility_trend_analysis.ipynb ← 波动率趋势分析
    └── model_results_summary.ipynb ← 模型结果汇总
```

## 安装和配置

### 1. 环境要求

- Python 3.8+
- 建议使用虚拟环境

### 2. 安装依赖

```bash
cd epu_experiments
pip install -r requirements.txt
```

### 3. 配置设置

项目的主要配置在 `config/config.py` 中：

- **时间范围**: 默认为2015-01-01到2025-06-15
- **API配置**: DeepSeek API的并发数、超时时间等
- **文件路径**: 各种数据文件的存储路径
- **中文字体**: 图表显示的字体配置

### 4. 数据准备

将波动率数据文件 `volatility_2014_2025.xlsx` 放置在项目根目录。

### 5. 项目完整性检查

在开始使用之前，可以运行完整性检查脚本验证所有文件和依赖是否准备就绪：

```bash
python check_project.py
```

该脚本会检查：

- 目录结构完整性
- Python文件完整性
- 依赖包安装情况
- 配置文件是否存在
- 外部数据文件路径是否正确

## 使用方法

### 命令行接口

项目提供了统一的命令行接口，支持模块化运行：

#### 1. 数据收集

```bash
# 收集维基百科数据
python main.py collect --wiki

# 收集人民日报数据  
python main.py collect --people

# 清洗新闻数据
python main.py collect --clean

# 执行所有数据收集步骤
python main.py collect --all
```

#### 2. EPU生成

```bash
# 生成EPU数据（自动包含聚合和合并步骤）
python main.py epu --generate
```

#### 3. 建模分析

```bash
# 运行所有模型
python main.py model --type all

# 运行特定模型
python main.py model --type lstm
```

#### 4. 回测分析

```bash
# 运行回测
python main.py backtest --strategy volatility
```

#### 5. 完整流程

```bash
# 运行完整实验流程
python main.py full --run-modeling --run-backtest
```

### Python代码调用

```python
from src.epu_generator.run_deepseek_parallel import DeepSeekParallelRunner
from src.merge.merge_epu_volatility import EPUVolatilityMerger

# 创建EPU生成器
runner = DeepSeekParallelRunner()

# 处理新闻数据生成EPU
results = runner.process_news_data(news_data_list, output_dir)

# 合并EPU和波动率数据
merger = EPUVolatilityMerger()
merged_df = merger.process_merge(epu_file, output_file)
```

## 核心功能详解

### 1. EPU指数生成

项目支持5种不同类型的EPU指数：

- **向好EPU**: 识别可能带来积极经济影响的政策不确定性
- **向坏EPU**: 识别可能带来消极经济影响的政策不确定性  
- **短期高频EPU**: 识别短期内高频变化的政策不确定性
- **长期低频EPU**: 识别长期内低频但影响深远的政策不确定性
- **综合EPU**: 所有类型政策不确定性的综合指标

### 2. 并行API调用

使用aiohttp和asyncio实现高效的并行API调用：

- 支持10个API密钥同时工作
- 自动重试和错误处理
- 请求速率控制和超时管理
- 详细的执行统计和日志

### 3. 数据处理流程

1. **新闻收集**: 从维基百科和人民日报获取每日新闻
2. **数据清洗**: 文本预处理、去噪、中文分词
3. **EPU生成**: 调用DeepSeek API生成EPU分数
4. **数据聚合**: 生成日度和月度EPU统计
5. **数据合并**: 与波动率数据合并形成最终数据集

### 4. 建模和回测

- **线性模型**: OLS回归和LASSO正则化
- **集成学习**: 随机森林和XGBoost
- **深度学习**: LSTM时序预测模型
- **策略回测**: 基于波动率预测的交易策略

## 输出数据格式

最终生成的数据表格包含以下列：

```
Date, 向好EPU, 向坏EPU, 短期高频EPU, 长期低频EPU, EPU,
volatility, Positive_Semivariance_Daily_Avg, Negative_Semivariance_Daily_Avg,
High_Frequency_Volatility_Daily_Avg, Low_Frequency_Volatility_Daily_Avg
```

## 注意事项

1. **API限制**: DeepSeek API有调用频率限制，请合理设置并发数
2. **数据质量**: 新闻数据的质量会影响EPU指数的准确性
3. **时间范围**: 建议不要设置过长的时间范围，避免API调用超时
4. **中文字体**: 确保系统安装了宋体字体以正确显示中文图表

## 许可证

本项目仅供学术研究使用。

## 联系方式

如有问题或建议，请联系项目开发团队。
