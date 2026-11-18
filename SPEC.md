# BIOSTAT 707 ILI Demo v0.1 — Codex Execution Plan

> 给 Codex：这是用户已选定题目后的第一轮 demo 执行任务。请阅读完整任务书，在当前工作空间实际实现、运行和检查；不要只返回计划。本文件交付时，模型尚未训练，性能尚未产生。
>
> Agentic workflow：如当前环境提供 superpowers 技能，可使用 superpowers:executing-plans 按任务推进；独立的数据／图表实现可以委派，模型选择和测试集使用由主执行者统一控制。工具或技能不可用时按本任务书原生执行，不假定能力存在。

**Goal：**构建一个可在普通 CPU 上运行的离线分析 demo：使用过去 12 周的地区 ILI 历史，预测其后第 2 周的 ILI 比例，对比 Persistence、Ridge、HistGradientBoosting 和小型 LSTM。

**Architecture：**一个共享的地区—周数据表，一套严格按时间划分的预测窗口，四种模型共用相同评估样本。生成英文 HTML 结果报告、预测明细和检查摘要，方便五人小组阅读和继续开发。

**Tech Stack：**Python、pandas、NumPy、scikit-learn、PyTorch、epiweeks、matplotlib、pytest；使用项目隔离环境，记录实际版本。优先复用可工作的 Python 3.11/3.12 环境，不修改全局环境。

**Spec：**本文件第 1–7 节为完整 demo 规格，第 8–10 节为执行与验收计划。无需依赖此前聊天、ChatGPT 临时目录或任何未提供的附件。

**日期：**2026-10-20。数据可访问性已做小规模实测；Codex 仍应在自己的运行环境下载并复核。

## Global Constraints

- 范围：本地离线预测 demo；不部署网站，不运行实时预报服务，不提交课程作业或推送远端仓库。
- 建立独立项目目录 biostat707-ili-demo；若当前目录已是该项目则复用。先检查现有文件及适用的 AGENTS.md，报告实际绝对路径；不覆盖已有无关文件或改动旧 Checkpoint 项目。
- 10 个 HHS 地区，2010w40–2019w39，目标为加权 ILI 比例 wili。
- lookback=12，horizon=2；使用 CDC/MMWR 周历；输入截至 origin，目标在 origin 后两周。
- Train=2010w40–2016w39；Validation=2016w40–2017w39；Test=2017w40–2019w39。
- 对每个样本，origin 和 target 必须同时落在同一分区；历史输入允许跨越分区起点。
- 训练参数和预处理仅由 Train 学得；Validation 负责有限选择和早停；Test 只用于最终评价。
- 不合并 Train+Validation 重新训练；不滚动重训；不根据测试结果改模型、特征、种子或超参数。
- 三个可训练模型共享相同的信息：12 周 wili、地区编码、目标周的已知年周期特征。Persistence 作为简单基线只使用最近值。
- 默认 CPU，最多使用 4 个计算线程，随机种子 2026；LSTM 最多 60 epochs。
- 所有性能必须来自本轮真实运行。模型未优于基线不构成失败，也不得隐藏。
- 这是“使用修订后历史 ILINet 数据的回顾性预测 demo”，不等同于使用当年实际发布数据的实时回测。
- 如果将 demo 纳入课程项目，应依课程规则如实记录 Codex 对实现、调试或检查的协助；不要沿用与实际工作不符的“仅用于 brainstorming”声明。

## Review Focus

这些是需要重点检查的实际风险；第 8 节给出对应检查。

1. CDC/MMWR 年界和第 53 周：12 周窗口与两周 horizon 必须连续，不能对 YYYYWW 做整数加减。
2. 分区边界：origin、target 双条件归属；早期验证／测试样本可以读取之前的合法历史。
3. 数据来源返回异常：HTTP 成功不等于数据成功；检查 API result、字段、地区、周覆盖、重复和缺失。
4. 标准化及模型选择：预处理和模型权重仅由 Train 拟合；Validation 用于规定的选择和早停。在固定输入特征时，仅改变传入评价器的 Test 标签不能改变训练配置或预测。
5. 输出一致性：四模型使用同一组测试样本，目标周对齐，模型重载预测与保存结果相符，报告指标可由预测明细复算。

## 1. 研究问题与 demo 定位

研究问题：

> 在固定的历史时间划分下，小型 LSTM 能否比持续性预测、正则化线性模型和提升树更准确地预测地区 ILI 比例？

ILI 是流感样就诊指标，不等于实验室确诊流感病例，也不等于人群感染率。wili 以百分数记录，例如 2.5 表示 2.5%；误差的单位是“百分点”，不是相对百分比。[S1]

第一轮只证明数据、时间索引、模型训练、评价和展示能正确串联。一个时期、一个随机种子的结果不足以证明某种模型普遍更好。

报告至少说明以下限制：

- 使用查询时取得的历史修订值，不能重建当时的全部信息可用性。
- horizon=2 相对最后输入的“观测周”，并非经过核实的实际发布日起两周。
- 地区之间和同地区相邻周并非独立样本；不把所有窗口当独立重复做显著性检验。
- 仅使用 2010–2019 年，结论不自动推广到疫情后。
- 这是连续预测 demo；分类 AUROC/AUPRC 不适用。正式课程项目对 discrimination/calibration 的要求尚需通过合适的研究设计满足，不随意二值化目标。

## 2. 数据获取与复核

### 2.1 首选数据入口

公开查询：

https://api.delphi.cmu.edu/epidata/fluview/?regions=hhs1,hhs2,hhs3,hhs4,hhs5,hhs6,hhs7,hhs8,hhs9,hhs10&epiweeks=201040-201939

无需使用 auth 参数或凭据。本次上游验证得到：

- HTTP 200，API result=1，message=success。
- 4,690 条记录，10 地区各 469 周。
- 首周 201040，末周 201939。
- region × epiweek 唯一，wili 无缺失。
- 响应约 1.26 MB。

这些数值是本次已读取的参考结果，不是 Codex 可直接复用的执行证据。请在自己的环境重新验证。

该接口是 legacy V3，官方正在迁移至 V5。[S2] 若实际不可访问，可查阅官方 V5 的 fluview_ilinet 等价来源，但必须保留同一地区层级、时间范围、wili 定义和唯一地区—周结构，记录字段映射与实际 URL。不得静默改用州级数据、私人插补数据或不同预测目标。

### 2.2 缓存和来源记录

- 首次下载原始响应到 data/raw/fluview_201040_201939.json；后续默认读取该快照。
- 记录来源 URL、UTC 获取时间、响应状态、字节数、SHA-256、字段和行数到 data/source_manifest.json。
- 保留 release_date、issue 等来源元数据，用于说明修订；这些字段不作为预测特征。
- 记录原始数据来自 CDC ILINet，经 CMU Delphi 提供。来源许可按官方 public-domain 说明记录，并提供 CDC 和 Delphi 的引用。[S1–S3]
- 不把大文件、隔离环境或模型缓存提交到 Git；为后续共享保留下载脚本与数据说明。
- 如果接口因权限、网络或服务错误不可用，记录真实错误。不要把网页文档中的 JSON 示例当实际下载的数据。

### 2.3 规范化数据

生成一个规范表，至少包含：

region、epiweek、week_end、season、wili、release_date、issue。

- region 必须为 hhs1–hhs10；编码顺序按数字排序，避免 hhs10 排在 hhs2 前。
- 使用 CDC/MMWR 周历解析 epiweek，例如 epiweeks 的 CDC 模式；week_end 为该周周六。[S4]
- 每地区按 week_end 排序，相邻观测须相隔 7 天。
- 保留真实第 53 周，不压缩或删除。
- season 按 week≥40 归入当年开始的流感季，否则归入前一年开始的季节。
- 对重复 region × epiweek、无效日期、缺周或无效 wili 产生明确诊断；本轮不通过任意插值或跨缺口拼接掩盖问题。
- 数据结构不满足该完整面板时，先完成诊断；若不能从同一官方来源修复，则报告具体阻塞，不继续生成虚假的成功结果。

## 3. 窗口、切分与信息可用性

### 3.1 窗口定义

对每个地区单独构造：

X_t = [wili_(t−11), …, wili_t]

y_t = wili_(t+2)

origin=t，target=t+2。这里的加减是连续日历周，不是 YYYYWW 整数运算。

每个样本至少保留：

sample_id、region、origin_week、origin_date、target_week、target_date、target_season、split、12 个滞后值、y_true。

其中 sample_id 由 region、origin_week、target_week 唯一确定。输入按由旧到新的顺序排列。

### 3.2 固定分区

| 分区 | origin 与 target 均须位于 | 预计每地区窗口数 | 预计总窗口数 |
|---|---|---:|---:|
| Train | 2010w40–2016w39 | 300 | 3,000 |
| Validation | 2016w40–2017w39 | 50 | 500 |
| Test | 2017w40–2019w39 | 102 | 1,020 |

预期边界：

- Train 首个 origin=2010w51、target=2011w01；最后 origin=2016w37、target=2016w39。
- Validation 首个 origin=2016w40、target=2016w42；最后 origin=2017w37、target=2017w39。
- Test 首个 origin=2017w40、target=2017w42；最后 origin=2019w37、target=2019w39。

先在完整地区时间线上构造窗口，再用 origin 和 target 双条件划分。各段最后两个 origin 的目标越界，因此不进入该段，也不挪到下一段。

Validation/Test 起始窗口的历史输入可以来自前一段。Test 中较晚的预测可以使用当时已经观察到的较早 Test 周 ILI；这是滚动产生预测输入，不是重新训练模型。Test 两季作为一个连续分区，不在两季之间额外清空历史。

预期数量用于发现错位，不得为了匹配数量删改数据。

### 3.3 特征与标准化

第一轮不引入天气、搜索趋势、其他地区同期值、num_patients 等额外特征。

三个可训练模型均使用：

1. 12 个历史 wili。
2. 固定的 10 维地区 one-hot。
3. 目标周的两个年周期特征：sin(2πd/365.2425)、cos(2πd/365.2425)，d 为 target_date 距 2000-01-01 的天数。目标日期在预测时已知，因此可使用。

为便于比较：

- 在 Train 时间段内的唯一地区—周 wili 上计算全局均值 μ 和总体标准差 σ（ddof=0）。
- 用同一 μ、σ 标准化历史 wili 和训练目标；地区编码、sin/cos 保持原值。
- Validation/Test 仅应用该变换；预测后还原到原始百分数单位，再计算指标。
- σ 非有限或等于 0 时报告错误。
- Ridge/树的输入为 24 维：12 个历史值 + 10 个地区编码 + 2 个周期特征。
- LSTM 序列输入形状为 N×12×1；最后隐藏状态与同样的 12 个静态／日历特征拼接。
- Persistence 直接在原始单位下输出 origin 对应的 wili。

## 4. 模型与有限调参

所有可训练模型为跨地区共享的 pooled model；不分别训练十套模型。所有模型使用完全相同的有效样本。

### 4.1 Persistence

预测值 = origin 时点的 wili。

这是必须报告的参照基线。

### 4.2 Ridge

- 带截距的 Ridge 回归。
- alpha 只试 0.1、1、10、100，共 4 个候选。
- 用 Validation 的地区等权 MAE 选择；完全相同时选较大的 alpha。
- 使用入选的、只在 Train 上拟合的模型，不额外合并 Validation 重训。

### 4.3 HistGradientBoostingRegressor

固定配置，不搜索：

- loss="squared_error"
- max_iter=100
- learning_rate=0.05
- max_leaf_nodes=7
- min_samples_leaf=20
- l2_regularization=1.0
- early_stopping=False
- random_state=2026

关闭内部自动验证式 early stopping，统一遵守本项目的时间切分。[S5]

### 4.4 小型 LSTM

- PyTorch nn.LSTM，input_size=1、hidden_size=16、num_layers=1、batch_first=True、dropout=0、bidirectional=False。[S6]
- 取最后时间步隐藏状态，与地区 one-hot 和两个周期特征拼接，线性输出一个数值。
- 每个窗口独立重置隐藏状态；不在不同地区或样本之间携带状态。
- MSE loss；Adam，learning_rate=0.001、weight_decay=0.0001。
- batch_size=32，最多 60 epochs，gradient clip norm=1.0。
- 固定种子 2026；已构造好的 Train 窗口可在 mini-batch 中打乱。
- 每 epoch 计算还原单位后的 Validation 地区等权 MAE；patience=8，改善阈值 0.00001 个百分点。
- 恢复最佳 Validation epoch 的权重，保存实际训练 epoch 数、最佳 epoch 和训练曲线。
- 测试集不参与早停。单个固定种子足够完成本轮。

如果 LSTM 表现较差，如实报告训练／验证曲线及可能原因；不为了超过基线而临时增加层数、种子、搜索或新架构。

## 5. 评价与图表

### 5.1 指标

主要指标：

Macro-MAE = 10 个地区各自 MAE 的算术平均，单位为百分点。

同时输出：

- pooled MAE 和 pooled RMSE。
- 每地区 MAE、RMSE。
- 每个测试流感季的 Macro-MAE 和 pooled RMSE。
- 相对 Persistence 的主要指标改善率：
  100 × (MAE_persistence − MAE_model) / MAE_persistence。
  分母为 0 时写 NA 并解释，不做除零。
- 各分区和各模型实际评估样本数。

报告 Validation 和 Test 的结果，但 Test 的结果只在模型选择完成后生成。不用 MAPE 作为主指标，不计算不适用的 AUROC/AUPRC，不把窗口独立性当作显著性检验前提。

不在主分析中裁剪负预测或做测试集后校准；记录任何负预测数量，以暴露模型行为。

### 5.2 最少三张图

1. test_timeseries.png：10 地区小多图，共享模型颜色；画真实值及四模型预测，横轴必须是 target_date，不是 origin_date。
2. model_comparison.png：四模型 Test Macro-MAE 柱图，附清楚的数值与单位；可在报告表格中给地区结果。
3. predicted_vs_observed.png：四模型一模型一面板，共用坐标尺度，加入 y=x 参考线。

额外保存 LSTM 的训练损失和 Validation MAE 曲线，作为诊断图。

预测—实际值图用于检查偏差，不能称为完整的概率预测校准。本轮不加入预测区间、conformal、峰值预警或分类阈值。

### 5.3 英文 HTML 报告

报告标题：

Regional ILI Forecasting Demo: Persistence, Ridge, Gradient Boosting, and LSTM

要求：

- 一页可顺序阅读的自包含 HTML，内嵌图像，不依赖外部网络资源。
- 包含研究问题、数据来源、修订数据限制、切分表、模型配置、指标表、图表和简短结果解释。
- 正文和图表以英文为主，方便小组讨论；Codex 最终进度回复用中文。
- 报告写实际结果，不预写“LSTM 更好”。
- 若某模型无法完成，报告明确标记 partial，并保留已完成的实际结果。

## 6. 建议的精简项目结构

可以在保持接口与功能的情况下做小幅整理，不建设复杂框架。

~~~text
biostat707-ili-demo/
  README.md
  DATA_SOURCES.md
  requirements.txt
  config/demo.json
  run_demo.py
  src/data.py
  src/windows.py
  src/models.py
  src/evaluate.py
  tests/test_data_windows.py
  tests/test_evaluation.py
  data/raw/
  data/source_manifest.json
  outputs/demo_v0_1/
~~~

核心数据接口：

- data.py：下载／读取缓存，返回规范地区—周表及来源信息。
- windows.py：消费规范表与固定配置，返回唯一索引的窗口表及分区摘要；只在这里定义窗口和切分规则。
- models.py：训练函数接收 Train 特征／目标与 Validation 特征／目标；预测函数仅接收特征，Test 标签只交给 evaluate.py。返回四模型预测、入选配置和可重载模型，不自行重新切分。
- evaluate.py：消费预测明细与来源／切分／配置，计算指标、绘图和生成报告。
- run_demo.py：连接以上步骤，设置种子，保存日志，运行关键检查。

运行入口统一为：

~~~bash
python run_demo.py --config config/demo.json --output-dir outputs/demo_v0_1
~~~

在实际可用的项目环境中给出从零安装和运行命令；锁定本次成功版本，不伪称已经在所有操作系统上验证。无需为了采用特定环境管理工具而重建一个已经可工作的隔离环境。

## 7. 必须保留的运行产物

在 outputs/demo_v0_1 下保留：

- data_audit.json：地区、周历、行数、缺失、重复、目标单位与来源快照信息。
- split_summary.csv：分区与地区样本数、origin/target 最小最大值。
- predictions.csv：split、sample_id、region、origin_week/date、target_week/date、target_season、y_true、model、y_pred。
- metrics.csv：Validation/Test 的总体、地区、季节评价及基线改善率，注明统计口径。
- model_selection.json：实际候选、Validation 结果、Ridge 入选 alpha、LSTM 最佳 epoch、种子及变换参数。
- models/：Ridge、树、LSTM 的本地模型及重载所需参数。
- figures/：上述图表。
- demo_report.html：英文自包含结果报告。
- run_summary.json：完成状态、运行环境、耗时、检查结果及具体未完成项。
- execution_log.txt：本轮实际命令及关键输出；不包含凭据或无关环境信息。

输出目录按本轮任务创建；重跑时不要把旧文件混入新结果。无需额外创建多层“认证”“审计协议”或大量重复状态文档。

## 8. 执行步骤与关键检查

### Task 1 — 数据与窗口

**Files：**config/demo.json、src/data.py、src/windows.py、tests/test_data_windows.py、data/source_manifest.json。

**Consumes：**第 2 节的官方公开响应与固定配置。

**Produces：**规范表、唯一窗口表、data_audit.json、split_summary.csv，供所有模型统一使用。

- [ ] 检查当前目录和环境，建立隔离项目，记录实际路径。
- [ ] 下载／缓存数据，核对官方许可与字段，输出数据审计。
- [ ] 用小型构造数据写并运行以下具有实际意义的检查，再实现对应逻辑：
  - CDC 第 53 周与跨年推进仍是每步 7 天，horizon=2 是 14 天。
  - 对已知递增序列，12 个输入由 t−11 到 t，标签准确等于 t+2。
  - 任一输入周不得晚于 origin；不同地区的数据不能组成同一窗口。
  - origin、target 双条件归属；越界目标被排除，合法的跨边界历史保留。
  - 发现重复地区—周或缺周时给出明确错误。
- [ ] 在真实数据上构造窗口，核对预计 3000／500／1020 及边界；若不同，定位原因。
- [ ] 保存数据与切分摘要后进入模型阶段。

### Task 2 — 基线、传统模型与 LSTM

**Files：**src/models.py、run_demo.py、config/demo.json。

**Consumes：**Task 1 的统一窗口；不得重新定义窗口。

**Produces：**三个可训练模型、Persistence 预测、变换参数、选择记录和训练曲线。

- [ ] 从 Train 唯一观测拟合 μ、σ，并保存参数。
- [ ] 跑通 Persistence；检查预测严格等于对应 origin 的观测值。
- [ ] 拟合 4 个 Ridge 候选与固定树模型，只用 Validation 选择 Ridge。
- [ ] 先检查 LSTM 输入／输出维度与一个 batch 的有限 loss，再完成限定训练。
- [ ] 恢复最佳 LSTM 权重；冻结模型配置和选择结果。
- [ ] 生成 Validation 与 Test 预测，四模型必须具有相同样本键。
- [ ] 用构造数据检查改变 Validation/Test 数值不会改变 Train 标准化参数；检查模型训练接口没有接收 Test 标签。
- [ ] 如遇实现错误，可修复并重跑必要步骤；不得借修复之名根据 Test 表现改变科学配置。报告中记录实质修复。

### Task 3 — 评价、展示与轻量复现

**Files：**src/evaluate.py、tests/test_evaluation.py、README.md、DATA_SOURCES.md。

**Consumes：**冻结模型的预测、配置、切分及来源信息。

**Produces：**第 7 节结果产物和一个可直接打开的报告。

- [ ] 对手工可核算的少量预测检查 MAE、RMSE、地区等权平均和基线改善公式。
- [ ] 检查每个模型每个 sample_id 恰好一条预测；缺行、重复行或非有限预测必须显式报错。
- [ ] 从保存的 predictions.csv 复算指标，与 metrics.csv 一致。
- [ ] 重载保存的模型，对同一 Test 特征再推理一次；与保存预测在原始单位下绝对容差 1e−6 内一致。此步骤不重新训练。
- [ ] 生成图表与自包含 HTML，实际查看图像，检查目标日期、单位、图例、表格和英文结论。
- [ ] 运行关键检查及完整流程一次，记录实际命令、退出状态和耗时。已有有效结果后不做无目的重复训练。
- [ ] 写明复现方法、来源、限制、Codex 实际协助范围及下一轮需要讨论的问题。

## 9. 验收与停止位置

完整 demo 的验收标准：

1. 官方真实数据成功取得，规范表和窗口时间定义正确。
2. 四模型均完成，所有指标来自同一组真实评估样本。
3. 无训练／调参使用 Test 的行为；修订数据限制被明确披露。
4. 预测可追踪至地区、origin、target；保存模型可重载；指标可复算。
5. HTML 和图表可阅读，结果与原始预测一致。
6. README 中的运行方法与实际执行一致。

不要求 LSTM 排名第一，不设未经依据的性能阈值。

完成后停止在本轮 demo 交付，不自动扩展到 GRU/Transformer、更多 horizon、更多 lookback、外部协变量、实时 vintage 重建、多种子大实验或正式课程提交。

如果某项无法完成：保留真实完成部分，状态写 PARTIAL 或 BLOCKED，指出具体文件、步骤和错误；不得把“代码写完”表述为“已成功运行”。

## 10. Codex 最终回报格式

请用中文，按以下顺序回报：

1. **状态与路径：**COMPLETE／PARTIAL／BLOCKED，实际项目绝对路径和报告位置。
2. **运行证据：**环境、关键命令、退出状态、总耗时；真实数据行数与切分样本数。
3. **结果表：**四模型的 Validation Macro-MAE、Test Macro-MAE、Test pooled RMSE、相对 Persistence 的改善。
4. **模型细节：**Ridge 入选 alpha；LSTM 最佳 epoch／总 epochs；是否有负值预测或训练异常。
5. **检查：**周历与边界、预处理范围、相同评估键、指标复算和模型重载是否通过。
6. **解释：**结果实际说明了什么、不能说明什么；若 LSTM 不好，结合曲线解释，暂不扩展搜索。
7. **下一轮讨论：**最多 3 个最值得解决的问题，不自行执行。
8. **可下载交付：**生成一个便于回传的 ZIP，包含源代码、依赖与配置、来源记录、约 1.26 MB 的实际原始数据快照、报告、图表、预测、指标、检查摘要及小型模型；排除 .venv、.git、包管理缓存和无关文件。保留来源与许可，便于后续用同一数据复核。

回传给 ChatGPT 后，下一轮先检查时间定义、数据与指标，再决定是否推进正式项目设计。

## Sources

[S1] Delphi：FluView ILINet Outpatient Illness，包含变量定义、地区层级、修订机制与公共领域许可。
https://cmu-delphi.github.io/delphi-epidata/api/v5-signals/fluview_ilinet.html

[S2] Delphi：FluView legacy API，包含查询参数、issue/lag、响应结构及 V5 迁移提示。
https://cmu-delphi.github.io/delphi-epidata/api/fluview.html

[S3] CDC Epidemic Prediction Initiative：历史发布版本数据。
https://cdcepi.github.io/flu/data.html

[S4] EpiWeeks：CDC／ISO 周历、跨年与周日期操作。
https://epiweeks.readthedocs.io/en/stable/usage.html

[S5] scikit-learn：HistGradientBoostingRegressor。
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html

[S6] PyTorch：LSTM。
https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html

