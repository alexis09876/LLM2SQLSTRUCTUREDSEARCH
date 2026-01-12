# LLM2SQLStructuredSearch

**LLM2SQLStructuredSearch** 是一个概念验证（POC）项目，用于探索如何将 **结构化搜索（Schema/Join Graph）+ 两段检索（Two-Phase Retrieval）+ 校验/纠偏闭环** 引入 Text-to-SQL 任务，从而提升大语言模型（如 GPT-4o / GPT-4.x）生成 SQL 的  **可控性、准确度、可解释性与可迭代修复能力** 。

项目基于  **MiniDev 数据集** ，对比两条路线：

* **Baseline（旧方案）** ：直接将自然语言/提示（payload）塞给 LLM，让其端到端一次性生成 SQL
* **Two-Phase（新方案）** ：**草图 → 结构搜索 → 校验 → 最小编辑修复 →（可选验证）** 的可控闭环流程

---

## 0. 一句话目标

给定一个自然语言问题（NLQ），系统输出一组可复盘的 SQL 结果：

1. 生成  **多个 SQL 草图/候选** （允许未知常量占位符）
2. 对候选抽取实体（表/列/join）并与 schema 对齐
3. 基于 schema embedding + join 图做  **两段检索** （表召回 → 列与 join 路径对齐）
4. 将“证据包 + allowed_joins + 违规清单”反馈给 LLM 做 **最小编辑修复**
5. （可选）对 SQL 做分级验证（L0/L1/L2/L3），失败则输出证据包进入人工审核

---

## 1. 动机：为什么需要结构化检索与验证闭环？

传统端到端 Text-to-SQL 常见失败原因：

* 数据库结构复杂：表/列数多，语义重叠明显
* LLM 容易“虚构实体”：生成不存在的表/列/关系
* JOIN / 外键路径复杂：多跳推理很脆弱
* 一次性生成缺乏“护栏”：很难程序化发现错误并纠偏

因此，本项目提出一条 **结构化搜索 + 验证增强** 路线：

让 LLM 更聚焦“意图草图”，把 **schema 对齐、路径合法性、静态规则、可选执行验证** 交给检索与程序化校验，从而让系统具备“可控收敛”的能力。

---

## 2. 核心思想：Two-Phase Structured Search Pipeline

### 2.1 草图阶段（Sketch / Candidates）

LLM 先理解意图，输出若干个 SQL 模板/候选：

* 允许未知常量用占位符（如 `:year`, `:start_date`, `:currency`）
* 候选要  **非同质化** ：不同 join 选择、不同聚合路径、不同筛选表达方式
* 输出建议使用  **严格 JSON** （便于后续自动处理、落盘、对比）

### 2.2 实体抽取（双通道：LLM + 解析器）

对每个候选 SQL 抽取结构实体：

* `tables_used`
* `columns_used`（可记录 `(table, column)`）
* `join_path`（left/right/on）

并进行双通道对照：

* **LLM 抽取** ：与候选生成过程统一格式输出
* **sqlglot（或类似解析器）抽取** ：程序化解析实体清单
* 两者不一致时记录为  **warning signal** （后续评分/惩罚项）

> 目标不是“完全相信 LLM”，而是尽可能把结构信息变成“可验证”的硬数据。

### 2.3 Schema 索引（Embedding + 元数据）

把数据库 schema（表/列/外键关系）构建为可检索资产：

* 表、列、外键/关系 → embedding
* 维护 Join Graph（FK 关系、多跳路径）
* 可扩展字段：synonyms、列类型、示例值、统计信息（行数/基数等）

### 2.4 两段检索（Two-Phase Retrieval）

对每个候选 SQL 的实体集合做递进式对齐：

**Stage 1：表召回**

* 根据草图中预测表/主题词，在 schema 索引中召回 Top-K 表

**Stage 2：列与 Join 路径对齐**

* 在候选表集合内召回列（Top-K）
* 在 Join Graph 中为跨表关系计算可行 join 路径（最短路 / 允许 FK 路径）
* 输出 `allowed_joins`（硬护栏）

并为每个候选产生  **证据包（evidence package）** ：

* 命中的表/列/join 及其分数
* 可替换建议（非法列→候选列）
* join 连接路径建议（不连通→最短 join 路径）

统一打分（示例）：

Score=0.4⋅avg(table_score)+0.4⋅avg(column_score)+0.2⋅avg(join_score)−PenaltiesScore = 0.4\cdot avg(table\_score) + 0.4\cdot avg(column\_score) + 0.2\cdot avg(join\_score) - Penalties**S**core**=**0.4**⋅**a**vg**(**t**ab**l**e**_**score**)**+**0.4**⋅**a**vg**(**co**l**u**mn**_**score**)**+**0.2**⋅**a**vg**(**j**o**in**_**score**)**−**P**e**na**lt**i**es**

### 2.5 最小编辑修复（LLM Minimal Edit）

将以下信息回馈给 LLM：

* 原候选 SQL / 草图实体清单
* 检索 Top-K 命中（证据包）
* `allowed_joins`
* 校验违规清单（见验证层）

要求 LLM  **只修改不合法的部分** （非法表/列/join），输出：

* 修复后的 SQL
* `confidence`
* `delta_notes`（变更说明与剩余疑点）

### 2.6 循环纠偏（带记忆与早停）

每轮修复后做校验与评分：

* 记录尝试轨迹，避免重复走坏路径
* 早停条件（示例）：
  * `schema_valid = true` 且总分 ≥ 阈值（如 0.75/0.8）
  * 达到最大轮次（2–3）
  * 连续两轮无提升 → 取历史最佳

失败则进入人工审核：输出候选 SQL + 证据包 + join 路径建议，降低人工负担。

---

## 3. （可选）验证层：分级快检 → 深检

* **L0 静态合规**
  * 表/列 ∈ 白名单（schema）
  * join 必须由 `allowed_joins` 串联
  * 禁 `SELECT *`
  * 有聚合则 GROUP BY 完整
* **L1 轻执行（安全执行）**
  * `EXPLAIN` / 带 `LIMIT` 的 dry-run
  * 超时保护
  * 捕获语法错误 / 对象不存在 / 代价异常
* **L2 语义快验**
  * 类型/聚合形态断言
  * “TopN/极值/时间窗口”是否体现在 SQL
  * 度量与维度来源合理性（事实表 vs 维表）
* **L3 抽样复核（可选）**
  * 对部分结果做子查询重算，验证口径

验证结果回馈给纠偏循环，用于调整 `confidence` 与是否转人工。

---

## 4. 项目结构（与你截图保持一致）

<pre class="overflow-visible! px-0!" data-start="3122" data-end="5388"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-text"><span><span>LLM2SQLStructuredSearch/
├── backend/
│   └── src/
│       ├── baseline/                 # 旧方案：直接提示生成 SQL
│       ├── config/
│       │   └── config.py             # 全局配置（路径/模型/数据集等）
│       ├── data_io/
│       │   ├── file_reader.py        # 统一读写封装
│       │   └── file_writer.py
│       ├── embedding/
│       │   ├── embedding_client.py
│       │   ├── column_names_embedder.py
│       │   ├── edge_names_embedder.py
│       │   └── output/
│       │       ├── columns_name_embeddings.json
│       │       └── edges_name_embeddings.json
│       ├── evaluator/
│       │   ├── sql_refiner.py
│       │   └── batch_sql_refiner.py
│       ├── llm/
│       │   └── chatgpt_client.py     # LLM 调用封装
│       ├── prompt/
│       │   ├── system_sql_candidates.txt
│       │   ├── user_sql_candidates_template.txt
│       │   ├── system_sql_refine.txt
│       │   └── user_sql_refine_template.txt
│       ├── query/
│       │   ├── sql_candidate_generator.py
│       │   ├── batch_sql_candidate_generator.py
│       │   ├── sql_structure_embedding_builder.py
│       │   ├── batch_sql_structure_embedding_builder.py
│       │   └── output/
│       │       ├── sql_candidates.json
│       │       ├── sql_candidates_0001.json
│       │       ├── sql_candidates_0002.json
│       │       ├── sql_candidates_0003.json
│       │       ├── sql_structure_embeddings.json
│       │       └── batch/
│       │           ├── sql_candidates_0002_sql_refinement_*.json
│       │           ├── sql_candidates_0002_structure_embedding_*.json
│       │           └── ...
│       ├── schema_parser/            # Schema 解析与 join 图构建
│       ├── twophase/
│       │   ├── sql_candidate_searcher.py
│       │   └── batch_sql_candidate_searcher.py
│       └── utils/
│           └── utils.py
├── data/
│   └── raw/
│       └── data_minidev/
│           └── MINIDEV/
│               ├── dev_databases/
│               ├── dev_tables.json
│               ├── mini_dev_mysql.json
│               ├── mini_dev_mysql_gold.sql
│               ├── mini_dev_postgresql.json
│               ├── mini_dev_postgresql_gold.sql
│               ├── mini_dev_sqlite.json
│               └── mini_dev_sqlite_gold.sql
├── tests/
├── main.py
├── README_cn.md
├── README_en.md
└── requirements.txt
</span></span></code></div></div></pre>

---

## 5. 产物（Outputs）与可复盘资产

你这个项目的一个核心价值就是： **每一步都落盘** ，便于 Debug / 对比 / A/B。

常见产物包括：

* `embedding/output/columns_name_embeddings.json`：列名 embedding
* `embedding/output/edges_name_embeddings.json`：关系/边名 embedding
* `query/output/sql_candidates_000X.json`：候选 SQL（生成阶段）
* `query/output/batch/sql_candidates_000X_sql_refinement_*.json`：候选精炼结果（修复阶段）
* `query/output/sql_structure_embeddings.json`：SQL 结构 embedding 汇总
* `query/output/batch/sql_candidates_000X_structure_embedding_*.json`：分批结构 embedding 结果
* （可扩展）`twophase` 输出检索证据包、allowed_joins、最终重排结果等

---

## 6. 快速开始（建议按你的工程习惯这样写）

### 6.1 安装依赖

<pre class="overflow-visible! px-0!" data-start="5985" data-end="6028"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>pip install -r requirements.txt
</span></span></code></div></div></pre>

### 6.2 配置环境变量（建议）

在项目根目录创建 `.env`（不要提交真实密钥）：

<pre class="overflow-visible! px-0!" data-start="6076" data-end="6111"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>OPENAI_API_KEY=YOUR_KEY
</span></span></code></div></div></pre>

### 6.3 运行入口

<pre class="overflow-visible! px-0!" data-start="6126" data-end="6152"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>python main.py
</span></span></code></div></div></pre>

> 如果你 `main.py` 目前是分阶段调用脚本，也完全 OK——README 可以保留“分步运行”章节（下一节）。

---

## 7. 分步运行（更适合实验与复盘）

> 具体命令以你脚本参数为准；你现在的目录已经具备“批处理实验”最佳实践：编号 + batch 输出。

1. **生成候选 SQL**

* `batch_sql_candidate_generator.py`
* 输出：`query/output/sql_candidates_000X.json`

2. **候选精炼 / 修复（Refine）**

* `batch_sql_refiner.py`
* 输出：`query/output/batch/sql_candidates_000X_sql_refinement_*.json`

3. **构建 SQL 结构 embedding**

* `batch_sql_structure_embedding_builder.py`
* 输出：`query/output/sql_structure_embeddings.json` & `query/output/batch/*structure_embedding*.json`

4. **Two-Phase 检索与重排**

* `batch_sql_candidate_searcher.py`
* 输出：建议写成 `query/output/batch/search_results_000X.json`（你可以后续统一命名）

---

## 8. 新方案 vs 旧方案：你要强调的对比点

### Baseline（旧方案）

* 直接把 payload 塞给 LLM 一次性生成 SQL
* 缺点：
  * LLM 不完全服从 payload，容易虚构表/列
  * 缺少硬护栏与验证闭环
  * 出错后难以自动修复、不可控

### Two-Phase Structured Search（新方案）

* LLM 只负责“草图意图”（更稳定）
* 检索 + 规则是硬护栏：
  * 两段检索
  * allowed_joins
  * 合规校验 + 惩罚项
* 可迭代可解释：
  * 证据包
  * 违规清单
  * 最小编辑修复轨迹
* 验证兜底：
  * 失败时转人工也高效（候选 + 证据 + 路径建议）

---

## 9. 数据集来源（你原文保留即可）

* MiniDev：轻量级跨库 Text-to-SQL 数据集，适合快速验证与开发
* Spider 1.0：经典跨库基准
* Spider 2.0：更复杂多表、多跳场景（未来扩展参考）

---

## 10. Roadmap（可选）

* 增加更严格的 SQL AST/逻辑计划结构表示
* 引入执行反馈（execution-guided）
* two-phase 的 Phase-2 增强重排器：规则打分/LLM 判别器/多策略融合
* 标准化评估：EM / execution accuracy / schema match rate
