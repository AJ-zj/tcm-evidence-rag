# 中医循证检索增强问答系统

面向中医古籍、教材与医案的**超长文本检索增强问答（RAG）系统**，针对三大失效模式设计针对性机制：

| 失效模式 | 对应机制 |
|---|---|
| **Embedding 截断**丢失长文信息 | 语义切分（标题/段落/句子分层）+ Token 回退硬切分，任何文本都不会被 512 token 上限丢弃 |
| **OCR 干扰**导致噪声证据 | 词表驱动形近字还原 + 噪声度评估 + 二次证据过滤（噪声上限/近重复去重） |
| **证据不足时的错误自信回答** | CoT+ReAct 逐轮检索决策：证据一致性 + 置信度门控，不足即拒答并给出可解释原因 |

```
┌────────────────────────── 摄取流水线 ──────────────────────────┐
│ PDF/MD/JSON/JSONL 加载 → OCR 清洗 → 语义切分+Token回退          │
│ → BGE 向量化 → FAISS-HNSW 索引 + BM25 索引（jieba 词表挂载）     │
└────────────────────────────────────────────────────────────────┘
┌────────────────────────── 查询时链路 ──────────────────────────┐
│ 用户问题 + 分层记忆上下文                                        │
│   ↓                                                            │
│ CoT+ReAct Agent ──逐轮判断──┐                                   │
│   │  Thought→Action→Observation                                 │
│   │  search / refine_search / resolve_conflict / refuse        │
│   ↓                                                            │
│ 混合召回（BM25+稠密）→ RRF 融合 → 特征重排 → 二次证据过滤         │
│   ↓                                                            │
│ 置信度/覆盖率/一致性门控 → 带引用作答 或 可解释拒答               │
└────────────────────────────────────────────────────────────────┘
```

## 快速开始

环境要求：Python 3.12（本仓库用 `.venv`），CPU 即可运行；可选接入 OpenAI 兼容 LLM。

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt

# 1) 构建实验语料（Huatuo26M-Lite 中医子集抽取 + OCR 噪声模拟 + 扫描版 PDF）
.venv/Scripts/python scripts/build_corpus.py --qa-count 600

# 2) 构建索引（清洗→切分→BGE向量化→FAISS-HNSW+BM25，约 2 分钟）
.venv/Scripts/python scripts/build_index.py

# 3) 生成 700 条评估测试集（含金标准证据引用 + 拒答测试题）
.venv/Scripts/python scripts/make_eval_set.py

# 4) 运行完整评估闭环（五组实验，产出 reports/eval_report.md）
.venv/Scripts/python scripts/run_eval.py            # --sample 150 可快速抽样

# 5) Web 演示
.venv/Scripts/python scripts/serve.py --port 8000   # 打开 http://127.0.0.1:8000

# 或 CLI 交互（/trace 看决策轨迹，/memory 看记忆状态，/mode 切换基线）
.venv/Scripts/python scripts/demo_chat.py
```

可选：复制 `.env.example` 为 `.env`，配置 `LLM_PROVIDER=openai_compatible` 与
`LLM_BASE_URL / LLM_API_KEY / LLM_MODEL` 即可切换为 LLM 生成（未配置时使用内置
离线抽取式回答引擎，全链路仍可运行与评估）。

## 系统设计

### 1. 多格式文档解析流水线

- **加载器**（`parsing/loaders.py`）：Markdown（古籍/医案）、教材 JSON（章节-条目-字段结构展平为层级文本）、QA JSONL（Huatuo26M-Lite 逐条成档）、PDF（pypdf 逐页提取）。`data/corpus/` 下按 `classics / textbook / cases / qa / ocr / pdf` 分类，共 11 个文件、610 篇文档。
- **OCR 清洗**（`parsing/ocr_clean.py`）：① 规范化（控制字符/乱码占位符清除、全角字母数字转半角、CJK 字符间插空格清除）；② **词表驱动的形近字还原**——仅当"替换后命中医药词表"时才纠正（如 甘革→甘草、五赃六腑→五脏六腑），避免误改正常文本；③ 噪声度评分（乱码占比 + 插空格占比 + 需纠正形近字占比），供证据过滤使用。
- **语义切分**（`parsing/chunking.py`）：四级回退——标题结构（条文/药味/方剂/医案天然边界）→ 段落 → 句子 → **Token 回退硬切分**（超长单句按预算切窗并保留重叠）。每块携带标题路径前缀（提升嵌入质量）、原文偏移（引用追溯）、切分方式标记。构建结果：1031→1067 块，全部 ≤ 480 token，无一块被 BGE 截断。

### 2. 混合召回 + 重排 + 二次过滤

- **双路召回**：FAISS-HNSW（`IndexHNSWFlat`，M=32，efSearch=128，归一化向量内积=余弦）+ BM25（jieba 精确分词 + 医药词表挂载，防止"桂枝汤"被切碎）。
- **RRF 融合**：`score(d)=Σ 1/(k+rank)`；通道分数保留**绝对量纲**（BM25 饱和归一 s/(s+10)，RRF 除以理论最大值），避免"垃圾结果的相对第一名"获得满分——这是抑制错误自信的关键细节。
- **重排三后端**（`retrieval/rerank.py`，配置 `retrieval.reranker` 切换）：`feature` 六特征线性融合（稠密相似度/BM25/RRF/医疗实体覆盖率/OCR 清洁度/来源先验）；`api` 远程 CrossEncoder（SiliconFlow `/v1/rerank`，BAAI/bge-reranker-v2-m3，与特征分加权融合，接口异常自动回退特征重排，**免下载本地模型**）；`cross_encoder` 本地 BGE-reranker（sentence-transformers）。
- **二次证据过滤**（`retrieval/evidence_filter.py`）：相关性下限、OCR 噪声上限、最短长度、字符 bigram Jaccard 近重复去重；被丢弃的证据保留原因（可解释）。

### 3. CoT+ReAct 动态检索决策 Agent

将"一次检索→生成"升级为逐轮判断（`agent/react_agent.py`）：

- **信号计算**（每轮观察）：查询实体覆盖率（NER）、查询-证据词面重叠（门控项）、重排绝对分、Top 证据关键陈述的两两 NLI 一致性。
- **决策策略**：`confidence = (0.65·relevance + 0.35·top_score) × (0.7 + 0.3·consistency)`；
  - 置信度 ≥ 0.45 且一致性 ≥ 0.50 → 带引用作答；
  - 检出证据矛盾 → `resolve_conflict`：检索第三方来源仲裁，仍矛盾则在答案中显式标注；
  - 置信度不足 → `refine_search`：补缺失实体 + 焦点词改写（按问题类型映射到 性味归经/组成/主治/使用注意 等检索焦点）；
  - 轮次耗尽：≥ 0.33 低置信警示作答，否则**拒答**——宁可拒答不给错误自信的回答。
- **多轮对话**：指代消解（"它的用法用量呢？"→ 用短期记忆近期实体补全查询）。
- LLM 可选：配置后生成 Thought 文本与证据约束下的答案（带〔n〕引用、禁止证据外知识）；离线时由确定性模板 + 抽取式引擎驱动，决策信号完全相同。

### 4. 分层记忆

- **短期**（`memory/short_term.py`）：滑动窗口保留最近 N 轮原文；超出部分按有效重要度压缩进滚动摘要（抽取式，可选 LLM 摘要）。`importance` 由医疗 NER 实体类型加权（方剂 1.0 / 中药 0.9 / 病证 0.85 / 症状 0.7…）并随轮次衰减（`decay^age`）；窗口 token 超预算时优先压缩衰减后重要度最低的轮次。追问改写：问题无实体/含指代词时自动锚定最近用户轮次实体补全查询。
- **长期**（`memory/long_term.py`）：写入门槛 = NER 重要度 ≥ 0.50；BGE 相似度 ≥ 0.88 判为重复 → 合并（hits+1、重要度取大）；相似但不同 → **NLI 冲突检测**（否定矛盾/领域反义对/剂量冲突三类规则式判定）→ 按 `newer_wins`/`importance_wins` 消解并双向登记冲突原因；召回按 相似度×重要度×新近度 加权。
- **持久化（双库设计）**：**SQLite**（`data/db/tcm_rag.db`）承担全部结构化数据——`chunks` 表（分块+元数据）、`memories` 表（长期记忆，含 embedding BLOB）、`sessions` 表（短期会话状态，每轮对话后即时 upsert）；**FAISS**（`data/index/hnsw.index`）承担向量索引，BM25 索引（pickle）由 chunks 可重建。写入即时落库、服务重启秒级恢复，旧 JSONL 持久化自动一次性迁移。

### 5. 评估闭环（RAGAS 风格）

- **测试集**（700 条，固定种子）：从语料自动生成带金标准块引用的题目——真实医患 QA、药味/方剂/病证模板题、古籍条文题、医案题、药→方多跳题，以及**经语料重叠校验的 30 条不可回答题**（自动剔除语料实际覆盖的候选，保证拒答标签诚实）。
- **指标**（`evaluation/`）：Recall@K / Precision@K / MRR / Top-1；RAGAS 四指标离线实现——faithfulness（答案陈述被上下文支持比例：bigram 覆盖 + 实体覆盖 + RuleNLI 不矛盾）、context precision（命中金标准的排序加权）、context recall、answer relevancy；另测拒答正确率、过度拒答率、**错误自信回答率**（作答但金标准证据不在上下文）。
- **可追溯**：全部端到端明细落盘 `reports/e2e_rows_*.jsonl`（逐题：检索块、信号值、轨迹、答案），支持逐条证据引用分析。

## 评估结果（实测）

> 复现命令：`scripts/run_eval.py`。语料 610 文档 / 1067 块；测试集 700 条（670 可答 + 30 不可答）；全程离线、固定种子可复现。
> 以下数字由本仓库代码实际运行产出，完整报告见 [`reports/eval_report.md`](reports/eval_report.md)，逐题明细见 `reports/e2e_rows_*.jsonl`。

### 实验一：检索架构 A/B（670 条可答题）

| 模式 | Recall@1 | Recall@5 | Recall@10 | Recall@30 | Top-1 | MRR |
|---|---|---|---|---|---|---|
| dense（纯向量基线） | 81.4% | 91.0% | 93.3% | 97.9% | 82.5% | 0.867 |
| bm25 | 81.9% | 89.6% | 92.4% | 96.9% | 83.3% | 0.869 |
| hybrid（+RRF） | 84.3% | 95.4% | 96.8% | 99.2% | 85.5% | 0.897 |
| hybrid + 特征重排 | 90.6% | 97.0% | 97.7% | 99.2% | 91.9% | 0.943 |
| **full + API CrossEncoder 重排** | **96.3%** | **98.9%** | **99.1%** | **99.1%** | **97.5%** | **0.983** |

（API 重排 = SiliconFlow `/v1/rerank` + BAAI/bge-reranker-v2-m3 与特征分融合，明细见 `reports/eval_results_reranker_api.json`。注：实验三/五的端到端评估基于特征重排运行。）

混合召回 + 重排 + 二次过滤使 **Top-1 提升 +15.0pp**（82.5%→97.5%，其中 CrossEncoder 重排贡献 +5.6pp）、**R@1 提升 +14.9pp**、MRR +0.116。

### 实验二：长文信息保留率 + OCR 清洗

- 信息保留率（含医药实体的信息句完整保留在单个块内的比例）：朴素截断 512 token **51.5%** → 语义切分+Token回退 **87.9%**（绝对 +36.4pp，信息丢失减少 75%）
- OCR 噪声文档（模拟扫描件，CER≈3.5%）：医药术语恢复率 **90.2% → 97.6%**（清洗后）

### 实验三：端到端 RAGAS 指标（static 一次检索 vs react 动态决策）

| 指标 | static 基线 | react Agent | 变化 |
|---|---|---|---|
| 忠实度 faithfulness | 94.0% | 94.0% | ≈ |
| 上下文精确度 context_precision | 94.3% | 94.4% | ≈ |
| 上下文召回率 context_recall | 97.2% | 97.3% | ≈ |
| 答案相关度 answer_relevancy | 80.1% | 80.1% | ≈ |
| **不可回答题拒答正确率** | **0.0%** | **90.0%** | **+90pp** |
| 可回答题过度拒答率 | 0.0% | 0.0% | 0 |
| 错误自信回答率（作答但金标准证据不在上下文） | 2.54% | 2.39% | -0.15pp |
| 平均检索轮次 | 1.00 | 1.10 | +0.10 |

核心结论：**证据不足时的错误自信回答被系统性抑制**——static 基线对全部 30 条超范围问题都强行作答，react Agent 拒答 27 条且零过度拒答（可答题召回不受影响）。

### 实验三·补充：忠实度对比——LLM 无证据自由生成 vs 证据约束生成

接入 LLM 后（XingChenAGI/Xing4.0-29B via SiliconFlow），同一批题（40 可答 + 8 不可答）两种生成方式对比，
裁判 context 统一为 RAG 检索证据（衡量"答案陈述是否可由证据推出"，明细见 `reports/faithfulness_experiment.json`）：

| 指标 | LLM 无证据自由生成 | RAG 证据约束生成 |
|---|---|---|
| **忠实度 faithfulness** | **56.0%** | **88.8%（+32.8pp）** |
| 平均不被证据支持的陈述数 | 13.3 / 16.5 条（81% 为证据外内容） | —（引用〔n〕可追溯） |
| 不可答题硬答率 | 62.5%（幻觉风险） | **0%（100% 拒答）** |

这是"系统忠实度 71%→85%"叙事的可复现版本：无证据直答正是错误自信的来源，证据约束 + 置信度门控把忠实度拉回 88.8%。

### 实验三·对照：官方 RAGAS 库复核（40 题）

使用 **ragas 官方库（v0.4.3）** 对同一检索管线（LLM 裁判与 embeddings 走 SiliconFlow：Qwen2.5-7B-Instruct + bge-m3）复核四指标，与内置离线实现相互印证（`scripts/run_ragas_official.py` → `reports/ragas_official.json`）：

| 指标 | 官方 RAGAS（LLM 裁判） | 内置离线实现（规则裁判） |
|---|---|---|
| faithfulness | **0.954** | 0.888（另一 40 题批次） |
| context_precision | **0.887** | 0.944 |
| context_recall | **0.891** | 0.973 |
| answer_relevancy | 0.386* | 0.801（代理口径不同） |

*官方 answer_relevancy 需要向 LLM 请求 n=3 个反向生成问题，SiliconFlow 不支持 n>1（报错"n must be 1"后内部降级行为不稳定），该列得分失真，见下方失效分析。

**指标差异来源分析（评估器校准实验）**：

1. **faithfulness（0.954 vs 0.888）**：不同批次（官方批答案由 Qwen2.5-7B 生成、离线批由 Xing4.0 生成），两套裁判口径（LLM 逐陈述归因 vs 规则覆盖+NLI）给出同量级结果——**相互印证，这是本对照的核心结论**。
2. **context_precision/recall（0.887/0.891 vs 0.944/0.973）**：离线实现按 chunk_id **硬命中**判定（金标准条目出现在检索结果即得分）；官方 LLM 裁判把金标准完整条目拆成陈述逐条归因到 contexts——条目被拆块/截断后部分陈述归因失败，语义判定天然更严。差距反映的是**金标准表示方式**的差异，不是检索质量变化。
3. **answer_relevancy（0.386 失真）**：官方算法要求一次生成 n=3 个反向问题，SiliconFlow 拒绝 n>1，ragas 内部降级行为不稳定导致打分失真。按同口径手工实现（n=1 反向生成 + 嵌入余弦）复测 40 题：**0.583**；去除答案中〔n〕引用标记后 **0.598**——引用标记仅贡献 -1.6pp，主因是 n=3 不可用。与离线版 0.801 的剩余差距是**指标定义本身不同**（反向问题-原问题嵌入余弦 vs 词面/实体覆盖率），两者不可直接互比。

结论：跨口径数字不能直接互比；同口径下两套裁判相互印证。评估器本身的失效分析（n>1 限制、金标准表示、口径差异）是评估闭环的一部分，明细见 `reports/ragas_official.json`。

### 实验四：延迟基准

- FAISS-HNSW 纯检索（ntotal=1067, dim=512）：P50=0.29ms / **P95=0.37ms** / P99=0.43ms
- 全管线（查询编码+双路召回+RRF+重排+二次过滤）：P50=25.6ms / **P95=39.4ms** / P99=44.2ms

### 实验五：分层记忆（40 组多轮对话 × 6 轮）

| 指标 | 记忆关闭 | 分层记忆 | 变化 |
|---|---|---|---|
| 指代轮命中率（"它的用法用量呢？"） | 2.5% | **71.2%** | **+68.8pp** |
| 指代轮错误率（作答但证据不对） | 77.5% | 28.7% | **-48.8pp** |
| 直接问题命中率（不受记忆影响） | 91.2% | 91.2% | 0 |

长期记忆写入统计：400 次尝试 → 63 新增 / 126 合并（近重复）/ 210 拒绝（重要度门槛）/ 1 次冲突检出并消解（NLI newer_wins）。

### 与项目目标声称的对照

| 声称（项目介绍） | 本复现实测 | 说明 |
|---|---|---|
| Recall@30 62%→84% | dense R@30 97.9% → full 99.1% | 本语料规模小、条目级 gold 较易命中，绝对值偏高；改进趋势一致 |
| Top-1 +13% | Top-1 82.5%→97.5%（**+15.0pp**，含 API CrossEncoder 重排） | 同量级偏强 |
| 长文信息保留率 +20% | 51.5%→87.9%（丢失减少 75%） | 本基线更严格（整篇截断） |
| Faiss-HNSW P95≈120ms | HNSW P95=0.37ms；全管线 P95=39.4ms | 远低于上限（语料规模 1k 块） |
| 记忆命中率 +10%、错误率 -13% | 指代轮命中率 +68.8pp、错误率 -48.8pp | 更大幅度的同方向改进 |
| 忠实度 71%→85%，上下文精确率/召回率 >90% | faithfulness 94%，context precision 94.4%，recall 97.3% | 抽取式回答 + 证据门控，>90% 达成 |

## 项目结构

```
├── config/default.yaml          # 全部超参（检索/Agent/记忆/评估）
├── data/
│   ├── corpus/                  # 语料（classics/textbook/cases/qa/ocr/pdf）
│   ├── corpus_ground_truth/     # OCR 金标准（清洁原文 + 扰动清单）
│   ├── db/tcm_rag.db            # SQLite：chunks / memories / sessions 三张表（双库设计·结构化侧）
│   ├── index/                   # FAISS-HNSW + BM25（双库设计·向量侧）
│   ├── raw/Huatuo26M-Lite/      # 原始数据集（177,702 条中文医疗 QA）
│   └── processed/               # vocab.json / eval_set.jsonl / manifest
├── src/tcm_rag/
│   ├── parsing/                 # loaders / ocr_clean / chunking
│   ├── embedding/               # BGE + 哈希回退
│   ├── indexing/                # FAISS-HNSW 封装（serialize 规避 Windows 非 ASCII 路径）+ ChunkStore
│   ├── retrieval/               # bm25 / hybrid(RRF) / rerank / evidence_filter
│   ├── ner/                     # 词典 Trie 最大匹配 NER + 重要度加权
│   ├── nli/                     # 规则式矛盾/蕴含判定
│   ├── llm/                     # OpenAI 兼容客户端 + 离线抽取式回答
│   ├── agent/                   # CoT+ReAct Agent + 会话编排
│   ├── memory/                  # short_term / long_term
│   ├── evaluation/              # dataset / metrics / runner
│   ├── api/                     # FastAPI + Web 演示界面
│   └── pipeline.py              # 构建与装配门面
├── scripts/                     # build_corpus / build_index / make_eval_set / run_eval / serve / demo_chat
├── tests/                       # 63 个单元测试
└── reports/                     # 评估报告与明细
```

## 配置要点

- **嵌入回退**：`EMBEDDING_PROVIDER=hash` 或 BGE 加载失败时自动降级为字符 n-gram 哈希嵌入（离线可运行，检索质量下降）。
- **HuggingFace 镜像**：默认走 `https://hf-mirror.com`（`HF_ENDPOINT`），国内可直接下载 `bge-small-zh-v1.5`。
- **LLM 接入**：任何 OpenAI 兼容接口（DashScope/vLLM/Ollama…）；不配置则纯离线。
- **阈值调参**：`config/default.yaml` 中 `agent.confidence_*_threshold`、`retrieval.secondary_filter.*` 均为独立可调项。

## 诚实说明（Limitations）

- 语料为**教学演示规模**（610 篇 / ~1,067 块）：古籍节选与教材条目为人工整理/改写（含模拟 OCR 扫描件），QA 子集来自 Huatuo26M-Lite（`label=中医科` 抽样）。检索指标在此规模语料上测得，不代表生产语料规模下的数值。
- 未配置 LLM 时答案为**抽取式**（忠实度高但表达生硬）；faithfulness/context precision 等生成指标为规则代理实现（bigram 覆盖 + RuleNLI），与 RAGAS 原版 LLM 裁判在语义上有差距；配置 LLM 后可启用 `LLMJudge` 复核。
- OCR 扰动为程序模拟（形近字替换/插空格/乱码注入），与真实扫描件的错误分布不完全一致。
- 规则式 NLI 只覆盖否定、领域反义对、剂量三类矛盾信号，复杂语义冲突需要模型式 NLI。

## 数据来源致谢

- [Huatuo-26M](https://github.com/FreedomIntelligence/Huatuo-26M)（Apache-2.0）：中文医疗 QA 数据集
- BGE embedding / reranker：[BAAI](https://huggingface.co/BAAI)
- 古籍原文依据通行整理本录入节选（《伤寒论》《金匮要略》《黄帝内经素问》《温病条辨》《神农本草经》），公版内容，仅供教学检索演示
