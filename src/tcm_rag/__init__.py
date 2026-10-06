"""中医循证检索增强问答系统（TCM Evidence-based RAG QA System）。

模块一览：
- parsing:    多格式文档解析（PDF/MD/JSON/JSONL）+ OCR 清洗 + 语义切分 + Token 回退
- embedding:  BGE 向量化（sentence-transformers）与离线哈希回退
- indexing:   FAISS-HNSW 向量索引与分块存储
- retrieval:  BM25 + 稠密向量混合召回、RRF 融合、重排、二次证据过滤
- ner/nli:    医疗 NER（词典最大匹配）与规则式 NLI 冲突检测
- llm:        OpenAI 兼容 LLM 客户端与离线抽取式回答器
- agent:      CoT + ReAct 动态检索决策 Agent（证据一致性与置信度控制）
- memory:     分层记忆（短期滑窗摘要 / 长期阈值入库 + 冲突消解）
- evaluation: RAGAS 风格自动化评估闭环
- api:        FastAPI 服务与 Web 演示界面
"""

__version__ = "1.0.0"
