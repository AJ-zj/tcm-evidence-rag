"""系统装配层：语料摄取（建索引）与运行时组装（RAGSystem 门面）。

build 流程：
  发现语料文件 → 多格式加载 → OCR 清洗 → 语义切分(Token回退)
  → BGE 向量化 → FAISS-HNSW 建索引 → BM25 建索引 → 持久化 artifacts

load 流程：从磁盘 artifacts 恢复全部组件，组装混合检索器与 Agent。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .agent.react_agent import EvidenceAgent
from .agent.session import ChatSession
from .config import Config, load_config
from .embedding import Embedder, create_embedder
from .indexing import ChunkStore, FaissHNSWIndex
from .llm.client import OpenAICompatibleLLM, create_llm
from .llm.extractive import ExtractiveAnswerer
from .ner.medical_ner import MedicalNER, build_vocab_from_corpus
from .nli.conflict import RuleNLI
from .parsing.chunking import SemanticChunker
from .parsing.loaders import discover_corpus_files, load_any
from .parsing.ocr_clean import OCRTextCleaner
from .retrieval.bm25 import BM25Index
from .retrieval.evidence_filter import SecondaryEvidenceFilter
from .retrieval.hybrid import HybridRetriever, RetrievalResult
from .retrieval.rerank import create_reranker
from .schema import AgentAnswer, Chunk, Document


class RAGSystem:
    """系统门面：检索 / 对话 / 会话管理。"""

    def __init__(
        self,
        cfg: Config,
        embedder: Embedder,
        store: ChunkStore,
        hnsw: FaissHNSWIndex,
        bm25: BM25Index,
        ner: MedicalNER,
        nli: RuleNLI,
        retriever: HybridRetriever,
        extractive: ExtractiveAnswerer,
        llm: OpenAICompatibleLLM | None,
        agent: EvidenceAgent,
        manifest: dict[str, Any] | None = None,
    ):
        self.cfg = cfg
        self.embedder = embedder
        self.store = store
        self.hnsw = hnsw
        self.bm25 = bm25
        self.ner = ner
        self.nli = nli
        self.retriever = retriever
        self.extractive = extractive
        self.llm = llm
        self.agent = agent
        self.manifest = manifest or {}
        self._sessions: dict[str, ChatSession] = {}
        self.db_path = self._db_path(cfg)

    @staticmethod
    def _db_path(cfg: Config) -> Path:
        """双库设计的结构化侧：SQLite 单文件（分块/长期记忆/会话三张表）。"""
        return cfg.path("data/db/tcm_rag.db")

    # ==================================================================
    # 构建（语料摄取 → 索引）
    # ==================================================================
    @classmethod
    def build(cls, cfg: Config | None = None, verbose: bool = True) -> "RAGSystem":
        cfg = cfg or load_config()
        t0 = time.time()
        log = (lambda *a: print(*a)) if verbose else (lambda *a: None)

        corpus_dir = cfg.path(cfg.get("project.corpus_dir", "data/corpus"))
        processed_dir = cfg.path(cfg.get("project.processed_dir", "data/processed"))
        index_dir = cfg.path(cfg.get("project.index_dir", "data/index"))
        processed_dir.mkdir(parents=True, exist_ok=True)

        # 1) 词表（种子 + 教材条目 + Huatuo 疾病词）
        vocab_path = processed_dir / "vocab.json"
        huatuo_file = corpus_dir / "qa" / "huatuo_tcm.jsonl"
        vocab = build_vocab_from_corpus(
            textbook_dir=corpus_dir / "textbook",
            huatuo_jsonl=huatuo_file if huatuo_file.exists() else None,
        )
        vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8")
        n_vocab = sum(len(v) for v in vocab.values())
        log(f"[1/6] 词表构建完成：{n_vocab} 词条 → {vocab_path}")

        ner = MedicalNER(
            vocab=vocab,
            weights=cfg.get("ner.weights"),
            base_importance=cfg.get("ner.base_importance", 0.15),
            max_entity_bonus=cfg.get("ner.max_entity_bonus", 0.85),
        )
        all_terms = [t for terms in vocab.values() for t in terms]

        # 2) 加载语料
        exts = cfg.get("parsing.supported_exts", [".md", ".json", ".jsonl", ".pdf", ".txt"])
        files = discover_corpus_files(corpus_dir, exts)
        docs: list[Document] = []
        for fp in files:
            try:
                docs.extend(load_any(fp))
            except Exception as e:  # noqa: BLE001
                log(f"  ! 跳过 {fp.name}: {e}")
        log(f"[2/6] 语料加载完成：{len(files)} 个文件 → {len(docs)} 篇文档")

        # 3) OCR 清洗 + 语义切分
        cleaner = OCRTextCleaner(
            vocab_terms=all_terms,
            enabled=cfg.get("parsing.ocr_clean.enabled", True),
        )
        ccfg = cfg.section("parsing").get("chunk", {})
        chunker = SemanticChunker(
            target_tokens=ccfg.get("target_tokens", 256),
            max_tokens=ccfg.get("max_tokens", 480),
            overlap_tokens=ccfg.get("overlap_tokens", 32),
            min_tokens=ccfg.get("min_tokens", 24),
        )
        chunks: list[Chunk] = []
        total_fixes = 0
        noise_before_sum = 0.0
        n_noisy_docs = 0
        for doc in docs:
            res = cleaner.clean(doc.text)
            total_fixes += res.n_fixes + res.n_space_removed + res.n_garbage
            if res.noise_before > 0.005:
                n_noisy_docs += 1
                noise_before_sum += res.noise_before
            chunks.extend(chunker.chunk(doc, cleaner))
        with open(processed_dir / "documents.jsonl", "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")
        log(f"[3/6] 清洗+切分完成：{len(chunks)} 个块（OCR 纠正 {total_fixes} 处，噪声文档 {n_noisy_docs} 篇）")

        # 4) 向量化
        embedder = create_embedder(cfg)
        t_emb = time.time()
        vectors = embedder.encode_documents(
            [c.text for c in chunks], batch_size=cfg.get("embedding.batch_size", 64)
        )
        emb_s = time.time() - t_emb
        log(f"[4/6] 向量化完成：{embedder.name} dim={vectors.shape[1]}，{len(chunks)} 块耗时 {emb_s:.1f}s")

        # 5) 索引构建
        hnsw = FaissHNSWIndex(
            dim=vectors.shape[1],
            m=cfg.get("index.hnsw_m", 32),
            ef_construction=cfg.get("index.ef_construction", 200),
            ef_search=cfg.get("index.ef_search", 128),
        )
        hnsw.add(vectors, [c.chunk_id for c in chunks])
        bm25 = BM25Index()
        bm25.load_jieba_dict(all_terms)
        bm25.build(chunks)

        # 双库：SQLite 存结构化数据（分块/记忆/会话），FAISS 存向量索引
        # reset=True：重建时清空旧块表（保留长期记忆与会话），保证与向量索引一致
        db_path = cls._db_path(cfg)
        store = ChunkStore(db_path=db_path, reset=True)
        store.add(chunks)
        hnsw.save(index_dir / "hnsw")
        bm25.save(index_dir / "bm25.pkl")
        log(f"[5/6] 双库持久化完成：SQLite({db_path.name}: {len(store)} chunks) + FAISS-HNSW(ntotal={hnsw.ntotal}) + BM25")

        # 6) 组装运行时
        nli = RuleNLI(ner)
        reranker = create_reranker(cfg, ner, embedder)
        fcfg = cfg.section("retrieval").get("secondary_filter", {})
        efilter = SecondaryEvidenceFilter(
            enabled=fcfg.get("enabled", True),
            min_rel_score=fcfg.get("min_rel_score", 0.30),
            ocr_noise_max=fcfg.get("ocr_noise_max", 0.18),
            min_chars=fcfg.get("min_chars", 24),
            dedup_jaccard=fcfg.get("dedup_jaccard", 0.72),
            final_top_k=cfg.get("retrieval.final_top_k", 5),
        )
        retriever = HybridRetriever(cfg, embedder, store, hnsw, bm25, reranker, efilter)
        extractive = ExtractiveAnswerer(ner)
        llm = create_llm(cfg)
        agent = EvidenceAgent(cfg, retriever, ner, nli, extractive, llm)

        manifest = {
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "build_seconds": round(time.time() - t0, 1),
            "n_files": len(files),
            "n_docs": len(docs),
            "n_chunks": len(chunks),
            "avg_chunk_tokens": round(float(np.mean([c.n_tokens for c in chunks])), 1) if chunks else 0,
            "max_chunk_tokens": max((c.n_tokens for c in chunks), default=0),
            "n_chunks_over_512": sum(1 for c in chunks if c.n_tokens > 512),
            "embedder": embedder.name,
            "dim": int(vectors.shape[1]),
            "ocr_fixes_total": total_fixes,
            "noisy_docs": n_noisy_docs,
            "vocab_size": n_vocab,
            "split_methods": _count_by(chunks, "split_method"),
            "source_types": _count_by(chunks, "source_type"),
            "llm": "openai_compatible" if llm else "offline_extractive",
        }
        (processed_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log(f"[6/6] 系统就绪（{manifest['build_seconds']}s）：{json.dumps({k: manifest[k] for k in ('n_docs','n_chunks','embedder','llm')}, ensure_ascii=False)}")
        return cls(cfg, embedder, store, hnsw, bm25, ner, nli, retriever, extractive, llm, agent, manifest)

    # ==================================================================
    # 加载（从磁盘 artifacts 恢复）
    # ==================================================================
    @classmethod
    def load(cls, cfg: Config | None = None, verbose: bool = False) -> "RAGSystem":
        cfg = cfg or load_config()
        log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
        processed_dir = cfg.path(cfg.get("project.processed_dir", "data/processed"))
        index_dir = cfg.path(cfg.get("project.index_dir", "data/index"))

        db_path = cls._db_path(cfg)
        if not db_path.exists():
            raise FileNotFoundError(
                f"未找到索引 artifacts（{db_path}）。请先运行 scripts/build_index.py"
            )

        vocab = json.loads((processed_dir / "vocab.json").read_text(encoding="utf-8"))
        ner = MedicalNER(
            vocab=vocab,
            weights=cfg.get("ner.weights"),
            base_importance=cfg.get("ner.base_importance", 0.15),
            max_entity_bonus=cfg.get("ner.max_entity_bonus", 0.85),
        )
        store = ChunkStore.load(db_path)
        embedder = create_embedder(cfg)
        hnsw = FaissHNSWIndex.load(index_dir / "hnsw", ef_search=cfg.get("index.ef_search", 128))
        bm25 = BM25Index.load(index_dir / "bm25.pkl")
        bm25.load_jieba_dict([t for terms in vocab.values() for t in terms])

        nli = RuleNLI(ner)
        reranker = create_reranker(cfg, ner, embedder)
        fcfg = cfg.section("retrieval").get("secondary_filter", {})
        efilter = SecondaryEvidenceFilter(
            enabled=fcfg.get("enabled", True),
            min_rel_score=fcfg.get("min_rel_score", 0.30),
            ocr_noise_max=fcfg.get("ocr_noise_max", 0.18),
            min_chars=fcfg.get("min_chars", 24),
            dedup_jaccard=fcfg.get("dedup_jaccard", 0.72),
            final_top_k=cfg.get("retrieval.final_top_k", 5),
        )
        retriever = HybridRetriever(cfg, embedder, store, hnsw, bm25, reranker, efilter)
        extractive = ExtractiveAnswerer(ner)
        llm = create_llm(cfg)
        agent = EvidenceAgent(cfg, retriever, ner, nli, extractive, llm)

        manifest = {}
        mf = processed_dir / "manifest.json"
        if mf.exists():
            manifest = json.loads(mf.read_text(encoding="utf-8"))
        log(f"系统加载完成：{len(store)} 块，embedder={embedder.name}，reranker={getattr(reranker, 'name', '?')}")
        return cls(cfg, embedder, store, hnsw, bm25, ner, nli, retriever, extractive, llm, agent, manifest)

    # ==================================================================
    # 运行时接口
    # ==================================================================
    def search(self, query: str, mode: str = "full", top_k: int | None = None) -> RetrievalResult:
        return self.retriever.retrieve(query, mode=mode, top_k=top_k)

    def session(self, session_id: str = "default") -> ChatSession:
        if session_id not in self._sessions:
            ltm = self._get_long_term_memory()
            scfg = self.cfg.section("memory").get("short_term", {})
            self._sessions[session_id] = ChatSession(
                session_id=session_id,
                agent=self.agent,
                ner=self.ner,
                long_term=ltm,
                stm_kwargs={
                    "window_turns": scfg.get("window_turns", 6),
                    "max_window_tokens": scfg.get("max_window_tokens", 900),
                    "importance_decay": scfg.get("importance_decay", 0.82),
                    "summary_max_tokens": scfg.get("summary_max_tokens", 220),
                },
                persist_db=self.db_path,
            )
        return self._sessions[session_id]

    def chat(self, session_id: str, message: str, decision_mode: str = "react") -> AgentAnswer:
        return self.session(session_id).chat(message, decision_mode=decision_mode)

    def reset_session(self, session_id: str) -> None:
        sess = self._sessions.pop(session_id, None)
        if sess is None:
            return
        # 重置会话 = 同时清除持久化状态（SQLite sessions 表 / 旧 json 存档）
        if sess.persist_db is not None:
            import sqlite3

            with sqlite3.connect(str(sess.persist_db)) as conn:
                conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
        elif sess.persist_path is not None and sess.persist_path.exists():
            sess.persist_path.unlink()

    # ---- 长期记忆（跨会话共享，持久化到 SQLite memories 表）----
    _ltm: Any = None

    def _get_long_term_memory(self):
        if self._ltm is None:
            from .memory.long_term import LongTermMemory

            lcfg = self.cfg.section("memory").get("long_term", {})
            db_path = self.db_path
            legacy_jsonl = (
                self.cfg.path(self.cfg.get("project.processed_dir", "data/processed"))
                / "long_term_memory.jsonl"
            )
            need_migrate = False
            if legacy_jsonl.exists():
                if db_path.exists():
                    import sqlite3 as _sq

                    with _sq.connect(str(db_path)) as conn:
                        try:
                            n_mem = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                        except _sq.OperationalError:
                            n_mem = 0
                    need_migrate = n_mem == 0
                else:
                    need_migrate = True
            if need_migrate:
                # 一次性迁移：旧 JSONL → SQLite memories 表
                legacy = LongTermMemory.load(
                    legacy_jsonl,
                    ner=self.ner,
                    embedder=self.embedder,
                    nli=self.nli,
                    merge_similarity=lcfg.get("merge_similarity", 0.88),
                    write_importance_threshold=lcfg.get("write_importance_threshold", 0.50),
                    recall_top_k=lcfg.get("recall_top_k", 3),
                    conflict_policy=lcfg.get("conflict_policy", "newer_wins"),
                )
                legacy.save(db_path)
            # load() 会从 SQLite 恢复已有记忆记录（构造函数只建表不读数）
            self._ltm = LongTermMemory.load(
                db_path,
                ner=self.ner,
                embedder=self.embedder,
                nli=self.nli,
                merge_similarity=lcfg.get("merge_similarity", 0.88),
                write_importance_threshold=lcfg.get("write_importance_threshold", 0.50),
                recall_top_k=lcfg.get("recall_top_k", 3),
                conflict_policy=lcfg.get("conflict_policy", "newer_wins"),
            )
            self._ltm_path = db_path
        return self._ltm

    def save_long_term_memory(self) -> None:
        if self._ltm is not None:
            self._ltm.save(self._ltm_path)


def _count_by(items: list[Chunk], attr: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items:
        key = str(getattr(it, attr))
        out[key] = out.get(key, 0) + 1
    return out
