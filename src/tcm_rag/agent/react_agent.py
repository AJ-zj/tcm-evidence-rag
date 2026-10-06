"""CoT + ReAct 动态检索决策 Agent。

将静态"一次检索→生成"升级为逐轮判断流程：

    Thought → Action(search/refine_search/resolve_conflict/finish/refuse)
            → Observation(证据统计信号) → 下一轮 Thought ...

决策信号（全部可解释、可追溯）：
- coverage：   查询中医疗实体被证据覆盖的比例（NER）
- top_score / mean_top3：重排后相关性
- consistency：Top 证据关键陈述两两 NLI 一致性（1 - 矛盾对占比）
- confidence = w1·top_score + w2·mean_top3 + w3·coverage + w4·consistency

控制策略：
- confidence ≥ answer_threshold 且 consistency ≥ consistency_threshold → 作答（带引用）
- 检出证据矛盾且仍有轮次 → resolve_conflict（检索第三方来源仲裁）
- confidence 不足且仍有轮次 → refine_search（实体/焦点词扩展改写）
- 轮次耗尽：confidence ≥ refusal_threshold → 低置信警示作答；否则拒答
  （宁可拒答也不给"错误自信"的回答）

LLM 可选：配置后用于生成 Thought 文本与证据约束下的最终答案；
未配置时 Thought 用确定性模板、答案用抽取式引擎，全链路仍可运行。
"""
from __future__ import annotations

import time
from typing import Any, Sequence

from ..llm.client import LLMError, OpenAICompatibleLLM
from ..llm.extractive import ExtractiveAnswerer
from ..memory.long_term import LongTermMemory
from ..memory.short_term import ShortTermMemory
from ..ner.medical_ner import MedicalNER
from ..nli.conflict import RuleNLI
from ..retrieval.hybrid import HybridRetriever, RetrievalResult
from ..schema import AgentAnswer, Evidence, TraceStep

# 问题类型 → 检索焦点词（查询改写用）
FOCUS_PATTERNS: list[tuple[tuple[str, ...], str]] = [
    (("组成", "哪些药", "什么药", "成分", "药物组成"), "组成 药味"),
    (("主治", "治什么", "用于", "适应症", "治疗哪些"), "主治 功用"),
    (("性味", "归经", "四气五味"), "性味归经"),
    (("剂量", "用量", "多少克", "怎么煎", "用法"), "用法用量"),
    (("出处", "哪本书", "记载", "出自", "原文"), "出处 原文"),
    (("禁忌", "注意", "不能吃", "慎用", "忌用", "副作用"), "使用注意 禁忌"),
    (("辨证", "证型", "怎么治", "治法", "用什么方", "选方"), "辨证论治 治法 方药"),
    (("病因", "病机", "为什么", "怎么回事"), "病因病机"),
    (("方解", "君臣佐使", "为何用"), "方解"),
    (("医案", "案例", "验案"), "医案 初诊 处方"),
]

QUESTION_WORDS = ("请问", "什么", "哪些", "如何", "怎么", "为什么", "为何", "吗", "呢", "?", "？")

# 指代词：多轮对话中触发基于短期记忆的查询补全
ANAPHORA_WORDS = (
    "它", "它们", "该药", "此药", "这味药", "该方", "此方", "本方", "上方", "这个方",
    "那个方", "这个", "那个", "前述", "上述", "此证", "该证", "这种", "他的", "她的",
)


def _bigram_set(text: str) -> set[str]:
    import re as _re

    chars = _re.sub(r"\s+", "", text)
    return {chars[i:i + 2] for i in range(len(chars) - 1)}


class EvidenceAgent:
    def __init__(
        self,
        cfg,
        retriever: HybridRetriever,
        ner: MedicalNER,
        nli: RuleNLI,
        extractive: ExtractiveAnswerer,
        llm: OpenAICompatibleLLM | None = None,
    ):
        self.cfg = cfg
        self.retriever = retriever
        self.ner = ner
        self.nli = nli
        self.extractive = extractive
        self.llm = llm

        self.max_rounds = cfg.get("agent.max_rounds", 3)
        self.answer_threshold = cfg.get("agent.confidence_answer_threshold", 0.45)
        self.refusal_threshold = cfg.get("agent.confidence_refusal_threshold", 0.22)
        self.consistency_threshold = cfg.get("agent.consistency_threshold", 0.50)
        self.coverage_target = cfg.get("agent.coverage_target", 0.66)

    # ==================================================================
    # 信号计算
    # ==================================================================
    def _top_evidences(self, pool: dict[str, Evidence], k: int = 5) -> list[Evidence]:
        return sorted(pool.values(), key=lambda e: e.score, reverse=True)[:k]

    def _claims_of(self, evidences: Sequence[Evidence], per_ev: int = 2) -> list[str]:
        """取每条证据中与证据标题最相关的前 per_ev 句作为"关键陈述"。"""
        claims: list[str] = []
        for ev in evidences:
            sents = self.extractive._split_evidence_sentences(ev.chunk.text)
            claims.extend(sents[:per_ev])
        return claims

    def compute_signals(self, question: str, pool: dict[str, Evidence]) -> dict[str, Any]:
        top = self._top_evidences(pool, k=5)
        q_ents = {e.text for e in self.ner.recognize(question)}

        if top:
            ev_text = " ".join(e.chunk.text for e in top)
            covered = {t for t in q_ents if t in ev_text}
            coverage = len(covered) / len(q_ents) if q_ents else 0.0
            # 词面重叠门控：查询内容词/字符 bigram 在证据中的覆盖率
            from ..retrieval.bm25 import tokenize

            q_tokens = {t for t in tokenize(question) if len(t) >= 2}
            token_overlap = (
                len(q_tokens & set(tokenize(ev_text))) / max(len(q_tokens), 1) if q_tokens else 0.0
            )
            q_bigrams = _bigram_set(question)
            ev_bigrams = _bigram_set(ev_text)
            bigram_cov = len(q_bigrams & ev_bigrams) / max(len(q_bigrams), 1)
            if q_ents:
                relevance = 0.5 * coverage + 0.25 * token_overlap + 0.25 * bigram_cov
            else:
                relevance = max(token_overlap, bigram_cov)
            top_score = top[0].score
        else:
            covered, coverage = set(), 0.0
            token_overlap, bigram_cov, relevance, top_score = 0.0, 0.0, 0.0, 0.0

        claims = self._claims_of(top[:4], per_ev=2)
        consistency, conflicts = self.nli.consistency_of(claims)

        # 置信度 = 相关性主项 × 一致性调节。
        # 相关性不足时，即使语料内部自洽（consistency=1）也无法获得高置信——
        # 这是抑制"错误自信回答"的关键门控。
        confidence = (0.65 * relevance + 0.35 * top_score) * (0.7 + 0.3 * consistency)
        return {
            "coverage": round(coverage, 4),
            "token_overlap": round(token_overlap, 4),
            "relevance": round(relevance, 4),
            "covered_entities": sorted(covered),
            "missing_entities": sorted(q_ents - covered),
            "top_score": round(top_score, 4),
            "consistency": round(consistency, 4),
            "conflicts": [
                {"reason": c.reason, "score": c.score} for c in conflicts
            ],
            "confidence": round(confidence, 4),
            "n_evidence": len(pool),
        }

    # ==================================================================
    # 查询分析与改写
    # ==================================================================
    def analyze_question(self, question: str) -> dict[str, Any]:
        ents = self.ner.entity_types(question)
        focus = ""
        for keywords, term in FOCUS_PATTERNS:
            if any(k in question for k in keywords):
                focus = term
                break
        qtype = "lookup"
        if focus:
            qtype = focus.split()[0]
        return {"entities": ents, "focus_terms": focus, "query_type": qtype}

    def initial_query(self, question: str, analysis: dict[str, Any]) -> str:
        q = question
        for w in QUESTION_WORDS:
            q = q.replace(w, " ")
        parts = [p for p in q.split() if p]
        core = "".join(parts) if parts else question
        focus = analysis.get("focus_terms", "")
        return f"{core} {focus}".strip()

    def resolve_anaphora(
        self, question: str, short_term: ShortTermMemory | None
    ) -> tuple[str, list[str]]:
        """多轮对话指代消解与追问补全：用短期记忆近期实体改写查询。

        触发条件（满足其一）：
        1. 问题含指代词（"它/这个方/此证…"）
        2. 问题不含任何词典实体且存在对话历史（"那该怎么调理？"这类口语追问）

        补全来源与优先级（关键：用户轮多为症状主诉，assistant 轮含上一轮的
        结论性实体——证型/方剂；"那该怎么调理"这类追问必须锚定后者）：
        1. assistant 最近轮的 formula / pattern / disease（结论实体）
        2. user 最近轮的 formula / herb / pattern / disease（用户在谈的对象）
        3. 两者的 symptom（症状词只做补充，避免词面主导检索）
        """
        if short_term is None or not short_term._active_turns():
            return question, []
        q_ents = self.ner.entity_types(question)
        has_anaphora = any(a in question for a in ANAPHORA_WORDS)
        if not has_anaphora and q_ents:
            return question, []          # 有实体的独立问题，无需补全

        asst_ents = short_term.recent_entities(last_n=1, role="assistant")
        user_ents = short_term.recent_entities(last_n=2, role="user")
        if not asst_ents and not user_ents:
            asst_ents = short_term.recent_entities(last_n=1)

        added: list[str] = []
        seen: set[str] = set()

        def _take(source: dict[str, list[str]], etypes: tuple[str, ...], limit: int) -> None:
            for etype in etypes:
                for term in source.get(etype, [])[:2]:
                    if term not in question and term not in seen:
                        seen.add(term)
                        added.append(term)
                        if len(added) >= limit:
                            return
                if len(added) >= limit:
                    return

        _take(asst_ents, ("formula", "pattern", "disease"), 3)
        _take(user_ents, ("formula", "herb", "pattern", "disease"), 3)
        _take(asst_ents, ("symptom",), 3)
        _take(user_ents, ("symptom", "western_disease"), 3)
        if not added:
            return question, []
        return question + " " + " ".join(added[:3]), added[:3]

    def refine_query(
        self, question: str, analysis: dict[str, Any], signals: dict[str, Any], history: list[str]
    ) -> str:
        """证据不足时的查询改写：补缺失实体 + 焦点词 + jieba 关键词。"""
        from ..retrieval.bm25 import tokenize

        missing = signals.get("missing_entities", [])
        focus = analysis.get("focus_terms", "")
        keywords = [t for t in tokenize(question) if len(t) >= 2][:6]
        candidate = " ".join(dict.fromkeys(keywords + missing + focus.split()))
        # 避免与历史查询重复：重复则加入同义焦点扩展
        if candidate in history:
            alt_focus = {
                "主治": "功效 应用 适应症",
                "组成": "药味 方剂组成",
                "使用注意": "禁忌 慎用 孕妇",
                "辨证论治": "证型 治法 代表方",
                "性味归经": "四气五味 归经",
                "出处": "来源 载于 首见",
            }.get(focus.split()[0] if focus else "", "中医 治疗 方药")
            candidate = f"{candidate} {alt_focus}"
        return candidate.strip()

    # ==================================================================
    # ReAct 主循环
    # ==================================================================
    def answer(
        self,
        question: str,
        decision_mode: str = "react",     # react | static（基线对照）
        short_term: ShortTermMemory | None = None,
        long_term: LongTermMemory | None = None,
    ) -> AgentAnswer:
        t_start = time.perf_counter()
        trace: list[TraceStep] = []
        pool: dict[str, Evidence] = {}
        query_history: list[str] = []

        # 记忆上下文
        memory_ctx = ""
        ltm_hits: list[dict] = []
        if short_term:
            stm_ctx = short_term.context()
            memory_ctx += (stm_ctx + "\n\n") if stm_ctx else ""
        if long_term:
            hits = long_term.recall(question)
            ltm_hits = [
                {"id": r.id, "content": r.content, "score": s} for r, s in hits
            ]
            if hits:
                memory_ctx += long_term.context_block(question) + "\n\n"

        # 指代消解（多轮对话）：用短期记忆补全问题后再分析/检索/评估
        effective_question, resolved_ents = self.resolve_anaphora(question, short_term)
        analysis = self.analyze_question(effective_question)
        query = self.initial_query(effective_question, analysis)

        # ---- Round 0: 首次检索 ----
        resolved_note = f"（指代消解补全：{'、'.join(resolved_ents)}）" if resolved_ents else ""
        thought0 = (
            f"问题涉及实体 {analysis['entities'] or '（无词典实体）'}{resolved_note}，"
            f"类型为[{analysis['query_type']}]。先用语义核心词+焦点词做一次混合检索，"
            f"再依据证据覆盖率/一致性/置信度决定是否需要补充检索。"
        )
        rr = self.retriever.retrieve(query, mode="full", query_vector=None)
        self._merge_pool(pool, rr)
        query_history.append(query)
        signals = self.compute_signals(effective_question, pool)
        obs0 = self._observation_text(rr, signals)
        trace.append(TraceStep(0, thought0, "search", {"query": query, "mode": "full"}, obs0))

        # ---- static 基线：一次检索直接作答，不做任何控制 ----
        if decision_mode == "static":
            top = self._top_evidences(pool, k=self.retriever.final_top_k)
            ans = self._generate(question, top, [], memory_ctx, force_answer=True, resolved_ents=resolved_ents)
            return self._finalize(
                question, ans, top, signals, trace, t_start,
                refused=False, refusal_reason="", rounds=1,
                ltm_hits=ltm_hits, stm=short_term, ltm=long_term,
            )

        # ---- ReAct 逐轮判断 ----
        decision = "finish"
        refusal_reason = ""
        conflict_notes: list[str] = []
        rounds_used = 1

        for rnd in range(1, self.max_rounds + 1):
            decision, reason = self._decide(signals, rnd)
            if decision == "finish":
                break
            if decision == "refuse":
                refusal_reason = reason
                break

            if decision == "resolve_conflict":
                conflict_notes = [c["reason"] for c in signals["conflicts"]]
                arb_q = self._conflict_query(effective_question, analysis, signals)
                thought = (
                    f"证据一致性 {signals['consistency']:.2f} 低于阈值 "
                    f"{self.consistency_threshold}，检出矛盾 {conflict_notes}。"
                    f"第{rnd}轮：检索第三方来源仲裁冲突。"
                )
                rr = self.retriever.retrieve(arb_q, mode="full")
                self._merge_pool(pool, rr)
                query_history.append(arb_q)
                action, action_input = "resolve_conflict", {"query": arb_q}
            else:  # refine_search
                new_q = self.refine_query(effective_question, analysis, signals, query_history)
                thought = (
                    f"置信度 {signals['confidence']:.2f} < 作答阈值 {self.answer_threshold}"
                    f"（覆盖率 {signals['coverage']:.2f}，缺失实体 {signals['missing_entities']}）。"
                    f"第{rnd}轮：改写查询补充检索。"
                )
                rr = self.retriever.retrieve(new_q, mode="full")
                self._merge_pool(pool, rr)
                query_history.append(new_q)
                action, action_input = "refine_search", {"query": new_q}

            signals = self.compute_signals(effective_question, pool)
            trace.append(TraceStep(rnd, thought, action, action_input, self._observation_text(rr, signals)))
            rounds_used = rnd + 1

        # ---- 终局判定 ----
        refused = False
        if decision == "refuse":
            refused = True
        elif signals["confidence"] < self.refusal_threshold:
            refused = True
            refusal_reason = (
                f"证据不足：置信度 {signals['confidence']:.2f} < 拒答阈值 {self.refusal_threshold}"
            )
        elif signals["confidence"] < self.answer_threshold:
            # 低置信警示作答（比直接拒答保留更多信息，但明确标注不确定性）
            conflict_notes.append(
                f"低置信作答（confidence={signals['confidence']:.2f} < {self.answer_threshold}），建议人工复核"
            )

        top = self._top_evidences(pool, k=self.retriever.final_top_k)
        if refused:
            thought = (
                f"轮次耗尽仍不满足作答条件（confidence={signals['confidence']:.2f}, "
                f"coverage={signals['coverage']:.2f}, consistency={signals['consistency']:.2f}）。"
                f"依据可控检索决策原则：证据不足时拒答，避免错误自信回答。"
            )
            trace.append(TraceStep(rounds_used - 1, thought, "refuse", {"signals": _compact(signals)}, refusal_reason))
            answer_text = (
                "抱歉，现有语料中未能检索到足以可靠回答该问题的证据。"
                + (f"（{refusal_reason}）" if refusal_reason else "")
                + (
                    "\n提示：这是对上一轮话题的追问，系统已尝试用上文实体补全检索但仍未命中足够证据。"
                    "请补充具体的症状名称、药名或方剂名（如\"肝阳上亢该用什么方\"），我将基于语料证据作答。"
                    if (short_term and short_term._active_turns())
                    else "\n建议：补充更权威的资料来源，或将问题拆解为更具体的中医术语后重试。"
                )
            )
            result = AgentAnswer(
                question=question, answer=answer_text, refused=True,
                refusal_reason=refusal_reason or "insufficient_evidence",
            )
            return self._finalize(
                question, result, top, signals, trace, t_start,
                refused=True, refusal_reason=refusal_reason, rounds=rounds_used,
                ltm_hits=ltm_hits, stm=short_term, ltm=long_term,
            )

        thought = (
            f"证据充分且一致（confidence={signals['confidence']:.2f} ≥ {self.answer_threshold}，"
            f"consistency={signals['consistency']:.2f}），基于 Top-{len(top)} 证据生成带引用的答案。"
        )
        trace.append(TraceStep(rounds_used - 1, thought, "finish", {"signals": _compact(signals)}, "生成最终答案"))
        ans = self._generate(
            question, top, conflict_notes, memory_ctx, force_answer=False, resolved_ents=resolved_ents
        )
        return self._finalize(
            question, ans, top, signals, trace, t_start,
            refused=False, refusal_reason="", rounds=rounds_used,
            ltm_hits=ltm_hits, stm=short_term, ltm=long_term,
        )

    # ==================================================================
    # 内部步骤
    # ==================================================================
    def _decide(self, signals: dict[str, Any], rnd: int) -> tuple[str, str]:
        """逐轮决策：finish | refine_search | resolve_conflict | refuse。"""
        conf = signals["confidence"]
        cons = signals["consistency"]
        if conf >= self.answer_threshold and cons >= self.consistency_threshold:
            return "finish", ""
        if signals["conflicts"] and rnd < self.max_rounds:
            return "resolve_conflict", ""
        if rnd < self.max_rounds and conf < self.answer_threshold:
            return "refine_search", ""
        if conf < self.refusal_threshold:
            return "refuse", f"证据不足：置信度 {conf:.2f} < 拒答阈值 {self.refusal_threshold}"
        return "finish", ""   # 轮次耗尽但高于拒答阈值 → 低置信作答

    def _conflict_query(self, question: str, analysis: dict[str, Any], signals: dict[str, Any]) -> str:
        ents = [t for terms in analysis["entities"].values() for t in terms]
        focus = analysis.get("focus_terms", "")
        return " ".join(dict.fromkeys(ents + focus.split() + ["禁忌", "注意"]))

    def _merge_pool(self, pool: dict[str, Evidence], rr: RetrievalResult) -> None:
        for ev in rr.evidences:
            prev = pool.get(ev.chunk.chunk_id)
            if prev is None or ev.score > prev.score:
                pool[ev.chunk.chunk_id] = ev

    def _observation_text(self, rr: RetrievalResult, signals: dict[str, Any]) -> str:
        top_lines = []
        for ev in rr.evidences[:3]:
            snippet = ev.chunk.text[:60].replace("\n", " ")
            top_lines.append(f"  - {ev.chunk.chunk_id} (score={ev.score:.3f}, noise={ev.chunk.ocr_noise:.2f}) {snippet}…")
        dropped = len(rr.dropped)
        return (
            f"检索返回 {len(rr.evidences)} 条证据（二次过滤丢弃 {dropped} 条）；"
            f"信号: coverage={signals['coverage']:.2f}, consistency={signals['consistency']:.2f}, "
            f"confidence={signals['confidence']:.2f}, 缺失实体={signals['missing_entities']}\n"
            + "\n".join(top_lines)
        )

    def _generate(
        self,
        question: str,
        evidences: Sequence[Evidence],
        conflict_notes: Sequence[str],
        memory_ctx: str,
        force_answer: bool,
        resolved_ents: Sequence[str] = (),
    ) -> AgentAnswer:
        """答案生成：LLM（证据约束 + 引用）或离线抽取式。"""
        if self.llm is not None:
            try:
                return self._generate_with_llm(
                    question, evidences, conflict_notes, memory_ctx, force_answer, resolved_ents
                )
            except LLMError:
                pass  # 回退抽取式
        ext = self.extractive.compose(question, evidences, conflict_notes)
        return AgentAnswer(
            question=question,
            answer=ext.answer,
            citations=[f"[{cid}]" for cid in ext.used_citations],
            evidences=[_evidence_dict(e) for e in evidences],
        )

    def _generate_with_llm(
        self,
        question: str,
        evidences: Sequence[Evidence],
        conflict_notes: Sequence[str],
        memory_ctx: str,
        force_answer: bool,
        resolved_ents: Sequence[str] = (),
    ) -> AgentAnswer:
        ev_blocks = []
        for i, ev in enumerate(evidences):
            ev_blocks.append(
                f"[{i + 1}] 来源: {ev.chunk.doc_id}#{ev.chunk.seq}《{ev.chunk.title}》\n{ev.chunk.text}"
            )
        system = (
            "你是中医循证问答助手。只允许依据给定证据回答，禁止使用证据之外的知识；"
            "每个论断后用〔n〕标注证据编号；证据不足以回答时明确说明不能回答。"
            "如证据间存在矛盾，须指出矛盾并给出倾向性判断依据。"
        )
        user_parts = []
        if resolved_ents:
            # 指代消解结论显式告知 LLM，避免其从证据池中自行猜测指代对象
            user_parts.append(
                f"【指代说明】问题中的\"它/此方/该药\"等指代对象为：{'、'.join(resolved_ents[:2])}（来自上文对话）。"
                "若用户询问其组成/功用，请针对该对象作答。"
            )
        if memory_ctx:
            user_parts.append(memory_ctx.strip())
        user_parts.append("【证据】\n" + "\n\n".join(ev_blocks))
        if conflict_notes:
            user_parts.append("【已检出的证据矛盾】\n" + "\n".join(f"- {n}" for n in conflict_notes))
        user_parts.append(f"【问题】{question}")
        if force_answer:
            user_parts.append("（基线模式：无论证据是否充分都请直接给出最可能的回答）")
        answer = self.llm.complete(system, "\n\n".join(user_parts))
        citations = [f"[{e.chunk.doc_id}#{e.chunk.seq}]" for e in evidences]
        return AgentAnswer(
            question=question,
            answer=answer,
            citations=citations,
            evidences=[_evidence_dict(e) for e in evidences],
        )

    def _best_quote(self, query: str, text: str) -> str:
        """LLM 生成模式的 quote 兜底：取证据中与查询词重叠最高的句子。"""
        sents = self.extractive._split_evidence_sentences(text)
        if not sents:
            return ""
        from ..retrieval.bm25 import tokenize as _tok

        q_tokens = set(_tok(query))
        best, best_n = "", -1
        for s in sents:
            n = len(q_tokens & set(_tok(s)))
            if n > best_n:
                best, best_n = s, n
        return best

    def _finalize(
        self,
        question: str,
        ans: AgentAnswer,
        top: Sequence[Evidence],
        signals: dict[str, Any],
        trace: list[TraceStep],
        t_start: float,
        refused: bool,
        refusal_reason: str,
        rounds: int,
        ltm_hits: list[dict],
        stm: ShortTermMemory | None,
        ltm: LongTermMemory | None,
    ) -> AgentAnswer:
        ans.confidence = signals["confidence"]
        ans.consistency = signals["consistency"]
        ans.coverage = signals["coverage"]
        ans.refused = refused
        ans.refusal_reason = refusal_reason
        ans.rounds_used = rounds
        ans.trace = trace
        # 行级高亮：无 quote 的证据（LLM 生成路径）用查询相关句兜底
        for e in top:
            if not e.quote:
                e.quote = self._best_quote(question, e.chunk.text)
        ans.evidences = [_evidence_dict(e) for e in top]
        if not ans.citations:
            ans.citations = [f"[{e.chunk.doc_id}#{e.chunk.seq}]" for e in top]
        ans.latency_ms = round((time.perf_counter() - t_start) * 1000, 1)
        ans.memory_used = {
            "ltm_hits": ltm_hits,
            "stm_summary_chars": len(stm.summary) if stm else 0,
            "stm_active_turns": len(stm._active_turns()) if stm else 0,
        }
        return ans


def _evidence_dict(ev: Evidence) -> dict[str, Any]:
    return {
        "chunk_id": ev.chunk.chunk_id,
        "doc_id": ev.chunk.doc_id,
        "title": ev.chunk.title,
        "source_type": ev.chunk.source_type,
        "score": round(ev.score, 4),
        "ocr_noise": ev.chunk.ocr_noise,
        "quote": ev.quote or "",
        "channel_scores": {k: round(v, 4) for k, v in ev.channel_scores.items()},
        "text": ev.chunk.text[:400],
        "citation": ev.citation,
    }


def _compact(signals: dict[str, Any]) -> dict[str, Any]:
    return {k: signals[k] for k in ("coverage", "relevance", "top_score", "consistency", "confidence", "missing_entities")}
