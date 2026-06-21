"""
rag_evaluator.py
================
美妆RAG项目专属评估模块

评估三大指标（全部通过 LLM-as-Judge，无需标注数据）：
  1. Retrieval Precision  — 检索到的chunks有多少是真正相关的？
  2. Answer Faithfulness  — 回答是否完全基于retrieved chunks，没有幻觉？
  3. Answer Relevance     — 回答是否真正解决了用户的问题？

额外指标：
  4. Rerank Improvement   — Rerank前后质量对比（专门针对你有rerank的架构）

使用方式：
    from rag_evaluator import RAGEvaluator, run_full_evaluation
    evaluator = RAGEvaluator(llm_client=your_llm_client)
    run_full_evaluation(evaluator, retriever, embed_fn, test_queries)
"""

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class RetrievalPrecisionResult:
    query: str
    score: float                    # 0.0 ~ 1.0，相关chunk占比
    relevant_count: int             # 被判定为相关的chunk数
    total_count: int                # 总chunk数
    reasoning: str                  # LLM的判断理由
    chunk_verdicts: list[dict]      # 每个chunk的单独判断


@dataclass
class FaithfulnessResult:
    query: str
    score: float                    # 0.0 ~ 1.0
    supported_claims: list[str]     # 有chunks支撑的claim
    unsupported_claims: list[str]   # 没有chunks支撑的claim（潜在幻觉）
    reasoning: str

@dataclass
class AnswerRelevanceResult:
    query: str
    score: float                    # 0.0 ~ 1.0
    addresses_intent: bool          # 是否回答了用户意图
    missing_aspects: list[str]      # 遗漏的方面
    reasoning: str


@dataclass
class RerankComparisonResult:
    query: str
    before_score: float             # rerank前的Retrieval Precision
    after_score: float              # rerank后的Retrieval Precision
    improved: bool
    improvement_delta: float        # after - before
    before_top3: list[str]          # rerank前top3的product_id
    after_top3: list[str]           # rerank后top3的product_id

@dataclass
class EvaluationRecord:
    """单条query的完整评估结果"""
    query: str
    retrieval_precision: RetrievalPrecisionResult | None = None
    faithfulness: FaithfulnessResult | None = None
    answer_relevance: AnswerRelevanceResult | None = None
    rerank_comparison: RerankComparisonResult | None = None
    final_answer: str = ""
    retrieved_chunks: list[str] = field(default_factory=list)
    status: str = "SUCCESS"


@dataclass
class EvaluationSummary:
    """整体评估汇总"""
    total_queries: int
    avg_retrieval_precision: float
    avg_faithfulness: float
    avg_answer_relevance: float
    avg_rerank_improvement: float
    rerank_improved_ratio: float    # rerank提升了多少比例的query
    failed_queries: list[str]
    records: list[EvaluationRecord]

# ---------------------------------------------------------------------------
# LLM Judge Prompts
# ---------------------------------------------------------------------------

RETRIEVAL_PRECISION_PROMPT = """You are evaluating a beauty product RAG system.

User Query: {query}

Retrieved Chunks (these are the chunks returned by the retriever):
{chunks}

Task: For EACH chunk, judge whether it is RELEVANT to answering the user's query.
A chunk is relevant if it contains information that would help answer the query.

For a query like "Show me highly rated cleansers under $30":
- RELEVANT: a chunk about a cleanser with price and rating info
- NOT RELEVANT: a chunk about a lipstick, or a chunk with no pricing info

Return ONLY valid JSON, no markdown fences:
{{
  "chunk_verdicts": [
    {{"chunk_index": 0, "relevant": true, "reason": "..."}}
  ],
  "overall_reasoning": "..."
}}"""


FAITHFULNESS_PROMPT = """You are evaluating whether an AI answer is grounded in the provided context.

User Query: {query}

Context (Retrieved Chunks):
{chunks}

AI Answer:
{answer}

Task: Break down the answer into individual claims, then check if each claim
is supported by the context above. A claim is "unsupported" if it introduces
information NOT found in any of the chunks (this is hallucination).

Return ONLY valid JSON, no markdown fences:
{{
  "supported_claims": ["claim that is in the context", "..."],
  "unsupported_claims": ["claim NOT in context = hallucination", "..."],
  "reasoning": "overall assessment"
}}"""


ANSWER_RELEVANCE_PROMPT = """You are evaluating whether an AI answer actually addresses the user's question.

User Query: {query}

AI Answer:
{answer}

Task: Judge if the answer genuinely addresses what the user asked.
Focus on: Does it fulfill the user's intent? Are there aspects of the query
that were ignored?

For example, if the user asked "Show me highly rated cleansers under $30" but
the answer recommends moisturizers, that's NOT relevant even if well-written.

Return ONLY valid JSON, no markdown fences:
{{
  "addresses_intent": true,
  "missing_aspects": ["list things the query asked for but answer didn't cover"],
  "score": 0.85,
  "reasoning": "..."
}}"""

RERANK_CHUNK_RELEVANCE_PROMPT = """You are evaluating chunk relevance for a beauty product search.

User Query: {query}

Chunk:
{chunk}

Is this chunk relevant to the query? Answer with ONLY valid JSON:
{{
  "relevant": true,
  "reason": "one sentence"
}}"""

# ---------------------------------------------------------------------------
# Core Evaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    LLM-as-Judge 评估器，专为你的 HybridRetriever 架构设计。

    Args:
        llm_client: 任何有 .complete(prompt: str) -> str 方法的客户端
                    （和你 HybridRetriever 里用的 llm_client 一样）
        delay_between_calls: 每次LLM调用之间的间隔秒数，避免rate limit
    """
    def __init__(self, llm_client, delay_between_calls: float = 0.5):
        self.llm = llm_client
        self.delay = delay_between_calls

    def _call_llm(self, prompt: str) -> dict:
        """调用LLM并解析JSON，带错误处理"""
        time.sleep(self.delay)
        try:
            raw = self.llm.complete(prompt)
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"LLM returned invalid JSON: {e}\nRaw: {raw[:200]}")
            return {}
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return {}
    
    # ------------------------------------------------------------------ #
    # 指标1: Retrieval Precision
    # ------------------------------------------------------------------ #
    def evaluate_retrieval_precision(
        self,
        query: str,
        chunks: list[str],
    ) -> RetrievalPrecisionResult:
        """
        评估检索精准度：retrieved chunks中有多少是真正相关的。

        Args:
            query:  用户原始问题
            chunks: retrieve()返回的chunk content列表
        """
        if not chunks:
            return RetrievalPrecisionResult(
                query=query, score=0.0, relevant_count=0, total_count=0,
                reasoning="No chunks retrieved", chunk_verdicts=[]
            )

        chunks_text = "\n\n".join(
            f"[Chunk {i}]:\n{c[:400]}" for i, c in enumerate(chunks)
        )
        prompt = RETRIEVAL_PRECISION_PROMPT.format(query=query, chunks=chunks_text)
        result = self._call_llm(prompt)

        verdicts = result.get("chunk_verdicts", [])
        relevant_count = sum(1 for v in verdicts if v.get("relevant", False))
        total = len(chunks)
        score = relevant_count / total if total > 0 else 0.0
        
        return RetrievalPrecisionResult(
            query=query,
            score=score,
            relevant_count=relevant_count,
            total_count=total,
            reasoning=result.get("overall_reasoning", ""),
            chunk_verdicts=verdicts,
        )

    # ------------------------------------------------------------------ #
    # 指标2: Answer Faithfulness
    # ------------------------------------------------------------------ #
    def evaluate_faithfulness(
        self,
        query: str,
        chunks: list[str],
        answer: str,
    ) -> FaithfulnessResult:
        """
        评估回答是否完全基于检索到的chunks，检测幻觉。

        Args:
            query:  用户原始问题
            chunks: 生成答案时用到的chunks
            answer: LLM生成的最终回答
        """
        if not answer:
            return FaithfulnessResult(
                query=query, score=0.0,
                supported_claims=[], unsupported_claims=["No answer generated"],
                reasoning="Empty answer"
            )
        
        chunks_text = "\n\n".join(f"[Chunk {i}]:\n{c[:400]}" for i, c in enumerate(chunks))
        prompt = FAITHFULNESS_PROMPT.format(query=query, chunks=chunks_text, answer=answer)
        result = self._call_llm(prompt)

        supported = result.get("supported_claims", [])
        unsupported = result.get("unsupported_claims", [])
        total_claims = len(supported) + len(unsupported)
        score = len(supported) / total_claims if total_claims > 0 else 1.0

        return FaithfulnessResult(
            query=query,
            score=score,
            supported_claims=supported,
            unsupported_claims=unsupported,
            reasoning=result.get("reasoning", ""),
        )
    
    # ------------------------------------------------------------------ #
    # 指标3: Answer Relevance
    # ------------------------------------------------------------------ #
    def evaluate_answer_relevance(
        self,
        query: str,
        answer: str,
    ) -> AnswerRelevanceResult:
        """
        评估回答是否真正解决了用户问题（端到端）。

        Args:
            query:  用户原始问题
            answer: LLM生成的最终回答
        """
        if not answer:
            return AnswerRelevanceResult(
                query=query, score=0.0, addresses_intent=False,
                missing_aspects=["No answer generated"], reasoning="Empty answer"
            )
        
        prompt = ANSWER_RELEVANCE_PROMPT.format(query=query, answer=answer)
        result = self._call_llm(prompt)

        return AnswerRelevanceResult(
            query=query,
            score=float(result.get("score", 0.0)),
            addresses_intent=bool(result.get("addresses_intent", False)),
            missing_aspects=result.get("missing_aspects", []),
            reasoning=result.get("reasoning", ""),
        )

    # ------------------------------------------------------------------ #
    # 指标4: Rerank Comparison（专为你的架构）
    # ------------------------------------------------------------------ #
    def evaluate_rerank_improvement(
        self,
        query: str,
        chunks_before: list[dict],
        chunks_after: list[dict],
        top_k: int = 5,
    ) -> RerankComparisonResult:
        """
        对比rerank前后，top-K chunks的相关性是否提升。
        直接插入你 smart_search() 里已有的 rerank comparison 逻辑。

        Args:
            query:         用户原始问题
            chunks_before: rerank前的chunk列表（dict，含 content 和 product_id）
            chunks_after:  rerank后的chunk列表
            top_k:         只评估前top_k个chunk
        """
        def _score_chunks(chunks: list[dict]) -> tuple[float, list[str]]:
            top = chunks[:top_k]
            if not top:
                return 0.0, []
            scores = []
            for chunk in top:
                prompt = RERANK_CHUNK_RELEVANCE_PROMPT.format(
                    query=query,
                    chunk=chunk.get("content", "")[:400]
                )
                res = self._call_llm(prompt)
                scores.append(1.0 if res.get("relevant", False) else 0.0)
            avg = sum(scores) / len(scores)
            pids = [c.get("product_id", "") for c in top]
            return avg, pids
        
        before_score, before_top3 = _score_chunks(chunks_before)
        after_score, after_top3 = _score_chunks(chunks_after)
        delta = after_score - before_score

        return RerankComparisonResult(
            query=query,
            before_score=before_score,
            after_score=after_score,
            improved=delta > 0,
            improvement_delta=delta,
            before_top3=before_top3,
            after_top3=after_top3,
        )
    
# ---------------------------------------------------------------------------
# Test Query Generator（用LLM自动生成测试问题，无需人工标注）
# ---------------------------------------------------------------------------

QUERY_GEN_PROMPT = """You are generating test queries for a Sephora beauty product RAG system.
The system handles two types of queries:
  1. Product recommendation with filters (e.g., price, rating, category)
  2. Vague/exploratory recommendations (e.g., "something popular")

Generate {n} diverse test queries covering these categories:
  - Price filter: "under $X", "between $X and $Y"
  - Rating filter: "highly rated", "best rated"
  - Category: cleansers, moisturizers, serums, foundations, lipsticks, etc.
  - Skin type: oily, dry, combination, sensitive
  - Vague: popular, trending, bestseller, new arrivals
  - Combined: multiple filters at once

Return ONLY valid JSON, no markdown fences:
{{
  "queries": [
    {{"query": "...", "type": "price_filter|rating_filter|category|skin_type|vague|combined"}}
  ]
}}"""

def generate_test_queries(llm_client, n: int = 30) -> list[dict]:
    """
    用LLM自动生成测试query集，无需人工标注。
    返回: [{"query": "...", "type": "..."}]
    """
    prompt = QUERY_GEN_PROMPT.format(n=n)
    try:
        raw = llm_client.complete(prompt)
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw)
        return data.get("queries", [])
    except Exception as e:
        logger.error(f"Query generation failed: {e}")
        # 返回一组默认测试query，保底用
        return [
            {"query": "Show me highly rated cleansers under $30", "type": "combined"},
            {"query": "Recommend me something popular", "type": "vague"},
            {"query": "Best moisturizer for oily skin", "type": "skin_type"},
            {"query": "New arrivals in skincare", "type": "vague"},
            {"query": "Sephora exclusive foundation under $50", "type": "combined"},
            {"query": "Trending serums with good reviews", "type": "combined"},
            {"query": "Fragrance-free toner for sensitive skin", "type": "skin_type"},
            {"query": "Bestselling lip products", "type": "vague"},
            {"query": "Limited edition eyeshadow palette", "type": "category"},
            {"query": "Highly rated sunscreen between $20 and $40", "type": "price_filter"},
        ]


# ---------------------------------------------------------------------------
# Pipeline Runner（把评估插入你现有的 smart_search 流程）
# ---------------------------------------------------------------------------

def run_full_evaluation(
    evaluator: RAGEvaluator,
    retriever,                          # 你的 HybridRetriever 实例
    embed_fn: Callable[[str], list],    # 你的 embedding 函数
    generate_answer_fn: Callable[[str, list], str],  # 你的 LLM 生成函数
    test_queries: list[dict] | None = None,
    n_auto_queries: int = 20,
    top_k_eval: int = 5,
    session_id: str = "eval_session",
) -> EvaluationSummary:
    """
    完整评估流程，直接对接你的 HybridRetriever。

    Args:
        evaluator:          RAGEvaluator 实例
        retriever:          你的 HybridRetriever 实例
        embed_fn:           query embedding 函数，例如 lambda q: model.encode(q).tolist()
        generate_answer_fn: 生成最终回答的函数，接收 (query, chunks) 返回 answer 字符串
        test_queries:       手动提供的测试query列表（None则自动生成）
        n_auto_queries:     自动生成的query数量
        top_k_eval:         评估前多少个chunk
        session_id:         日志用的session标识

    Returns:
        EvaluationSummary（含所有指标的汇总和每条query的详细结果）

    Example:
        def my_embed(query):
            return embedding_model.encode(query).tolist()

        def my_generate(query, chunks):
            context = "\\n".join(c["content"] for c in chunks[:5])
            return llm_client.complete(f"Query: {query}\\nContext: {context}\\nAnswer:")

        summary = run_full_evaluation(
            evaluator=RAGEvaluator(llm_client),
            retriever=hybrid_retriever,
            embed_fn=my_embed,
            generate_answer_fn=my_generate,
        )
        print_evaluation_report(summary)
    """
    # 自动生成测试query（如果没有提供）
    if test_queries is None:
        print("Generating test queries with LLM...")
        test_queries = generate_test_queries(evaluator.llm, n=n_auto_queries)
        print(f"Generated {len(test_queries)} test queries")

    records: list[EvaluationRecord] = []
    failed: list[str] = []

    for i, item in enumerate(test_queries):
        query = item["query"] if isinstance(item, dict) else item
        print(f"\n[{i+1}/{len(test_queries)}] Evaluating: {query}")

        record = EvaluationRecord(query=query)

        try:
            # ── Step 1: 获取embedding ──────────────────────────────────
            query_embedding = embed_fn(query)

            # ── Step 2: 运行你的 smart_search，同时捕获rerank前后数据 ──
            # 注意：这里我们需要rerank前后的数据，所以直接调用底层方法
            parsed_intent = retriever.parse_intent(query)
            candidate_ids = retriever._prefilter_by_sql(parsed_intent.get("filters", []))

            has_data, raw_results, status = retriever.search_with_judge(
                query, query_embedding, session_id, candidate_ids=candidate_ids
            )
            record.status = status

            if not has_data:
                print(f"  ⚠️  No results: {status}")
                failed.append(query)
                records.append(record)
                continue

            # 应用review filters
            review_filters = parsed_intent.get("review_filters", [])
            if review_filters:
                raw_results = retriever.apply_filters(raw_results, review_filters)

            # 保存rerank前的结果
            chunks_before_rerank = raw_results[:top_k_eval]

            # 执行rerank
            reranked_results = retriever.rerank(query, raw_results)
            chunks_after_rerank = reranked_results[:top_k_eval]

            # ── Step 3: 提取chunk内容 ──────────────────────────────────
            chunk_contents_before = [c["content"] for c in chunks_before_rerank]
            chunk_contents_after = [c["content"] for c in chunks_after_rerank]
            record.retrieved_chunks = chunk_contents_after

            # ── Step 4: 生成最终回答 ───────────────────────────────────
            answer = generate_answer_fn(query, chunks_after_rerank)
            record.final_answer = answer
            print(f"  ✓ Got answer ({len(answer)} chars)")

            # ── Step 5: 评估三大指标 ───────────────────────────────────
            print("  → Evaluating Retrieval Precision...")
            record.retrieval_precision = evaluator.evaluate_retrieval_precision(
                query, chunk_contents_after
            )
            print(f"     Score: {record.retrieval_precision.score:.2f} "
                  f"({record.retrieval_precision.relevant_count}/{record.retrieval_precision.total_count} relevant)")

            print("  → Evaluating Answer Faithfulness...")
            record.faithfulness = evaluator.evaluate_faithfulness(
                query, chunk_contents_after, answer
            )
            print(f"     Score: {record.faithfulness.score:.2f} "
                  f"({len(record.faithfulness.unsupported_claims)} unsupported claims)")

            print("  → Evaluating Answer Relevance...")
            record.answer_relevance = evaluator.evaluate_answer_relevance(query, answer)
            print(f"     Score: {record.answer_relevance.score:.2f}")

            # ── Step 6: Rerank对比（你架构的专属评估）─────────────────
            print("  → Evaluating Rerank Improvement...")
            record.rerank_comparison = evaluator.evaluate_rerank_improvement(
                query, chunks_before_rerank, chunks_after_rerank, top_k=3
            )
            delta = record.rerank_comparison.improvement_delta
            direction = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
            print(f"     {direction} {record.rerank_comparison.before_score:.2f} → "
                  f"{record.rerank_comparison.after_score:.2f} (Δ{delta:+.2f})")

        except Exception as e:
            logger.error(f"Evaluation failed for query '{query}': {e}", exc_info=True)
            record.status = f"ERROR: {e}"
            failed.append(query)

        records.append(record)

# ── 计算汇总指标 ──────────────────────────────────────────────────────
    valid = [r for r in records if r.status == "SUCCESS"]

    def _avg(values):
        return sum(values) / len(values) if values else 0.0

    avg_precision = _avg([r.retrieval_precision.score for r in valid if r.retrieval_precision])
    avg_faith = _avg([r.faithfulness.score for r in valid if r.faithfulness])
    avg_relevance = _avg([r.answer_relevance.score for r in valid if r.answer_relevance])

    rerank_records = [r.rerank_comparison for r in valid if r.rerank_comparison]
    avg_rerank_delta = _avg([r.improvement_delta for r in rerank_records])
    rerank_improved_ratio = (
        sum(1 for r in rerank_records if r.improved) / len(rerank_records)
        if rerank_records else 0.0
    )

    return EvaluationSummary(
        total_queries=len(test_queries),
        avg_retrieval_precision=avg_precision,
        avg_faithfulness=avg_faith,
        avg_answer_relevance=avg_relevance,
        avg_rerank_improvement=avg_rerank_delta,
        rerank_improved_ratio=rerank_improved_ratio,
        failed_queries=failed,
        records=records,
    )

# ---------------------------------------------------------------------------
# Report Printer
# ---------------------------------------------------------------------------

def print_evaluation_report(summary: EvaluationSummary):
    """打印可读的评估报告"""
    sep = "=" * 60

    print(f"\n{sep}")
    print("  RAG EVALUATION REPORT — Beauty Product System")
    print(sep)
    print(f"  Total Queries Evaluated : {summary.total_queries}")
    print(f"  Failed / Skipped        : {len(summary.failed_queries)}")
    print(sep)
    print("  CORE METRICS (LLM-as-Judge, 0.0 ~ 1.0)")
    print(sep)
    print(f"  1. Retrieval Precision  : {summary.avg_retrieval_precision:.2f}  "
          "← 检索到的chunks有多少是相关的")
    print(f"  2. Answer Faithfulness  : {summary.avg_faithfulness:.2f}  "
          "← 回答有多少来自chunks（越高越少幻觉）")
    print(f"  3. Answer Relevance     : {summary.avg_answer_relevance:.2f}  "
          "← 回答是否真正解决了用户问题")
    print(sep)
    print("  RERANK ANALYSIS")
    print(sep)
    print(f"  Avg Precision Δ (rerank): {summary.avg_rerank_improvement:+.2f}  "
          "(正数=rerank有效提升)")
    print(f"  Queries Improved by Rerank: "
          f"{summary.rerank_improved_ratio*100:.0f}%")
    print(sep)

    # 按指标找出最差的query，方便定向优化
    valid = [r for r in summary.records if r.status == "SUCCESS"]

    if valid:
        worst_precision = min(
            (r for r in valid if r.retrieval_precision),
            key=lambda r: r.retrieval_precision.score,
            default=None
        )
        worst_faith = min(
            (r for r in valid if r.faithfulness),
            key=lambda r: r.faithfulness.score,
            default=None
        )

        print("  WORST CASES (需要重点优化)")
        print(sep)
        if worst_precision:
            print(f"  Lowest Retrieval Precision ({worst_precision.retrieval_precision.score:.2f}):")
            print(f"    Query: {worst_precision.query}")
            print(f"    Reason: {worst_precision.retrieval_precision.reasoning[:100]}")
        if worst_faith:
            print(f"  Lowest Faithfulness ({worst_faith.faithfulness.score:.2f}):")
            print(f"    Query: {worst_faith.query}")
            if worst_faith.faithfulness.unsupported_claims:
                print(f"    Hallucinated: {worst_faith.faithfulness.unsupported_claims[0][:80]}")
        print(sep)

    if summary.failed_queries:
        print(f"  Failed Queries:")
        for q in summary.failed_queries:
            print(f"    - {q}")
        print(sep)


def save_evaluation_results(summary: EvaluationSummary, output_path: str = "eval_results.json"):
    """将评估结果保存为JSON，方便后续分析"""
    data = {
        "summary": {
            "total_queries": summary.total_queries,
            "avg_retrieval_precision": summary.avg_retrieval_precision,
            "avg_faithfulness": summary.avg_faithfulness,
            "avg_answer_relevance": summary.avg_answer_relevance,
            "avg_rerank_improvement": summary.avg_rerank_improvement,
            "rerank_improved_ratio": summary.rerank_improved_ratio,
            "failed_queries": summary.failed_queries,
        },
        "records": [
            {
                "query": r.query,
                "status": r.status,
                "final_answer": r.final_answer[:300],
                "retrieval_precision": asdict(r.retrieval_precision) if r.retrieval_precision else None,
                "faithfulness": asdict(r.faithfulness) if r.faithfulness else None,
                "answer_relevance": asdict(r.answer_relevance) if r.answer_relevance else None,
                "rerank_comparison": asdict(r.rerank_comparison) if r.rerank_comparison else None,
            }
            for r in summary.records
        ]
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")


# ---------------------------------------------------------------------------
# 使用示例
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    使用示例 — 把下面的占位符替换成你自己的实例

    from app.database.session import get_db
    from app.retriever import HybridRetriever
    from app.llm import YourLLMClient          # 你自己的LLM客户端
    from sentence_transformers import SentenceTransformer

    # 1. 初始化你现有的组件
    db = next(get_db())
    llm = YourLLMClient()
    embed_model = SentenceTransformer("your-embedding-model")
    retriever = HybridRetriever(db=db, llm_client=llm)

    # 2. 定义embedding函数（一行）
    embed_fn = lambda q: embed_model.encode(q).tolist()

    # 3. 定义answer生成函数（用你现有的generator）
    def generate_answer(query: str, chunks: list[dict]) -> str:
        context = "\\n\\n".join(c["content"] for c in chunks[:5])
        return llm.complete(
            f"You are a Sephora beauty advisor.\\n"
            f"User: {query}\\n"
            f"Context:\\n{context}\\n"
            f"Answer:"
        )

    # 4. 一行运行完整评估
    evaluator = RAGEvaluator(llm_client=llm)
    summary = run_full_evaluation(
        evaluator=evaluator,
        retriever=retriever,
        embed_fn=embed_fn,
        generate_answer_fn=generate_answer,
        n_auto_queries=20,      # 自动生成20条测试query
    )

    # 5. 查看结果
    print_evaluation_report(summary)
    save_evaluation_results(summary, "eval_results.json")
    """
    print("Import this module and call run_full_evaluation() to start evaluation.")
    print("See the __main__ block for a complete usage example.")
