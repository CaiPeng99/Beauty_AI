"""
eval_runner.py
运行方式: python eval_runner.py
"""
from app.database.session import get_db
from app.rag.retriever import HybridRetriever
from app.rag.evaluator import RAGEvaluator, run_full_evaluation, print_evaluation_report, save_evaluation_results
from openai import OpenAI
from app.config import ZhiPu_API_KEY, LLM_MODEL
from sentence_transformers import SentenceTransformer

# ── 你已有的组件，直接复用 ────────────────────────────────────
db = next(get_db())

# 复用 agent_core.py 里同一个 client
llm_client_raw = OpenAI(
    api_key=ZhiPu_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4"
)

# 包一层，让它有 .complete() 方法（evaluator需要）
class ZhipuAdapter:
    def complete(self, prompt: str) -> str:
        resp = llm_client_raw.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()

llm_adapter = ZhipuAdapter()

# 你的embedding模型（换成你实际用的）
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embed_fn = lambda q: embed_model.encode(q).tolist()

# HybridRetriever 复用
retriever = HybridRetriever(db=db, llm_client=llm_adapter)

# ── 定义 answer 生成函数 ──────────────────────────────────────
def generate_answer(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join(c["content"] for c in chunks[:5])
    return llm_adapter.complete(
        f"You are a Sephora beauty advisor.\n"
        f"User: {query}\n"
        f"Context:\n{context}\n"
        f"Answer:"
    )

# ── 跑评估 ───────────────────────────────────────────────────
if __name__ == "__main__":
    evaluator = RAGEvaluator(llm_client=llm_adapter)

    summary = run_full_evaluation(
        evaluator=evaluator,
        retriever=retriever,
        embed_fn=embed_fn,
        generate_answer_fn=generate_answer,
        session_id="eval_session",
        n_auto_queries=20,
    )

    print_evaluation_report(summary)
    save_evaluation_results(summary, "eval_results.json")

    db.close()