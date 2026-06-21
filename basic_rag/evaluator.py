"""
📚: Evaluator — Measuring How Good Your RAG System Is
=============================================================
Without evaluation, you're flying blind. Did changing chunk_size from 500
to 300 make things better or worse? The evaluator tells you.

We measure THREE things:

1. RETRIEVAL PRECISION — Did we find the RIGHT chunks?
   "Of the chunks retrieved, how many came from the expected source?"
   This tests the RETRIEVER (embedder + vector store + search).

2. ANSWER FAITHFULNESS — Is the answer GROUNDED in the context?
   "Does the answer only use information from the retrieved chunks?"
   This tests the GENERATOR (prompt + LLM). A low score means hallucination.

3. ANSWER RELEVANCE — Does the answer actually ADDRESS the question?
   "Does this answer the question that was asked?"
   This tests the END-TO-END pipeline. The answer might be grounded but off-topic.

📚 WHY ALL THREE?
Each metric tests a different part of the pipeline:

  Metric              Tests           Bad score means
  ─────────────────   ─────────────   ──────────────────────────
  Retrieval Precision Retriever       Fix chunking or search
  Faithfulness        Generator       Fix system prompt
  Relevance           End-to-end      Fix both

📚 LLM-AS-JUDGE:
For faithfulness and relevance, we use the same LLM (gpt-5-mini) as a "judge".
We ask it: "On a scale of 0.0 to 1.0, how faithful/relevant is this answer?"
This is a common industry pattern (used by RAGAS, DeepEval, etc.) — imperfect
but practical and fast.
"""
import json
import os
import re
from datetime import datetime

from rag.retriever import Retriever
from rag.vector_store import VectorStore
from rag.generator import generate, build_prompt
from rag.pipeline import DEFAULT_INDEX_PATH

# ── Test Question Loading ─────────────────────────────────────
def load_test_questions(path: str = "eval/test_questions.json") -> list[dict]:
    """
    Load evaluation test questions from a JSON file.

    Each question should have:
      - "question": str — the query to test
      - "expected_source": str|null — which file should appear in results
      - "expected_keywords": list[str] — keywords that should be in the answer

    Args:
        path: Path to the test questions JSON file

    Returns:
        List of test question dicts

    Raises:
        FileNotFoundError: If the test file doesn't exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test questions file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
# ── Metric 1: Retrieval Precision ─────────────────────────────

def compute_retrieval_precision(
    retrieved_chunks: list[dict],
    expected_source: str | None,
) -> float | None:
    """
    Compute retrieval precision: fraction of retrieved chunks from the expected source.

    📚: Precision = (relevant results) / (total results)
    A high precision means most retrieved chunks are actually relevant.
    A low precision means we're retrieving a lot of noise.

    We use a relaxed path match: if the chunk source ENDS WITH the expected
    source path, it counts as a match. This handles different path prefixes
    (e.g., "sample_data/policies/refund-policy.md" matches "policies/refund-policy.md").

    Args:
        retrieved_chunks: List of retrieval results
        expected_source: Expected source file path, or None for unanswerable questions

    Returns:
        Precision score (0.0 to 1.0), or None if expected_source is None
    """
    # 📚: For unanswerable questions (expected_source=None), we can't
    # measure precision because we don't know which source is "correct".
    if expected_source is None:
        return None
    if not retrieved_chunks:
        return 0.0
    
    # Count how many chunks match the expected source
    # Use endswith() for flexible path matching
    hits = sum(
        1 for chunk in retrieved_chunks
        if chunk.get("metadata", {}).get("source", "").replace("\\", "/").endswith(
            expected_source.replace("\\", "/")
        )
    )
    return hits / len(retrieved_chunks)

# ── LLM-as-Judge Helpers ──────────────────────────────────────

def parse_llm_score(text: str) -> float:
    """
    Extract a numeric score (0.0 to 1.0) from LLM judge output.

    📚: The LLM returns a text response, but we need a number.
    We look for the LAST decimal or integer in the response and clamp
    it to [0.0, 1.0]. This is robust to different response formats:
      - "0.8"
      - "Score: 0.85"
      - "The answer is well grounded. I'd rate it 0.9."

    Args:
        text: Raw LLM response text

    Returns:
        A float between 0.0 and 1.0
    """
    # Find all numbers (decimals or integers, possibly negative) in the text
    numbers = re.findall(r"-?\d+\.?\d*", text)

    if not numbers:
        return 0.0
    
    # Take the last number (most likely to be the final score)
    score = float(numbers[-1])

    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, score))

def build_faithfulness_prompt(context: str, answer: str) -> str:
    """
    Build a prompt asking the LLM to judge answer faithfulness.

    📚: Faithfulness checks if the answer is GROUNDED in the context.
    A faithful answer only states things that appear in the provided context.
    An unfaithful answer adds information from the LLM's training data (hallucination) or makes up facts not in the context.

    Args:
        context: The retrieved context text
        answer: The generated answer

    Returns:
        A prompt string for the LLM judge
    """
    return f"""You are an evaluation judge. Your task is to score how FAITHFUL an answer is to the provided context.

    Faithfulness means the answer ONLY contains information that is supported by the context.
    - Score 1.0: Every claim in the answer is directly supported by the context.
    - Score 0.5: Some claims are supported, but the answer adds information not in the context.
    - Score 0.0: The answer contains claims that contradict or are not found in the context.

    Context:
    {context}

    Answer:
    {answer}

    Respond with ONLY a score between 0.0 and 1.0. Nothing else."""


def build_relevance_prompt(question: str, answer: str) -> str:
    """
    Build a prompt asking the LLM to judge answer relevance.

    📚: Relevance checks if the answer ADDRESSES the question.
    An answer might be perfectly grounded in context but completely off-topic.
    For example, if asked "What is the return policy?" and the answer talks
    about shipping, it's faithful (if shipping info is in context) but NOT relevant.

    Args:
        question: The original user question
        answer: The generated answer

    Returns:
        A prompt string for the LLM judge
    """
    return f"""You are an evaluation judge. Your task is to score how RELEVANT an answer is to the question.

    Relevance means the answer directly addresses what the question is asking.
    - Score 1.0: The answer directly and completely addresses the question.
    - Score 0.5: The answer partially addresses the question or includes irrelevant information.
    - Score 0.0: The answer does not address the question at all.

    Question:
    {question}

    Answer:
    {answer}

    Respond with ONLY a score between 0.0 and 1.0. Nothing else."""

# ── Internal Helpers ──────────────────────────────────────────

def _retrieve_for_question(question: str, top_k: int = 5, threshold: float = 0.3, search_mode: str = "vector", use_reranker: bool = False) -> list[dict]:
    """
    Retrieve chunks for a single question using the current index.

    This is a helper that creates a fresh retriever for each question.
    Separated out so it can be easily mocked in tests.
    """
    store = VectorStore()
    if os.path.exists(DEFAULT_INDEX_PATH):
        store.load(DEFAULT_INDEX_PATH)
    
    retriever = Retriever(store, top_k=topk, threshold=threshold, search_mode=search_mode, use_reranker=use_reranker)
    return retriever.retrieve(question)

def _call_llm_judge(prompt: str) -> str:
    """
    Call the LLM to get a judge score.

    📚: We use the SAME generate() function as the main RAG pipeline,
    but with a different prompt. The LLM acts as a judge — evaluating
    quality rather than answering a question. This is called "LLM-as-Judge"
    and is standard practice in the industry (RAGAS, DeepEval, etc.).

    We use non-streaming mode here since we just need the score, not
    a pretty streaming output.
    """
    # Build messages for the judge
    messages = [
        {"role": "system", "content": "You are an evaluation judge. Respond with only a numeric score."},
        {"role": "user", "content": prompt},
    ]

    # Use generator's internal API call (non-streaming)
    from rag.generator import _get_api_key, CHAT_API_URL
    import requests

    api_key = _get_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"messages": messages, "stream": False}

    response = requests.post(CHAT_API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"]

# ── Main Evaluation Runner ────────────────────────────────────


def run_evaluation(
    test_questions: list[dict],
    top_k: int = 5,
    threshold: float = 0.3,
    verbose: bool = False,
    search_mode: str = "vector",
    use_reranker: bool = False,
) -> dict:
    """
    Run a full evaluation across all test questions.

    📚: For each question, we:
      1. Retrieve chunks (same as a normal query)
      2. Compute retrieval precision (did we find the right source?)
      3. Generate an answer (same as a normal query)
      4. Ask the LLM to judge faithfulness (is the answer grounded?)
      5. Ask the LLM to judge relevance (does it answer the question?)

    The results are aggregated into averages and also stored per-question
    so you can see exactly where the system struggles.

    Args:
        test_questions: List of test question dicts from test_questions.json
        top_k: Number of chunks to retrieve per question
        threshold: Minimum similarity threshold
        verbose: Print detailed results per question

    Returns:
        Dict with "summary" (aggregated metrics) and "individual" (per-question)
    """
    individual_results = []

    # Accumulators for averaging
    precision_scores = []
    faithfulness_scores = []
    relevance_scores = []
    keyword_hit_rates = []

    total = len(test_questions)
    for i, tq in enumerate(test_questions):
        question = tq["question"]
        expected_source = tq.get("expected_sourse")
        expected_keywords = tq.get("expected_keywords", [])

        print(f"\n{'─' * 60}")
        print(f"📝 [{i+1}/{total}] {question}")
        print(f"{'─' * 60}")

        # ── Step 1: Retrieve ──────────────────────────────────
        retrieved = _retrieve_for_question(question, top_k=top_k, threshold=threshold, search_mode=search_mode, use_reranker=use_reranker)

        # ── Step 2: Retrieval Precision ───────────────────────
        precision = compute_retrieval_precision(retrieved, expected_source)
        if precision is not None:
            precision_scores.append(precision)
        
        # ── Step 3: Generate Answer ───────────────────────────
        if retrieved:
            # Build context text from retrieved chunks
            context_text = "\n\n".join(chunk["text"] for chunk in retrieved)

            # Generate answer (non-streaming for evaluation)
            print("  🤖 Generating answer...")
            answer = generate(question, retrieved, stream=False)
        else:
            context_text = ""
            answer = "No relevant documents found."
        
        # ── Step 4: Faithfulness (LLM Judge) ──────────────────
        if retrieved and answer:
            try:
                print("  ⚖️  Judging faithfulness...")
                faith_prompt = build_faithfulness_prompt(context_text, answer)
                faith_response = _call_llm_judge(faith_prompt)
                faithfulness = parse_llm_score(faith_response)
            except Exception as e:
                print(f"  ⚠️  Faithfulness judge failed: {e}")
                faithfulness = None
        else:
            faithfulness = None
        
        if faithfulness is not None:
            faithfulness_scores.append(faithfulness)
        
        # ── Step 5: Relevance (LLM Judge) ─────────────────────
        if answer and answer != "No relevant documents found.":
            try:
                print("  ⚖️  Judging relevance...")
                rel_prompt = build_relevance_prompt(question, answer)
                rel_response = _call_llm_judge(rel_prompt)
                relevance = parse_llm_score(rel_response)
            except Exception as e:
                print(f"  ⚠️  Relevance judge failed: {e}")
                relevance = None
        else:
            relevance = None
        
        if relevance is not None:
            relevance_scores.append(relevance)
        
        # ── Step 6: Keyword Check ─────────────────────────────
        # 📚: Keyword checking is a simple sanity check — if the
        # expected keywords don't appear in the answer, something mightbe wrong. 
        # It's not a metric per se, but a useful signal.
        if expected_keywords and answer:
            answer_lower = answer.lower()
            hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
            keyword_rate = hits / len(expected_keywords) if expected_keywords else 0.0
            keyword_hit_rates.append(keyword_rate)
        else:
            keyword_rate = None

        # ── Store result ──────────────────────────────────────
        result = {
            "question": question,
            "expected_source": expected_source,
            "expected_keywords": expected_keywords,
            "answer": answer,
            "sources_retrieved": [
                chunk.get("metadata", {}).get("source", "?") for chunk in retrieved
            ],
            "top_score": retrieved[0]["score"] if retrieved else 0.0,
            "retrieval_precision": precision,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "keyword_hit_rate": keyword_rate,
        }
        individual_results.append(result)

        # ── Verbose output ────────────────────────────────────
        if verbose:
            print(f"\n  📊 Precision:    {precision if precision is not None else 'N/A'}")
            print(f"  📊 Faithfulness: {faithfulness if faithfulness is not None else 'N/A'}")
            print(f"  📊 Relevance:    {relevance if relevance is not None else 'N/A'}")
            print(f"  📊 Keywords:     {keyword_rate if keyword_rate is not None else 'N/A'}")
            print(f"  💬 Answer: {answer[:150]}...")
        else:
            status = "✅" if (precision is not None and precision > 0) else ("⏭️" if precision is None else "❌")
            print(f"  {status} P:{_fmt(precision)} F:{_fmt(faithfulness)} R:{_fmt(relevance)} K:{_fmt(keyword_rate)}")
    
    # ── Aggregate Summary ─────────────────────────────────────
    summary = {
        "num_questions": total,
        "avg_retrieval_precision": _safe_avg(precision_scores),
        "avg_faithfulness": _safe_avg(faithfulness_scores),
        "avg_relevance": _safe_avg(relevance_scores),
        "avg_keyword_hit_rate": _safe_avg(keyword_hit_rates),
        "num_precision_scored": len(precision_scores),
        "num_faithfulness_scored": len(faithfulness_scores),
        "num_relevance_scored": len(relevance_scores),
    }

    return {
        "summary": summary,
        "individual": individual_results,
        "settings": {
            "top_k": top_k,
            "threshold": threshold,
            "search_mode": search_mode,
            "use_reranker": use_reranker,
            "timestamp": datetime.now().isoformat(),
        },
    }

def _safe_avg(scores: list[float]) -> float:
    """Compute average, returning 0.0 for empty lists."""
    return sum(scores) / len(scores) if scores else 0.0

def _fmt(val) -> str:
    """Format a score for display, handling None."""
    return f"{val:.2f}" if val is not None else "N/A "

# ── Results Persistence ───────────────────────────────────────

def save_results(results: dict, output_dir: str = "eval/results") -> str:
    """
    Save evaluation results to a timestamped JSON file.

    📚: Saving each evaluation run lets you compare results over time.
    Did changing chunk_size help? Did switching to hybrid search improve
    retrieval precision? You can answer these by comparing result files.

    Args:
        results: The results dict from run_evaluation()
        output_dir: Directory to save results in

    Returns:
        Path to the saved file
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return filepath

def print_scorecard(results: dict):
    """
    Print a pretty scorecard of evaluation results.

    📚 LEARN: The scorecard gives you a quick overview:
      - Overall averages for each metric
      - Per-question breakdown so you can find weak spots
      - Settings used so you know how to reproduce
    """
    summary = results["summary"]

    print("\n" + "=" * 60)
    print("📊 EVALUATION SCORECARD")
    print("=" * 60)

    print(f"\n  Questions evaluated:  {summary['num_questions']}")
    print(f"\n  {'Metric':<25} {'Score':>8}  {'Scored':>8}")
    print(f"  {'─' * 25} {'─' * 8}  {'─' * 8}")
    print(f"  {'Retrieval Precision':<25} {summary['avg_retrieval_precision']:>8.2f}  {summary['num_precision_scored']:>5}/{summary['num_questions']}")
    print(f"  {'Answer Faithfulness':<25} {summary['avg_faithfulness']:>8.2f}  {summary['num_faithfulness_scored']:>5}/{summary['num_questions']}")
    print(f"  {'Answer Relevance':<25} {summary['avg_relevance']:>8.2f}  {summary['num_relevance_scored']:>5}/{summary['num_questions']}")
    print(f"  {'Keyword Hit Rate':<25} {summary['avg_keyword_hit_rate']:>8.2f}")

    # ── Interpretation ────────────────────────────────────────
    print(f"\n  {'─' * 45}")

    p = summary['avg_retrieval_precision']
    f = summary['avg_faithfulness']
    r = summary['avg_relevance']

    if p >= 0.7 and f >= 0.7 and r >= 0.7:
        print("  🎉 Overall: Looking good! All metrics are healthy.")
    elif p < 0.5:
        print("  ⚠️  Retrieval is weak — consider tuning chunk_size, overlap, or threshold.")
    elif f < 0.5:
        print("  ⚠️  Answers have low faithfulness — the LLM may be hallucinating. Check the system prompt.")
    elif r < 0.5:
        print("  ⚠️  Answers have low relevance — they may be off-topic. Check retrieval + prompt.")
    else:
        print("  📈 Room for improvement — see individual results for weak spots.")

    print("=" * 60)




