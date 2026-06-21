"""
📚: Reranker — The Second, Smarter Pass
===============================================
Initial retrieval (vector search or hybrid) is fast but rough. It scores
each chunk INDEPENDENTLY — the query embedding is compared to each chunk
embedding separately. Neither the vector store nor BM25 actually READS
the query and chunk together.

A reranker fixes this. It takes each (query, chunk) pair and asks:
  "How relevant is THIS specific passage to THIS specific question?"

📚 TWO-STAGE ARCHITECTURE:
Why not just use the reranker for everything? Because it's expensive:
  - Stage 1 (fast):   Retrieve top-20 candidates with vector/BM25
                       Cost: 1 embedding API call
  - Stage 2 (precise): Rerank those 20 → keep best 5
                       Cost: 20 LLM API calls

This is the standard industry pattern:
  - Fast, cheap retrieval narrows 305 chunks → 20 candidates
  - Slow, precise reranking narrows 20 candidates → 5 best

📚 LLM-AS-RERANKER:
We use the same gpt-5-mini model as our reranker. We ask it to score
each passage's relevance to the query on a 0-10 scale. In production,
companies often use specialized cross-encoder models (like Cohere Rerank
or a fine-tuned BERT), but LLM-as-reranker works well and needs no
extra model deployment.
"""
import re
import requests
from rag.generator import _get_api_key, CHAT_API_URL


# ── Prompt Building ───────────────────────────────────────────
def build_rerank_prompt(query: str, passage: str) -> str:
    """
    Build a prompt asking the LLM to score passage relevance.

    📚: The prompt is carefully designed:
      1. Clear role: "You are a relevance judge"
      2. Specific scale: 0-10 with anchor descriptions
      3. Both query AND passage provided together — this is the key
         advantage over embedding-based search, which processes them
         separately

    Args:
        query: The user's question
        passage: The chunk text to evaluate

    Returns:
        A prompt string for the LLM
    """
    return f"""You are a relevance judge. Score how relevant the passage is to the question.

Score scale:
- 0: Completely irrelevant, no connection to the question
- 3: Tangentially related but doesn't answer the question
- 5: Somewhat relevant, contains related information
- 7: Relevant, contains information that helps answer the question
- 10: Perfectly relevant, directly answers the question

Question: {query}

Passage: {passage}

Respond with ONLY a single number between 0 and 10. Nothing else."""

# ── Score Parsing ─────────────────────────────────────────────


def parse_rerank_score(text: str) -> float:
    """
    Extract a 0-10 score from LLM rerank output.

    📚: Similar to the evaluator's parse_llm_score, but on a
    0-10 scale instead of 0-1. We find the last number in the response
    and clamp it to [0, 10].

    Args:
        text: Raw LLM response

    Returns:
        A float between 0 and 10
    """
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if not numbers:
        return 0
    score = float(numbers[-1])
    return max(0, min(10, score))

# ── LLM Call ──────────────────────────────────────────────────

def _call_rerank_llm(prompt: str) -> str:
    """
    Call the LLM to get a rerank score.

    📚: Each (query, chunk) pair gets its own LLM call. This is
    why reranking is expensive — 20 chunks means 20 API calls. But each
    call is very fast because the response is just a single number.

    Separated out so it can be easily mocked in tests.
    """
    api_key = _get_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [
            {"role": "system", "content": "You are a relevance scoring system. Respond with only a number."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    response = requests.post(CHAT_API_URL, headers=headers, json=payload, timeout=15)
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"]

# ── Main Rerank Function ─────────────────────────────────────

def rerank(
    query: str,
    chunks: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Rerank retrieved chunks using the LLM as a relevance judge.

    📚: This is the two-stage retrieval pattern:
      Stage 1 already happened (vector/BM25 → top-20 candidates)
      Stage 2 is this function: score each candidate with the LLM → keep top-5

    For each chunk, we:
      1. Build a prompt: "Score 0-10 how relevant this passage is to the question"
      2. Call the LLM
      3. Parse the score
      4. Sort by score and return top_k

    Args:
        query: The user's question
        chunks: Retrieved chunks from Stage 1 (each has text, metadata, score)
        top_k: Number of chunks to keep after reranking

    Returns:
        List of chunks sorted by rerank score (highest first).
        Each chunk gets a new "rerank_score" field (0-10).
        The original "score" field is preserved.
    """
    if not chunks:
        return []
    print(f"  🔄 Reranking {len(chunks)} chunks...")

    scored_chunks = []
    for i,chunk in enumerate(chunks):
        prompt = build_rerank_prompt(query, chunk["text"])
        try:
            response = _call_rerank_llm(prompt)
            score = parse_rerank_score(response)
        except Exception as e:
            print(f"  ⚠️  Rerank failed for chunk {i}: {e}")
            score = 0  # Fallback: keep chunk but with lowest priority
        
        scored_chunks.append({
            "text": chunk["text"],
            "metadata": chunk.get("metadata", {}),
            "score": chunk.get("score", 0),  # Preserve original retrieval score
            "rerank_score": score,
        })
    # Sort by rerank score (highest first)
    scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

    top = scored_chunks[:top_k]
    if top:
        print(f"  ✅ Reranked: top score {top[0]['rerank_score']}/10, "
              f"kept {len(top)}/{len(chunks)} chunks")

    return top
