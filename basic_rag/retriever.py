"""
📚 LEARN: Retriever — Finding the Right Context for the LLM
=============================================================
The retriever is the "R" in RAG — Retrieval-Augmented Generation.

When a user asks a question, the retriever:
  1. Embeds the question into a vector (using the same model as the documents)
  2. Searches the vector store for the most similar chunks
  3. Returns the top-k chunks ranked by relevance

These chunks become the "context" that we inject into the LLM's prompt,
so it can answer based on YOUR documents instead of guessing.

📚 SEARCH MODES:
  - "vector"  — Semantic search only (cosine similarity on embeddings)
  - "keyword" — BM25 keyword search only (term frequency + IDF)
  - "hybrid"  — Both vector + BM25, merged with Reciprocal Rank Fusion (RRF)

📚 WHY HYBRID?
Vector search finds semantically similar text but can miss exact keywords.
BM25 finds exact keyword matches but misses synonyms.
Hybrid combines both for the best retrieval quality. This is what Azure AI Search, Elasticsearch, and most production systems use.

📚 WHY SAME EMBEDDING MODEL?
The query MUST be embedded with the same model used for the documents.
Both vectors must live in the same "vector space" for cosine similarity
to be meaningful. Using different models would be like measuring one thing
in miles and another in kilograms — the comparison makes no sense.
"""
from rag.embedder import embed_query
from rag.vector_store import VectorStore
from rag.bm25 import BM25, reciprocal_rank_fusion
from rag.reranker import rerank

class Retriever:
    """
    Retrieves the most relevant document chunks for a given query.

    📚 LEARN: The retriever is a thin orchestration layer. It doesn't do
    much on its own — it delegates:
      - Embedding → embedder.py
      - Searching → vector_store.py

    But it's valuable because it:
      1. Provides a clean interface: query in → ranked results out
      2. Handles configuration (top_k, threshold) in one place
      3. Keeps the pipeline modular — you can swap the vector store
         without changing the rest of the code
    """
    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        threshold: float = 0.0,
        search_mode: str = "vector",
        use_reranker: bool = False,
    ):
        """
        Args:
            vector_store: A populated VectorStore instance
            top_k: Maximum number of chunks to retrieve (default: 5)
            threshold: Minimum similarity score to include (default: 0.0)
            search_mode: "vector", "keyword", or "hybrid" (default: "vector")
            use_reranker: If True, rerank results with LLM (default: False)
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.threshold = threshold
        self.search_mode = search_mode
        self.use_reranker = use_reranker

        # 📚: Build a BM25 index from the same data in the vector store.
        # This lets us do keyword search alongside vector search without storing the data twice — BM25 just needs the text and metadata.
        self.bm25 = BM25()
        # if keyword/hybrid search & vector db has data -> contruct key word search
        if search_mode in ("keyword", "hybrid") and vector_store.entries:
            # extract text + metadata from vector db, not storing original data
            bm25_docs = [
                {"text": e["text"], "metadata": e["metadata"]}
                for e in vector_store.entries
            ]
            self.bm25.index(bm25_docs)
    
    def retrieve(self, query: str) -> list[dict]:
        """
        Retrieve the most relevant chunks for a natural language query.

        📚 LEARN: This method supports three search modes:

        "vector" (default):
          query → embed → cosine similarity search
          Good at finding semantically similar text (synonyms, paraphrases).

        "keyword" (BM25 only):
          query → tokenize → BM25 score
          Good at finding exact keyword matches (product names, codes).

        "hybrid" (both + RRF merge):
          query → [vector search] + [BM25 search] → Reciprocal Rank Fusion
          Best of both worlds. Chunks ranking well in BOTH methods get boosted.

        Args:
            query: The user's question in natural language

        Returns:
            List of dicts, each with:
              - "text": the chunk content
              - "metadata": source, chunk_index, etc.
              - "score": similarity score (cosine for vector, BM25 for keyword, RRF for hybrid)
            Sorted by score descending (most relevant first)
        """
        print(f"  🔍 Embedding query: \"{query[:60]}...\"" if len(query) > 60
              else f"  🔍 Embedding query: \"{query}\"")
        
         # ====================== Mode 1：key word search（BM25）======================
        if self.search_mode == "keyword":
            # BM25 only — no embedding needed
            # 📚: When reranking with BM25, we retrieve more candidates (leave reranking more space)
            # initially (top_k * 4) so the reranker has a wider pool to pick from.
            retrieve_k = self.top_k * 4 if self.use_reranker else self.top_k
            results = self.bm25.search(query, top_k=retrieve_k)
            print(f"  📄 Found {len(results)} relevant chunks via BM25 "
                  f"(top score: {results[0]['score']:.4f})" if results
                  else "  ⚠️  No relevant chunks found!")
            
            return self._maybe_rerank(query, results)
        
        # Vector search (needed for both "vector" and "hybrid" modes)
        query_vector = embed_query(query) # convert usr questions into vector
        # make sure the number use for retrieve
        retrieve_k = self.top_k * 4 if (self.search_mode == "hybrid" or self.use_reranker) else self.top_k
        # use similarity search to vector
        vector_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=retrieve_k,
            threshold=self.threshold,
        )
        # ========= Mode 2：vector search ======================
        if self.search_mode == "vector":
            print(f"  📄 Found {len(vector_results)} relevant chunks "
                  f"(top score: {vector_results[0]['score']:.4f})" if vector_results
                  else "  ⚠️  No relevant chunks found!")
            return self._maybe_rerank(query, vector_results)
        
        # ========= Mode 3：hybrid search (vector + keyword) ======================
        # 1. use BM25 search
        bm25_results = self.bm25.search(query, top_k =self.top_k * 4)
        # 2. use RRF algo to combine two search results (ranking)
        fused = reciprocal_rank_fusion(vector_results, bm25_results, top_k=self.top_k*4 if self.use_reranker else self.top_k)
        print(f"  📄 Found {len(fused)} relevant chunks via hybrid search "
              f"(vector: {len(vector_results)}, BM25: {len(bm25_results)}, "
              f"top RRF: {fused[0]['score']:.4f})" if fused
              else "  ⚠️  No relevant chunks found!")
        return self._maybe_rerank(query, fused)

    def _maybe_rerank(self, query: str, results: list[dict]) -> list[dict]:
        """
        Apply LLM reranking if enabled, otherwise return as-is.

        📚: This is the two-stage pattern:
          Stage 1 already retrieved top-N candidates (fast, cheap)
          Stage 2 (this step): rerank N → keep top_k (slow, precise)
        """
        # not reranking/no results -> return 
        if not self.use_reranker or not results:
            return results
        # reranking -> use llm to rerank, save top_k
        return rerank(query, results, top_k=self.top_k)
    