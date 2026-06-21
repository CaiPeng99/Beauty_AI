"""
📚: BM25 — The Classic Keyword Search Algorithm (from scratch)
=====================================================================
BM25 (Best Matching 25) is the algorithm behind Elasticsearch, Apache Lucene,
and most search engines you've ever used. It's the industry-standard keyword
search algorithm.

📚 WHY BM25 AND NOT JUST "word in text"?
Simple keyword matching ("does the word appear?") treats every match the same.
BM25 is smarter — it considers THREE things:

  1. TERM FREQUENCY (TF): How often the query word appears in the chunk.
     More mentions = likely more relevant. But with diminishing returns!
     Going from 0→1 mentions is a big deal. 10→11 barely matters.

  2. INVERSE DOCUMENT FREQUENCY (IDF): How rare the word is across ALL chunks.
     "overfitting" appearing in 1 out of 100 chunks → very informative.
     "the" appearing in 99 out of 100 chunks → tells you nothing.
     IDF gives rare, distinctive words higher weight.

  3. LENGTH NORMALIZATION: A 50-word chunk that mentions "return" is probably
     more focused than a 500-word chunk that mentions it once. BM25 adjusts
     for document length so short, focused chunks aren't penalized.

📚 THE BM25 FORMULA:
For each query term q in the query Q, and each document D:

  score(Q, D) = Σ  IDF(q) × [ TF(q,D) × (k1 + 1) ]
               q∈Q          [ ─────────────────────── ]
                             [ TF(q,D) + k1 × (1 - b + b × |D|/avgdl) ]

Where:
  k1 = 1.2  (controls TF saturation — how quickly extra mentions stop mattering)
  b  = 0.75 (controls length normalization — 0 = ignore length, 1 = full normalization)
  |D|   = length of document D (in words)
  avgdl = average document length across the corpus

These k1 and b values are the standard defaults used by Elasticsearch and Lucene.
"""
import math
import re

# ── Stopwords ─────────────────────────────────────────────────

# 📚: Stopwords are extremely common words that carry almost no
# meaning for search. Removing them makes BM25 more effective because
# it focuses on the meaningful words. This list covers English basics.
STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "be", "was", "were",
    "are", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "shall",
    "this", "that", "these", "those", "i", "you", "he", "she", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "its", "our", "their", "not", "no", "nor", "so", "if", "then",
    "than", "too", "very", "just", "about", "up", "out", "all", "also",
})

# ── Tokenizer ─────────────────────────────────────────────────


def tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25: lowercase, remove punctuation, drop stopwords.

    📚: Tokenization is the first step in any text search system.
    We convert free-form text into a list of "tokens" (meaningful words).

    Steps:
      1. Lowercase: "Return" → "return"
      2. Remove punctuation: "policy." → "policy"
      3. Split on whitespace: "return policy" → ["return", "policy"]
      4. Remove stopwords: drop "the", "a", "is", etc.

    This is a simple tokenizer. Production systems might also do:
      - Stemming ("running" → "run")
      - Lemmatization ("better" → "good")
      - N-grams ("machine learning" as one token)

    Args:
        text: Raw text to tokenize

    Returns:
        List of clean, lowercase tokens
    """

    if not text:
        return []
    
    # Lowercase and split on non-alphanumeric characters
    words = re.findall(r"[a-z0-9]+", text.lower())

    # Remove stopwords
    return [w for w in words if w not in STOPWORDS]

# ── BM25 Class ────────────────────────────────────────────────


class BM25:
    """
    BM25 keyword search index built from scratch.

    📚: BM25 works in two phases:
      1. INDEX PHASE: Analyze all documents to compute IDF scores
         (how rare each word is across the corpus)
      2. SEARCH PHASE: For a query, score each document using TF × IDF
         with length normalization

    Usage:
        bm25 = BM25()
        bm25.index(documents)  # Build the index
        results = bm25.search("return policy", top_k=5)  # Search
    """
    # 📚: These are the standard BM25 hyperparameters.
    # k1 controls TF saturation (higher = TF matters more)
    # b controls length normalization (0 = off, 1 = full)
    K1 = 1.2
    B = 0.75

    def __init__(self):
        self.documents: list[dict] = []  # Original docs
        self.doc_tokens: list[list[str]] = []  # Tokenized docs
        self.doc_lengths: list[int] = []  # Length of each doc in tokens
        self.avg_doc_length: float = 0.0
        self.idf: dict[str, float] = {}  # IDF score per term
        self.n_docs: int = 0
    
    def index(self, documents):
        """
        Build the BM25 index from a list of documents.

        📚: Indexing computes two things we'll need at search time:

        1. TOKENIZED DOCUMENTS: Each doc converted to a list of tokens.
           We store this so we can quickly count term frequency at search time.

        2. IDF SCORES: For each unique word, how rare it is across the corpus.
           IDF = log((N - n(q) + 0.5) / (n(q) + 0.5) + 1)
           Where:
             N    = total number of documents
             n(q) = number of documents containing word q

        Args:
            documents: List of dicts with at least "text" and "metadata" keys
        """
        self.documents = documents
        self.n_docs = len(documents)

        # Step 1: Tokenize all documents
        self.documents = [tokenize(doc["text"]) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / max(self.n_docs, 1)

        # Step 2: Count document frequency for each term
        # df[word] = number of documents containing that word
        df: dict[str, int] = {}
        for tokens in self.doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] = df.get(token, 0) + 1
        
        # Step 3: Compute IDF for each term
        # 📚: The IDF formula penalizes common words and boosts rare ones.
        # log((N - df + 0.5) / (df + 0.5) + 1)
        #   - Word in ALL docs: log((N-N+0.5)/(N+0.5)+1) ≈ log(1) ≈ 0 (worthless)
        #   - Word in 1 doc:    log((N-1+0.5)/(1+0.5)+1) ≈ log(N) (valuable)
        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log(
                (self.n_docs - freq + 0.5) / (freq + 0.5) + 1
            )
    
    def _score_document(self, query_tokens: list[str], doc_idx: int) -> float:
        """
        Compute BM25 score for a single document against query tokens.

        📚: For each query term, we compute:
          score += IDF(term) × (TF × (k1+1)) / (TF + k1 × (1 - b + b × dl/avgdl))

        Where TF is how many times the term appears in this document.
        The denominator includes length normalization: longer docs get
        a slightly lower score per term occurrence.
        """
        doc_tokens = self.doc_tokens[doc_idx]
        doc_len = self.doc_lengths[doc_idx]

        # Count term frequencies in this document
        tf: dict[str, int] = {}
        for token in doc_tokens:
            tf[token] = tf.get(token, 0) + 1
        
        score = 0.0
        for qt in query_tokens:
            if qt not in self.idf:
                continue # Query term not in corpus
                
            term_freq = tf.get(qt, 0)
            if term_freq == 0:
                continue

            idf = self.idf[qt]
            # 📚: The BM25 TF component with saturation and length norm.
            # Numerator: TF × (k1 + 1) — linear in TF, scaled by k1
            # Denominator: TF + k1 × (1 - b + b × dl/avgdl)
            #   When dl = avgdl: denominator simplifies to TF + k1
            #   When dl > avgdl: denominator grows → score decreases (length penalty)
            #   When dl < avgdl: denominator shrinks → score increases (short doc bonus)
            numerator = term_freq * (self.K1 + 1)
            denominator = term_freq + self.K1 * (
                1 - self.B + self.B * doc_len / max(self.avg_doc_length, 1)
            )

            score += idf * (numerator / denominator)
        return score

def search(self, query: str, top_k: int = 5) -> list[dict]:
    """
    Search the corpus for documents matching the query.

    Args:
        query: Natural language query string
        top_k: Number of results to return

    Returns:
        List of dicts with "text", "metadata", "score", sorted by score desc.
        Only returns results with score > 0 (at least one query term matches).
    """
    if not self.documents:
        return []
    
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    # Score every document
    scored = []
    for i in range(self.n_docs):
        score = self._score_document(query_tokens, i)
        if score > 0:
            score.append({
                "text": self.documents[i]["text"],
                "metadata": self.documents[i].get("metadata", {}),
                "score": score,
            })
    # Sort by score descending and return top-k
    score.sort(key=lambda x: x["score"])
    return scored[:top_k]

    def __len__(self) -> int:
        return self.n_docs

# ── Reciprocal Rank Fusion ────────────────────────────────────


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    top_k: int = None,
) -> list[dict]:
    """
    Merge vector search and BM25 results using Reciprocal Rank Fusion (RRF).

    📚: The problem — you can't just add vector scores and BM25 scores:
      - Vector scores are 0.0 to 1.0 (cosine similarity)
      - BM25 scores are 0 to 25+ (unbounded)
    Adding them would let BM25 dominate.

    RRF solves this by using RANKS instead of scores:
      RRF_score(chunk) = 1/(k + rank_vector) + 1/(k + rank_bm25)

    Where k=60 is a constant that prevents top-ranked items from dominating
    too much. A chunk that ranks well in BOTH methods gets the highest
    combined score.

    📚 WHY k=60?
    The original RRF paper (Cormack et al., 2009) tested values from 1 to 100.
    k=60 gave the best results across their benchmarks. It's now the standard
    default used by Elasticsearch, Azure AI Search, and others.

    Args:
        vector_results: Results from vector search (with "text", "metadata", "score")
        bm25_results: Results from BM25 search
        k: RRF constant (default: 60)
        top_k: Max results to return (default: all)

    Returns:
        Merged results sorted by RRF score, with "score" set to the RRF score
    """
    # 📚: We use (text, source) as a unique key to identify chunks.
    # This lets us detect when the same chunk appears in both result lists.
    rrf_scores: dict[str, dict] = {}  # key → {chunk_data, rrf_score}

    def _chunk_key(chunk: dict) -> str:
        """Create a unique key for deduplication."""
        source = chunk.get("metadata", {}).get("source", "")
        # Use text + source to identify unique chunks
        return f"{source}::{chunk['text'][:100]}"
    
    # Score vector results by rank
    for rank, chunk in enumerate(vector_results):
        key = _chunk_key(chunk)
        if key not in rrf_scores:
            rrf_scores[key] = {
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {}),
                "rrf_score": 0.0,
            }
        rrf_scores[key]["rrf_score"] += 1.0 / (k + rank + 1)
    
    # Score BM25 results by rank
    for rank, chunk in enumerate(bm25_results):
        key = _chunk_key(chunk)
        if key not in rrf_scores:
            rrf_scores[key] = {
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {}),
                "rrf_score": 0.0,
            }
        rrf_scores[key]["rrf_score"] += 1.0 / (k + rank + 1)
    
    # Convert to list and sort by RRF score
    results = [
        {
            "text": v["text"],
            "metadata": v["metadata"],
            "score": v["rrf_score"],
        }
        for v in rrf_scores.values()
    ]

    results.sort(key=lambda x: x["score"], reverse=True)

    if top_k is not None:
        return results[:top_k]
    return results