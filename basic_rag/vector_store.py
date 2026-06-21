"""
📚: Vector Store — Your Homemade Search Engine
=====================================================
A vector store is a database optimized for storing and searching vectors.
In production, people use FAISS, ChromaDB, Pinecone, etc. But we're building
one FROM SCRATCH so you understand exactly how it works.

The core idea is simple:
  1. Store a list of {text, vector, metadata} entries
  2. When searching, compute cosine similarity between the query vector
     and every stored vector
  3. Return the top-k most similar entries

📚 COSINE SIMILARITY — The Math Behind Semantic Search
=======================================================
Cosine similarity measures the angle between two vectors:

  cosine_sim(A, B) = (A · B) / (|A| × |B|)

Where:
  A · B  = dot product = sum of (a_i × b_i) for each dimension
  |A|    = magnitude = sqrt(sum of a_i²)

The result is a number between -1 and 1:
  1.0  = identical direction (same meaning)
  0.0  = perpendicular (unrelated)
  -1.0 = opposite direction (opposite meaning)

Why cosine and not Euclidean distance? Because cosine similarity cares about
DIRECTION, not magnitude. Two paragraphs about cats will point in a similar
direction regardless of whether one is 10 words or 100 words.
"""
import json
import math


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors manually.

    📚 LEARN: We implement this from scratch — no numpy! Here's the math:

    Step 1: Dot product — multiply corresponding elements and sum them.
      A = [1, 2, 3], B = [4, 5, 6]
      A · B = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32

    Step 2: Magnitudes — the "length" of each vector.
      |A| = sqrt(1² + 2² + 3²) = sqrt(14) ≈ 3.74
      |B| = sqrt(4² + 5² + 6²) = sqrt(77) ≈ 8.77

    Step 3: Divide.
      cosine_sim = 32 / (3.74 × 8.77) ≈ 0.974

    Args:
        vec_a: First vector (list of floats)
        vec_b: Second vector (list of floats)

    Returns:
        Cosine similarity score between -1.0 and 1.0
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vectors must have same length: {len(vec_a)} vs {len(vec_b)}"
        )

    # Step 1: Dot product
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))

    # Step 2: Magnitudes
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    # Step 3: Divide (guard against zero-length vectors)
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)

class VectorStore:
    """
    A simple in-memory vector store built from scratch.

    📚 LEARN: This is conceptually what ChromaDB, FAISS, and Pinecone do,
    but stripped down to the essentials. Our store is just a Python list
    of dictionaries. Search is a brute-force loop computing cosine similarity
    against every entry.

    This is O(n) per search — it checks every vector. Production vector DBs
    use clever data structures (e.g., HNSW graphs, IVF indexes) to make
    search sub-linear. But for thousands of chunks, brute-force is fine
    and much easier to understand.
    """
    def __init__(self):
        # 📚: Each entry in the store is a dict with:
        #   "text"      — the original chunk text
        #   "embedding" — the vector (list of floats)
        #   "metadata"  — source file, chunk index, etc.
        self.entries: list[dict] = []
    
    def add(self, text: str, embedding: list[float], metadata: dict = None):
        """
        Add a single entry to the vector store.

        Args:
            text: The chunk text
            embedding: The vector representation
            metadata: Optional dict with source, chunk_index, etc.
        """
        self.entries.append({
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {},
        })
    
    def add_chunks(self, chunks: list[dict]):
        """
        Add multiple chunks (from embed_chunks) to the store at once.

        This is the main way to populate the store. It expects chunk dicts
        that have "text", "embedding", and optionally other metadata fields.

        Args:
            chunks: List of chunk dicts from embedder.embed_chunks()
        """
        for chunk in chunks:
            metadata = {
                k: v for k,v in chunk.items()
                if k not in ("text", "embedding")
            }
            self.add(
                text=chunk["text"],
                embedding=chunk["embedding"],
                metadata=metadata,
            )

    def search(
        self, query_vector: list[float], top_k: int = 5, threshold: float = 0.0
    ) -> list[dict]:
        """
        Find the top-k most similar entries to the query vector.

        📚 LEARN: This is BRUTE-FORCE search — we compute cosine similarity
        between the query and EVERY vector in the store, then sort by score.

        It's O(n × d) where:
          n = number of stored entries
          d = vector dimensions (1536)

        For 10,000 entries: 10,000 × 1536 ≈ 15 million multiplications.
        Sounds like a lot, but modern CPUs do billions per second — it takes
        about 100ms. For millions of entries, you'd need an index (FAISS etc.)

        Args:
            query_vector: The query embedding vector
            top_k: Number of results to return (default: 5)
            threshold: Minimum similarity score (default: 0.0, meaning return all)

        Returns:
            List of dicts with: text, metadata, score — sorted by score descending
        """
        if not self.entries:
            return []
        
        # 📚: Score every entry against the query
        scored = []
        for entry in self.entries:
            score = cosine_similarity(query_vector, entry["embedding"])
            if score >= threshold:
                scored.append({
                    "text": entry["text"],
                    "metadata": entry["metadata"],
                    "score": score,
                })
        # 📚: Sort by similarity score (highest first) and take top-k.
        # This is the "retrieval" in Retrieval-Augmented Generation!
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
    
    def save(self, path:str):
        """
        Save the vector store to a JSON file.

        📚 LEARN: We persist the store as JSON so you can inspect it with
        any text editor. Production systems use binary formats (faster, smaller),
        but JSON is perfect for learning — you can literally open the file
        and see the vectors.

        Args:
            path: File path to save to (e.g., "my_index.json")
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2)
        print(f"  💾 Saved {len(self.entries)} entries to {path}")

    def load(self, path: str):
        """
        Load a vector store from a JSON file.

        This replaces the current store contents with the loaded data.

        Args:
            path: File path to load from
        """
        with open(path, "r", encoding="utf-8") as f:
            self.entries = json.load(f)
        print(f"  📂 Loaded {len(self.entries)} entries from {path}")
    

    def __len__(self) -> int:
        return len(self.entries)
    
    def list_sources(self) -> list[str]:
        """
        List all unique document sources stored in the index.

        Returns:
            List of source paths/URLs
        """
        sources = set()
        for entry in self.entries:
            source = entry.get("metadata", {}).get("source", "unkown")
            source.add(source)
        return sorted(source)
    
# ── Manual testing ─────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import tempfile

    print("=" * 60)
    print("VECTOR STORE TEST — Cosine Similarity from Scratch")
    print("=" * 60)

    # 📚: Let's create some fake vectors to test with.
    # In real usage, these come from the embedder. But for testing,
    # we use simple vectors where the math is easy to verify.
    store = VectorStore()

    # Three "documents" with simple 3D vectors
    store.add(
        text="The cat sat on the mat",
        embedding=[1.0, 0.0, 0.0],   # Points along x-axis
        metadata={"source": "cats.txt", "chunk_index": 0},
    )
    store.add(
        text="A kitten rested on a rug",
        embedding=[0.9, 0.1, 0.0],   # Mostly x-axis (similar to cat!)
        metadata={"source": "cats.txt", "chunk_index": 1},
    )
    store.add(
        text="Stock prices rose 5%",
        embedding=[0.0, 0.0, 1.0],   # Points along z-axis (different!)
        metadata={"source": "finance.txt", "chunk_index": 0},
    )

    print(f"\nStore has {len(store)} entries")
    print(f"Sources: {store.list_sources()}")

    # Search with a vector similar to "cat" entries
    print("\n--- Searching for 'cat-like' query [0.95, 0.05, 0.0] ---")
    results = store.search(query_vector=[0.95, 0.05, 0.0], top_k=3)
    for r in results:
        print(f"  Score: {r['score']:.4f} | {r['text']}")

    # Test save/load
    print("\n--- Testing save/load ---")
    tmp_path = os.path.join(tempfile.gettempdir(), "test_store.json")
    store.save(tmp_path)

    store2 = VectorStore()
    store2.load(tmp_path)
    print(f"  Loaded store has {len(store2)} entries")

    # Verify loaded store gives same results
    results2 = store2.search(query_vector=[0.95, 0.05, 0.0], top_k=3)
    assert results[0]["text"] == results2[0]["text"], "Save/load broke search!"
    print("  ✅ Save/load round-trip works correctly!")

    # Clean up
    os.remove(tmp_path)