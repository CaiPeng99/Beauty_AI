"""
📚 LEARN: Pipeline — Wiring It All Together
=============================================
This module connects all the pieces into a complete RAG system:

  INGEST FLOW:
    Source file/URL → Loader → Chunker → Embedder → Vector Store (save to disk)

  QUERY FLOW:
    User question → Embedder → Retriever (search store) → Generator → Answer

Think of the pipeline as the "conductor" — each module is an instrument,
and the pipeline makes sure they play in the right order.
"""
import os
from rag.loader import load_document
from rag.chunker import create_chunks
from rag.embedder import embed_chunks
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.generator import generate_with_sources

# Default paths for the vector store index
DEFAULT_INDEX_PATH = "rag_index.json"

class RAGPipeline:
    """
    End-to-end RAG pipeline: ingest documents and answer questions.

    📚 LEARN: This class is the "glue" that connects all our modules.
    It manages the vector store lifecycle (create, populate, save, load)
    and provides simple ingest() and query() methods.
    """
    def __init__(
        self,
        index_path: str = DEFAULT_INDEX_PATH,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        top_k: int = 5,
        threshold: float = 0.3,
        search_mode: str = "vector",
        use_reranker: bool = False,
    ):
        """
        Args:
            index_path: Where to save/load the vector store
            chunk_size: Target size for text chunks
            chunk_overlap: Overlap between consecutive chunks
            top_k: Number of chunks to retrieve per query
            threshold: Minimum similarity score for retrieved chunks
            search_mode: "vector", "keyword", or "hybrid" (default: "vector")
            use_reranker: If True, rerank results with LLM (default: False)
        """
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.threshold = threshold
        self.search_mode = search_mode
        self.use_reranker = use_reranker

        # Load existing index if it exists, otherwise create empty store 
        self.store = VectorStore()
        if os.path.exists(index_path):
            self.store.load(index_path)
    
    def ingest(self, source: str) -> int:
        """
        Ingest a document: load → chunk → embed → store.

        📚: This is the "indexing" phase. You run this once for each
        document. After ingestion, the chunks and their vectors are saved
        to disk so you don't need to re-embed them every time.

        Args:
            source: File path or URL to ingest

        Returns:
            Number of chunks created
        """
        print(f"\n📥 Ingesting: {source}")

        # Step 1: Load the document
        print("  1️⃣  Loading document...")
        doc = load_document(source)
        print(f"     → {doc['type']} document, {len(doc['text'])} characters")

        # Step 2: Chunk the text
        print("  2️⃣  Chunking text...")
        chunks = create_chunks(
            text=doc["text"],
            source=doc["source"],
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
            strategy="sentence",
        )
        print(f"     → {len(chunks)} chunks (avg {sum(len(c['text']) for c in chunks) // max(len(chunks), 1)} chars each)")

        # Step 3: Embed the chunks
        print("  3️⃣  Embedding chunks...")
        embedded_chunks = embed_chunks(chunks)

        # Step 4: Add to vector store and save
        print("  4️⃣  Adding to vector store...")
        self.store.add_chunks(embedded_chunks)
        self.store.save(self.index_path)

        print(f"  ✅ Done! Store now has {len(self.store)} total chunks.")
        return len(chunks)

def query(self, question: str, verbose: bool = False) -> dict:
    """
    Ask a question and get an answer grounded in your documents.

    📚 LEARN: This is the "query" phase — the whole point of RAG:
        1. Embed the question
        2. Find similar chunks (retrieval)
        3. Send chunks + question to LLM (generation)
        4. Return answer with sources

    Args:
        question: Natural language question
        verbose: If True, show retrieved chunks before the answer

    Returns:
        Dict with "answer", "sources", and "chunks_used"
    """
    if len(self.store) == 0:
        print("⚠️  No documents ingested yet! Run 'ingest' first.")
        return {"answer": "No documents indexed.", "sources": [], "chunks_used": []}

    # Step 1 & 2: Retrieve relevant chunks
    retriever = Retriever(
            self.store,
            top_k=self.top_k,
            threshold=self.threshold,
            search_mode=self.search_mode,
            use_reranker=self.use_reranker,
    )
    results = retriever.retrieve(question)
    
    if not results:
        return {
            "answer": "No relevant documents found for your question.",
            "sources": [],
            "chunks_used": [],
        }

    # Show retrieved chunks if verbose
    if verbose:
        print(f"\n📄 Retrieved {len(results)} chunks:")
        for i, r in enumerate(results):
            source = r["metadata"].get("source", "?")
            print(f"  [{i+1}] (score: {r['score']:.4f}) {source}")
            print(f"      {r['text'][:100]}...")
        print()

    # Step 3: Generate answer
    print("\n💬 Answer:")
    result = generate_with_sources(question, results, stream=True)

    return result

def list_documents(self) -> list[str]:
    """List all documents that have been ingested."""
    return self.store.list_sources()

def auto_ingest(self, sample_dir: str = "sample_data"):
    """
    Auto-ingest all supported files from a directory on first run.

    📚: This is a convenience method for new users. On first run,
    the vector store is empty, so we automatically ingest all sample
    documents. This gives users something to query immediately without
    manual setup.

    Args:
        sample_dir: Directory to scan for documents (default: sample_data/)
    """
    if len(self.store) > 0:
        return  # Already have data — skip

    if not os.path.isdir(sample_dir):
        return  # No sample_data folder — skip
    
    # Find all supported files (including subdirectories)
    supported_extensions = (".txt", ".md", ".pdf")
    files = []
    for root, dirs, filenames in os.walk(sample_dir):
        for f in sorted(filenames):
            if f.lower().endswith(supported_extensions):
                files.append(os.path.join(root, f))
    
    if not files:
        return 
    
    print("=" * 60)
    print("🚀 First run detected! Auto-ingesting sample documents...")
    print("=" * 60)

    for file_path in files:
        self.ingest(file_path)

    print(f"\n✅ Auto-ingested {len(files)} documents. Ready to query!")








