"""
Text Chunking — Breaking Documents Into Searchable Pieces
====================================================================
Imagine you have a 50-page document about ML. A user asks:
"What is overfitting?" The answer is probably in 1-2 paragraphs, not all 50
pages. So we need to:

  1. Split the document into small "chunks" (a few hundred characters each)
  2. Later, embed each chunk into a vector
  3. When the user asks a question, find the most relevant chunks

This module implements three chunking strategies, from simple to smart:

  1. Fixed-size chunking    — split every N characters (simplest)
  2. Overlapping chunking   — fixed-size but with overlap between chunks
  3. Sentence-aware chunking — split on sentence boundaries (smartest)

📚 WHY OVERLAP?
Imagine this text is split at exactly character 500:
  "...Overfitting happens when the model memori|zes the training data..."
The word "memorizes" got cut in half! And worse, chunk 1 ends with half a
thought and chunk 2 starts with the other half. If the user asks about
overfitting, neither chunk alone has the complete answer.

Important:!!!!
Overlap fixes this: chunk 1 ends at position 500, but chunk 2 starts at
position 300 (200 chars of overlap). Now the sentence about memorizing
appears in BOTH chunks, so at least one of them will match the query.
that's why we need checking overlap in chunk_sentences()
"""
import re

def chunk_fixed_size(text: str, chunk_size: int = 500) -> list[str]:
    """
    Split text into fixed-size chunks by character count.

    This is the simplest possible chunking strategy. It just slices the text
    every `chunk_size` characters. It's fast but dumb — it can cut words and
    sentences in half.

    📚: This is our "baseline" chunker. We build it first to understand
    the basic idea, then improve it with overlap and sentence-awareness.

    Args:
        text: The input text to chunk
        chunk_size: Maximum characters per chunk (default: 500)

    Returns:
        A list of text chunks
    """
    if not text:
        return []
    
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size].strip()
        if chunk: # skip empty chunks
            chunks.append(chunk)
    return chunks

def chunk_with_overlap(
    text: str, chunk_size: int = 500, overlap: int = 100
) -> list[str]:
    """
    Split text into fixed-size chunks with overlap between consecutive chunks.

    Each chunk overlaps with the previous one by `overlap` characters. This
    ensures that sentences spanning a chunk boundary appear fully in at least
    one chunk.

    📚: Think of it like a sliding window moving across the text:
    - Window size = chunk_size (how much text each chunk contains)
    - Step size = chunk_size - overlap (how far the window moves each time)

    Example with chunk_size=10, overlap=3:
      Text: "ABCDEFGHIJKLMNOP"
      Chunk 1: "ABCDEFGHIJ"  (pos 0-9)
      Chunk 2: "HIJKLMNOP"   (pos 7-15)  ← "HIJ" overlaps with chunk 1

    Args:
        text: The input text to chunk
        chunk_size: Maximum characters per chunk (default: 500)
        overlap: Number of overlapping characters (default: 100)

    Returns:
        A list of text chunks
    """
    if not text:
        return []
    
    if overlap >= chunk_size:
        raise ValueError(
            f"Overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )
    
    chunks = []
    # 📚: The step size determines how far we advance between chunks.
    # step = chunk_size - overlap. If chunk_size=500 and overlap=100, we
    # advance 400 chars each time, so each chunk shares 100 chars with the next.
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        chunk = text[i:i+chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        # Stop if we've reached the end of the text
        if i + chunk_size >= len(text):
            break
    return chunks

def chunk_sentences(text: str, chunk_size: int = 500, overlap: int = 100
) -> list[str]:
    """
    Split text into chunks at sentence boundaries, with overlap.

    This is the smartest chunker. Instead of cutting at arbitrary character
    positions, it splits on sentence endings (periods, question marks, etc.)
    and then groups sentences into chunks that fit within chunk_size.

    if current_length + sentence_length <= chunk_size, we continuing add to current_chunk_sentences
    else, if >, we make sure the next chunk has overlap with the size we want from the previous chunk

    📚 : This produces much better chunks for RAG because:
    1. No words or sentences get cut in half
    2. Each chunk contains complete thoughts
    3. The LLM will get coherent context, not sentence fragments

    Args:
        text: The input text to chunk
        chunk_size: Target maximum characters per chunk (default: 500)
        overlap: Approximate overlap in characters (default: 100)

    Returns:
        A list of text chunks, each containing complete sentences
    """
    if not text:
        return []
    # 📚: This regex splits text into sentences. It looks for:
    # [.!?]  — sentence-ending punctuation
    # \s+    — followed by whitespace
    # The (?<=...) is a "lookbehind" — it matches the position AFTER the
    # punctuation without consuming it, so the period stays with its sentence.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip()) # split into one sentence 

    # Remove empty strings that might result from splitting
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []
    
    chunks = [] # final result
    current_chunk_sentences = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        # If adding this sentence would exceed the chunk_size,
        # save the current chunk and start a new one
        if current_length + sentence_length > chunk_size and current_chunk_sentences:
            # Save the current chunk
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(chunk_text)

            # 📚: For overlap, we keep some sentences from the end of the current chunk to start the next chunk. 
            # We walk backwards through the sentences, adding them until we've accumulated enough overlap.
            # we control the size of overlapping
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_chunk_sentences):
                if overlap_length + len(s) > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_length += len(s)
            
            current_chunk_sentences = overlap_sentences
            current_length = overlap_length
        # if above if's current_chunk_sentences is empty(means the 1st sentence is big), we append to current_chunk_sentences
        # we will not cut any complete sentences
        current_chunk_sentences.append(sentence)
        current_length += sentence_length
    
    # Don't forget the last chunk!
    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences)
        # Only add if it's not a duplicate of the last chunk
        if not chunks or chunk_text != chunks[-1]:
            chunks.append(chunk_text)
    return chunks

def create_chunks(
    text: str,
    source: str,
    chunk_size: int = 500,
    overlap: int = 100,
    strategy: str = "sentence",
) -> list[dict]:
    """
    Main entry point: chunk text and attach metadata to each chunk.

    📚: Metadata is crucial in RAG! When the system retrieves a relevant
    chunk, you want to know WHERE it came from — which document, what position.
    This enables "source attribution" — showing the user not just an answer,
    but where the answer was found.

    Args:
        text: The text to chunk
        source: The source document path/URL (for metadata)
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks in characters
        strategy: "fixed", "overlap", or "sentence"

    Returns:
        A list of dicts, each with:
          - "text": the chunk text
          - "source": where it came from
          - "chunk_index": position in the document (0, 1, 2, ...)
          - "char_start": starting character position in original text
          - "char_end": ending character position in original text
    """
    # Choose the chunking strategy
    if strategy == "fixed":
        raw_chunks = chunk_fixed_size(text, chunk_size)
    elif strategy == "overlap":
        raw_chunks = chunk_with_overlap(text, chunk_size, overlap)
    elif strategy == "sentence":
        raw_chunks = chunk_sentences(text, chunk_size, overlap)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'fixed', 'overlap', or 'sentence'")
    
    # 📚: We attach metadata to each chunk. The char_start/char_end
    # fields let us find where this chunk lives in the original document.
    # This is done with text.find() which locates the chunk's position.
    enriched_chunks = []
    search_start = 0
    for i,chunk_text in enumerate(raw_chunks):
        # Find where this chunk appears in the original text
        char_start = text.find(chunk_text[:50], search_start)
        if char_start == -1:
            char_start = search_start # Fallback if exact match not found
        char_end = char_start + len(chunk_text)
        search_start = char_start + 1 # Move past this position for next search

        enriched_chunks.append({
            "text": chunk_text,
            "source": source,
            "chunk_index": i,
            "char_start": char_start,
            "char_end": char_end,
        })

    return enriched_chunks

# Manual testing
if __name__ == "__main__":
    sample_text = (
        "Machine learning is a subset of artificial intelligence. "
        "It enables computers to learn from data. "
        "Instead of writing rules by hand, you provide examples. "
        "The algorithm finds patterns in the data. "
        "Supervised learning uses labeled training data. "
        "Each example has an input and a known correct output. "
        "Unsupervised learning finds patterns in unlabeled data. "
        "There are no correct answers to learn from. "
        "Reinforcement learning involves an agent interacting with an environment. "
        "The agent learns by receiving rewards or penalties."
    )

    print("=" * 60)
    print("STRATEGY: fixed (chunk_size=200)")
    print("=" * 60)
    for chunk in chunk_fixed_size(sample_text, chunk_size=200):
        print(f"  [{len(chunk):3d} chars] {chunk[:80]}...")
    print()
    
    print("=" * 60)
    print("STRATEGY: overlap (chunk_size=200, overlap=50)")
    print("=" * 60)
    for chunk in chunk_with_overlap(sample_text, chunk_size=200, overlap=50):
        print(f"  [{len(chunk):3d} chars] {chunk[:80]}...")
    print()

    print("=" * 60)
    print("STRATEGY: sentence (chunk_size=200, overlap=50)")
    print("=" * 60)
    for chunk in chunk_sentences(sample_text, chunk_size=200, overlap=50):
        print(f"  [{len(chunk):3d} chars] {chunk[:80]}...")
    print()

    print("=" * 60)
    print("WITH METADATA (sentence strategy)")
    print("=" * 60)
    chunks = create_chunks(sample_text, source="test.md", chunk_size=200, overlap=50)
    for c in chunks:
        print(f"  Chunk {c['chunk_index']}: chars {c['char_start']}-{c['char_end']}")
        print(f"    {c['text'][:80]}...")
        print()
