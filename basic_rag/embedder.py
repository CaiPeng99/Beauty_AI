"""
📚: Embeddings — Turning Text Into Numbers
==================================================
This is where the "magic" of RAG happens. An embedding converts a piece of
text into a vector — a list of numbers (e.g., 1536 floating-point values).

Why is this useful? Because:
  - "The cat sat on the mat" → [0.02, -0.15, 0.83, ...]
  - "A kitten was sitting on a rug" → [0.03, -0.14, 0.81, ...]  (similar!)
  - "Stock prices rose 5% today" → [0.91, 0.22, -0.45, ...]     (different!)

Texts with similar MEANING get similar vectors, even if they use completely
different words. This is called "semantic similarity" and it's why vector
search is so much better than keyword search for RAG.

📚 HOW WE CALL THE API:
We use the raw Azure AI Foundry REST API via the `requests` library — no SDK.
This means you'll see exactly what HTTP request goes out and what comes back.
The endpoint is configured via .env and looks like:
  POST https://<resource>.cognitiveservices.azure.com/openai/deployments/<model>/embeddings?api-version=...

The request body looks like:
  {
    "input": ["chunk 1 text", "chunk 2 text", ...]
  }

The response looks like:
  {
    "data": [
      {"embedding": [0.02, -0.15, ...], "index": 0},
      {"embedding": [0.03, -0.14, ...], "index": 1},
    ],
    "usage": {"total_tokens": 42}
  }
"""

import os
import time
import requests
from dotenv import load_dotenv

# 📚: load_dotenv() reads the .env file and sets environment variables.
# This keeps your API key out of source code — NEVER hardcode secrets!
load_dotenv()

# ── Configuration ──────────────────────────────────────────────

# 📚: Azure AI Foundry uses a different URL pattern than OpenAI.
# Instead of a single endpoint + model in the body, Azure bakes the model
# (deployment name) into the URL itself. The API key goes in an "api-key"
# header instead of "Authorization: Bearer ...".
EMBEDDING_API_URL = os.getenv(
    "AZURE_EMBEDDING_ENDPOINT",
    "https://your-resource-name.cognitiveservices.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15",
)

# 📚: text-embedding-3-large produces 3072-dimensional vectors.
# That means each chunk becomes a list of 3072 numbers. "3-large" is higher
# quality than "3-small" (1536 dims) — it captures more nuance in meaning.
EMBEDDING_DIMENSIONS = 3072

# 📚: The API accepts multiple texts in one request (batching).
# Sending 50 chunks at once is faster and cheaper than 50 individual calls.
# But there's a limit — we cap at 100 texts per batch to stay within
# API limits and avoid timeouts.
MAX_BATCH_SIZE = 100

# Retry configuration for rate-limit handling
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

def _get_api_key() -> str:
    """
    Get an auth token for Azure AI Foundry.

    📚: We support two authentication methods:
    1. AZURE_API_KEY in .env — simple, static key from Azure Portal
    2. Azure CLI token — dynamic, uses your logged-in Azure identity

    If no API key is set, we fall back to Azure CLI (`az account get-access-token`).
    This is great for local development — you just need to be logged in with `az login`.
    """
    api_key = os.getenv("AZURE_API_KEY")
    if api_key:
        return api_key

    # Fall back to Azure CLI token
    try:
        import subprocess
        result = subprocess.run(
            ["az", "account", "get-access-token", "--resource",
             "https://cognitiveservices.azure.com", "--query", "accessToken", "-o", "tsv"],
            capture_output=True, text=True, timeout=15, shell=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass

    raise ValueError(
        "No auth found! Either:\n"
        "  1. Set AZURE_API_KEY in .env\n"
        "  2. Run 'az login' to use Azure CLI auth"
    )

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of text strings into embedding vectors.

    This function handles batching automatically — if you pass 200 texts,
    it splits them into batches of MAX_BATCH_SIZE and makes multiple API calls.

    📚: This is a raw HTTP POST request to Azure AI Foundry's REST API.
    We're doing what the `openai` Python SDK does internally, but explicitly —
    so you can see every header, every JSON field, every response.

    Args:
        texts: List of text strings to embed

    Returns:
        List of vectors (each vector is a list of 1536 floats)

    Raises:
        ValueError: If API key is missing
        requests.HTTPError: If the API returns an error
    """
    if not texts:
        return []

    api_key = _get_api_key()

    # 📚: HTTP headers tell the server who we are and what we're sending.
    # Azure AI Foundry accepts Bearer token authentication, same pattern as
    # OpenAI's API — the key goes in the "Authorization" header.
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json", 
    }

    all_embeddings = []

    # 📚: Batching — instead of one API call per chunk, we send many
    # chunks at once. This is faster (fewer network round-trips) and the API
    # is designed for it. We split into batches of MAX_BATCH_SIZE.
    for batch_start in range(0, len(texts), MAX_BATCH_SIZE):
        batch = texts[batch_start : batch_start + MAX_BATCH_SIZE]

        # 📚: For Azure, the model is already in the URL (deployment name),
        # so we only need to send the input texts in the body.
        payload = {
            "input": batch,
        }

        # 📚: Retry logic with exponential backoff. If the API is busy
        # (rate-limited), it returns HTTP 429. Instead of crashing, we wait
        # and try again. Each retry waits longer (5s, 10s, 20s) — this is
        # called "exponential backoff" and it's polite to the server.
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    EMBEDDING_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=30,
                )

                # 📚: HTTP status codes:
                # 200 = success
                # 429 = rate limited (too many requests) — we should retry
                # 401 = bad API key
                # 400 = bad request (e.g., text too long)
                if response.status_code == 429:
                    wait_time = RETRY_DELAY_SECONDS * (2 ** attempt)
                    print(f"  ⏳ Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                break  # Success — exit retry loop
            except requests.exceptions.RequestException as e:
                if attempt == MAX_RETRIES - 1:
                    raise  # Final attempt failed — re-raise the error
                wait_time = RETRY_DELAY_SECONDS * (2 ** attempt)
                print(f"  ⚠️  Request failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        # 📚: The API response is JSON. The "data" field contains a list
        # of embedding objects, each with an "embedding" (the vector) and an
        # "index" (which input text it corresponds to).
        result = response.json()

        # Sort by index to ensure order matches our input order
        sorted_data = sorted(result["data"], key=lambda x: x["index"])
        batch_embeddings = [item["embedding"] for item in sorted_data]
        all_embeddings.extend(batch_embeddings)

        # Show progress for large batches
        if len(texts) > MAX_BATCH_SIZE:
            done = min(batch_start + MAX_BATCH_SIZE, len(texts))
            print(f"  📊 Embedded {done}/{len(texts)} chunks...")
    return all_embeddings

def embed_query(query: str) -> list[float]:
    """
    Embed a single query string into a vector.

    📚: We embed the user's question using the SAME model that we used
    to embed the document chunks. This is critical — both must be in the same
    "vector space" for similarity comparison to work. If you used different
    models, the vectors would be incomparable (like comparing temperatures
    in Celsius vs. Fahrenheit without conversion).

    Args:
        query: The user's question

    Returns:
        A single vector (list of floats)
    """
    vectors = embed_texts([query])
    return vectors[0]

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed a list of chunk dicts (from chunker.py) and add vectors to each.

    This is the main entry point that connects the chunker to the embedder.
    It takes the output of create_chunks() and adds an "embedding" field to
    each chunk dict.

    Args:
        chunks: List of chunk dicts from chunker.create_chunks()
                Each must have a "text" key.

    Returns:
        The same list of dicts, each now also containing an "embedding" key
        with the vector (list of 1536 floats).
    """
    if not chunks:
        return []
    
    # Extract just the text from each chunk for embedding
    texts = [chunk["text"] for chunk in chunks]

    print(f"  🔢 Embedding {len(texts)} chunks via Azure AI Foundry...")
    vectors = embed_texts(texts)

    # Add the embedding vector to each chunk dict
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    print(f"  ✅ Done! Each chunk now has a {len(vectors[0])}-dimensional vector.")
    return chunks

if __name__ == '__main__':
    print("=" * 60)
    print("EMBEDDER TEST — Raw OpenAI API call")
    print("=" * 60)

    test_texts = [
        "The cat sat on the mat.",
        "A kitten was resting on a rug.",
        "Stock prices rose 5% today.",
    ]

    print(f"\nEmbedding {len(test_texts)} texts...")
    vectors = embed_texts(test_texts)

    for text, vec in zip(test_texts, vectors):
        print(f"\n  Text: \"{text}\"")
        print(f"  Vector dims: {len(vec)}")
        print(f"  First 5 values: {vec[:5]}")

    # 📚: Let's compute cosine similarity manually to prove that
    # similar texts get similar vectors! (We'll build this properly in
    # vector_store.py, but here's a quick preview.)
    import math

    def cosine_sim(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0

    print("\n" + "=" * 60)
    print("SIMILARITY CHECK")
    print("=" * 60)
    sim_cat_kitten = cosine_sim(vectors[0], vectors[1])
    sim_cat_stocks = cosine_sim(vectors[0], vectors[2])
    print(f"  'cat on mat' vs 'kitten on rug':   {sim_cat_kitten:.4f}  (should be HIGH)")
    print(f"  'cat on mat' vs 'stock prices':    {sim_cat_stocks:.4f}  (should be LOW)")
