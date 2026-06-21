"""
📚: Generator — The "AG" in RAG (Augmented Generation)
==============================================================
This is the final step: we take the retrieved context chunks and the user's
question, build a prompt, and send it to the LLM to generate an answer.

The key insight of RAG is prompt engineering:
  - WITHOUT RAG: "What is overfitting?" → LLM answers from training data (may hallucinate)
  - WITH RAG: "Given these excerpts from the user's docs: [chunks], answer: What is overfitting?"
    → LLM answers grounded in the actual documents

📚 HOW WE CALL THE CHAT API:
We use the raw Azure AI Foundry REST API:
  POST https://<resource>.cognitiveservices.azure.com/openai/deployments/<model>/chat/completions?api-version=...

The request body looks like:
  {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant..."},
      {"role": "user", "content": "Context: [chunks]\\nQuestion: [query]"}
    ],
    "stream": true
  }

With "stream": true, the API sends tokens one at a time, so the user sees
the answer appear progressively instead of waiting for the full response.
"""
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────

# 📚: Azure AI Foundry bakes the model (deployment name) into the URL.
# The api-version query parameter tells Azure which API version to use.
CHAT_API_URL = os.getenv(
    "AZURE_CHAT_ENDPOINT",
    "https://your-resource-name.cognitiveservices.azure.com/openai/deployments/gpt-5-mini/chat/completions?api-version=2025-04-01-preview",
)

# 📚: The system message sets the LLM's behavior. It tells the model
# to answer ONLY from the provided context and to cite sources. This is
# crucial for RAG — without this instruction, the LLM might ignore the
# context and make up answers from its training data.
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Rules:
1. Answer ONLY based on the context provided below. Do not use your training data.
2. If the context doesn't contain enough information to answer, say "I don't have enough information in the provided documents to answer this question."
3. Keep answers concise and accurate.
4. At the end of your answer, cite which source(s) you used."""

def _get_api_key() -> str:
    """Get an auth token for Azure AI Foundry (API key or Azure CLI fallback)."""
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

    raise ValueError("No auth found! Either:\n"
        "  1. Set AZURE_API_KEY in .env\n"
        "  2. Run 'az login' to use Azure CLI auth"
    )

def build_prompt(query: str, context_chunks: list[dict]) -> list[dict]:
    """
    Build the chat messages array from the query and retrieved context.

    📚 LEARN: This is where "Augmentation" happens — we inject the retrieved
    document chunks directly into the prompt. The LLM sees them as context
    and uses them to generate a grounded answer.

    The prompt structure is:
      System: "You are a helpful assistant that answers from context..."
      User: "Context:\n[chunk 1]\n[chunk 2]\n\nQuestion: [query]"

    Args:
        query: The user's question
        context_chunks: List of retrieval results (each has text, metadata, score)

    Returns:
        A list of message dicts ready for the OpenAI Chat API
    """
    # 📚: We format each chunk with its source and relevance score
    # so the LLM knows where each piece of information came from.
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        source = chunk.get("metadata", {}).get("source", "unknown")
        score = chunk.get("score", 0.0)
        context_parts.append(
            f"[Source: {source} | Relevance: {score:.2f}]\n{chunk['text']}"
        )
    context_text = "\n\n---\n\n".join(context_parts)

    # 📚: The messages format is what OpenAI's Chat API expects.
    # "system" sets behavior, "user" is the actual prompt.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {query}",
        },
    ]
    return messages

def generate(query: str, context_chunks: list[dict], stream: bool = True) -> str:
    """
    Generate an answer using the Azure AI Foundry Chat API.

    📚 LEARN: We call the API with stream=True, which uses Server-Sent Events
    (SSE). Instead of waiting for the full answer, the API sends one token at
    a time. Each line looks like:
      data: {"choices": [{"delta": {"content": "The"}}]}
      data: {"choices": [{"delta": {"content": " answer"}}]}
      data: {"choices": [{"delta": {"content": " is"}}]}
      data: [DONE]

    We print each token immediately, so the user sees the answer build up
    in real time — just like ChatGPT's typing effect.

    Args:
        query: The user's question
        context_chunks: Retrieved context from the retriever
        stream: Whether to stream tokens (default: True)

    Returns:
        The complete generated answer as a string
    """
    api_key = _get_api_key()
    messages = build_prompt(query, context_chunks)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "messages": messages,
        "stream": stream,
        # 📚 : Some models (like gpt-5-mini) only support the default
        # temperature. We omit it here so the API uses its default.
    }

    # 📚: temperature controls randomness in the LLM's output:
    #   0.0 = always pick the most likely next token (deterministic)
    #   1.0 = sample from the full distribution (more creative/random)
    # For RAG, we want factual answers, so we use a low temperature (0.3).
    if stream:
        return _generate_streaming(headers, payload)
    else:
        return _generate_non_streaming(headers, payload)
    

def _generate_streaming(headers: dict, payload: dict) -> str:
    """
    Call the API with streaming and print tokens as they arrive.

    📚: Streaming uses Server-Sent Events (SSE). The response is not
    a single JSON object — it's a series of lines, each starting with "data: ".
    We parse each line, extract the token, and print it immediately.
    """
    response = requests.post(
        CHAT_API_URL,
        headers=headers,
        json=payload,
        stream=True, # Don't download the whole response at once
        timeout=60
    )
    if response.status_code != 200:
        print(f"  ❌ API Error {response.status_code}: {response.text[:500]}")
    response.raise_for_status()

    full_response = []

    # 📚: iter_lines() reads the response one line at a time as
    # the server sends them. Each meaningful line starts with "data: ".
    for line in response.iter_lines(decode_unicode=True):
        if not line or not iter.startswith("data: "):
            continue

        data_str = line[6:] # Remove "data: " prefix
        if data_str == '[Done]':
            break
        try:
            data = json.loads(data_str)
            # 📚: In streaming mode, each chunk has a "delta" instead
            # of "message". The delta contains just the NEW content (one or
            # a few tokens). We accumulate these to build the full answer.
            delta = data["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                print(content, end="", flush=True)
                full_response.append(content)
        except (json.JSONDecodeError, KeyError, IndexError):
            continue  # Skip malformed chunks
    
    print()  # Newline after streaming is done
    return "".join(full_response)

def _generate_non_streaming(headers: dict, payload: dict) -> str:
    """
    Call the API without streaming — wait for the full response.
    """
    response = requests.post(
        CHAT_API_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()
    answer = result["choices"][0]["messages"]["content"]
    return answer

def generate_with_sources(
    query: str, context_chunks: list[dict], stream: bool = True
) -> dict:
    """
    Generate an answer and return it with source attribution.

    📚: Source attribution is what makes RAG trustworthy. Instead of
    just giving an answer, we also show WHICH documents were used. The user
    can then verify the answer against the original sources.

    Args:
        query: The user's question
        context_chunks: Retrieved context chunks

    Returns:
        A dict with:
          - "answer": the generated text
          - "sources": list of source files used
          - "chunks_used": the context chunks with scores
    """
    answer = generate(query, context_chunks, stream=stream)
    # Extract unique sources from the chunks
    sources = list(set(
        chunk.get("metadata", {}).get("source", "unkown")
        for chunk in context_chunks
    ))

    return {
        "answer": answer,
        "sources": sorted(sources),
        "chunks_used": context_chunks,
    }
