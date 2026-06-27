# Beauty_AI
# Beauty AI — RAG + MCP + AI Agent for Beauty Product Q&A and Content Publishing

An end-to-end AI system that answers beauty product questions and publishes social content to Notion and Twitter, built on a hybrid RAG pipeline with intent routing, cross-encoder reranking, and automated evaluation.

> **Stack:** Python · PostgreSQL + pgvector + TSVECTOR · Sentence Transformers · CrossEncoder · Claude API · MCP SDK · Notion API · Twitter API · Ragas

---

## What It Does

Users type a natural language prompt. The agent classifies the intent, routes it to the right pipeline, and either returns a grounded answer or publishes content to an external platform.

```
"Recommend a foundation for oily skin under $30"
    → Intent: select_product
    → SQL pre-filter: primary_category = Makeup, price_usd ≤ 30
    → Hybrid RAG: BM25 + cosine → RRF → CrossEncoder rerank
    → Aggregate: chunks collapsed to products, rec_ratio boost applied
    → Returns ranked product list with review-backed reasoning

"Show me new Sephora-exclusive skincare"
    → Intent: select_by_attribute
    → Pure SQL: new=1, sephora_exclusive=1, primary_category=Skincare
    → Returns products sorted by loves_count, no RAG needed

"Write an Instagram caption for this moisturizer and save to Notion"
    → Intent: publish_notion
    → RAG retrieves product context
    → LLM generates platform-specific copy with real review stats
    → MCP tool writes structured entry to Notion database
```

---

## Architecture

The system has two product-finding paths and one content path, all routing through a central intent classifier.

```
User Prompt
     │
     ▼
Intent Recognition  (LLM-based, 7 intent classes)
     │
     ├─── select_by_attribute ──► Pure SQL on Product table
     │                              (new, limited_edition, brand, category…)
     │                              Sort by loves_count / rating
     │                              └─► Return products
     │
     ├─── select_product ────────► RAG Pipeline  (see Module 1)
     │    unknown                   └─► Return ranked products
     │
     └─── publish_notion  ───────► RAG Pipeline
          publish_twitter            └─► Generate content  (see Module 3)
          save_local                       └─► MCP Tool call (see Module 3)
          generate_content
```

Two paths exist because they serve different query types.
`select_by_attribute` handles structured attribute queries ("new arrivals", "Sephora exclusive") where the answer is a deterministic SQL lookup.
`select_product` handles open-ended queries ("best moisturizer for dry skin") where meaning needs to be retrieved, not filtered.

---

## Module 1 — RAG Pipeline (`select_product` path)

### Stage 1 — Intent Parsing

Before any retrieval, the raw query is parsed into a structured filter dict by an LLM call:

```python
# Output schema
{
  "filters":        [{"field": "price_usd", "op": "lte", "value": 30}],
  "review_filters": [{"field": "skin_type", "op": "eq",  "value": "oily"}],
  "sort_by":        "loves_count",
  "intent":         "product_search"
}
```

Product-level conditions (`price_usd`, `primary_category`, `brand_name`, …) go into `filters`. Review-level conditions (`skin_type`, `skin_tone`, `is_recommended`, …) go into `review_filters`. They are applied at different stages of the pipeline.

One edge case handled explicitly: if both `primary_category` and `secondary_category` appear in filters, `primary_category` is dropped — the more specific secondary category gives better SQL recall without double-filtering.

### Stage 2 — SQL Pre-filter

Structured filters are applied directly to the `Product` table via SQLAlchemy before any vector search runs. This produces a `candidate_ids` set.

```python
# Example: price_usd ≤ 30 AND primary_category = 'Skincare'
q = db.query(Product.product_id)
q = q.filter(Product.price_usd <= 30)
q = q.filter(Product.primary_category.ilike('Skincare'))
candidate_ids = {r[0] for r in q.all()}
```

`review_filters` (skin type, skin tone) are pushed down to the `BeautyVectorStore` table, which has indexed columns for these fields. The two sets are intersected before retrieval begins.

If SQL pre-filter returns zero products, the pipeline short-circuits immediately — no embedding or BM25 computation is wasted.

### Stage 3 — Hybrid Retrieval: BM25 + Cosine → RRF

Two independent retrieval passes run in parallel, both scoped to `candidate_ids`:

**Dense (pgvector cosine distance)**
```sql
SELECT * FROM beauty_vector_store
WHERE product_id = ANY(:ids)
ORDER BY embedding <=> :query_vec
LIMIT 200;
```
Strong at semantic matching — catches paraphrases and concept-level similarity.

**Sparse (PostgreSQL TSVECTOR + BM25)**
```sql
SELECT * FROM beauty_vector_store
WHERE product_id = ANY(:ids)
  AND search_tsv @@ websearch_to_tsquery('english', :query)
ORDER BY ts_rank(search_tsv, websearch_to_tsquery('english', :query)) DESC
LIMIT 200;
```
Strong at exact keyword matching — critical for brand names, ingredient terms, and product codes that embeddings tend to blur.

Both ranked lists are merged with **Reciprocal Rank Fusion (RRF, k=60)**:

```python
rrf_score = 1/(60 + bm25_rank) + 1/(60 + vector_rank)
```

RRF is rank-based rather than score-based, which naturally handles the scale mismatch between cosine similarity (0–1) and BM25 scores (unbounded) without normalisation. The fused list is the input to reranking.

### Stage 4 — CrossEncoder Reranking

The top chunks from RRF are scored by a CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), which sees the query and each chunk together rather than independently:

```python
pairs = [[query, chunk["content"]] for chunk in rrf_results]
scores = rerank_model.predict(pairs)  # raw logits
```

Each chunk gets a `rerank_score` written back into its dict. This two-stage design — broad bi-encoder retrieval, precise cross-encoder scoring — is standard in production retrieval systems. The bi-encoder retrieves at scale; the cross-encoder selects precisely.

### Stage 5 — Aggregate: Chunks → Products

After reranking, the result list is still at **chunk level** (individual review snippets). This stage folds them back into **product level** — the unit users actually care about.

For each product in the results:

1. **Batch query** all its reviews from `BeautyVectorStore` (one SQL call for all products, not N+1)
2. Compute `rec_ratio` and `avg_rating` from the full review set
3. Apply a **Wilson lower-bound correction** on `rec_ratio` to penalise products with very few reviews (1 positive review ≠ 100% recommendation rate)
4. Compute `composite_score`:

```python
boost = 1.0 + 0.3 * (wilson_rec - 0.5) + 0.1 * ((avg_rating / 5.0) - 0.5)
boost = max(0.7, min(1.3, boost))          # cap boost between 0.7–1.3
composite_score = max_rerank_score * boost  # semantic score stays dominant
```

The CrossEncoder's `rerank_score` is the base — semantic relevance is the primary signal. `boost` is capped at ±30% so it adjusts but never overrides relevance. Products with fewer than `min_rec_count` positive reviews or `rec_ratio ≤ 0.5` are filtered out.

`min_rec_count` is dynamic: if the entire recall set has fewer than 30 chunks (sparse category), the threshold drops to 1 to avoid returning nothing for niche products.

---

## Module 2 — AI Agent

The agent (`run_workflow`) is the orchestration layer. It does not run a free-form loop — it is a deterministic router with memory and guardrails.

### Intent Classes

| Intent | Pipeline | Description |
|---|---|---|
| `select_product` | RAG pipeline | Open-ended product search |
| `select_by_attribute` | Pure SQL | Structured attribute filtering |
| `publish_notion` | RAG + MCP | Generate content → Notion |
| `publish_twitter` | RAG + MCP | Generate content → Twitter |
| `save_local` | RAG + MCP | Generate content → local file |
| `generate_content` | RAG + MCP | Content generation only |
| `qa` | Redirect | Routes to `/chat` endpoint |
| `unknown` | RAG pipeline | Falls through to `select_product` |

### Memory

Two memory layers run across turns:

- **Short-term memory** (`get_short_memory`): recent conversation turns, injected into content generation prompts so the LLM has context about what was just discussed
- **Long-term memory** (`get_long_memory`): semantically relevant past interactions retrieved by similarity to the current query, used in publish flows to maintain style consistency

### Guardrails

The pipeline has explicit stopping conditions at multiple stages:

- **SQL returns 0 products** → short-circuit, return `"NO_CONTENT"` before any embedding runs
- **Recall returns < 3 chunks** → `"INSUFFICIENT_RESULTS"`, trigger clarification
- **Aggregate produces 0 ranked products** → `"NO_RECOMMENDED_PRODUCTS"`, trigger fallback
- **Fallback** → if the category has products but no review data, return top-5 by `loves_count` with a note that review data is unavailable
- **Max retry** → after `MAX_RETRY` clarification attempts, return `"end"` status and stop asking

Each branch returns a typed status dict (`step`, `intent`, `result` / `message`) so the frontend can render the right UI state without parsing free text.

---

## Module 3 — MCP Tools

Four tools are registered via `@ToolRegistry.register`. The agent calls them by name; it never accesses external APIs directly.

| Tool | What it does |
|---|---|
| `select_product` | RAG smart search — full pipeline from intent parsing to ranked products |
| `generate_content` | LLM content generation with platform-specific rules and review stats injected |
| `publish_social` | Publishes to Twitter or Notion; LLM generates hashtags dynamically; writes a `PublishRecord` to the database |
| `write_local_file` | Saves content as a timestamped `.md` file |
| `write_local_file_mcp` | Same as above but called via MCP stdio protocol — used when the file server runs as a separate process |

The MCP stdio tools (`write_local_file_mcp`, `list_local_files_mcp`, `read_local_file_mcp`) spawn a child `file_server.py` process, communicate over stdin/stdout, and return structured results. This demonstrates the MCP client-server pattern: the tool boundary is a protocol, not a function call.

Content generation (`generate_content`) injects real recommendation statistics into the prompt:

```
[Real User Review Summary]
87% of users recommend this product,
average rating 4.3/5, based on 142 verified reviews
```

Platform-specific rules are enforced in the prompt: Instagram gets rich copy with hashtags; Twitter is capped at 280 characters with minimal tags. The LLM is instructed to use only provided product information and never invent ingredients or effects.

---

## Module 4 — Evaluation

A standalone eval harness (`tests/eval_pipeline.py`) measures quality across the retrieval and generation stages using a curated golden dataset of 30 query–answer pairs.

### Metrics

| Metric | What it catches | Method |
|---|---|---|
| **Retrieval Precision@5** | Retriever or reranker surfacing irrelevant chunks | Ground-truth chunk IDs vs. top-5 retrieved |
| **Answer Faithfulness** | LLM hallucinating product details not in context | LLM-as-judge (Claude) |
| **Answer Relevance** | Full pipeline drifting from the user's actual intent | LLM-as-judge (Claude) |

Evaluation uses the **Ragas** framework for standardised scoring, with a custom LLM-as-judge layer for domain-specific checks (e.g., whether the generated copy avoids unverified ingredient claims).

Results are written to `eval_results.json` after each run, enabling regression tracking as the pipeline evolves.

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Claude API (`claude-sonnet-4-6`) |
| Agent orchestration | Custom intent router + MCP Python SDK |
| Vector database | PostgreSQL + `pgvector` (cosine distance) |
| Full-text index | PostgreSQL `TSVECTOR` + `websearch_to_tsquery` + GIN index |
| Retrieval fusion | Reciprocal Rank Fusion (RRF, k=60) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Evaluation | Ragas + custom LLM-as-judge harness |
| Social publishing | Notion REST API v1 · Twitter API v2 |
| ORM | SQLAlchemy |
| Config | `python-dotenv` |


---

## Setup

**Prerequisites:** PostgreSQL ≥ 15 with `pgvector` extension, Python 3.11+

```bash
# 1. Clone and install
git clone https://github.com/CaiPeng99/Beauty_AI.git
cd Beauty_AI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env

# 3. Initialise database
python app/init_db.py

# 4. Ingest data
python app/data_process/etl.py

# 5. Create Notion database (one-time)
python tests/create_notion_db.py

# 6. Run the agent
python app/main.py

# 7. Run evaluation
python tests/eval_pipeline.py   # outputs eval_results.json
```

**Environment variables:**

```
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
POSTGRES_URL=postgresql://user:password@localhost:5432/beauty_ai
NOTION_TOKEN=
NOTION_PARENT_PAGE_ID=
NOTION_URL=https://api.notion.com/v1/databases
TWITTER_BEARER_TOKEN=
OUTPUT_DIR=./output
```

---

## License

MIT
