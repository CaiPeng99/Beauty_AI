'''
Include：BM25 + 向量召回、分数判断、无结果分支、自动记录未知请求
'''
import json
import numpy as np
from collections import defaultdict
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from sqlalchemy.orm import Session
 
from app.database.models import BeautyVectorStore
from app.config import SIMILARITY_THRESHOLD, TOP_K_RECALL
from app.common.logger import unknown_logger
from app.database.models import Product

import logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reranker (module-level singleton — loaded once at import time)
# ---------------------------------------------------------------------------
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------------------------------------------------------------------
# Field registry
# ---------------------------------------------------------------------------
# Maps every queryable attribute to (source_table, python_cast).
# source_table = "product" | "review"
# To add a new field: append one line here — nothing else changes.
FIELD_SOURCE: dict[str, tuple[str, type]] = {
    # ── product_info fields ──────────────────────────────────────────────
    "rating":            ("product", float),
    "price_usd":         ("product", float),
    "sale_price_usd":    ("product", float),
    "value_price_usd":   ("product", float),
    "loves_count":       ("product", int),
    "reviews":           ("product", int),
    "new":               ("product", int),
    "limited_edition":   ("product", int),
    "sephora_exclusive": ("product", int),
    "online_only":       ("product", int),
    "out_of_stock":      ("product", int),
    "child_count":       ("product", int),
    "child_max_price":   ("product", float),
    "child_min_price":   ("product", float),
    "brand_name":        ("product", str),
    "primary_category":  ("product", str),
    "secondary_category":("product", str),
    "tertiary_category": ("product", str),
    "highlights":        ("product", str),
    "size":              ("product", str),
    # ── review fields ────────────────────────────────────────────────────
    "is_recommended":         ("review", int),
    "helpfulness":            ("review", float),
    "total_feedback_count":   ("review", int),
    "total_pos_feedback_count":("review", int),
    "total_neg_feedback_count":("review", int),
    "skin_type":   ("review", str),
    "skin_tone":   ("review", str),
    "eye_color":   ("review", str),
    "hair_color":  ("review", str),
}
 
# Supported comparison operators
OPS: dict[str, callable] = {
    "eq":       lambda a, b: a == b,
    "neq":      lambda a, b: a != b,
    "gte":      lambda a, b: a >= b,
    "lte":      lambda a, b: a <= b,
    "gt":       lambda a, b: a > b,
    "lt":       lambda a, b: a < b,
    "contains": lambda a, b: b.lower() in str(a).lower(),
}

# Product-level fields that live in the Product table (not meta_info chunks)
PRODUCT_TABLE_FIELDS = {
    "primary_category", "secondary_category", "tertiary_category",
    "brand_name", "price_usd", "sale_price_usd", "value_price_usd",
    "rating", "loves_count", "reviews", "new", "limited_edition",
    "sephora_exclusive", "online_only", "out_of_stock",
    "child_count", "child_max_price", "child_min_price", "size",
}

# 🔥 新增：BeautyVectorStore 表中可下推的 review 字段（带独立列索引）
    # 如果以后加了 skin_concern 列，只需在这里添加，代码一处改动全局生效
REVIEW_TABLE_FIELDS = {
    "skin_type", "skin_tone"
}

 
# Review-level fields that live in chunk meta_info
CHUNK_LEVEL_FIELDS = {
    "is_recommended", "helpfulness", "total_feedback_count",
    "total_pos_feedback_count", "total_neg_feedback_count",
    "skin_type", "skin_tone", "eye_color", "hair_color",
}
 
# Prompt template for intent parsing (filled at runtime)
INTENT_PARSE_PROMPT = """You are a beauty product query parser.
Extract structured filter conditions from the user's query.
 
Available product fields (from product_info):
  rating, price_usd, sale_price_usd, value_price_usd, loves_count, reviews,
  new, limited_edition, sephora_exclusive, online_only, out_of_stock,
  child_count, child_max_price, child_min_price,
  brand_name, primary_category, secondary_category, tertiary_category,
  highlights, size
 
Available review fields (from review data):
  is_recommended, helpfulness, total_feedback_count,
  total_pos_feedback_count, total_neg_feedback_count,
  skin_type, skin_tone, eye_color, hair_color
 
Operator options: eq | neq | gte | lte | gt | lt | contains
 
Return ONLY valid JSON — no markdown fences, no explanation:
{{
  "filters": [
    {{"field": "...", "op": "...", "value": ...}}
  ],
  "review_filters": [
    {{"field": "...", "op": "...", "value": ...}}
  ],
  "sort_by": "<field_name or null>",
  "sort_order": "desc",
  "intent": "product_search | review_analysis | post_generation"
}}
 
Rules:
- Use "filters" for product-level conditions, "review_filters" for review-level.
- Boolean flags are integers: 1 = true, 0 = false.
- Available primary_category values: {categories}
  Use primary_category when user mentions one of these broad categories.
- Available secondary_category values: {secondary_categories}
  Use secondary_category when user mentions a specific product type.
  Pick the closest matching value from this list.
- If the user says "not out of stock" → {{"field":"out_of_stock","op":"eq","value":0}}
- If no condition applies to a category, return an empty list for it.
- Do not invent fields that are not listed above.
- IMPORTANT: "review_filters" must ONLY contain review fields (is_recommended, helpfulness, skin_type, etc). NEVER put primary_category, secondary_category, brand_name, or any product field into "review_filters".
- Do NOT add "is_recommended" to review_filters. Recommendation filtering is handled automatically in aggregation.

User query: {query}"""
 

def tokenize(text: str):
    return text.lower().split()

def _parse_meta(meta_str: str) -> dict:
    """Safely parse a JSON metadata string stored in the DB."""
    if not meta_str:
        return {}
    try:
        return json.loads(meta_str)
    except Exception:
        return {}
 
 
def _cast_value(raw, cast: type):
    """Cast a raw metadata value to the expected Python type."""
    if cast is int:
        return int(float(raw))   # handles "1.0" → 1
    return cast(raw)
 

class HybridRetriever:
    """
    Hybrid BM25 + vector retriever with Pre-filter then RAG pattern:
      - SQL pre-filter narrows candidate set when structured conditions exist
      - BM25 + vector recall runs only within the candidate set
      - CrossEncoder reranking
      - Per-product recommendation aggregation
    """
    def __init__(self, db: Session, llm_client=None):
        """
        Args:
            db:         SQLAlchemy session connected to BeautyVectorStore.
            llm_client: Any client with a `.complete(prompt) -> str` method.
                        Used for intent parsing.  Pass None to skip parsing
                        and supply intent dicts manually.
        """
        self.db = db
        self.llm_client = llm_client                     # ← 用参数，不用全局变量
        self.bm25: BM25Okapi | None = None
        self.corpus: list[str] = []
        self.vector_data: list[dict] = []
        self._load_data()
        # 启动时从数据库读取真实的 category 列表
        self.known_categories = self._load_categories()
        self.known_secondary_categories = self._load_secondary_categories()
    
    def _load_categories(self) -> list[str]:
        # from app.database.models import Product
        rows = self.db.query(Product.primary_category).distinct().all()
        return [r[0] for r in rows if r[0]]
    
    def _load_secondary_categories(self) -> list[str]:
        rows = self.db.query(Product.secondary_category).distinct().all()
        return [r[0] for r in rows if r[0]]
    
    def _load_data(self):
        """Load the full knowledge base and initialise BM25."""
        records = self.db.query(BeautyVectorStore).all()
        for r in records:
            self.corpus.append(r.content)
            self.vector_data.append({
                "content": r.content,
                "embedding": np.array(r.embedding),
                "product_id": r.product_id,
                "chunk_type":   getattr(r, "chunk_type", ""),
                "is_recommended": getattr(r, "is_recommended", None),
                "rating":       getattr(r, "rating", 0.0),
                "meta_info":    getattr(r, "meta_info", "{}"),
            })
        tokenized_corpus = [tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    # ------------------------------------------------------------------ #
    # Intent parsing                                                       #
    # ------------------------------------------------------------------ #
    def parse_intent(self, query: str) -> dict:
        """
        Use the LLM to convert a free-text query into a structured filter dict.
 
        Returns a dict with keys: filters, review_filters, sort_by,
        sort_order, intent.  Falls back to an empty-filter dict on error.
        """
        empty = {"filters": [], "review_filters": [], "sort_by": None,
                 "sort_order": "desc", "intent": "product_search"}
        
        if self.llm_client is None:
            return empty
        
        cat_list = ", ".join(self.known_categories[:30])
        sec_cat_list = ", ".join(self.known_secondary_categories[:50])
        prompt = INTENT_PARSE_PROMPT.format(
            query=query,
            categories=cat_list,
            secondary_categories=sec_cat_list,
        )
        
        # prompt = INTENT_PARSE_PROMPT.format(query=query)
        try:
            raw = self.llm_client.complete(prompt)
            # Strip accidental markdown fences
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)
            # Validate top-level keys
            for key in ("filters", "review_filters"):
                if not isinstance(parsed.get(key), list):
                    parsed[key] = []
            return parsed
        
        except Exception as exc:
            unknown_logger.warning(f"parse_intent failed for query='{query}': {exc}")
            return empty
    
    # ------------------------------------------------------------------ #
    # SQL pre-filter                                                        #
    # ------------------------------------------------------------------ #
    def _prefilter_by_sql(self, filters: list[dict], review_filters: list[dict] = None) -> set[str] | None:
        """
        Apply product-table filters via SQL and return matching product_ids.
        Returns None if no product-table filters exist (caller should use full corpus).
        
        扩展版预过滤：
        1. 从 Product 表按结构化条件（品类/价格）筛出 product_id 集合 A
        2. 从 BeautyVectorStore 表按肤质/肤色筛出 product_id 集合 B
        3. 返回 A ∩ B (交集)
        返回 None 表示“无任何过滤，全量检索”；返回空 set 表示“被筛空”。

        维度	你的原方案（内存后过滤）	                               新方案（SQL 前置过滤）
        性能	向量库先捞 Top-K，再内存遍历过滤 → 浪费向量检索算力	    用 BeautyVectorStore 独立列索引直接筛出 ID，毫秒级完成
        召回率	如果 Top-K 里没油皮，直接返回 0 条	                 保证所有油皮产品都在候选集里（只要数据库有），向量检索只看这些 ID
        扩展性	硬编码 apply_filters 逻辑	                      白名单 REVIEW_TABLE_FIELDS，加字段只需改一行
        数据库压力	无 SQL JOIN，不卡 reviews 大表	               只查轻量级的 BeautyVectorStore（预聚合表），且有 product_id + skin_type 联合索引，极快
    """
        product_filters = [f for f in filters if f["field"] in PRODUCT_TABLE_FIELDS]

        print(f"DEBUG product_filters: {product_filters}")  # ← 加这行
        
        if not product_filters:
            return None
 
        q = self.db.query(Product.product_id)
        for rule in product_filters:
            field = rule["field"]
            op    = rule["op"]
            value = rule["value"]

            print(f"DEBUG checking field={field}, hasattr={hasattr(Product, field)}")  # ← 加这行
 
            if not hasattr(Product, field):
                continue
 
            col = getattr(Product, field)
 
            if op == "eq":
                # For string fields use case-insensitive match
                if isinstance(value, str):
                    q = q.filter(col.ilike(value))
                else:
                    q = q.filter(col == value)
            elif op == "neq":
                q = q.filter(col != value)
            elif op == "gte":
                q = q.filter(col >= value)
            elif op == "lte":
                q = q.filter(col <= value)
            elif op == "gt":
                q = q.filter(col > value)
            elif op == "lt":
                q = q.filter(col < value)
            elif op == "contains":
                q = q.filter(col.ilike(f"%{value}%"))
 
        # result = {r[0] for r in q.all()}
        # print(f"DEBUG candidate_ids count: {len(result)}")
        # return result

        product_ids = {r[0] for r in q.all()}
        # 如果 Product 表已经筛不出任何东西，直接返回空集（短路，不用查 BeautyVectorStore）
        if not product_ids:
            return set()
        
        # ====================================================
        # 阶段 2：处理 Review 层面的过滤（基于 BeautyVectorStore 表）
        # ====================================================
        review_ids = None
        
        if review_filters:
            # 只处理在白名单中的 review 字段
            valid_review_filters = [
                rf for rf in review_filters 
                if rf["field"] in self.REVIEW_TABLE_FIELDS
            ]
            
            if valid_review_filters:
                # 查询 BeautyVectorStore 表，找出符合肤质条件的 product_id（去重）
                bvs_q = self.db.query(BeautyVectorStore.product_id).distinct()
                
                for rule in valid_review_filters:
                    field = rule["field"]
                    op    = rule["op"]
                    value = rule["value"]
                    
                    # 安全检查：确保 BeautyVectorStore 有这个字段
                    if not hasattr(BeautyVectorStore, field):
                        continue
                    col = getattr(BeautyVectorStore, field)
                    
                    if op == "eq":
                        bvs_q = bvs_q.filter(col == value)  # 肤质是精确匹配，不要 ilike
                    elif op == "in":
                        bvs_q = bvs_q.filter(col.in_(value))
                    # 如果未来有范围查询（如 rating），可扩展 gte/lte
                
                review_ids = {r[0] for r in bvs_q.all()}
                
                # 如果 BeautyVectorStore 里都找不到任何匹配肤质的产品，直接返回空集
                if not review_ids:
                    return set()

            # ====================================================
            # 阶段 3：合并结果（取交集）
            # ====================================================
            # 情况1：没有任何过滤（Product 和 Review 都没限制）→ 返回 None，允许全量检索
            if product_ids is None and review_ids is None:
                return None
            
            # 情况2：只有 Product 过滤
            if review_ids is None:
                return product_ids
            
            # 情况3：只有 Review 过滤
            if product_ids is None:
                return review_ids
            
            # 情况4：两者都有 → 取交集
            return product_ids & review_ids
            

    
    # ------------------------------------------------------------------ #
    # Core retrieval                                                       #
    # ------------------------------------------------------------------ #
    def _cosine(self, v1, v2) -> float:
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        return float(np.dot(v1, v2) / (n1 * n2)) if n1 and n2 else 0.0
    
    def retrieve(
            self,
            query: str,
            query_embedding,
            candidate_ids: set[str] | None = None,   # ← 新增参数
        ) -> list[dict]:
        """
        BM25 keyword recall → vector semantic re-scoring.
 
        If candidate_ids is provided, only chunks belonging to those
        product_ids are considered (Pre-filter then RAG pattern).
        """
        # 1. BM25 关键词召回
        tokenized_query = tokenize(query)
        bm25_scores     = self.bm25.get_scores(tokenized_query)
 
        # When candidate_ids is given, mask out non-candidate chunks
        if candidate_ids is not None:
            for i, item in enumerate(self.vector_data):
                if item["product_id"] not in candidate_ids:
                    bm25_scores[i] = -1  # exclude from top-k
 
        # Pick top-k by BM25 (or more if candidate set is large)
        k = max(TOP_K_RECALL, len(candidate_ids) * 3) if candidate_ids else TOP_K_RECALL
        bm25_top = np.argsort(bm25_scores)[::-1][:k]

        # 2. 向量语义召回 + 打分
        results = []
        for idx in bm25_top:
            if bm25_scores[idx] < 0:
                break  # rest are masked out
            item  = self.vector_data[idx]
            score = self._cosine(query_embedding, item["embedding"])
            results.append({**item, "score": score})
        # 3. 按相似度排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    

    def search_with_judge(
        self,
        query: str,
        query_embedding,
        session_id: str,
        candidate_ids: set[str] | None = None,
    ):
        """
        Retrieval with threshold gating and unknown-query logging.
        Returns: (has_data, result_list, status_code)
        """
        results = self.retrieve(query, query_embedding, candidate_ids=candidate_ids)
        if not results:
            unknown_logger.warning(f"session:{session_id} | query:{query} | no recall results")
            return False, [], "NO_CONTENT"
 
        max_score = results[0]["score"]
        if max_score < SIMILARITY_THRESHOLD:
            unknown_logger.warning(
                f"session:{session_id} | query:{query} | "
                f"top score {max_score:.3f} < threshold"
            )
            return False, results, "LOW_SIMILARITY"
 
        return True, results, "SUCCESS"
 

    # ------------------------------------------------------------------ #
    # Dynamic filter engine (chunk-level review fields only)              #
    # ------------------------------------------------------------------ #
    def apply_filters(self, results: list[dict], filters: list[dict]) -> list[dict]:
        """
        Only filters review-level fields from chunk meta_info.
        Product-level fields are handled by SQL pre-filter.
        """
        if not filters:
            return results
 
        out = []
        for item in results:
            meta   = _parse_meta(item.get("meta_info", "{}"))
            passed = True
            for rule in filters:
                field = rule.get("field", "")
                op    = rule.get("op", "eq")
                value = rule.get("value")
 
                # Skip product-level fields — already handled by SQL pre-filter
                if field not in CHUNK_LEVEL_FIELDS:
                    continue
 
                config = FIELD_SOURCE.get(field)
                if config is None:
                    continue
 
                _, cast = config
                raw = meta.get(field)
                if raw is None:
                    passed = False
                    break
 
                op_fn = OPS.get(op)
                if op_fn is None:
                    continue
 
                try:
                    if not op_fn(_cast_value(raw, cast), _cast_value(value, cast)):
                        passed = False
                        break
                except Exception:
                    passed = False
                    break
 
            if passed:
                out.append(item)
        return out

    # ------------------------------------------------------------------ #
    # Reranking                                                            #
    # ------------------------------------------------------------------ #
 
    def rerank(self, query: str, results: list[dict]) -> list[dict]:
        """CrossEncoder rerank — returns results sorted by cross-attention score."""
        if not results:
            return []
        pairs = [[query, item["content"]] for item in results]
        scores = rerank_model.predict(pairs)
        return [item for item, _ in
                sorted(zip(results, scores), key=lambda x: x[1], reverse=True)]
 

    # ------------------------------------------------------------------ #
    # Recommendation aggregation                                                #
    # ------------------------------------------------------------------ #
    def get_product_review_stat(self, product_id: str) -> dict:
        """
        Query the DB directly for all reviews of a product and compute:
          rec_count, not_rec_count, avg_rating
        """
        rows = self.db.query(BeautyVectorStore).filter(
            BeautyVectorStore.product_id == product_id,
            BeautyVectorStore.chunk_type == "review"
        ).all()

        rec, not_rec = 0, 0
        rating_sum, rating_cnt = 0.0, 0
        for r in rows:
            if getattr(r, "is_recommended", 0) == 1:
                rec += 1
            else:
                not_rec += 1
            if getattr(r, "rating", 0) > 0:
                rating_sum += r.rating
                rating_cnt += 1
 
        return {
            "rec_count":     rec,
            "not_rec_count": not_rec,
            "avg_rating":    rating_sum / rating_cnt if rating_cnt else 0.0,
        }
 
    
    # 新增：按推荐状态快速筛选评论
    def filter_by_recommend(self, results: list, is_rec: int) -> list:
        """Return only chunks where is_recommended matches the given value."""
        return [i for i in results if i.get("is_recommended") == is_rec]
    
    def aggregate_and_rank(
        self,
        results: list[dict],
        min_rec_count: int = 3,
        rec_weight: float = 0.6,
        rating_weight: float = 0.4,
    ) -> list[dict]:
        """
        Group retrieved chunks by product, compute a composite score, and
        return only products that clear the recommendation bar.
 
        composite_score = rec_ratio * rec_weight + (avg_rating / 5) * rating_weight
 
        Args:
            min_rec_count:  Minimum number of positive reviews required.
            rec_weight:     Weight of recommendation ratio in composite score.
            rating_weight:  Weight of normalised avg rating in composite score.
 
        Returns a list of dicts (sorted by composite_score desc):
          {
            product_id, product_name, brand_name,
            composite_score, rec_ratio, avg_rating,
            rec_count, not_rec_count,
            top_chunks  # up to 3 positive review chunks for post generation
          }
        """
        # Group chunks by product
        by_product: dict[str, list[dict]] = defaultdict(list)
        for item in results:
            by_product[item["product_id"]].append(item)
        
        # 一次查询所有product的reviews，而不是循环查
        all_pids = list(by_product.keys())
        all_reviews = self.db.query(BeautyVectorStore).filter(
            BeautyVectorStore.product_id.in_(all_pids),
            BeautyVectorStore.chunk_type == "review"
        ).all()

        # 在Python里按product_id分组统计
        # stats: dict[str, dict] = defaultdict(lambda: {"rec": 0, "not_rec": 0, "rating_sum": 0.0, "rating_cnt": 0})
        # for r in all_reviews:
        #     s = stats[r.product_id]
        #     if getattr(r, "is_recommended", 0) == 1:
        #         s["rec"] += 1
        #     else:
        #         s["not_rec"] += 1
        #     if getattr(r, "rating", 0) > 0:
        #         s["rating_sum"] += r.rating
        #         s["rating_cnt"] += 1

        # 后续逻辑不变，只是用stats[pid]替换get_product_review_stat(pid)
 
        ranked = []
        for pid, chunks in by_product.items():
            stat = self.get_product_review_stat(pid)
            # stat = stats[pid]
            rec, not_rec = stat["rec_count"], stat["not_rec_count"]
            total = rec + not_rec
            print(f"DEBUG aggregate pid={pid}, rec={rec}, not_rec={not_rec}, total={total}")  # ← 加这行
 
            # Hard filters: need enough evidence and majority positive
            if total == 0 or rec < min_rec_count:
                print(f"  ↳ DROPPED: not enough reviews")  # ← 加这行
                continue
            rec_ratio = rec / total
            if rec_ratio <= 0.5:
                print(f"  ↳ DROPPED: rec_ratio={rec_ratio:.2f} too low")  # ← 加这行
                continue
 
            composite = rec_ratio * rec_weight + (stat["avg_rating"] / 5.0) * rating_weight
 
            # Keep top positive review chunks for post generation
            pos_chunks = self.filter_by_recommend(chunks, is_rec=1)
            # Take the top-3 by rerank score (already sorted from rerank step)
            top_chunks = pos_chunks[:3]
 
            # Pull display fields from the first chunk's metadata
            first_meta = _parse_meta(chunks[0].get("meta_info", "{}"))
 
            ranked.append({
                "product_id":      pid,
                "product_name":    first_meta.get("product_name", pid),
                "brand_name":      first_meta.get("brand_name", ""),
                "composite_score": composite,
                "rec_ratio":       rec_ratio,
                "avg_rating":      stat["avg_rating"],
                "rec_count":       rec,
                "not_rec_count":   not_rec,
                "top_chunks":      top_chunks,
            })
 
        ranked.sort(key=lambda x: x["composite_score"], reverse=True)
        return ranked
 
    # ------------------------------------------------------------------ #
    # Top-level entry point                                                #
    # ------------------------------------------------------------------ #
 
    def smart_search(
        self,
        query: str,
        query_embedding,
        session_id: str,
        parsed_intent: dict | None = None,
        min_rec_count: int = 3,
    ) -> tuple[bool, list, str]:
        """
        Full pipeline: SQL pre-filter → retrieve → review filter → rerank → aggregate.
 
        Args:
            query:          Raw user query string.
            query_embedding: Pre-computed query embedding vector.
            session_id:     For logging / tracing.
            parsed_intent:  Pre-parsed intent dict (skips LLM call if supplied).
                            If None and llm_client is set, calls parse_intent().
            min_rec_count:  Forwarded to aggregate_and_rank().
 
        Returns: (has_data, ranked_products, status_code)
        """
        # 1. Parse intent
        intent = parsed_intent if parsed_intent is not None else self.parse_intent(query)
 
        # 2. SQL pre-filter: narrow candidate set using product-table conditions
        # candidate_ids = self._prefilter_by_sql(intent.get("filters", []))

        # 改成 👇 同时传入 review_filters
        candidate_ids = self._prefilter_by_sql(
            filters=intent.get("filters", []),
            review_filters=intent.get("review_filters", [])
        )
        
        if candidate_ids is not None and len(candidate_ids) == 0:
            # SQL filter returned zero products — no point doing RAG
            unknown_logger.warning(
                f"session:{session_id} | query:{query} | SQL pre-filter returned 0 products"
            )
            return False, [], "NO_CONTENT"
 
        # 3. Hybrid recall within candidate set + threshold gate
        has_data, results, tip = self.search_with_judge(
            query, query_embedding, session_id, candidate_ids=candidate_ids
        )
        if not has_data:
            return False, [], tip
        
        print(f"DEBUG after recall: {len(results)} chunks")
        print(f"DEBUG products in results: {set(r['product_id'] for r in results)}")

 
        # 4. Apply review-level filters (chunk meta_info)
        # review_filters = intent.get("review_filters", [])
        # if review_filters:
        #     results = self.apply_filters(results, review_filters)
        #     if not results:
        #         unknown_logger.warning(
        #             f"session:{session_id} | query:{query} | "
        #             "all results filtered out after review filter"
        #         )
        #         return False, [], "FILTERED_EMPTY"
 
        # 5. Rerank
        # results = self.rerank(query, results)

        # rerank comparison
        results_before_rerank = results[:5]
        results = self.rerank(query, results)
        results_after_rerank = results[:5]
        logger.info(f"RERANK COMPARISON:")
        logger.info(f"Before: {[r['product_id'] for r in results_before_rerank]}")
        logger.info(f"After:  {[r['product_id'] for r in results_after_rerank]}")
 
        print(f"DEBUG before aggregate: {len(results)} chunks")
        by_pid = {}
        for r in results:
            by_pid.setdefault(r['product_id'], []).append(r)
        for pid, chunks in by_pid.items():
            print(f"  product_id={pid}, chunk_count={len(chunks)}")

        # 6. Aggregate per-product recommendation stats
        ranked = self.aggregate_and_rank(results, min_rec_count=min_rec_count)
 
        if not ranked:
            return False, [], "NO_RECOMMENDED_PRODUCTS"
 
        return True, ranked, "SUCCESS"
 