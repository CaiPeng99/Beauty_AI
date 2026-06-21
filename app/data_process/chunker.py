"""
chunker.py
美妆场景专属分块：按单产品 / 单评论独立分块

每个函数返回一个 ChunkResult，包含：
  - content   : 用于 embedding 的自然文本
  - meta      : 完整原始字段的 dict（存入 BeautyVectorStore.meta_info）
  - hot_fields: 高频筛选字段（直接写入 BeautyVectorStore 的独立列）

入库调用方式示例（见文件底部 build_vector_record）：
  chunk = split_product_chunk(row)
  db.add(BeautyVectorStore(
      content      = chunk.content,
      embedding    = embed(chunk.content),
      product_id   = chunk.hot_fields["product_id"],
      chunk_type   = "product",
      rating       = chunk.hot_fields["rating"],
      is_recommended = chunk.hot_fields["is_recommended"],
      skin_type    = chunk.hot_fields["skin_type"],
      skin_tone    = chunk.hot_fields["skin_tone"],
      meta_info    = json.dumps(chunk.meta, ensure_ascii=False),
  ))
"""

import json
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class ChunkResult:
    content: str          # 自然文本，用于 embedding
    meta: dict            # 所有原始字段，序列化后存 meta_info
    hot_fields: dict      # 高频筛选字段，写入 BeautyVectorStore 独立列


# ---------------------------------------------------------------------------
# 辅助：安全转换
# ---------------------------------------------------------------------------

def _float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def _int(val, default=0) -> int:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default

def _str(val, default="") -> str:
    if val is None:
        return default
    return str(val).strip()

def _list_str(val) -> str:
    """把 list 或逗号字符串统一转为可读字符串（用于 content 文本）"""
    if isinstance(val, list):
        return ", ".join(str(v) for v in val if v)
    return _str(val)


# ---------------------------------------------------------------------------
# Product chunk
# ---------------------------------------------------------------------------

def split_product_chunk(row: dict) -> ChunkResult:
    """
    Product Info 单块：结构化字段 → 自然文本 + 完整 meta。
    row 来自 product_info.csv 的一行（dict 格式）。
    """
    # ── 自然文本（embedding 用）──────────────────────────────────────────
    content = (
        f"【Product Info】\n"
        f"Product ID：{_str(row.get('product_id'))}\n"
        f"Product Name：{_str(row.get('product_name'))}\n"
        f"Brand Name：{_str(row.get('brand_name'))}\n"
        f"Loves Count：{_int(row.get('loves_count'))}\n"
        f"Rating：{_float(row.get('rating'))}\n"
        f"Reviews：{_int(row.get('reviews'))}\n"
        f"Size：{_str(row.get('size'))}\n"
        f"Variation：{_str(row.get('variation_desc'))}\n"
        f"Ingredients：{_list_str(row.get('ingredients'))}\n"
        f"Price：{_float(row.get('price_usd'))} USD"
        f"，Sale Price：{_float(row.get('sale_price_usd'))} USD\n"
        f"Limited Edition：{_int(row.get('limited_edition'))}"
        f"，New：{_int(row.get('new'))}\n"
        f"Online Only：{_int(row.get('online_only'))}"
        f"，Out Of Stock：{_int(row.get('out_of_stock'))}\n"
        f"Sephora Exclusive：{_int(row.get('sephora_exclusive'))}\n"
        f"Highlights：{_list_str(row.get('highlights'))}\n"
        f"Category：{_str(row.get('primary_category'))}"
        f" > {_str(row.get('secondary_category'))}"
        f" > {_str(row.get('tertiary_category'))}"
    )

    # ── 完整 meta（apply_filters 用）─────────────────────────────────────
    # 所有字段都转为基础 Python 类型，确保 json.dumps 不报错
    meta = {
        "product_id":          _str(row.get("product_id")),
        "product_name":        _str(row.get("product_name")),
        "brand_id":            _str(row.get("brand_id")),
        "brand_name":          _str(row.get("brand_name")),
        "loves_count":         _int(row.get("loves_count")),
        "rating":              _float(row.get("rating")),
        "reviews":             _int(row.get("reviews")),
        "size":                _str(row.get("size")),
        "variation_type":      _str(row.get("variation_type")),
        "variation_value":     _str(row.get("variation_value")),
        "variation_desc":      _str(row.get("variation_desc")),
        "ingredients":         _list_str(row.get("ingredients")),
        "price_usd":           _float(row.get("price_usd")),
        "value_price_usd":     _float(row.get("value_price_usd")),
        "sale_price_usd":      _float(row.get("sale_price_usd")),
        "limited_edition":     _int(row.get("limited_edition")),
        "new":                 _int(row.get("new")),
        "online_only":         _int(row.get("online_only")),
        "out_of_stock":        _int(row.get("out_of_stock")),
        "sephora_exclusive":   _int(row.get("sephora_exclusive")),
        "highlights":          _list_str(row.get("highlights")),
        "primary_category":    _str(row.get("primary_category")),
        "secondary_category":  _str(row.get("secondary_category")),
        "tertiary_category":   _str(row.get("tertiary_category")),
        "child_count":         _int(row.get("child_count")),
        "child_max_price":     _float(row.get("child_max_price")),
        "child_min_price":     _float(row.get("child_min_price")),
    }

    # ── 高频筛选列（写入 BeautyVectorStore 独立列）───────────────────────
    hot_fields = {
        "product_id":    meta["product_id"],
        "rating":        meta["rating"],
        "is_recommended": 0,        # 产品块没有推荐字段，固定 0
        "skin_type":     "",
        "skin_tone":     "",
    }

    return ChunkResult(content=content, meta=meta, hot_fields=hot_fields)


# ---------------------------------------------------------------------------
# Review chunk
# ---------------------------------------------------------------------------

def split_review_chunk(row: dict) -> ChunkResult:
    """
    User Review 单块：评论字段 → 自然文本 + 完整 meta。
    row 来自 review.csv 的一行（dict 格式）。
    """
    # ── 自然文本 ──────────────────────────────────────────────────────────
    content = (
        f"【User Review】\n"
        f"Product ID：{_str(row.get('product_id'))}\n"
        f"Product Name：{_str(row.get('product_name'))}\n"
        f"Brand Name：{_str(row.get('brand_name'))}\n"
        f"Rating：{_float(row.get('rating'))}"
        f"，Recommended：{_int(row.get('is_recommended'))}\n"
        f"Skin Type：{_str(row.get('skin_type'))}"
        f"，Skin Tone：{_str(row.get('skin_tone'))}"
        f"，Eye Color：{_str(row.get('eye_color'))}"
        f"，Hair Color：{_str(row.get('hair_color'))}\n"
        f"Review Title：{_str(row.get('review_title'))}\n"
        f"Review Text：{_str(row.get('review_text'))}"
    )

    # ── 完整 meta ─────────────────────────────────────────────────────────
    meta = {
        "product_id":               _str(row.get("product_id")),
        "product_name":             _str(row.get("product_name")),
        "brand_name":               _str(row.get("brand_name")),
        "price_usd":                _float(row.get("price_usd")),
        "author_id":                _str(row.get("author_id")),
        "rating":                   _float(row.get("rating")),
        "is_recommended":           _int(row.get("is_recommended")),
        "helpfulness":              _float(row.get("helpfulness")),
        "total_feedback_count":     _int(row.get("total_feedback_count")),
        "total_pos_feedback_count": _int(row.get("total_pos_feedback_count")),
        "total_neg_feedback_count": _int(row.get("total_neg_feedback_count")),
        "submission_time":          _str(row.get("submission_time")),
        "skin_type":                _str(row.get("skin_type")),
        "skin_tone":                _str(row.get("skin_tone")),
        "eye_color":                _str(row.get("eye_color")),
        "hair_color":               _str(row.get("hair_color")),
        "review_title":             _str(row.get("review_title")),
        "review_text":              _str(row.get("review_text")),
    }

    # ── 高频筛选列 ────────────────────────────────────────────────────────
    hot_fields = {
        "product_id":     meta["product_id"],
        "rating":         meta["rating"],
        "is_recommended": meta["is_recommended"],
        "skin_type":      meta["skin_type"],
        "skin_tone":      meta["skin_tone"],
    }

    return ChunkResult(content=content, meta=meta, hot_fields=hot_fields)


# ---------------------------------------------------------------------------
# 入库辅助：把 ChunkResult 转为 BeautyVectorStore ORM 对象
# ---------------------------------------------------------------------------

def build_vector_record(chunk: ChunkResult, embedding: list, chunk_type: str):
    """
    将 ChunkResult + embedding 向量组装为 BeautyVectorStore ORM 实例。

    Args:
        chunk:      split_product_chunk / split_review_chunk 的返回值
        embedding:  已计算好的向量（list[float]，长度 384）
        chunk_type: "product" 或 "review"

    Usage:
        from app.database.models import BeautyVectorStore
        record = build_vector_record(chunk, embedding_vector, "product")
        db.add(record)
        db.commit()
    """
    from app.database.models import BeautyVectorStore  # 延迟导入，避免循环依赖

    return BeautyVectorStore(
        content        = chunk.content,
        embedding      = embedding,
        product_id     = chunk.hot_fields["product_id"],
        chunk_type     = chunk_type,
        rating         = chunk.hot_fields["rating"],
        is_recommended = chunk.hot_fields["is_recommended"],
        skin_type      = chunk.hot_fields["skin_type"],
        skin_tone      = chunk.hot_fields["skin_tone"],
        meta_info      = json.dumps(chunk.meta, ensure_ascii=False),
    )
