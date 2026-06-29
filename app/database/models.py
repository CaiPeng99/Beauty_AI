"""
models.py
Beauty RAG — SQLAlchemy ORM 模型定义

修改说明（相对原版）：
  1. Product.product_id 从 primary_key=True 改为 unique=True（避免双主键冲突）
  2. 删除 Product.prices_usd（重复字段）
  3. trigger DDL 中 ingredients/highlights 从 ARRAY 改为 TEXT 拼接，兼容 TSVECTOR
"""

import json
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY, Boolean, Column, DateTime, DDL, Float, ForeignKey,
    Index, Integer, String, Text, event, text,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSON

from app.database.session import Base


# ===========================================================================
# Product
# ===========================================================================
class Product(Base):
    __tablename__ = "products"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    # product_id = Column(String, primary_key=True, index=True) 之前的
    # ↓ 改为 unique=True，不再做主键（避免 SQLAlchemy 双主键冲突）
    product_id = Column(String, unique=True, index=True, nullable=False)

    product_name = Column(String)
    brand_name   = Column(String)
    brand_id     = Column(String)
    loves_count  = Column(Integer)
    rating       = Column(Float)
    reviews      = Column(Integer)

    size            = Column(String,  nullable=True)
    variation_value = Column(String,  nullable=True)
    variation_type  = Column(String,  nullable=True)
    variation_desc  = Column(Text,    nullable=True)

    price_usd       = Column(Float, nullable=True)
    value_price_usd = Column(Float, nullable=True)
    sale_price_usd  = Column(Float, nullable=True)

    limited_edition  = Column(Integer, default=0)
    new              = Column(Integer, default=0)
    online_only      = Column(Integer, default=0)
    sephora_exclusive= Column(Integer, default=0)
    out_of_stock     = Column(Integer, default=0)

    highlights         = Column(ARRAY(String), nullable=True)
    primary_category   = Column(String)
    secondary_category = Column(String)
    tertiary_category  = Column(String, nullable=True)
    ingredients        = Column(ARRAY(String), nullable=True)

    child_count     = Column(Integer, default=0, nullable=True)
    child_max_price = Column(Float, nullable=True)
    child_min_price = Column(Float, nullable=True)

    embedding   = Column(Vector(384))
    search_tsv  = Column(TSVECTOR)

    __table_args__ = (
        Index("idx_product_search_tsv",       "search_tsv",       postgresql_using="gin"),
        Index("idx_product_embedding",
              text("embedding vector_cosine_ops"), postgresql_using="hnsw"),
        Index("idx_product_loves",  "loves_count"),
        Index("idx_product_rating", "rating"),
        Index("idx_product_new",              "new",
              postgresql_where=(text("new = 1"))),
        Index("idx_product_sephora_exclusive","sephora_exclusive",
              postgresql_where=(text("sephora_exclusive = 1"))),
        Index("idx_product_out_of_stock",     "out_of_stock",
              postgresql_where=(text("out_of_stock = 0"))),
    )

    def __repr__(self):
        return f"<Product(product_id={self.product_id}, name={self.product_name})>"


# ===========================================================================
# Review
# ===========================================================================

class Review(Base):
    __tablename__ = "reviews"

    review_id  = Column(Integer, primary_key=True, autoincrement=True)
    author_id  = Column(String)
    rating     = Column(Integer)
    is_recommended = Column(Integer, default=0)
    helpfulness    = Column(Float,   nullable=True)

    total_feedback_count     = Column(Integer)
    total_neg_feedback_count = Column(Integer)
    total_pos_feedback_count = Column(Integer)

    submission_time = Column(DateTime)
    review_text     = Column(Text)
    review_title    = Column(Text)

    skin_tone  = Column(String, nullable=True)
    eye_color  = Column(String)
    skin_type  = Column(String)
    hair_color = Column(String, nullable=True)

    product_id   = Column(String)
    product_name = Column(String)
    brand_name   = Column(String)
    price_usd    = Column(Float)

    embedding  = Column(Vector(384))
    search_tsv = Column(TSVECTOR)

    __table_args__ = (
        Index("idx_reviews_search_tsv", "search_tsv", postgresql_using="gin"),
        Index("idx_reviews_embedding",
              text("embedding vector_cosine_ops"), postgresql_using="hnsw"),
        Index("idx_reviews_product_id",     "product_id"),
        Index("idx_reviews_rating",         "rating"),
        Index("idx_reviews_submission_time","submission_time"),
        Index("idx_reviews_skin_type",      "skin_type"),
        Index("idx_reviews_skin_tone",      "skin_tone"),
    )

    def __repr__(self):
        return (
            
            f"<Review(review_id={self.review_id}, "
            f"product_id={self.product_id}, rating={self.rating})>"
        )


# ===========================================================================
# BeautyVectorStore — RAG 知识库
# ===========================================================================

class BeautyVectorStore(Base):
    __tablename__ = "beauty_vector_store"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    content    = Column(Text,        nullable=False)
    embedding  = Column(Vector(384), nullable=False)
    product_id = Column(String(100), nullable=False)
    chunk_type = Column(String(50),  nullable=False)   # "product" | "review"

    # 高频筛选列（独立列，走索引最快）
    rating         = Column(Float,   default=0.0)
    is_recommended = Column(Integer, default=0)        # 1 推荐 / 0 不推荐
    skin_type      = Column(String(50), default="")
    skin_tone      = Column(String(50), default="")

    search_tsv = Column(TSVECTOR)  # ← 加这行

    # 全量元数据（所有原始字段 JSON 序列化，供 apply_filters 动态过滤）
    meta_info  = Column(Text, default="{}")

    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("idx_bvs_prod_type", "product_id", "chunk_type"),
        Index("idx_bvs_rec",    "is_recommended"),
        Index("idx_bvs_rating", "rating"),
    )


# ===========================================================================
# ChatMemory — 会话长期记忆
# ===========================================================================

class ChatMemory(Base):
    __tablename__ = "chat_memory"

    id         = Column(Integer,     primary_key=True)
    session_id = Column(String(100), index=True)
    user_query = Column(Text)
    ai_response= Column(Text)
    summary    = Column(Text)
    embedding  = Column(Vector(384))
    created_at = Column(DateTime, default=datetime.now)


# ===========================================================================
# PublishRecord — 社媒发布记录
# ===========================================================================

class PublishRecord(Base):
    __tablename__ = "publish_record"

    id             = Column(Integer,    primary_key=True)
    platform       = Column(String(20))                   # "twitter" | "instagram"
    product_id     = Column(String(100))
    content        = Column(Text)
    tags           = Column(JSON)
    publish_status = Column(String(20))                   # "success" | "fail"
    publish_time   = Column(DateTime, default=datetime.now)


# ===========================================================================
# UnknownQueryLog — 未命中查询日志
# ===========================================================================

class UnknownQueryLog(Base):
    __tablename__ = "unknown_query_log"

    id          = Column(Integer,     primary_key=True, autoincrement=True)
    user_query  = Column(Text,        nullable=False)
    fail_reason = Column(String(100))  # "NO_CONTENT" | "LOW_SIMILARITY" | "FILTERED_EMPTY" | "NO_RECOMMENDED_PRODUCTS"
    session_id  = Column(String(100))
    create_time = Column(DateTime, default=datetime.now)


# ===========================================================================
# PostgreSQL 触发器：自动维护 search_tsv
# ===========================================================================

# 注意：ingredients / highlights 是 ARRAY(String)，TSVECTOR 需要先 array_to_string 转换
_trigger_product_sql = DDL("""
    CREATE OR REPLACE FUNCTION product_info_tsvector_update() RETURNS TRIGGER AS $$
    BEGIN
        NEW.search_tsv :=
            setweight(to_tsvector('english', COALESCE(NEW.product_name, '')), 'A') ||
            setweight(to_tsvector('english', COALESCE(NEW.brand_name, '')), 'B') ||
            setweight(to_tsvector('english',
                COALESCE(array_to_string(NEW.ingredients, ' '), '')), 'C') ||
            setweight(to_tsvector('english',
                COALESCE(array_to_string(NEW.highlights, ' '), '')), 'C');
        RETURN NEW;
    END
    $$ LANGUAGE plpgsql;

    CREATE TRIGGER trigger_product_info_tsvector
        BEFORE INSERT OR UPDATE OF product_name, brand_name, ingredients, highlights
        ON products
        FOR EACH ROW EXECUTE FUNCTION product_info_tsvector_update();
""")

_trigger_review_sql = DDL("""
    CREATE OR REPLACE FUNCTION reviews_tsvector_update() RETURNS TRIGGER AS $$
    BEGIN
        NEW.search_tsv :=
            setweight(to_tsvector('english', COALESCE(NEW.review_title, '')), 'A') ||
            setweight(to_tsvector('english', COALESCE(NEW.review_text, '')), 'B');
        RETURN NEW;
    END
    $$ LANGUAGE plpgsql;

    CREATE TRIGGER trigger_reviews_tsvector
        BEFORE INSERT OR UPDATE OF review_title, review_text
        ON reviews
        FOR EACH ROW EXECUTE FUNCTION reviews_tsvector_update();
""")

event.listen(Product.__table__, "after_create", _trigger_product_sql)
event.listen(Review.__table__,  "after_create", _trigger_review_sql)
