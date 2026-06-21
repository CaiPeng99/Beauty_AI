"""
etl.py
数据入库全流程：CSV 清洗 → 主表写入 → 向量化 → BeautyVectorStore 入库

修改说明：
  - split_product_chunk / split_review_chunk 现在返回 ChunkResult 对象
    (包含 .content / .meta / .hot_fields)，不再是纯字符串
  - 使用 build_vector_record() 辅助函数组装 ORM 对象，统一写法
  - ETL 里不再手动拼 meta_info，由 chunker 负责
"""

import os
import json
import ast
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Optional
from sqlalchemy.orm import Session

from app.config import (
    PRODUCT_CSV_PATH, REVIEW_FOLDER, REVIEW_FILE_SUFFIX,
    BATCH_SIZE, FULL_REBUILD,
)
from app.database.session import get_db
from app.database.models import Product, Review, BeautyVectorStore
# ↓ 新版 chunker：返回 ChunkResult，并提供 build_vector_record 辅助函数
from app.data_process.chunker import split_product_chunk, split_review_chunk, build_vector_record
from app.common.logger import logger

# 强制 HF 镜像
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

from sentence_transformers import SentenceTransformer
# _encoder = SentenceTransformer("all-MiniLM-L6-v2")  # 384 维，全局单例

_encoder: SentenceTransformer | None = None
 
def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder

# 路径
PRODUCT_CSV_PATH = "data/product_info.csv"
REVIEW_FOLDER = "data/"  # 文件夹，不是单个文件
REVIEW_FILE_SUFFIX = "review"  # 所有包含 review 的csv

DB_BATCH_SIZE = 2000   # 业务主表批次
VEC_BATCH_SIZE = BATCH_SIZE  # 向量表批次（config 里配置）

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def embed_text(text: str) -> list:
    return _encoder.encode(text).tolist()


def safe_int(val, default=0) -> int:
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default
    
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """通用数据清洗：空值填充、类型统一"""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("")
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(0)
    bool_map = {"True": 1, "False": 0, True: 1, False: 0, 1: 1, 0: 0}
    for col in df.columns:
        if any(k in col.lower() for k in ("limited", "new", "exclusive", "online_only", "out_of_stock")):
            df[col] = df[col].map(bool_map).fillna(0).astype(int)
    return df


# OpenAI
# def embed_text(text:str) -> list:
#     """单条文本向量化"""
#     resp = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
#     return resp.data[0].embedding

    # """带限流+重试的向量化"""
    # delay = 0.5
    # for _ in range(retry_times):
    #     try:
    #         resp = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    #         time.sleep(EMBEDDING_RATE_LIMIT_SLEEP)
    #         return resp.data[0].embedding
    #     except (APIError, APIConnectionError) as e:
    #         logger.warning(f"Embedding接口异常，延迟{delay}s重试: {str(e)}")
    #         time.sleep(delay)
    #         delay *= 2
    # raise Exception(f"文本向量化连续{retry_times}次调用失败")


def parse_array_field(value) -> Optional[List[str]]:
    """安全地将数组字符串或列表转换为字符串列表"""
    if value is None:
        return None
    if isinstance(value, list):
        # 已经是列表，但需要确保每个元素是字符串（处理可能的字符列表情况）
        if all(isinstance(item, str) and len(item) == 1 for item in value):
            # 这种情况是错误地将字符串拆成了字符，尝试合并回字符串？
            # 更好的做法是去源头修复，此处可尝试将字符列表拼成字符串再解析
            joined = ''.join(value)   # 例如 "['Alcohol', 'Water']"
            return parse_array_field(joined)
        # 正常列表，过滤掉空字符串或清洗
        return [str(item) for item in value if item]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        if value.startswith("["):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(value)
                    if isinstance(parsed, list):
                        return [str(i) for i in parsed]
                except Exception:
                    pass
        return [value]
    return [str(value)]

def _bulk_commit(db: Session, records: list, label: str):
    """批量提交，失败时回滚并 raise。"""
    try:
        db.bulk_save_objects(records)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"{label} 批量入库失败，已回滚: {e}")
        raise

# ---------------------------------------------------------------------------
# Product ETL
# ---------------------------------------------------------------------------

def import_product_data():
    db: Session = next(get_db())
    logger.info("开始导入产品数据...")

    if FULL_REBUILD:
        logger.info("FULL_REBUILD：清空产品表 & 产品向量")
        db.query(BeautyVectorStore).filter(BeautyVectorStore.chunk_type == "product").delete()
        db.query(Product).delete()
        db.commit()

    df = pd.read_csv(PRODUCT_CSV_PATH)
    df = clean_df(df)

    # ── 1. 写入业务主表 ────────────────────────────────────────────────────
    product_records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="产品主表"):
        product_records.append(Product(
            product_id         = str(row["product_id"]),
            product_name       = row["product_name"],
            brand_id           = str(row["brand_id"]),
            brand_name         = row["brand_name"],
            loves_count        = safe_int(row["loves_count"]),
            rating             = safe_float(row["rating"]),
            reviews            = safe_int(row["reviews"]),
            size               = row["size"],
            variation_type     = row["variation_type"],
            variation_value    = row["variation_value"],
            variation_desc     = row["variation_desc"],
            ingredients        = parse_array_field(row["ingredients"]),
            price_usd          = safe_float(row["price_usd"]),
            value_price_usd    = safe_float(row["value_price_usd"]),
            sale_price_usd     = safe_float(row["sale_price_usd"]),
            limited_edition    = safe_int(row["limited_edition"]),
            new                = safe_int(row["new"]),
            online_only        = safe_int(row["online_only"]),
            out_of_stock       = safe_int(row["out_of_stock"]),
            sephora_exclusive  = safe_int(row["sephora_exclusive"]),
            highlights         = parse_array_field(row["highlights"]),
            primary_category   = row["primary_category"],
            secondary_category = row["secondary_category"],
            tertiary_category  = row["tertiary_category"],
            child_count        = safe_int(row["child_count"]),
            child_max_price    = safe_float(row["child_max_price"]),
            child_min_price    = safe_float(row["child_min_price"]),
        ))

    _bulk_commit(db, product_records, "产品主表")
    logger.info(f"产品主表导入完成，共 {len(product_records)} 条")

    # ── 2. 向量化 + 写入向量库 ─────────────────────────────────────────────
    # ★ 关键改动：split_product_chunk 现在返回 ChunkResult，不是字符串
    vector_records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="产品向量化"):
        chunk = split_product_chunk(row.to_dict())        # ChunkResult
        emb   = embed_text(chunk.content)                  # 384-dim list
        vector_records.append(
            build_vector_record(chunk, emb, "product")    # → BeautyVectorStore ORM
        )
        if len(vector_records) >= VEC_BATCH_SIZE:
            _bulk_commit(db, vector_records, "产品向量")
            vector_records.clear()

    if vector_records:
        _bulk_commit(db, vector_records, "产品向量（尾批）")

    logger.info("✅ 产品向量库导入完成")

# ---------------------------------------------------------------------------
# Review ETL
# ---------------------------------------------------------------------------

def import_review_data():
    db: Session = next(get_db())
    logger.info("开始导入评论数据...")

    if FULL_REBUILD:
        logger.info("FULL_REBUILD：清空评论表 & 评论向量")
        db.query(BeautyVectorStore).filter(BeautyVectorStore.chunk_type == "review").delete()
        db.query(Review).delete()
        db.commit()

    review_files = [
        f for f in os.listdir(REVIEW_FOLDER)
        if f.lower().endswith(".csv") and REVIEW_FILE_SUFFIX in f.lower()
    ]
    logger.info(f"待处理评论文件：{review_files}")

    for file_idx, f in enumerate(review_files):
        file_path = os.path.join(REVIEW_FOLDER, f)
        logger.info(f"处理文件: {f} ({file_idx + 1}/{len(review_files)})")
        df = pd.read_csv(file_path)
        df = clean_df(df)

        review_records = []
        vector_records = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"解析 {f}"):
            row_dict = row.to_dict()

            # ── 业务主表 ──────────────────────────────────────────────────
            try:
                review_records.append(Review(
                    author_id                = str(row["author_id"]),
                    rating                   = safe_float(row["rating"]),
                    is_recommended           = safe_int(row["is_recommended"]),
                    helpfulness              = safe_float(row["helpfulness"]),
                    total_feedback_count     = safe_int(row["total_feedback_count"]),
                    total_neg_feedback_count = safe_int(row["total_neg_feedback_count"]),
                    total_pos_feedback_count = safe_int(row["total_pos_feedback_count"]),
                    submission_time          = pd.to_datetime(row["submission_time"], errors="coerce"),
                    review_text              = row["review_text"],
                    review_title             = row["review_title"],
                    skin_tone                = row["skin_tone"],
                    eye_color                = row["eye_color"],
                    skin_type                = row["skin_type"],
                    hair_color               = row["hair_color"],
                    product_id               = str(row["product_id"]),
                    product_name             = row["product_name"],
                    brand_name               = row["brand_name"],
                    price_usd                = safe_float(row["price_usd"]),
                ))
            except Exception as e:
                logger.warning(f"跳过脏数据行: {e}")
                continue

            # ── 向量库 ────────────────────────────────────────────────────
            # ★ 关键改动：split_review_chunk 返回 ChunkResult
            chunk = split_review_chunk(row_dict)
            emb   = embed_text(chunk.content)
            vector_records.append(
                build_vector_record(chunk, emb, "review")
            )

            # 分批提交
            if len(review_records) >= DB_BATCH_SIZE:
                _bulk_commit(db, review_records, f"评论主表({f})")
                review_records.clear()
            if len(vector_records) >= VEC_BATCH_SIZE:
                _bulk_commit(db, vector_records, f"评论向量({f})")
                vector_records.clear()

        # 文件收尾
        if review_records:
            _bulk_commit(db, review_records, f"评论主表({f})-尾批")
        if vector_records:
            _bulk_commit(db, vector_records, f"评论向量({f})-尾批")

    logger.info("✅ 评论主表 + 评论向量库 导入完成")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_full_etl():
    logger.info("===== ETL 全量任务开始 =====")
    import_product_data()
    import_review_data()
    logger.info("===== ETL 全量任务完成 =====")


if __name__ == "__main__":
    run_full_etl()
