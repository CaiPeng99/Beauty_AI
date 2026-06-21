import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from sqlalchemy.orm import Session
from app.database.models import BeautyVectorStore
from app.config import SIMILARITY_THRESHOLD, TOP_K_RECALL
from app.common.logger import unknown_logger

rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_results(query: str, doc_list):
    """CrossEncoder rerank — returns results sorted by cross-attention score."""
    if not doc_list:
        return []
    pairs = [[query, doc.content] for doc in doc_list]
    scores = rerank_model.predict(pairs)
    # 按分数降序排序
    sorted_pairs = sorted(zip(doc_list, scores), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_pairs]
