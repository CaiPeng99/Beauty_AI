"""
memory.py
短期记忆（Redis）+ 长期记忆（PostgreSQL）

修改说明：
  - summary_text：修复 message= 拼写错误（→ messages=）
  - save_long_memory：embedding 改用 SentenceTransformer（与项目其他地方一致）
    不再依赖 OpenAI Embeddings API
  - get_short_memory：返回值统一为 str（Redis 可能返回 bytes，解码处理）
"""

import redis
from sqlalchemy.orm import Session
from datetime import datetime
from openai import OpenAI

from app.config import REDIS_URL, LLM_MODEL, ZhiPu_API_KEY
from app.database.models import ChatMemory
from app.common.logger import logger

import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer
# _encoder = SentenceTransformer("all-MiniLM-L6-v2")  # 384 维，全局单例

_encoder: SentenceTransformer | None = None
 
def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder

# client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# ZhiPu LLM client（用于摘要生成）
client = OpenAI(
    api_key=ZhiPu_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4",
)

redis_client = redis.from_url(REDIS_URL, decode_responses=True)  # decode_responses=True 自动解码

# 超过此长度触发摘要入长期记忆库
MAX_SHORT_MEMORY_LEN = 2000

'''
short-term memory in Redis, long-term memory in DB
'''
# ---------------------------------------------------------------------------
# 短期记忆（Redis，TTL 24h）
# ---------------------------------------------------------------------------

def get_short_memory(session_id: str) -> str:
    "Get Redis Short-term memory"
    key = f"chat:mem:{session_id}"
    val = redis_client.get(key)
    return val if val else ""   # decode_responses=True 已自动 decode

def append_short_memory(session_id: str, new_msg: str):
    """追加一条消息到短期记忆。"""
    key = f"chat:mem:{session_id}"
    old  = get_short_memory(session_id)
    full = f"{old}\n{new_msg}".strip()
    redis_client.set(key, full, ex=86400)   # 1 天过期

def clear_short_memory(session_id: str):
    """清空指定会话的短期记忆（用于会话结束或测试）。"""
    redis_client.delete(f"chat:mem:{session_id}")

# ---------------------------------------------------------------------------
# 长期记忆（PostgreSQL ChatMemory 表）
# ---------------------------------------------------------------------------

def _summary_text(text:str) -> str:
    """用 LLM 对长对话做摘要。"""
    resp = client.chat.completions.create(   # ★ 修复：message= → messages=
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": f"Briefly summarize the following conversation in 2-3 sentences:\n\n{text}",
        }],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

def save_long_memory(db: Session, session_id: str, user_q: str, ai_a: str):
    """
    当对话内容超过 MAX_SHORT_MEMORY_LEN 时，
    生成摘要 + 向量并写入长期记忆表。

    ★ 修复：embedding 改用 SentenceTransformer（384 维），
      与 Product / Review / BeautyVectorStore 保持一致，
      不再调用 OpenAI Embeddings API。
    """
    full_content = f"User: {user_q}\nAI: {ai_a}"
    if len(full_content) <= MAX_SHORT_MEMORY_LEN:
        return   # 不够长，不入库

    try:
        summary = _summary_text(full_content)
        # ★ 使用 SentenceTransformer 生成 384 维向量
        emb = _get_encoder().encode(summary).tolist()

        mem = ChatMemory(
            session_id  = session_id,
            user_query  = user_q,
            ai_response = ai_a,
            summary     = summary,
            embedding   = emb,
        )
        db.add(mem)
        db.commit()
        logger.info(f"[{session_id}] 长期记忆已写入")
    except Exception as e:
        db.rollback()
        logger.error(f"[{session_id}] 长期记忆写入失败: {e}", exc_info=True)

def get_long_memory(db: Session, session_id: str, query: str, top_k: int = 3) -> str:
    """
    用向量相似度从长期记忆中召回与当前 query 最相关的历史摘要。
    返回拼接后的字符串，供 agent prompt 使用。
    """
    try:
        query_emb = _get_encoder().encode(query).tolist()
        # pgvector 余弦距离检索（越小越相似）
        rows = (
            db.query(ChatMemory)
            .filter(ChatMemory.session_id == session_id)
            .order_by(ChatMemory.embedding.cosine_distance(query_emb))
            .limit(top_k)
            .all()
        )
        if not rows:
            return ""
        return "\n".join(
            f"[{r.created_at.strftime('%Y-%m-%d')}] {r.summary}" for r in rows
        )
    except Exception as e:
        logger.warning(f"get_long_memory 查询失败: {e}")
        return ""

