"""
memory.py
短期记忆（Redis）+ 长期记忆（PostgreSQL + pgvector）

架构说明：
  - Short-term memory：以 session_id 为 key 存 Redis，TTL 24h
    作用：同一次对话内的上下文（指代消解、上下文延续）

  - Long-term memory：以 user_id 为 key 存 PostgreSQL
    作用：跨 session 的用户偏好（肤质、预算、品牌偏好等）
    触发：每轮对话结束后，由 LLM 判断是否有值得记住的信息
    召回：每次 session 开始时，注入 system prompt

  user_id vs session_id：
    user_id    —— 标识"这个人"，跨 session 不变（前端 localStorage 生成；
                  测试阶段用固定 mock 值）
    session_id —— 标识"这次对话"，新对话重新生成

修改记录（相对原版）：
  1. save_long_memory：触发条件从"长度 > 2000"改为"LLM 语义判断"
  2. get_long_memory：过滤字段从 session_id 改为 user_id（跨 session 召回）
  3. 新增 maybe_save_memory：对外暴露的统一入口，agent 每轮调用
  4. 新增 build_system_prompt：把长期记忆注入 system prompt 的工具函数
  5. _summary_text 保留但仅内部使用；原 message= 拼写错误已修复
"""

import json
import redis
from sqlalchemy.orm import Session
from datetime import datetime
from openai import OpenAI
 
from app.config import REDIS_URL, LLM_MODEL, ZhiPu_API_KEY
from app.database.models import ChatMemory
from app.common.logger import logger
 
from sentence_transformers import SentenceTransformer
 
# ---------------------------------------------------------------------------
# Singleton encoder
# ---------------------------------------------------------------------------
 
_encoder: SentenceTransformer | None = None
 
def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim
    return _encoder
 
 
# ---------------------------------------------------------------------------
# LLM client (ZhiPu, used for memory extraction)
# ---------------------------------------------------------------------------
 
client = OpenAI(
    api_key=ZhiPu_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4",
)
 
redis_client = redis.from_url(REDIS_URL, decode_responses=True)
 
 
# ---------------------------------------------------------------------------
# Prompt: ask the LLM whether a conversation turn is worth remembering
# ---------------------------------------------------------------------------
 
MEMORY_EXTRACTION_PROMPT = """
You are the memory manager for a beauty assistant.
Read the conversation turn below and decide whether it contains user preference
information worth storing for the long term.
 
Examples WORTH saving:
- Skin type (dry, oily, combination, sensitive)
- Budget range
- Brands or ingredients the user explicitly likes or dislikes
- Usage habits or special requirements
 
Examples NOT worth saving:
- One-off questions ("How much does this product cost?", "Look up XX for me")
- Small talk or thank-you messages
 
Conversation:
{conversation}
 
If there is something worth saving, return:
{{"should_save": true, "memory": "one concise sentence describing the user preference"}}
 
If there is nothing worth saving:
{{"should_save": false, "memory": null}}
 
Return JSON only. No other text.
""".strip()
 
 
# ===========================================================================
# Short-term Memory (Redis, session-scoped, TTL 24 h)
# ===========================================================================
 
def get_short_memory(session_id: str) -> str:
    """Return the conversation history for the current session."""
    val = redis_client.get(f"chat:mem:{session_id}")
    return val if val else ""
 
 
def append_short_memory(session_id: str, new_msg: str) -> None:
    """Append one message to the current session's short-term memory."""
    key = f"chat:mem:{session_id}"
    old  = get_short_memory(session_id)
    full = f"{old}\n{new_msg}".strip()
    redis_client.set(key, full, ex=86400)  # TTL 24 h
 
 
def clear_short_memory(session_id: str) -> None:
    """Delete the short-term memory for a session (end of session or testing)."""
    redis_client.delete(f"chat:mem:{session_id}")
 
 
# ===========================================================================
# Long-term Memory (PostgreSQL, user-scoped, persistent)
# ===========================================================================
 
def _extract_memory_from_llm(user_q: str, ai_a: str) -> str | None:
    """
    Ask the LLM whether this conversation turn contains information
    worth storing in long-term memory.
    Returns the extracted preference string, or None if not worth saving.
    """
    conversation = f"User: {user_q}\nAI: {ai_a}"
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{
                "role": "user",
                "content": MEMORY_EXTRACTION_PROMPT.format(conversation=conversation),
            }],
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        result = json.loads(raw)
        if result.get("should_save") and result.get("memory"):
            return result["memory"]
    except Exception as e:
        logger.warning(f"[memory extraction] LLM call failed, skipping this turn: {e}")
    return None
 
 
def maybe_save_memory(
    db: Session,
    user_id: str,
    session_id: str,
    user_q: str,
    ai_a: str,
) -> bool:
    """
    Public entry point called by the agent after each conversation turn.
    The LLM decides whether the turn is worth writing to long-term memory.
 
    Args:
        db:         SQLAlchemy session
        user_id:    Stable user identifier (unchanged across sessions)
        session_id: Current conversation ID (used for log tracing only)
        user_q:     User's input this turn
        ai_a:       Agent's reply this turn
 
    Returns:
        True  = written to long-term memory
        False = not worth saving, skipped
    """
    memory_text = _extract_memory_from_llm(user_q, ai_a)
    if not memory_text:
        return False
 
    try:
        emb = _get_encoder().encode(memory_text).tolist()
        mem = ChatMemory(
            user_id     = user_id,
            session_id  = session_id,
            user_query  = user_q,
            ai_response = ai_a,
            summary     = memory_text,  # extracted preference, not a raw summary
            embedding   = emb,
        )
        db.add(mem)
        db.commit()
        logger.info(f"[{user_id}|{session_id}] Long-term memory saved: {memory_text}")
        return True
    except Exception as e:
        db.rollback()
        logger.error(f"[{user_id}|{session_id}] Failed to save long-term memory: {e}", exc_info=True)
        return False
 
 
def get_long_memory(
    db: Session,
    user_id: str,
    query: str,
    top_k: int = 3,
) -> str:
    """
    Retrieve the most relevant past preferences for a user using vector similarity.
    Filters by user_id, so results are valid across sessions.
 
    Returns a newline-joined string for use in build_system_prompt.
    """
    try:
        query_emb = _get_encoder().encode(query).tolist()
        rows = (
            db.query(ChatMemory)
            .filter(ChatMemory.user_id == user_id)   # user-scoped, not session-scoped
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
        logger.warning(f"get_long_memory query failed (user={user_id}): {e}")
        return ""
 
 
# ===========================================================================
# System prompt injection (called by the agent at the start of each session)
# ===========================================================================
 
def build_system_prompt(db: Session, user_id: str, current_query: str) -> str:
    """
    Build a system prompt that includes the user's long-term memory.
    Call this on the first turn of every session.
 
    Example output:
        You are a professional beauty assistant helping users choose the right
        beauty products. Tailor your recommendations to the user's skin type,
        budget, and preferences.
 
        What you already know about this user:
        [2026-06-01] Dry skin, budget under $50, avoids alcohol-based products
        [2026-06-15] Prefers Japanese brands, especially SK-II
    """
    base = (
        "You are a professional beauty assistant helping users choose the right beauty products.\n"
        "Always tailor your recommendations to the user's skin type, budget, and stated preferences."
    )
 
    long_mem = get_long_memory(db, user_id, current_query, top_k=3)
    if long_mem:
        base += f"\n\nWhat you already know about this user:\n{long_mem}"
 
    return base
