"""
main.py
Beauty RAG + MCP + AI Agent — 本地命令行入口

修改说明：
  1. import json 移到文件顶部
  2. 数据库 session 用 try/finally 确保关闭，避免连接池泄漏
  3. 新增 safe_json_dumps：处理 numpy float / ORM 对象 / datetime 等不可序列化类型
  4. 每轮对话加异常兜底，单次报错不中断整个循环
  5. 删除重复的 get_db import
"""

import json
import numpy as np
from datetime import datetime
from fastapi import Depends
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.agent.agent_core import BeautyAgent
from app.common.logger import logger


# ---------------------------------------------------------------------------
# 自定义 JSON 序列化器
# ---------------------------------------------------------------------------

class _SafeEncoder(json.JSONEncoder):
    """
    处理 result dict 里可能出现的非标准类型：
      - numpy 数值（int64 / float32 / float64 等）
      - datetime → ISO 字符串
      - SQLAlchemy ORM 对象 → __dict__ 过滤掉 _sa_instance_state
      - 其他无法序列化的对象 → str() 兜底
    """
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        # SQLAlchemy ORM 实例
        if hasattr(obj, "__dict__") and hasattr(obj, "_sa_instance_state"):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        return str(obj)


def safe_json_dumps(data) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, cls=_SafeEncoder)


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

def main():
    db: Session = next(get_db())
    try:
        session_id = "local_test_001"
        # agent      = BeautyAgent(db=db, session_id=session_id)
        agent = BeautyAgent(db=db, session_id=session_id, user_id=user_id)

        print("✅ Beauty Assistant activated！Enter your question（q to quit）：")

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye Bye！")
                break

            if user_input.lower() in ("q", "quit", "exit"):
                print("Bye Bye！")
                break
            if not user_input:
                continue

            def stream_print(msg: str):
                print(f"  → {msg}")

            try:
                result = agent.run_workflow(
                    query=user_input,
                    stream_cb=stream_print,
                    retry_count=0,
                )
                print("\n===== Result =====")
                print(safe_json_dumps(result))

            except KeyboardInterrupt:          # ← 新增：捕获 Ctrl+C
                print("\nBye Bye！")
                break                          # 跳出 while，进入 finally 关闭 db

            except Exception as e:
                logger.error(f"run_workflow 异常: {e}", exc_info=True)
                print(f"\n⚠️  Something went wrong: {e}\nPlease try again.")

    finally:
        db.close()   # 确保 session 归还连接池


if __name__ == "__main__":
    main()


# 在终端临时跑，或加到 main.py 里临时测试
# from app.database.session import get_db
# from app.database.models import Product, BeautyVectorStore

# db = next(get_db())

# # 1. 看有多少 fragrance 产品
# rows = db.query(Product.product_id, Product.product_name, Product.primary_category)\
#     .filter(Product.primary_category.ilike('%fragrance%'))\
#     .limit(10).all()
# print("Fragrance products:", rows)

# # 2. 随便取一个 product_id，看它有多少条评论
# if rows:
#     pid = rows[0].product_id
#     reviews = db.query(BeautyVectorStore)\
#         .filter(BeautyVectorStore.product_id == pid,
#                 BeautyVectorStore.chunk_type == 'review').all()
#     print(f"Reviews for {pid}: {len(reviews)}")
#     rec = sum(1 for r in reviews if getattr(r, 'is_recommended', 0) == 1)
#     print(f"Recommended count: {rec}")

# # 检查所有 fragrance 产品的评论数
# frag_ids = [r.product_id for r in 
#     db.query(Product.product_id)
#     .filter(Product.primary_category.ilike('%fragrance%')).all()]

# print(f"Total fragrance products: {len(frag_ids)}")

# for pid in frag_ids[:10]:
#     count = db.query(BeautyVectorStore)\
#         .filter(BeautyVectorStore.product_id == pid,
#                 BeautyVectorStore.chunk_type == 'review').count()
#     if count > 0:
#         print(f"{pid}: {count} reviews")

# print("Done")









''' OR'''
# from fastapi import FastAPI
# from app.database.session import get_db
# from app.agent.agent_core import BeautyAgent
# from pydantic import BaseModel

# app = FastAPI(title="Beauty RAG Agent API")

# class AgentReq(BaseModel):
#     session_id: str
#     user_query: str
#     retry_count: int = 0

# @app.post("/agent/run")
# def run_agent(req: AgentReq):
#     db = next(get_db())
#     agent = BeautyAgent(db=db, session_id=req.session_id)
#     def stream_cb(msg):
#         print(msg)
#     res = agent.run_workflow(req.user_query, stream_cb, req.retry_count)
#     return res

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


'''下面可能是错的'''

# app = FastAPI(title="Beauty AI Social Publish System")
# client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# @app.post("/chat")
# def chat(query: str, session_id: str, db: Session = Depends(get_db)):
#     try:
#         # 1. 问题向量化
#         embed_resp = client.embeddings.create(input=query, model="text-embedding-3-small")
#         query_emb = embed_resp.data[0].embedding

#         # 2. 混合检索 + 合法性判断
#         retriever = HybridRetriever(db)
#         has_data, results, tip = retriever.search_with_judge(query, query_emb, session_id)

#         # ========== Branch 1：No matched data / Insufficient Similarities ==========
#         if not has_data:
#             if tip == "NO_CONTENT":
#                 save_unknown_query(query, "No matched product data", session_id)
#                 final_prompt = NO_DATA_PROMPT.format(query=query)
#             else:
#                 save_unknown_query(query, "相似度低于阈值", session_id)
#                 final_prompt = NO_DATA_PROMPT.format(query=query)

#             llm_resp = client.chat.completions.create(
#                 model=LLM_MODEL,
#                 messages=[{"role":"user", "content": final_prompt}]
#             )
#             return {"code": 0, "reply": llm_resp.choices[0].message.content, "status": "no_data"}

#         # ========== Branch 2：retrieve efficient data，normal answer ==========
#         context = "\n".join([i["content"] for i in results])
#         final_prompt = RAG_NORMAL_PROMPT.format(context=context, query=query)
#         llm_resp = client.chat.completions.create(
#             model=LLM_MODEL,
#             messages=[{"role":"user", "content": final_prompt}]
#         )
#         return {"code": 0, "reply": llm_resp.choices[0].message.content, "status": "success"}

#     except Exception as e:
#         logger.error(f"请求异常: {str(e)}")
#         # branch 3：Anomaly system/can't understand goal
#         final_prompt = UNKNOWN_INTENT_PROMPT
#         llm_resp = client.chat.completions.create(
#             model=LLM_MODEL,
#             messages=[{"role":"user", "content": final_prompt}]
#         )
#         return {"code": -1, "reply": llm_resp.choices[0].message.content, "status": "unknown_intent"}
    


'''下面的先不看'''
# Agent 流式接口（选品/发布全流程 + SSE）
# @app.get("/agent/stream")
# async def agent_stream(
#     query: str = Query(...),
#     session_id: str = Query(...),
#     db: Session = Depends(get_db)
# ):
#     step_msgs = []
#     agent = BeautyAgent(db)

#     # 流式回调，收集步骤信息
#     def stream_callback(msg: str):
#         step_msgs.append(msg)

#     # 执行Agent流程
#     result = agent.run_workflow(query, stream_cb=stream_callback)
#     step_msgs.append(f"【执行完成】结果：{str(result)}")

#     # 返回SSE流式响应
#     return StreamingResponse(
#         sse_stream_generator(step_msgs),
#         media_type="text/event-stream"
#     )