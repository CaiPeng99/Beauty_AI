"""
agent_core.py
Agent 核心：意图识别 → 工作流编排 → 工具调用

修改说明：
  - 删除 unknown 意图的写死关键词修正逻辑
    （由 HybridRetriever.parse_intent 的 LLM 动态解析替代）
  - unknown 统一走 select_product 兜底，让 smart_search 决定有没有匹配结果
  - publish_twitter / publish_ins / save_local 复用同一个内部函数 _run_publish_flow
    避免大量重复代码
"""

from typing import Optional, Callable, Dict, Any
from sqlalchemy.orm import Session
from openai import OpenAI
from app.config import OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL, ZhiPu_API_KEY
from app.agent.tools import ToolRegistry
from app.common.logger import logger

from app.agent.memory import get_short_memory, append_short_memory, save_long_memory, get_long_memory

from app.rag.retriever import HybridRetriever

from app.agent.llm_client import llm_adapter

# OPENAI
# client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


client = OpenAI(
    api_key=ZhiPu_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4"
)

# 选品类意图统一归类，收敛到同一个工具
SELECT_PRODUCT_INTENTS = {"select_hot", "select_new", "select_exclusive", "select_product"}

# intention recognition prompt 【previous】
# INTENT_PROMPT = """
# You need to identify the user's intent and return only the corresponding keyword:
# 1. Select trending popular products → intent:select_hot
# 2. Select newly launched products → intent:select_new
# 3. Select Sephora exclusive limited products → intent:select_exclusive
# 4. General product search by skin type/effect/price/brand → intent:select_product
# 5. Generate social media copy only → intent:generate_content
# 6. Publish finished copy to X(Twitter) → intent:publish_twitter
# 7. Publish finished copy to Instagram → intent:publish_ins
# 8. Save copy to local markdown file → intent:save_local
# 9. Q&A / product ingredient / usage inquiry → intent:qa
# 10. Unrecognized irrelevant request → intent:unknown

# Rule:
# Any demand of searching, filtering, or recommending beauty products → return intent:select_product.
# User input: {query}
# Return only intent:xxx, no extra words.
# """

INTENT_PROMPT = """
Identify the user's intent and return only the keyword:

1. User asks for products with specific attributes (new arrivals, hot/trending,
   exclusive, limited edition, on sale, specific brand, category, price range,
   skin type, etc.) → intent:select_by_attribute

2. User wants open-ended recommendations ("what's good for dry skin",
   "best anti-aging serum", "recommend me something") → intent:select_product

3. Generate social media copy → intent:generate_content
4. Publish to X(Twitter) → intent:publish_twitter
5. Publish to Instagram → intent:publish_ins
6. Save copy locally → intent:save_local
7. Q&A / ingredient / usage inquiry → intent:qa
8. Unrecognized → intent:unknown

User input: {query}
Return only intent:xxx, no extra words.
"""

class BeautyAgent:
    def __init__(self, db: Session, session_id: str):
        self.db = db
        self.tools = ToolRegistry.tools
        self.session_id = session_id
    
    # ------------------------------------------------------------------ #
    # 意图识别                                                             #
    # ------------------------------------------------------------------ #
    def recognize_intent(self, query:str) -> str:
        '''识别用户意图，增加异常兜底'''
        try:
            # resp = client.chat.completions.create(
            #     model=LLM_MODEL,
            #     messages=[{"role": "user", "content": INTENT_PROMPT.format(query=query)}]
            # )
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": INTENT_PROMPT.format(query=query)}],
                temperature=0.1,
            )
            raw    = resp.choices[0].message.content.strip()
            intent = raw.replace("intent:", "").strip()
            logger.info(f"[{self.session_id}] 意图识别: {intent} | query: {query}")
            return intent
        except Exception as e:
            logger.error(f"意图识别失败: {e}")
            return "unknown"
    
    # ------------------------------------------------------------------ #
    # 内部：选品 → 文案生成 → 发布/存档 的通用流程                        #
    # ------------------------------------------------------------------ #

    def _run_publish_flow(
        self,
        query: str,
        platform: str,
        stream_cb: Optional[Callable[[str], None]],
    ) -> Dict[str, Any]:
        """
        publish_twitter / publish_ins / save_local 的公共逻辑：
          选品 → 生成文案 → 发布或存档
        platform: "twitter" | "instagram" | "local"
        """
        if stream_cb:
            stream_cb("【Step 2】Use RAG to find suitable products...")

        res = self.tools["select_product"](
            db=self.db,
            session_id=self.session_id,
            user_query=query,
            retry_count=0,
        )
        product_list = res["data"].get("products", [])
        if not product_list:
            return {
                "step": "clarify",
                "message": res.get("message", "未找到匹配产品，请重新描述需求"),
            }

        prod      = product_list[0]
        prod_name = prod.get("product_name", "")

        if stream_cb:
            stream_cb(f"【Step 3】Generating copy for {prod_name}...")

        gen_res = self.tools["generate_content"](
            db=self.db,
            session_id=self.session_id,
            user_query=query,
            product_info=prod,
            platform=platform,
            style="种草",
        )
        if gen_res["status"] != "success":
            return {"step": "error", "message": f"文案生成失败: {gen_res['message']}"}

        content = gen_res["data"]["content"]

        # 本地存档
        if platform == "local":
            if stream_cb:
                stream_cb("【Step 4】Saving to local file...")
            save_res = self.tools["write_local_file"](
                product_name=prod_name,
                platform="local",
                content=content,
            )
            return {
                "step": "done",
                "product": prod,
                "save_result": save_res,
                "message": save_res["message"],
            }

        # 社媒发布
        if stream_cb:
            stream_cb(f"【Step 4】Publishing on {platform}...")
        publish_res = self.tools["publish_social"](
            db=self.db,
            user_query=query,
            product=prod,
            content=content,
            platform=platform,
        )
        return {
            "step": "done",
            "product": prod,
            "publish": publish_res,
        }

    # ------------------------------------------------------------------ #
    # 内部：记忆保存（每轮结束统一调用，避免重复）                         #
    # ------------------------------------------------------------------ #

    def _save_memory(self, query: str, result: Dict[str, Any]):
        """
        每轮工作流结束后调用：
          - 追加到 Redis 短期记忆（所有轮次）
          - 超过 2000 字符时，摘要后存入 PostgreSQL 长期记忆
        result 只取关键字段存储，避免把大 dict 整个写进 Redis。
        """
        # 只保留对下一轮有用的关键信息，不存原始大 dict
        step    = result.get("step", "")
        message = result.get("message", "")
        products = []
        if "result" in result:
            products = [
                p.get("product_name", "")
                for p in result["result"].get("data", {}).get("products", [])[:3]
            ]
        elif "product" in result:
            products = [result["product"].get("product_name", "")]

        summary_line = f"User: {query} | Step: {step}"
        if products:
            summary_line += f" | Products: {', '.join(products)}"
        if message:
            summary_line += f" | Msg: {message}"

        try:
            append_short_memory(self.session_id, summary_line)
            save_long_memory(self.db, self.session_id, query, summary_line)
        except Exception as e:
            logger.warning(f"[{self.session_id}] 记忆保存失败（不中断流程）: {e}")

    # ------------------------------------------------------------------ #
    # 主工作流                                                             #
    # ------------------------------------------------------------------ #
    def run_workflow(
        self,
        query: str,
        stream_cb: Optional[Callable[[str], None]] = None,
        retry_count: int = 0,
    ) -> Dict[str, Any]:

        """
        执行完整工作流
        stream_cb: SSE 流式回调函数
        """

        intent = self.recognize_intent(query)

        if stream_cb:
            stream_cb(f"【Step 1】Intent identified: {intent}")

        # ── 读取记忆（在所有分支之前）─────────────────────────────────────
        # 短期记忆已经在 mcp_tools.generate_content 里通过 get_short_memory() 注入 prompt
        # 这里额外读长期记忆，传给需要它的分支（publish 流程的文案生成）
        long_mem = get_long_memory(self.db, self.session_id, query)
        if long_mem and stream_cb:
            stream_cb("【Memory】Loaded relevant past context")

        if intent == "select_by_attribute":
            retriever = HybridRetriever(self.db, llm_client=llm_adapter)
            parsed_intent = retriever.parse_intent(query)
            
            res = self.tools["select_by_attribute"](
                db=self.db,
                session_id=self.session_id,
                parsed_intent=parsed_intent,
            )
            
            if res["status"] == "success":
                product_list = res["data"].get("products", [])
                if stream_cb:
                    stream_cb(f"【Success】Found {len(product_list)} products")
                result = {"step": "done", "intent": intent, "result": res}
            else:
                if stream_cb:
                    stream_cb(f"【Clarify Needed】{res['message']}")
                result = {"step": "clarify", "intent": intent, "message": res["message"]}
            
            self._save_memory(query, result)
            return result

        # ── 选品类（含 unknown → 让 smart_search 决定有无结果）────────────
        elif intent in SELECT_PRODUCT_INTENTS or intent == "unknown":
            if stream_cb:
                stream_cb("【Step 2】Running RAG smart search...")

            res = self.tools["select_product"](
                db=self.db,
                session_id=self.session_id,
                user_query=query,
                retry_count=retry_count,
            )
            product_list = res["data"].get("products", [])

            if res["status"] == "success":
                if stream_cb:
                    stream_cb(f"【Success】Found {len(product_list)} recommended products")
                result = {"step": "done", "intent": intent,
                          "retry_count": retry_count, "result": res}
                self._save_memory(query, result)
                return result

            elif res["status"] == "clarify":
                if stream_cb:
                    stream_cb(f"【Clarify Needed】{res['message']}")
                result = {"step": "clarify", "intent": intent,
                          "message": res["message"], "retry_count": res["retry_count"]}
                self._save_memory(query, result)
                return result

            elif res["status"] == "end":
                if stream_cb:
                    stream_cb(f"【End】{res['message']}")
                result = {"step": "end", "intent": intent, "message": res["message"]}
                self._save_memory(query, result)
                return result

            else:
                msg = res.get("message", "产品检索异常")
                if stream_cb:
                    stream_cb(f"【Error】{msg}")
                result = {"step": "error", "intent": intent, "message": msg}
                self._save_memory(query, result)
                return result

        # ── 发布到 X ────────────────────────────────────────────────────
        elif intent == "publish_twitter":
            result = self._run_publish_flow(query, "twitter", stream_cb, long_mem)
            result["intent"] = intent
            self._save_memory(query, result)
            return result

        # ── 发布到 Instagram ─────────────────────────────────────────────
        elif intent == "publish_ins":
            result = self._run_publish_flow(query, "instagram", stream_cb, long_mem)
            result["intent"] = intent
            self._save_memory(query, result)
            return result

        # ── 本地存档 ─────────────────────────────────────────────────────
        elif intent == "save_local":
            result = self._run_publish_flow(query, "local", stream_cb, long_mem)
            result["intent"] = intent
            self._save_memory(query, result)
            return result

        # ── 问答 ─────────────────────────────────────────────────────────
        elif intent == "qa":
            result = {"step": "qa", "intent": intent,
                      "message": "Please use the /chat endpoint for Q&A."}
            self._save_memory(query, result)
            return result
        
        elif intent == "generate_content":
            result = self._run_publish_flow(query, "instagram", stream_cb)
            result["intent"] = intent
            self._save_memory(query, result)
            return result

        # ── 兜底 ─────────────────────────────────────────────────────────
        else:
            result = {"step": "unknown", "intent": intent,
                      "message": "I don't quite understand. Could you ask me in a different way?"}
            self._save_memory(query, result)
            return result
