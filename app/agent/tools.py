"""
tools.py
Tool Registration Center — 选品 / 文案生成 / 社媒发布 / 本地存档

修改说明：
  - select_product 改用 smart_search（意图解析 + 动态过滤 + 聚合推荐）
  - HybridRetriever 传入 llm_client，由 retriever 内部完成 parse_intent
  - 选品结果从 aggregate_and_rank 的 ranked_products 结构中提取
  - 补全 encoder 单例（与 etl.py 保持一致，避免重复加载）
"""

import os
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database.models import Product, PublishRecord
from app.social_publish.twitter_api import twitter_publisher
from app.social_publish.instagram_api import ig_publisher
from app.common.logger import logger
from app.rag.retriever import HybridRetriever
from app.agent.memory import get_short_memory
from openai import OpenAI
from app.config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, EMBEDDING_MODEL, LLM_MODEL, ZhiPu_API_KEY,
    MAX_RETRY, OUTPUT_DIR
)

from app.agent.llm_client import llm_adapter
import re

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 强制指定HF镜像 & 延长超时
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["SENTENCE_TRANSFORMERS_TIMEOUT"] = "30"

# 全局单例
from sentence_transformers import SentenceTransformer

# 懒加载单例
encoder: SentenceTransformer | None = None
 
def _get_encoder() -> SentenceTransformer:
    global encoder
    if encoder is None:
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return encoder

# encoder = SentenceTransformer('all-MiniLM-L6-v2')  # 384维

# ZhiPu LLM client（文案生成 / 标签生成 / 字段识别）
llm_client_raw = OpenAI(
    api_key=ZhiPu_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4",
)


os.makedirs(OUTPUT_DIR, exist_ok=True)

# 最大反问次数

DEFAULT_FIELDS = ["product_id", "product_name", "brand_name", "rating", "price_usd"]
FILTER_OPS = {
    "eq":       lambda a, b: a == b,
    "neq":      lambda a, b: a != b,
    "gte":      lambda a, b: a >= b,
    "lte":      lambda a, b: a <= b,
    "gt":       lambda a, b: a > b,
    "lt":       lambda a, b: a < b,
    "contains": lambda a, b: b.lower() in str(a).lower(),
}

# ---------------------------------------------------------------------------
# LLM client 适配器：让 HybridRetriever 的 llm_client.complete() 接口可用
# ---------------------------------------------------------------------------
# class ZhiPuLLMAdapter:
#     """
#     HybridRetriever 期望 llm_client.complete(prompt) -> str。
#     ZhiPu 用 OpenAI 兼容接口，封装一层即可。
#     """
#     def __init__(self, client: OpenAI, model: str):
#         self.client = client
#         self.model  = model

#     def complete(self, prompt: str) -> str:
#         resp = self.client.chat.completions.create(
#             model=self.model,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.1,   # 意图解析要稳定
#             timeout=20.0,    # ← 加这行，20秒超时
#         )
#         return resp.choices[0].message.content.strip()


# _llm_adapter = ZhiPuLLMAdapter(llm_client_raw, LLM_MODEL)


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """MCP Tools Registration Center"""
    tools: dict = {}

    @classmethod
    def register(cls, name: str):
        def decorator(func):
            cls.tools[name] = func
            return func
        return decorator

# ======================================================================
# 工具1：终极智能选品 —— 支持反问 + 次数超限结束 + 动态字段（完全不写死）
# ======================================================================
@ToolRegistry.register("select_product")
def select_product(
    db: Session,
    session_id: str, 
    user_query: str,   
    retry_count: int = 0
) -> dict:
    """
    RAG 智能选品：
      1. parse_intent()   — LLM 解析结构化过滤条件
      2. _prefilter_by_sql() — SQL 预过滤缩小候选集（在 retriever 内部）
      3. smart_search()   — 在候选集内做向量召回 → review 过滤 → rerank → 聚合
      4. fallback         — 无结果时按 loves_count 兜底
      5. LLM 动态识别返回字段
    """
    # ======================
    # 步骤1：RAG混合检索找产品
    # ======================
    try:
        logger.info(f"[{session_id}] 执行产品检索，query: {user_query}")
        # 1. 调用 RAG 检索（逻辑放在rag，不写在tool）
        # embed_resp = client.embeddings.create(input=user_query, model=EMBEDDING_MODEL)
        # query_emb = embed_resp.data[0].embedding

        # ── 1. Embed 查询 ──────────────────────────────────────────────────
        query_emb = _get_encoder().encode(user_query).tolist()

        # ── 2. 构建 Retriever（传入 llm_adapter 以支持 parse_intent）──────
        retriever = HybridRetriever(db, llm_client=llm_adapter)

        # ── 3. 解析意图（预先解析，smart_search 直接用，避免重复调用 LLM）──
        intent_parsed = retriever.parse_intent(user_query)

        print(f"DEBUG parsed_intent: {intent_parsed}")

        # 保险：如果 filters 为空，检查 query 是否包含已知 category
        if not intent_parsed.get("filters"):
            query_lower = user_query.lower()
            primary_matched = False  # 🔥 加一个旗帜变量
            
            for cat in retriever.known_categories:
                if re.search(r'\b' + re.escape(cat.lower()) + r'\b', query_lower):
                    intent_parsed["filters"].append({
                        "field": "primary_category",
                        "op": "eq",
                        "value": cat
                    })
                    primary_matched = True  # 🔥 标记已匹配
                    break

            # 🔥 修改这里的条件：只有没匹配到 primary 时，才尝试 secondary
            if not primary_matched:  # 这才是你注释里写的“只在没匹配到 primary 时才加”
                for sec_cat in retriever.known_secondary_categories:
                    if re.search(r'\b' + re.escape(sec_cat.lower()) + r'\b', query_lower):
                        intent_parsed["filters"].append({
                            "field": "secondary_category",
                            "op": "eq",
                            "value": sec_cat
                        })
                        break
            
        print(f"DEBUG parsed_intent after fix: {intent_parsed}")

        # ── 4. smart_search：SQL预过滤 + 向量召回 + review过滤 + rerank + 聚合
        has_data, ranked_products, tip = retriever.smart_search(
            query=user_query,
            query_embedding=query_emb,
            session_id=session_id,
            parsed_intent=intent_parsed,   # 传入已解析的 intent，避免重复调用

        )

        # for debug
        print(f"DEBUG tip: {tip}")
        print(f"DEBUG ranked count: {len(ranked_products)}")
        if ranked_products:
            print(f"DEBUG top product: {ranked_products[0]['product_name']}, "
                  f"score={ranked_products[0]['composite_score']:.3f}")

        # ── 5. 无结果 fallback ────────────────────────────────────────────
        if not has_data:
            # 提取品类信息用于 fallback 查询
            category_filter = next(
                (f for f in intent_parsed.get("filters", [])
                 if f["field"] in ("primary_category", "secondary_category")),
                None
            )
            category_val   = category_filter["value"].lower() if category_filter else None
            category_field = category_filter["field"] if category_filter else None
 
            if category_val:
                # 该品类按 loves_count 兜底
                if category_field == "secondary_category":
                    filter_cond = Product.secondary_category.ilike(f"%{category_val}%")
                else:
                    filter_cond = Product.primary_category.ilike(f"%{category_val}%")
                fallback_products = (
                    db.query(Product)
                    .filter(filter_cond)
                    .filter(Product.out_of_stock != 1)
                    .order_by(Product.loves_count.desc())
                    .limit(5).all()
                )
                fallback_msg = (
                    f"暂无该品类的用户评论数据，"
                    f"以下是 {category_val} 品类中收藏数最高的产品，供参考"
                )
            else:
                # 全库按 loves_count 兜底
                fallback_products = (
                    db.query(Product)
                    .filter(Product.out_of_stock != 1)
                    .order_by(Product.loves_count.desc())
                    .limit(5).all()
                )
                fallback_msg = "未能精确匹配您的需求，以下是全站收藏数最高的产品，供参考"
 
            if fallback_products:
                product_list = []
                for p in fallback_products:
                    item = {f: getattr(p, f) for f in DEFAULT_FIELDS if hasattr(p, f)}
                    item["_rec_ratio"]  = None
                    item["_avg_rating"] = None
                    item["_rec_count"]  = 0
                    product_list.append(item)
                return {
                    "status": "success",
                    "data":    {"products": product_list},
                    "message": fallback_msg,
                }
 
            # fallback 也没找到 → 反问或结束
            if retry_count >= MAX_RETRY:
                return {
                    "status":      "end",
                    "data":        {},
                    "message":     "多次检索未匹配到合适产品，可更换需求重新提问",
                    "retry_count": retry_count,
                }
            return {
                "status":      "clarify",
                "data":        {},
                "message":     "未找到匹配产品，请补充肤质、功效、价格、品牌等偏好",
                "retry_count": retry_count + 1,
            }
 
        # ── 6. 查 Product 主表获取完整字段 ───────────────────────────────
        product_ids = [p["product_id"] for p in ranked_products if p.get("product_id")]
        products_db = (
            db.query(Product)
            .filter(Product.product_id.in_(product_ids))
            .all()
        )
        pid_order = {pid: idx for idx, pid in enumerate(product_ids)}
        products_db.sort(key=lambda p: pid_order.get(p.product_id, 999))
 
        # ── 7. LLM 动态识别返回字段 ──────────────────────────────────────
        field_prompt = """
You are a beauty product information parser.
Based on the user's question, tell me which product fields to return.
Choose ONLY from:
  product_id, product_name, brand_name, loves_count, rating,
  price_usd, sale_price_usd, ingredients, size, highlights,
  limited_edition, new, sephora_exclusive, online_only,
  out_of_stock, primary_category, secondary_category
 
User question: {query}
Output field names separated by commas. No extra text.
        """.strip()
 
        raw_fields    = llm_adapter.complete(field_prompt.format(query=user_query))
        needed_fields = [f.strip() for f in raw_fields.split(",") if f.strip()]
        if not needed_fields:
            needed_fields = DEFAULT_FIELDS
 
        # ── 8. 构建返回列表 ───────────────────────────────────────────────
        ranked_map   = {p["product_id"]: p for p in ranked_products}
        product_list = []
        for p in products_db:
            item = {}
            for field in needed_fields:
                if hasattr(p, field):
                    item[field] = getattr(p, field)
            stats = ranked_map.get(p.product_id, {})
            item["_rec_ratio"]  = round(stats.get("rec_ratio", 0), 2)
            item["_avg_rating"] = round(stats.get("avg_rating", 0), 2)
            item["_rec_count"]  = stats.get("rec_count", 0)
            product_list.append(item)
 
        return {
            "status": "success",
            "data":    {"products": product_list},
            "message": f"匹配到 {len(product_list)} 个推荐产品",
        }
 
    except Exception as e:
        logger.error(f"select_product 异常: {e}", exc_info=True)
        return {"status": "error", "data": {}, "message": f"检索失败：{e}"}
 


# ---------------------------------------------------------------------------
# 工具 2：生成社媒文案
# ---------------------------------------------------------------------------

@ToolRegistry.register("generate_content")
def generate_content(
    db: Session,
    session_id: str,
    user_query: str,
    product_info: dict,
    platform: str,
    style: str = "种草"
) -> dict:
    """结合对话历史、产品、平台生成合规美妆文案"""
    try:
        # 读取短期对话记忆
        chat_history = get_short_memory(session_id)

        # 把推荐统计一并放进 prompt，让文案有数据支撑
        rec_info = ""
        if product_info.get("_rec_ratio"):
            rec_info = (
                f"\n【真实用户评价摘要】"
                f"{int(product_info['_rec_ratio'] * 100)}% 用户推荐，"
                f"平均评分 {product_info['_avg_rating']:.1f}/5，"
                f"共 {product_info['_rec_count']} 条好评"
            )

        # prompt = f"""
        #     【历史对话上下文】
        #     {chat_history}

        #     【用户需求】
        #     {user_query}

        #     【产品完整信息】
        #     {product_str}

        #     【发布平台】{platform}
        #     【文案风格】{style}

        #     写作规则：
        #     1. 仅使用提供产品信息，禁止编造成分/功效
        #     2. Instagram：文案丰富、适度表情，末尾带标签
        #     3. X(Twitter)：短句简洁，标签精简
        #     4. 自然种草，禁止夸大、医疗类宣传词
        #     5. 语言贴合海外美妆社群，可读性强
        #     仅输出最终文案，不要多余解释。
        # """.strip()

        prompt = f"""
[Historical Conversation Context]
{chat_history}

[User Request]
{user_query}

[Complete Product Information]
{product_info}{rec_info}

[Publishing Platform]{platform}
[Copywriting Style]{style}

Writing Rules:
1. Use only the product information above; do not invent ingredients or effects.
2. Instagram: Rich copy with moderate emojis, include hashtags at the end.
3. X (Twitter): Short, concise sentences ≤280 characters, minimal hashtags.
4. Quote real user review data to increase credibility.
5. Natural种草 (soft selling), avoid exaggerated or medical/promotional claims.
6. Use language that fits the overseas beauty community, ensure readability.
Output only the final copy, no extra explanation.
""".strip()

        resp = llm_client_raw.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        content = resp.choices[0].message.content.strip()
        return {"status": "success", "data": {"content": content}, "message": "文案生成完成"}

    except Exception as e:
        logger.error(f"generate_content 异常: {e}", exc_info=True)
        return {"status": "error", "data": {}, "message": f"文案生成失败：{e}"}


# ---------------------------------------------------------------------------
# 工具 3：社交平台发布
# ---------------------------------------------------------------------------

@ToolRegistry.register("publish_social")
def publish_social(
    db: Session,
    user_query: str,
    product: dict,
    content: str,
    platform: str
) -> dict:
    try:
        # ------------------------------------------------------------------
        # 【核心】LLM 动态生成标签（完全不写死）
        # ------------------------------------------------------------------
        tag_prompt = """
Based on user needs, product info, and platform, generate 3-5 English hashtags.
Do NOT include #. Separate with commas.
User needs: {q}
Product: {p}
Platform: {plat}
Output tags only.
        """.strip()

        raw_tags = llm_adapter.complete(
            tag_prompt.format(q=user_query, p=product, plat=platform)
        )
        tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
        if not tags:
            tags = ["Beauty", "Skincare", "Makeup", "Sephora"]

        # 平台发布
        if platform == "twitter":
            publish_res = twitter_publisher.publish(content, tags)
        elif platform == "instagram":
            publish_res = ig_publisher.publish_post(content, tags)
        else:
            return {"status": "fail", "data": {}, "message": f"不支持的发布平台: {platform}"}


        # 写入发布记录
        record = PublishRecord(
            platform       = platform,
            product_id     = product.get("product_id", ""),
            content        = content,
            tags           = tags,
            publish_status = publish_res.get("status", "unknown"),
        )
        db.add(record)
        db.commit()

        return {
            "status": publish_res.get("status"),
            "data": {"platform": platform, "content": content, "tags": tags},
            "message": "发布完成，已存档记录",
        }

    except Exception as e:
        db.rollback()
        logger.error(f"publish_social 异常: {e}", exc_info=True)
        return {"status": "fail", "data": {}, "message": f"发布失败：{e}"}

# ---------------------------------------------------------------------------
# 工具 4：本地文件写入
# ---------------------------------------------------------------------------

@ToolRegistry.register("write_local_file")
def write_local_file(
    product_name: str,
    platform: str,
    content: str
) -> dict:
    try:
        safe_name = product_name.replace(" ", "_").replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"{platform}_{safe_name}_{timestamp}.md"
        full_path = os.path.join(OUTPUT_DIR, filename)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(f"===== 存档时间：{datetime.now()} =====\n")
            f.write(f"平台：{platform}\n产品：{product_name}\n\n")
            f.write(content)

        return {"status": "success", "data": {"file_path": full_path}, "message": "文案本地存档完成"}

    except Exception as e:
        logger.error(f"write_local_file 异常: {e}", exc_info=True)
        return {"status": "fail", "data": {}, "message": f"文件写入失败：{e}"}

# ------------------------------------------------------------------ #
# MCP 版本：供外部项目调用，与原 write_local_file 并存
# ------------------------------------------------------------------ #
 
MCP_SERVER_PATH = os.path.join(os.path.dirname(__file__), "../../mcp_servers/file_server.py")
 
async def _call_mcp_file_server(tool_name: str, arguments: dict) -> dict:
    """底层：启动 MCP Server 进程并调用指定工具。"""
    server_params = StdioServerParameters(
        command="python",
        args=[MCP_SERVER_PATH],
        env={"OUTPUT_DIR": OUTPUT_DIR},  # 复用项目已有的 OUTPUT_DIR
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            # result.content 是 list[TextContent]
            text = result.content[0].text if result.content else ""
            return {
                "status": "success" if "✅" in text else "fail",
                "message": text,
            }

@ToolRegistry.register("write_local_file_mcp")
def write_local_file_mcp(
    product_name: str,
    platform: str,
    content: str,
) -> dict:
    """
    通过 MCP Server 保存文案到本地文件。
    功能与 write_local_file 完全一致，但走 MCP 协议，
    方便外部项目复用同一个 file_server.py。
    """
    try:
        return asyncio.run(_call_mcp_file_server(
            tool_name="write_file",
            arguments={
                "product_name": product_name,
                "platform": platform,
                "content": content,
            },
        ))
    except Exception as e:
        logger.error(f"write_local_file_mcp 异常: {e}", exc_info=True)
        return {"status": "fail", "data": {}, "message": f"MCP 调用失败：{e}"}
    
@ToolRegistry.register("list_local_files_mcp")
def list_local_files_mcp(platform: str = "") -> dict:
    """通过 MCP Server 列出已保存的文件。"""
    try:
        return asyncio.run(_call_mcp_file_server(
            tool_name="list_files",
            arguments={"platform": platform},
        ))
    except Exception as e:
        logger.error(f"list_local_files_mcp 异常: {e}", exc_info=True)
        return {"status": "fail", "message": f"MCP 调用失败：{e}"}
 
 
@ToolRegistry.register("read_local_file_mcp")
def read_local_file_mcp(filename: str) -> dict:
    """通过 MCP Server 读取已保存的文件内容。"""
    try:
        return asyncio.run(_call_mcp_file_server(
            tool_name="read_file",
            arguments={"filename": filename},
        ))
    except Exception as e:
        logger.error(f"read_local_file_mcp 异常: {e}", exc_info=True)
        return {"status": "fail", "message": f"MCP 调用失败：{e}"}

@ToolRegistry.register("select_by_attribute")
def select_by_attribute(db: Session, session_id: str, parsed_intent: dict) -> dict:

    # print(f"DEBUG parsed_intent: {parsed_intent}")  # ← 加这行

    # 复用已有的 parse_intent，动态解析过滤条件
    retriever = HybridRetriever(db)   # 不需要传 llm_client
    candidate_ids = retriever._prefilter_by_sql(parsed_intent.get("filters", []))
    
    if candidate_ids is not None and len(candidate_ids) == 0:
        return {"status": "clarify", "data": {}, "message": "未找到符合条件的产品"}
    
    # 根据 sort_by 决定排序
    sort_field = parsed_intent.get("sort_by") or "loves_count"
    sort_col = getattr(Product, sort_field, Product.loves_count)
    
    q = db.query(Product).filter(Product.out_of_stock != 1)
    if candidate_ids is not None:
        q = q.filter(Product.product_id.in_(candidate_ids))
    
    # print(f"DEBUG candidate_ids count in tool: {len(candidate_ids) if candidate_ids is not None else 'None'}")
    products = q.order_by(sort_col.desc()).limit(10).all()
    # print(f"DEBUG first product new={products[0].new if products else 'no products'}")
        
    if not products:
        # fallback：放宽条件，去掉 out_of_stock 限制
        products = q.order_by(sort_col.desc()).limit(10).all()
    
    product_list = [
        {f: getattr(p, f) for f in DEFAULT_FIELDS if hasattr(p, f)}
        for p in products
    ]
    return {
        "status": "success",
        "data": {"products": product_list},
        "message": f"找到 {len(product_list)} 款产品",
    }