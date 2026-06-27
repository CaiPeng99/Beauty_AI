"""
notion_publisher.py
Beauty RAG Agent — Notion 发布模块

依赖：
    pip install notion-client python-dotenv

环境变量（.env）：
    NOTION_TOKEN          # Integration secret token
    NOTION_DATABASE_ID    # 目标 Database 的 ID

Notion Database 需包含以下 properties（类型见注释）：
    Name            → Title
    Platform        → Select
    Product ID      → Rich Text
    Content         → Rich Text
    Tags            → Multi-select
    Publish Status  → Select   ("draft" | "published" | "fail")
    Created At      → Date

使用示例：
    publisher = NotionPublisher()
    result = publisher.publish_post(content="Foundation review!", tags=["beauty"])
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from notion_client import Client
from notion_client.errors import APIResponseError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class NotionPublisher:
    """
    将 AI 生成的美妆文案写入 Notion Database。

    每次调用 publish_post() 会在 Database 中新建一条 Page，
    包含：标题、平台、产品ID、文案正文、hashtag、发布状态、时间戳。
    """

    def __init__(
        self,
        token: Optional[str] = None,
        database_id: Optional[str] = None,
    ):
        self.token       = token       or os.getenv("NOTION_TOKEN")
        self.database_id = database_id or os.getenv("NOTION_DATABASE_ID")
        self._validate_credentials()
        self.client = Client(auth=self.token)

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def publish_post(
        self,
        content: str,
        tags: list[str],
        product_id: str = "",
        platform: str = "notion",
        status: str = "draft",
    ) -> dict:
        """
        在 Notion Database 新建一条文案记录。

        Args:
            content:    LLM 生成的文案正文
            tags:       hashtag 列表（不含 #）
            product_id: 关联的产品 ID（可选）
            platform:   发布目标平台标注，默认 "notion"
            status:     "draft" | "published" | "fail"

        Returns:
            {"status": "success"|"fail", "page_id": str|None,
             "page_url": str|None, "message": str}
        """
        title = self._make_title(content)
        logger.info(f"[NotionPublisher] Creating page: {title[:60]}...")

        try:
            response = self.client.pages.create(
                parent={"database_id": self.database_id},
                properties=self._build_properties(
                    title=title,
                    platform=platform,
                    product_id=product_id,
                    tags=tags,
                    status=status,
                ),
                children=self._build_body(content, tags),
            )

            page_id  = response["id"]
            page_url = response.get("url", f"https://notion.so/{page_id.replace('-', '')}")
            logger.info(f"[NotionPublisher] ✅ Page created: {page_url}")

            return {
                "status":   "success",
                "page_id":  page_id,
                "page_url": page_url,
                "message":  "Notion page created successfully",
            }

        except APIResponseError as e:
            logger.error(f"[NotionPublisher] API error: {e}")
            return {"status": "fail", "page_id": None, "page_url": None,
                    "message": f"Notion API error — {e.message}"}

        except Exception as e:
            logger.error(f"[NotionPublisher] Unexpected error: {e}", exc_info=True)
            return {"status": "fail", "page_id": None, "page_url": None,
                    "message": f"Unexpected error — {e}"}

    def update_status(self, page_id: str, status: str) -> dict:
        """
        更新已有 Page 的发布状态（比如 Twitter 发布成功后回写 "published"）。

        Args:
            page_id: Notion Page ID
            status:  "draft" | "published" | "fail"
        """
        try:
            self.client.pages.update(
                page_id=page_id,
                properties={"Publish Status": {"select": {"name": status}}},
            )
            return {"status": "success", "message": f"Status updated to '{status}'"}
        except APIResponseError as e:
            return {"status": "fail", "message": f"Notion API error — {e.message}"}

    # ------------------------------------------------------------------
    # 内部方法：构建 Notion API payload
    # ------------------------------------------------------------------

    def _build_properties(
        self,
        title: str,
        platform: str,
        product_id: str,
        tags: list[str],
        status: str,
    ) -> dict:
        """构建 Database properties（对应 Notion Database 的列）"""
        return {
            # Title 列（必须有）
            "Name": {
                "title": [{"text": {"content": title}}]
            },
            # Select 列
            "Platform": {
                "select": {"name": platform}
            },
            # Rich Text 列
            "Product ID": {
                "rich_text": [{"text": {"content": product_id}}]
            },
            # Rich Text 列（正文也存一份在 properties，方便 filter）
            "Content": {
                "rich_text": [{"text": {"content": content_preview(title)}}]
            },
            # Multi-select 列
            "Tags": {
                "multi_select": [{"name": t} for t in tags if t.strip()]
            },
            # Select 列
            "Publish Status": {
                "select": {"name": status}
            },
            # Date 列
            "Created At": {
                "date": {"start": datetime.now(timezone.utc).isoformat()}
            },
        }

    @staticmethod
    def _build_body(content: str, tags: list[str]) -> list:
        """
        构建 Page body（Block children）：
          - 段落正文
          - 分割线
          - hashtag callout
        """
        hashtag_line = "  ".join(f"#{t}" for t in tags if t.strip())

        blocks = [
            # 正文段落（按换行拆分，每段一个 paragraph block）
            *[
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": para}}]
                    },
                }
                for para in content.split("\n") if para.strip()
            ],
            # 分割线
            {"object": "block", "type": "divider", "divider": {}},
        ]

        # hashtag callout（有 tag 时才加）
        if hashtag_line:
            blocks.append({
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": [{"type": "text", "text": {"content": hashtag_line}}],
                    "icon": {"emoji": "🏷️"},
                    "color": "pink_background",
                },
            })

        return blocks

    @staticmethod
    def _make_title(content: str, max_len: int = 60) -> str:
        """用文案首句作为页面标题，超长截断。"""
        first_line = content.strip().split("\n")[0]
        return first_line[:max_len] + ("…" if len(first_line) > max_len else "")

    def _validate_credentials(self):
        missing = [
            name for name, val in {
                "NOTION_TOKEN":       self.token,
                "NOTION_DATABASE_ID": self.database_id,
            }.items() if not val
        ]
        if missing:
            raise EnvironmentError(
                f"[NotionPublisher] Missing credentials: {', '.join(missing)}\n"
                "Set them in your .env file or as environment variables."
            )


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def content_preview(text: str, max_len: int = 200) -> str:
    """截取前 200 字符用于 properties 预览（Notion Rich Text 有上限）。"""
    return text[:max_len] + ("…" if len(text) > max_len else "")


# ---------------------------------------------------------------------------
# 模块级单例
# ---------------------------------------------------------------------------

notion_publisher = NotionPublisher()