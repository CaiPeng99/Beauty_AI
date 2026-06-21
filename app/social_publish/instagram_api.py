import requests
from typing import Dict, Optional
from app.config import INSTAGRAM_CFG
from app.common.logger import logger

BASE_URL = "https://graph.instagram.com/v18.0"

class InstagramPublisher:
    def __init__(self):
        self.access_token = INSTAGRAM_CFG["access_token"]
        self.business_id = INSTAGRAM_CFG["business_id"]

    def publish_post(self, caption: str, tags: list = None, image_url: str = None) -> Dict:
        """发布Ins图文帖子，无图则仅发布文字动态"""
        try:
            tags = tags or []
            tag_str = " ".join([f"#{t}" for t in tags])
            full_caption = f"{caption}\n{tag_str}".strip()

            # 1. 创建媒体容器
            create_url = f"{BASE_URL}/{self.business_id}/media"
            params = {
                "caption": full_caption,
                "access_token": self.access_token
            }
            if image_url:
                params["image_url"] = image_url

            res = requests.post(create_url, params=params)
            res_data = res.json()
            if "id" not in res_data:
                raise Exception(f"创建媒体失败: {res_data}")

            media_id = res_data["id"]
            # 2. 发布媒体
            publish_url = f"{BASE_URL}/{self.business_id}/media_publish"
            publish_params = {
                "creation_id": media_id,
                "access_token": self.access_token
            }
            requests.post(publish_url, params=publish_params)

            logger.info(f"Instagram发布成功，媒体ID: {media_id}")
            return {"status": "success", "platform": "instagram", "content": full_caption}
        except Exception as e:
            logger.error(f"Instagram发布失败: {str(e)}")
            return {"status": "fail", "platform": "instagram", "error": str(e)}

# 单例实例
ig_publisher = InstagramPublisher()