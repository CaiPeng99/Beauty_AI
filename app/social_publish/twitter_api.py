import tweepy
from typing import Dict, Optional
from app.config import TWITTER_CFG
from app.common.logger import logger

class TwitterPublisher:
    def __init__(self):
        auth = tweepy.OAuth1UserHandler(
            consumer_key=TWITTER_CFG["api_key"],
            consumer_secret=TWITTER_CFG["api_secret"],
            access_token=TWITTER_CFG["access_token"],
            access_token_secret=TWITTER_CFG["access_secret"]
        )
        self.api = tweepy.API(auth)
        self.client = tweepy.Client(
            consumer_key=TWITTER_CFG["api_key"],
            consumer_secret=TWITTER_CFG["api_secret"],
            access_token=TWITTER_CFG["access_token"],
            access_token_secret=TWITTER_CFG["access_secret"]
        )

    def publish(self, content: str, tags: list = None) -> Dict:
        """发布推文"""
        try:
            tags = tags or []
            tag_str = " ".join([f"#{t}" for t in tags])
            full_content = f"{content}\n{tag_str}".strip()
            resp = self.client.create_tweet(text=full_content)
            logger.info(f"X发布成功，推文ID: {resp.data['id']}")
            return {"status": "success", "platform": "twitter", "content": full_content}
        except Exception as e:
            logger.error(f"X发布失败: {str(e)}")
            return {"status": "fail", "platform": "twitter", "error": str(e)}

# 单例实例
twitter_publisher = TwitterPublisher()