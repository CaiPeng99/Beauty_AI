import os
from dotenv import load_dotenv

load_dotenv()

# 数据库
PG_DATABASE_URL = (
    f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD', "")}"
    f"@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}"
)

# ========== 文件路径 ==========
PRODUCT_CSV_PATH = "data/product_info.csv"
REVIEW_FOLDER = "data/"
REVIEW_FILE_SUFFIX = "review"
OUTPUT_DIR = "outputs"

# ========== ETL 配置 ==========
# BATCH_SIZE = 20 # before
BATCH_SIZE = 100
EMBEDDING_RATE_LIMIT_SLEEP = 0.1
FULL_REBUILD = False  # True=全量重建清空旧数据

MAX_RETRY = 2

# Redis
REDIS_URL = f"redis://{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}/0"

# LLM & Embedding
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
# LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# 智谱
ZhiPu_API_KEY = "50156fd8035d4392a10c0a35e5f61d7b.zWfgO6ZnWFG9DkXG"
# LLM_MODEL = "glm-4.7-flash"
LLM_MODEL = "glm-4-flash"

# 检索阈值（判断「无数据」核心参数）
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD"))
TOP_K_RECALL = int(os.getenv("TOP_K_RECALL"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK"))

# 社交平台
TWITTER_CFG = {
    "api_key": os.getenv("TWITTER_API_KEY"),
    "api_secret": os.getenv("TWITTER_API_SECRET"),
    "access_token": os.getenv("TWITTER_ACCESS_TOKEN"),
    "access_secret": os.getenv("TWITTER_ACCESS_SECRET"),
}
INSTAGRAM_CFG = {
    "access_token": os.getenv("INSTAGRAM_ACCESS_TOKEN"),
    "business_id": os.getenv("INSTAGRAM_BUSINESS_ID"),
}

# 时区
TIMEZONE = os.getenv("TIMEZONE")




# from pydantic_settings import BaseSettings
# import os

# class Setting(BaseSettings):
#     # 数据库
#     DATABASE_URL: str = os.getenv("DATABASE_URL")
    
#     # OpenAI
#     OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    
#     # 社交平台
#     X_API_KEY: str = os.getenv("X_API_KEY")
#     X_API_SECRET: str = os.getenv("X_API_SECRET")
#     X_ACCESS_TOKEN: str = os.getenv("X_ACCESS_TOKEN")
#     X_ACCESS_SECRET: str = os.getenv("X_ACCESS_SECRET")
    
#     # RAG
#     RAG_SCORE_THRESHOLD: float = 0.5
#     TOP_K: int = 5
    
#     # 服务
#     ENV: str = "dev"

# settings = Settings()