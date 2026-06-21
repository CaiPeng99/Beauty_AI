import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from app.database.session import get_db
from app.database.models import UnknownQueryLog

# 全局日志
logger = logging.getLogger("beauty_system")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler("system.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# 【重点】未知问题专属日志
unknown_logger = logging.getLogger("unknown_query")
unknown_logger.setLevel(logging.WARNING)
unknown_handler = RotatingFileHandler("unknown_query.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
unknown_handler.setFormatter(formatter)
unknown_logger.addHandler(unknown_handler)

# 写入数据库日志函数
def save_unknown_query(query: str, reason: str, session_id: str):
    db = next(get_db())
    log = UnknownQueryLog(
        user_query=query,
        fail_reason=reason,
        session_id=session_id
    )
    db.add(log)
    db.commit()