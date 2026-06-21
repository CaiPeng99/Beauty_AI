from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.agent.tools import ToolRegistry
from app.config import TIMEZONE
from app.common.logger import logger

scheduler = BackgroundScheduler(timezone=TIMEZONE)

def auto_daily_publish():
    """每日定时发布爆款产品到 X"""
    return