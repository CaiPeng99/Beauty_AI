from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.config import PG_DATABASE_URL

engine = create_engine(PG_DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def create_all_tables():
    Base.metadata.create_all(bind=engine)
    '''
    读取所有继承自 Base 的模型类（你之前在 models.py 中已改为 from app.database.session import Base）。
    根据模型中的 __tablename__ 和字段定义，生成对应的 CREATE TABLE SQL 语句。
    通过 engine 连接到数据库并执行这些 SQL 语句。
    如果表已经存在，默认不会重复创建（取决于参数，默认 checkfirst=True，所以不会报错）。
    '''

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()