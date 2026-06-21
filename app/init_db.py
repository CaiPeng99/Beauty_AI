from app.database.session import create_all_tables
from app.database import models  # 这会自动执行 models.py，注册所有模型

if __name__ == "__main__":
    create_all_tables()
    print("数据表全部创建完成")