
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sqlalchemy.ext.declarative import declarative_base

from utils.logger import setup_logger

logger = setup_logger(__name__)

Base = declarative_base()

class DatabaseManager:
    """数据库管理器"""
    def __init__(self, config):
        self.config = config
        self.engine = None
        self.SessionLocal = None
        self.init_database()

    def init_database(self):
        """创建数据库连接"""

        connection_string = f"mysql+pymysql://{self.config.MYSQL_USER}:{self.config.MYSQL_PASSWORD}@{self.config.MYSQL_HOST}:{self.config.MYSQL_PORT}/{self.config.MYSQL_DATABASE}?charset=utf8mb4"
        try:
            self.engine = create_engine(connection_string, echo=False)
            # 创建所有的表
            Base.metadata.create_all(self.engine)
            self.SessionLocal = sessionmaker(bind=self.engine)
            print("数据库连接成功...")
        except Exception as e:
            print(f"数据库连接失败: {e}")
            raise

    def get_session(self):
        """获取数据库连接session"""
        return self.SessionLocal()

