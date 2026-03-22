"""用户信息相关操作"""

from sqlalchemy.orm import Session

from config.common import Config
from dao.db_manager import DatabaseManager
from model.user_model import UserInDB, User
from utils.logger import setup_logger

logger = setup_logger(__name__)

"""
模拟用户数据库，这里使用字典来模拟数据库存储，包含一个默认管理员账户
"""
# fake_users_db = {
#     "root": {
#         "username": "root",
#         "full_name": "Administrator",
#         "email": "root@163.com",
#         # 这是"root123"的bcrypt哈希值
#         "hashed_password": "$argon2id$v=19$m=65536,t=3,p=4$mJtdIxdxJCESA561Ng5gNw$YVcDK5kpItT8HMYUq+tCHX4OfiyHQ6B/23fcNHvXGvg",
#         "disabled": False,
#     }
# }

db_manager = DatabaseManager(Config())

class UserDAO:
    """用户模型与数据库连接层"""

    def __init__(self):
        self.db: Session = None


    def get_user(self, username: str):
        """
        从数据库中获取用户信息
    
        Args:
            username (str): 用户名

        """
        self.db = db_manager.get_session()
        try:
            user = self.db.query(UserInDB).filter(UserInDB.username == username).first()
            if user:
                return user
            return None
        finally:
            self.db.close()


    def update_user(self, user: User) :
        """
        更新数据库中用户信息

        Args:
            user: 用户更新信息

        """

        self.db = db_manager.get_session()
        try:
            db_user = self.db.query(UserInDB).filter(UserInDB.username == user.username).first()

            if user.email is not None and user.email != "":
                db_user.email = user.email
            if user.full_name is not None and user.full_name != "":
                db_user.full_name = user.full_name
            self.db.commit()
            self.db.refresh(db_user)
            return User(username=db_user.username,
                        email=db_user.email,
                        full_name=db_user.full_name,
                        disabled=db_user.disabled)

        finally:
            self.db.close()

    def save_user(self, user: UserInDB):
        """
        更新数据库中用户信息
        Args:
            user: 用户更新信息

        """

        self.db = db_manager.get_session()
        try:
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            return user

        finally:
            self.db.close()

    def delete_user(self, username: str):
        """
        删除数据库中用户信息
        软删除：disabled=True
        Args:
            username (str): 用户名
        """
        self.db = db_manager.get_session()
        try:
            db_user = self.db.query(UserInDB).filter(UserInDB.username == username).first()

            db_user.disabled = True
            self.db.commit()
            self.db.refresh(db_user)
            return db_user

        finally:
            self.db.close()

    def list_user(self):
        """
        查询数据库中所有用户信息

        """
        self.db = db_manager.get_session()
        try:
            users = self.db.query(UserInDB)
            if users:
                return users
            return None
        finally:
            self.db.close()