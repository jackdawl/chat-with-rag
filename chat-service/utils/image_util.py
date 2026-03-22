import base64
from pathlib import Path

from config.common import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

def image_to_base64(path: str) -> str:
    """将图片文件转为 Base64 字符串，方便前端 <img src='data:...'> 内联展示"""

    try:
        file_name = Path(path).name
        file_path = Config.PDF_IMAGE_DIR + rf"\{file_name}"
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"图片转Base64失败: {e}")
        return ""