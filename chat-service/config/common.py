
from pathlib import Path
from dotenv import load_dotenv
import os
load_dotenv()

class Config:
    """应用配置类"""

    # 嵌入模型
    EMBEDDING_MODEL_PATH: str = r"D:\tools\LLM\local_model\BAAI\bge-large-zh-v1___5"

    # 文档处理配置
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TITLE_EXTRACTOR_NODES: int = 5

    # 检索配置
    SIMILARITY_TOP_K: int = 5
    RERANK_TOP_K: int = 3
    SIMILARITY_CUTOFF: float = 0.5
    RERANK_MODEL_PATH: str = r"D:\tools\LLM\local_model\BAAI\bge-reranker-large"

    # 存储配置
    PROJECT_ROOT: Path = Path(__file__).parent.parent  # 项目根目录
    CHROMA_PERSIST_DIR: str = str(PROJECT_ROOT / "data/chroma_db")
    BM25_PERSIST_DIR: str = str(PROJECT_ROOT / "data/storage_bm25")
    DOCUMENTS_DIR: str = str(PROJECT_ROOT / "data/documents")
    DEFAULT_PERSIST_DIR: str = str(PROJECT_ROOT / "data/storage")
    PDF_IMAGE_DIR: str = str(PROJECT_ROOT / "data/image")
    LOG_DIR: str = str(PROJECT_ROOT / "logs")

    # 支持的文件类型
    SUPPORTED_FILE_TYPES: list = [".txt", ".pdf", ".docx", ".md"]

    # Redis 配置
    REDIS_HOST: str = "127.0.0.1"
    REDIS_PORT: int = 6379
    REDIS_NAMESPACE_INDEX_STORE: str = "redis_index"
    REDIS_NAMESPACE_DOCUMENT_STORE: str = "redis_docs"
    REDIS_NAMESPACE_INGESTION_CACHE: str = "redis_cache"

    # MySQL配置
    MYSQL_HOST = "localhost"
    MYSQL_PORT = 3306
    MYSQL_USER = "root"
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
    MYSQL_DATABASE = "llama_rag"

    @classmethod
    def validate_api_key(cls) -> bool:
        """验证API密钥是否存在"""
        return cls.API_KEY is not None and len(cls.API_KEY.strip()) > 0

