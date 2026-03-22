from typing import Optional

from dotenv import load_dotenv
import os

load_dotenv()

class DashScopeConfig:
    """阿里云DashScope配置信息"""
    api_key: Optional[str] = os.getenv("DASHSCOPE_API_KEY") # 配置在根目录下.env 文件或者 Pycharm 的环境变量里
    base_url: Optional[str] = os.getenv("DASHSCOPE_BASE_URL")
    model_name: str = "qwen3-max-2026-01-23" # qwen3.5 模型会有问题，响应很慢，导致content 为null
    max_tokens = 2000
    temperature: float = 0.1