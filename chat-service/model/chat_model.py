"""聊天相关的数据模型"""
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional, Dict, Any

from config.dash_scope import DashScopeConfig


config = DashScopeConfig()

class ChatMessage(BaseModel):
    """
    聊天消息数据模型
    用于表示对话中的单条消息，包含角色、内容和时间戳
    Fields：
        role：消息角色: user(用户), assistant(AI助手), system(系统)
    """
    role: str
    content: str
    sources: List[str]  # 引用来源（文档名/片段/分数等）


class ChatRequest(BaseModel):
    """
    客户端聊天请求数据模型
    定义了客户端发送聊天请求时需要包含的所有参数

    Fields：
        query：用户问题
        model：要使用的AI模型名称，默认使用配置中的模型
        temperature：温度值(千问：0-2)，控制回答的随机性
        max_tokens：最大生成token数，限制回答长度
        knowledge_bool：是否要开启知识库模式
    """
    query: str
    model: Optional[str] = config.model_name
    temperature: Optional[float] = config.temperature
    max_tokens: Optional[int] = config.max_tokens
    knowledge_bool: bool = False


class ChatResponse(BaseModel):
    """
    服务端聊天响应数据模型（非流式）
    请求的响应包含完整的AI回答和使用统计

    Fields：
        message：更新后的完整消息历史
    """
    messages: ChatMessage


class ClearRequest(BaseModel):
    """
    清空会话请求

    Fields：
        message：会话ID
    """
    session_id: str