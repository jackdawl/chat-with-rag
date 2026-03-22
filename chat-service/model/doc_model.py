from typing import List

from pydantic import BaseModel


class UploadResponse(BaseModel):
    """上传/摄取结果"""
    status: str  # "success" | "processing" | "failed"
    message: str  # 友好提示文本
    processed_files: List[str] = []  # 成功处理的文件名列表


class DocsListResponse(BaseModel):
    """文档列表响应"""
    documents: List[str]