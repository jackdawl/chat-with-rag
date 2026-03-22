from pydantic import BaseModel


class CommonResponse(BaseModel):
    """通用状态响应"""
    status: str
    message: str