"""
FastAPI 应用入口：
- 配置 CORS（跨域）
- 注册子路由
- 提供健康检查接口
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from web.router import doc_router, user_router, chat_router
from utils.logger import setup_logger

logger = setup_logger(__name__)

# 创建 FastAPI 应用实例，title/version 会显示在自动生成的文档页面
app = FastAPI(title="RAG API", version="1.0.0")

# CORS 配置：开发阶段允许所有来源；生产环境务必收紧 allow_origins 白名单
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有自定义头
)

# 通过 include_router 挂载子路由，统一加上 /api 前缀与 tags
app.include_router(doc_router.router, prefix="/api/doc", tags=["document"])
app.include_router(chat_router.router, prefix="/api/chat", tags=["chat"])
app.include_router(user_router.router, prefix="/api/user", tags=["user"])


@app.get("/api/health")
def health():
    """
    健康检查：用于存活探针与基本可用性验证。
    Kubernetes/监控系统可定期调用该接口。
    """
    return {"status": "ok", "message": "healthy"}


if __name__ == "__main__":
    import uvicorn

    # 打印启动信息
    print("=" * 50)
    print("服务启动中...")
    print("=" * 50)
    print("Web界面: http://localhost:8100")
    print("API文档: http://localhost:8100/docs")
    print("=" * 50)


    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8100,
        reload=True,
        log_level="info",
        # debug=True,
    )


