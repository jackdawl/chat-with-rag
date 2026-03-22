# LlamaIndex RAG 实战项目

## 项目概述

这是一个基于 LlamaIndex 框架构建的RAG系统，提供了完整的文档处理、检索和对话功能。
该系统支持多种格式的文档（PDF、TXT、DOCX、Markdown）上传、处理，并通过混合检索策略（向量检索 + BM25）提供高质量的问答服务。

### 核心功能

- **多格式文档处理**：支持PDF、TXT、DOCX、Markdown等多种文档格式的上传和解析。
- **混合检索策略**：结合向量检索和BM25关键词检索，提升检索精度。
- **文档重排序**：使用专用的重排序模型优化检索结果质量。
- **流式回答生成**：支持SSE流式返回生成内容，提供更好的用户体验。
- **用户认证机制**：基于JWT的安全认证。
- **历史记录优化**：自动管理和优化会话历史长度，防止超长历史导致的性能问题。

## 技术架构

### 系统架构

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  客户端      │ ──→  │ FastAPI API │ ──→  │ RAG Service │
└─────────────┘      └─────────────┘      └─────┬───────┘
                                               │
                       ┌───────────────────────┼───────────────────────┐
                       ▼                       ▼                       ▼
           ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
           │  Document       │     │  RAG            │     │  LLM            │
           │  Ingestion      │     │  Workflow       │     │  Model          │
           │  Pipeline       │     │                 │     │                 │
           └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
                    │                       │                       │
                    ▼                       ▼                       ▼
           ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
           │  ChromaDB       │     │  Redis          │     │  DashScope      │
           │  (向量存储)      │     │  (文档存储)     │     │  (文生模型)     │
           └─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 主要模块

1. **API层**：基于FastAPI提供RESTful API接口。
2. **服务层**：封装核心RAG逻辑，处理请求适配。
3. **核心层**：包含文档摄取、检索生成工作流和应用管理。
4. **存储层**：使用ChromaDB、Redis和MySql进行数据持久化。

## 项目结构

```
chat-service/
├── config/                     # 配置
│   ├── __init__.py
│   │── common.py               # 应用配置类
│   │── dash_scope.py           # 千问模型配置类
│   └── jwt.py                  # JWT配置类
├── core/                       # 核心业务逻辑
│   ├── __init__.py
│   ├── app.py                  # RAG应用主类
│   ├── document_ingestion.py   # 文档摄取管道
│   ├── document_manager.py     # 文档管理器
│   ├── events.py               # 事件定义
│   ├── pdf_parser4llm.py       # PDF解析器
│   └── workflow.py             # RAG工作流
├── dao/                        # 用户数据访问层
│   ├── __init__.py
│   ├── db_manager.py           # MySql数据库管理器
│   ├── user_dao.py             # 用户数据访问类
├── model/                      # 数据模型
│   ├── __init__.py
│   ├── chat_model.py           # 聊天数据模型
│   ├── common_model.py         # 通用数据模型
│   ├── doc_model.py            # 文档数据模型
│   ├── user_model.py           # 用户数据模型
├── requirements.txt            # 项目依赖
└── utils/                      # 工具类
│   ├── __init__.py
│   ├── image_util.py           # 图片处理工具
│   ├── jwt_util.py             # JWT处理工具
│   ├── logger.py               # 日志处理工具
│   ├── pwd_util.py             # 密码处理工具
├── web/                        # FastAPI应用
│   ├── __init__.py
│   ├── router/                # FastAPI路由
│   │   ├── __init__.py      
│   │   ├── chat_router.py     # 聊天相关接口
│   │   ├── doc_router.py      # 文档相关接口
│   │   └── user_router.py     # 用户相关接口
│   └── service/               # 服务层
│       ├── __init__.py 
│       └── rag_service.py     # RAG服务
├── main.py                    # 应用入口
├── requirements.txt           # 项目依赖
└── README.md                  # 项目文档
```

## 技术栈

### 核心框架
- **LlamaIndex**: RAG应用框架。
- **FastAPI**: 高性能Web框架。
- **Pydantic**: 数据验证和设置管理。

### 模型与向量库
- **DashScope LLM**: 生成模型（通义千问）
- **HuggingFace Embedding**: 本地嵌入模型
- **ChromaDB**: 向量数据库
- **Redis**: 文档和索引存储
- **MySql**: 用户存储

### 检索优化
- **BM25Retriever**: 关键词检索
- **SentenceTransformerRerank**: 文档重排序

## 核心功能模块

### 1. 文档摄取管道 (DocumentIngestionPipeline)

负责文档的上传、解析、切分、向量化和索引构建：

- 支持多格式文档处理
- 文档分块和向量化
- 索引持久化存储
- 缓存机制优化

### 2. RAG工作流 (RAGWorkflow)

实现检索增强生成的核心流程：

- 混合检索策略（向量检索 + BM25）
- 检索结果重排序
- 上下~~文构建和回答生成~~

### 3. RAG应用管理 (RAGApplication)

应用层面的功能整合：

- 多会话管理
- 流式/非流式查询支持
- 历史记录优化（防止超长历史导致的性能问题）
- 错误处理和日志记录


## 安装与部署

### 前置条件

- Python 3.10+
- Redis服务
- MySql服务
- CUDA环境（推荐，用于加速嵌入模型）

### 安装步骤

1. 克隆项目
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 配置环境变量（在.env文件中）：
   ```
   DASHSCOPE_API_KEY=your_api_key_here
   ...
   ```
4. 启动应用：
   ```bash
   python  main.py
   ```
