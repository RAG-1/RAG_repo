from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from RAGModule import RAGModule as DjangoRAGModule
from RAGModule.python_RAG_module import PythonRAG, RAGConfig as PythonRAGConfig, MessageRequest
from RAGModule.nodejs_RAG_module import NodeJSRAG, NodeJSRAGConfig
from dotenv import load_dotenv

load_dotenv()

# 각 언어별 RAG 시스템 초기화
django_rag = DjangoRAGModule()

python_config = PythonRAGConfig(k_documents_per_namespace=3, search_type="mmr")
python_rag = PythonRAG(python_config)

nodejs_config = NodeJSRAGConfig(k_documents=2, search_type="mmr")
nodejs_rag = NodeJSRAG(nodejs_config)

app = FastAPI(title="Multi-RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageRequest(BaseModel):
    query: str
    language: str = "python"

@app.post("/chat")
async def chat_endpoint(req: MessageRequest) -> dict:
    """채팅 엔드포인트"""
    try:
        # MessageRequest 객체 생성 (각 모듈의 인터페이스에 맞게)
        message_req = MessageRequest(query=req.query, language=req.language)
        
        if req.language.lower() == "python":
            reply = python_rag.stream(message_req)
            return {"reply": reply, "system": "python_advanced"}
        elif req.language.lower() == "django":
            reply = django_rag.stream(message_req)
            return {"reply": reply, "system": "django_basic"}
        elif req.language.lower() == "nodejs" or req.language.lower() == "node":
            # Node.js RAG 사용
            reply = nodejs_rag.stream(message_req)
            return {"reply": reply, "system": "nodejs_multi_version"}
        else:
            return {"reply": "지원하지 않는 언어입니다. (python, django, nodejs)"}
    except Exception as e:
        return {"reply": f"시스템 오류: {str(e)}"}

@app.get("/systems")
async def get_systems():
    """사용 가능한 시스템 목록"""
    return {
        "available_systems": ["python", "django", "nodejs"],
        "python": {
            "type": "advanced",
            "features": ["namespace_routing", "parallel_search", "plan_based"],
            "index": "python"
        },
        "django": {
            "type": "basic", 
            "features": ["simple_retrieval"],
            "index": "django"
        },
        "nodejs": {
            "type": "multi_version",
            "features": ["version_parallel_search", "multi_index"],
            "indexes": nodejs_rag.document_retriever.available_indexes if nodejs_rag.document_retriever.available_indexes else ["no_indexes_found"]
        }
    }

@app.get("/health")
@app.get("/")   
async def health_check():
    return {
        "status": "ok", 
        "systems": ["python", "django", "nodejs"],
        "nodejs_indexes": len(nodejs_rag.document_retriever.available_indexes)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
