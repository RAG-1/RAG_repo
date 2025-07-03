from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from RAGModule import RAGModule as DjangoRAGModule
from RAGModule.python_RAG_module import RAGModule as PythonRAGModule, MessageRequest
from dotenv import load_dotenv

load_dotenv()

# 기존 RAG (Django용)
django_rag = DjangoRAGModule()

# 새로운 Python RAG (기존과 동일한 인터페이스)
python_rag = PythonRAGModule()

app = FastAPI()

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
    # MessageRequest 객체 생성
    message_req = MessageRequest(query=req.query, language=req.language)
    
    if req.language.lower() == "python":
        # Python 전용 RAG 사용
        reply = python_rag.stream(message_req)
        return {"reply": reply}
    elif req.language.lower() == "django":
        # 기존 Django RAG 사용
        reply = django_rag.stream(message_req)
        return {"reply": reply}
    else:
        return {"reply": "지원하지 않는 언어입니다."}

@app.get("/health")
@app.get("/")   
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
