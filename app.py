from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from RAGModule import RAGModule, MessageRequest  # MessageRequest import 추가
from dotenv import load_dotenv

load_dotenv()
rag = RAGModule()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat_endpoint(req: MessageRequest) -> dict:
    return {"reply" : rag.stream(req)}

@app.get("/health")
@app.get("/")   
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)