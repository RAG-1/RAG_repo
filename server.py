from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_prac import rag_prac

# ─── 1) 라이브러리 ─────────────────────────────────────

rag = rag_prac() # 
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# class ChatMessage(BaseModel):
#     role: str
#     content: str
class MessageRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(req: MessageRequest):
    # qa = RetrievalQA.from_chain_type(llm=chat_upstage,
    #                                  chain_type="stuff",
    #                                  retriever=pinecone_retriever,
    #                                  return_source_documents=True)

    #result = rag.answer(req.message)
    # return {"reply": result['result']}
    # return {"reply": rag.answer(req.message)["doc_answer"]}
    return {"reply": "helo"}


@app.get("/health")
@app.get("/")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)