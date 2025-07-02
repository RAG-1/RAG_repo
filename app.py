from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# from rag_prac import rag_prac
from RAGModule.RAGModule import RAGModule
from dotenv import load_dotenv
# ─── 1) 라이브러리 ─────────────────────────────────────

load_dotenv()
# rag_prac = rag_prac() # test
rag = RAGModule()
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
    language: str = "python"  # 기본값을 "python"으로 설정

@app.post("/chat")
async def chat_endpoint(req: MessageRequest) -> dict:

    #return rag.stream(req.message, req.language)
    return {"reply": "helo"}



@app.get("/health")
@app.get("/")   
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

    # qa = RetrievalQA.from_chain_type(llm=chat_upstage,
    #                                  chain_type="stuff",
    #                                  retriever=pinecone_retriever,
    #                                  return_source_documents=True)

    # result = rag.answer(req.message)
    # index = get_vector_index(req.language)
    # context = index.similarity_search(req.message)
    # answer = llm.generate_answer(context, req.message)
    # return {"reply": answer}
    # print(f"[chat] {req.message} -> {result}")
    # return {"reply": result}