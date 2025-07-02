import os

# ─── 0) Telemetry 끄기 (반드시 가장 위에) ──────────────
from dotenv import load_dotenv
load_dotenv()
os.environ["LANGCHAIN_TELEMETRY"] = "0"
os.environ["CHROMA_TELEMETRY"]   = "0"

# ─── 1) 환경변수 세팅 ──────────────────────────────────
# (HF Inference API)
# export HUGGINGFACEHUB_API_TOKEN="hf_xxx-your-token-xxx"
# (OpenAI API)
# export OPENAI_API_KEY="sk-xxx-your-key-xxx"
# os.environ['HUGGINGFACEHUB_API_TOKEN'] = ''

# ─── 2) 공통 문서 로드 & 전처리 ─────────────────────────
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter           import RecursiveCharacterTextSplitter

loader   = UnstructuredURLLoader(urls=["https://www.openssh.com/manual.html"])
raw_docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs     = splitter.split_documents(raw_docs)

# ─── 3) 벡터스토어 구축 (Flan-T5 & Chat 둘 다 공통 사용) ───
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings    import OpenAIEmbeddings
from langchain_chroma     import Chroma

# Flan-T5 테스트용 임베딩
hf_emb     = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# GPT-4o-Mini 테스트용 임베딩 (OpenAI)
# openai_emb = OpenAIEmbeddings()

# Chroma 인스턴스 (telemetry는 env var 로 이미 꺼둠)
# settings = Settings()        # uses defaults
# settings.anonymized_telemetry = True  # turn it off

# chroma = Chroma(
#     persist_directory="sshd_chroma_db",
#     embedding_function=hf_emb,
#     # client_settings=settings   # note: not `settings=` but `client_settings=`
# )
# print("[1/4] Chroma client created")

# ─── 2) Sanity check your `docs` ────────────────────────
from langchain.schema import Document
print("docs is a list:", isinstance(docs, list))
print("Each element is Document?:", all(isinstance(d, Document) for d in docs))
print("Number of chunks:", len(docs))
for i, d in enumerate(docs, 1):
    print(f" chunk {i} length:", len(d.page_content), "chars")

    
# print(docs)
print("Total chunks:", len(docs))

# chroma.aadd_documents(docs)

# print(f"[2/4] Indexed {len(docs)} chunks")
# # Chroma 0.4+ 자동 퍼시스트 → 아래 호출 생략 가능
# print("[3/4] Persisted to disk")
# retriever = chroma.as_retriever(search_kwargs={"k": 3})
# print("[4/4] Retriever ready\n")

from langchain_community.vectorstores import FAISS

# build in memory
faiss_store = FAISS.from_documents(docs, hf_emb)
print("FAISS index ready ✔️")
retriever  = faiss_store.as_retriever(search_kwargs={"k":3})

# ─── 4a) Flan-T5 파이프라인 정의 ───────────────────────

# 1) HF Inference API 클라이언트 생성
from huggingface_hub import InferenceClient
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1) HF Inference API 클라이언트 생성
print(f"env val: {os.getenv('HUGGINGFACEHUB_API_TOKEN')}")
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# 2) 원격 LLM 호출용 RunnableLambda 정의
remote_llm = RunnableLambda(
    lambda state: client.chat_completion(
        # 메시지 목록으로 system+user 역할 지정
        messages=[
            {"role": "system", "content": state["context"]},
            {"role": "user",   "content": state["query"]}
        ],
        model="deepseek-ai/DeepSeek-R1",
        temperature=0.0,
        max_tokens=512
    ).choices[0].message["content"]  # 응답 텍스트 추출
)

# 3) 파이프라인 조합 (context+query → chat_completion → str)
flan_pipeline = (
    RunnablePassthrough()  # 상태 dict: {"query":…, "context":…}
    | remote_llm           # -> HF remote chat_completion 호출
    | StrOutputParser()    # -> 최종 문자열 변환
)

flan_parser  = StrOutputParser()

# (B) Runnable 파이프라인 조합
flan_pipeline = (
    RunnablePassthrough()  # 상태 dict 받기       # prompt 문자열 생성
    | remote_llm           # HF inference 호출
    | flan_parser          # str 로 파싱
)

# ─── 4b) GPT-4o-Mini 파이프라인 정의 ───────────────────
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts           import ChatPromptTemplate

# (A) ChatOpenAI + parser
# chat_llm      = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
chat_template = ChatPromptTemplate.from_template(
    "아래 문서 컨텍스트를 참고하여 질문 “{query}”에 대해\n"
    "간결하고 명확하게 설명해 주세요.\n\n"
    "{context}"
)
chat_parser   = StrOutputParser()

# (B) Runnable 파이프라인 조합
# chat_pipeline = (
#     RunnablePassthrough()
#     | chat_template
#     | chat_llm
#     | chat_parser
# )

# ─── 5) LangGraph 노드 정의 & 그래프 컴파일 ─────────────
from langchain_community.utilities import SerpAPIWrapper
from langgraph.graph             import START, END, StateGraph

# 공통 retrieve node
def retrieve(state):
    docs = retriever.get_relevant_documents(state["query"])
    print(f"[retrieve] Found {len(docs)} relevant chunks")
    print(docs)
    state["context"] = "\n---\n".join(d.page_content for d in docs)
    return state

# Flan-T5 grade node
def grade_flan(state):
    out = flan_pipeline.invoke(state)
    state["doc_answer"] = out.strip()
    state["need_web"]   = False
    return state

# GPT-4o-Mini grade node
def grade_chat(state):
    # out = chat_pipeline.invoke(state)
    # state["doc_answer"] = out.strip()
    state["need_web"]   = False
    return state

# generate node (공통)
def generate(state):
    return state["doc_answer"]

# –– Flan-T5 그래프
wf_flan = StateGraph(dict, initial_state_type=dict)
wf_flan.add_node("retrieve",        retrieve)
wf_flan.add_node("grade_documents", grade_flan)
wf_flan.add_node("generate",        generate)
wf_flan.add_edge(START,             "retrieve")
wf_flan.add_edge("retrieve",        "grade_documents")
wf_flan.add_edge("grade_documents", "generate")
wf_flan.add_edge("generate",        END)
app_flan = wf_flan.compile()

# –– GPT-4o-Mini 그래프
wf_chat = StateGraph(dict, initial_state_type=dict)
wf_chat.add_node("retrieve",        retrieve)
wf_chat.add_node("grade_documents", grade_chat)
wf_chat.add_node("generate",        generate)
wf_chat.add_edge(START,             "retrieve")
wf_chat.add_edge("retrieve",        "grade_documents")
wf_chat.add_edge("grade_documents", "generate")
wf_chat.add_edge("generate",        END)
app_chat = wf_chat.compile()

class rag_prac:
    def __init__(self):
        self.app_flan = app_flan
        # self.app_flan = app_chat

    def answer(self, query):
        return self.app_flan.invoke({"query": query})


# ─── 6) 실행 예시 ────────────────────────────────────
if __name__ == "__main__":
    # q = "What does PermitRootLogin do in sshd_config?"
    
    while True:
        q = input("문서에 대한 질문입력 > ")
        if ("/exit" in q) : break

        print("▶ Flan-T5 pipeline answer:")
        # .invoke() 로 최종 문자열 반환
        ans1 = app_flan.invoke({"query": q})
        print(ans1, "\n")

    # print("▶ GPT-4o-Mini pipeline answer:")
    # ans2 = app_chat.invoke({"query": q})
    # print(ans2)
