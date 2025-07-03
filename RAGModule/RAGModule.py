 # index = get_vector_index(req.language)
    # context = index.similarity_search(req.message)
    # answer = llm.generate_answer(context, req.message)

from langchain_openai import OpenAI
# from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    

from pinecone import Pinecone

import os
from pydantic import BaseModel
# from langchain.runnables import RunnablePassthrough

load_dotenv()


class MessageRequest(BaseModel):
    query: str
    language: str = "python"  # Default to "python"

def build_all():
    # Build the RAG pipeline: retriever -> prompt -> llm -> output parser
    llm = OpenAI(
        api_key=os.getenv("UPSTAGE_API_KEY"), base_url="https://api.upstage.ai/v1/solar"
    )
    # llm = OpenAI(model_name="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    embedding_upstage = UpstageEmbeddings(model="embedding-query", api_key=os.getenv("UPSTAGE_API_KEY"))

    vectordb = PineconeVectorStore(index=pc.Index("django"), embedding=embedding_upstage, namespace="")
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2})

    prompt = PromptTemplate(
        input_variables=["language", "query", "context"],
        template="""
        너는 최신 소프트웨어 프레임워크에 대해 학습을 도와주는 전문가 챗봇이야.  
        너의 지식은 외부 공식문서를 벡터로 검색하여 얻은 정보(RAG)를 기반으로만 구성되어 있어.  
        다음 원칙을 반드시 지켜줘:

        1. 사용자 질문에 대해 **제공된 문서 내용 내에서만** 답변해.  
        → 문서에 없는 정보는 “공식 문서에 해당 내용이 없습니다” 또는 “정확하지 않아 답변드리기 어렵습니다”라고 솔직히 말해.

        2. 답변은 **정확하고 실용적인 코드 예시**, **최신 문법 안내**, **버전 차이 설명**, **사용자 실수 방지 팁** 위주로 해줘.  
        → 예시 코드에는 항상 어떤 버전에서 동작하는지 명시해줘.

        3. 답변의 말투는 친절하지만 군더더기 없이 **간결하고 명확하게** 유지해.  
        → 장황한 설명보다는 핵심 문장을 우선 제시해.

        4. 여러 프레임워크 버전이 섞여 있는 문서일 수 있으므로, 항상 답변에 **문서에 언급된 버전 정보**를 포함해줘.

        5. 잘못된 사용법이나 비추천 기능이 있다면, 공식 문서를 근거로 하여 경고해줘.

        6. 사용자가 프레임워크 이름을 정확히 말하지 않더라도, 제공된 문서 기반으로 유추 가능한 경우 정답을 제시하고, 불확실한 경우에는 되묻거나 유보해.

        문서는 기술 문서이므로, 기술적인 표현을 이해하고 요약할 수 있어야 해.  
        공식 문서 외의 지식이나 일반적인 추측은 절대 하지 마.  
        답변의 품질은 검색된 문서에서의 정확성, 실용성, 그리고 간결한 전달이 핵심이야.

        current target language is {language}.
        Please provide a detailed answer to the following question(query), with following contexts from the provided documents:
        query : {query}
        context : {context}
        """,
    )

   
    def retrieve_step(state):
        docs = retriever.get_relevant_documents(state["query"])
        context = "\n---\n".join(d.page_content for d in docs)
        return {**state, "context": context}

    rag = (
        RunnablePassthrough()
        | RunnableLambda(retrieve_step)
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag


class RAGModule:
    def __init__(self):
        self.rag = build_all()

    def stream(self, req: MessageRequest) -> str:
        # Pass the request as a dict directly to the pipeline
        return self.rag.invoke(req.model_dump())