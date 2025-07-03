from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.globals import set_verbose

from pinecone import Pinecone
from pydantic import BaseModel
import os

load_dotenv()
set_verbose(True)

class FixedUpstageEmbeddings(UpstageEmbeddings):
    def embed_query(self, text: str) -> list[float]:
        # Upstage expects `input` to be a list of strings, not a single string
        response = self.client.create(
            model=self.model,      # the model name you passed in __init__
            input=[text]           # wrap your text in a list
        )
        # extract the embedding from the first item
        return response.data[0].embedding



class MessageRequest(BaseModel):
    query: str
    language: str = "python"

def build_all():
    # 1) LLM 세팅
    llm = OpenAI(
        model_name="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=2048
    )

    # 2) Memory 세팅
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="query",
        output_key="answer",
        return_messages=False
    )

    # 3) 이전 대화 불러오기 (memory)
    load_memory = RunnableLambda(
        lambda state: {
            **state,
            "memory": memory.load_memory_variables({ "query": state["query"] })["chat_history"]
        }
    )

     # → 여기에 디버그 라벨 추가
    debug_memory = RunnableLambda(
        lambda state: (
            print(f"[DEBUG] Loaded memory for query '{state['query']}':\n{state['memory']}"),
            state
        )[1]
    )

    # 4) 벡터스토어 & RAG retriever
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    # embedding = UpstageEmbeddings(model="embedding-query", api_key=os.getenv("UPSTAGE_API_KEY"))
    embedding = FixedUpstageEmbeddings(model="embedding-query",
                                       api_key=os.getenv("UPSTAGE_API_KEY"))

    vectordb = PineconeVectorStore(index=pc.Index("django"), embedding=embedding)
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 8})

    def retrieve_step(state):
        docs = retriever.get_relevant_documents(state["query"])
        return {
            **state,
            "context": "\n---\n".join(d.page_content for d in docs)
        }

    # 5) PromptTemplate 정의 (memory + context 포함)
    prompt = PromptTemplate(
        input_variables=["language", "query", "memory", "context"],
        template="""
Question: {query}

You are a {language} expert.
look **memory** and **context** to clarify user's question. 
from question, follow user's fulfillment like language(eg. explain in korean, english..., etc) 
or some order (eg. explain step by step, or just answer, etc).

when returning answer, Remove any preamble or greeting. Start your output EXACTLY at the <think> tag.
**Begin** every response with your chain-of-thought, built-in system prompt wraped in `<think>...</think>`.
then output the answer in ## Answer section.

so answer format always be like this: 

<think> </think>
## Answer

below the ## Answer section, u can do freely write your answer, while...

NO HALF ANSWERS, NO HALLUCINATION, NO ASSUMPTIONS, NO GUESSING.
context is the most relevant information (latest features of {language}) from the documentation for the answer.
always try to answer with latest features of {language}, except when the user asks about older versions of {language}.
so if context is enough to make answer, just return answer. 
if lacks the answer, and if the answer must be related to the older version, 
make answer based on your knowledge.
missing info, or despite above procedures, yet the answer is not valid, reply: “This information is not fully explained by the provided documentation.”  
Cite source files (e.g., django/models.rst).  
Format code in ```python``` blocks.
always mention the version you are referring to.

<<<MEMORY>>>
{memory}
<<<END_MEMORY>>>

<<<CONTEXT>>>
{context}
<<<END_CONTEXT>>>

""",
    )

    # 6) Prompt 생성
    make_prompt = RunnableLambda(
        lambda state: {
            **state,
            "prompt": prompt.format(
                language=state["language"],
                query=state["query"],
                memory=state["memory"],
                context=state["context"]
            )
        }
    )


    # 7) LLM 호출
    def safe_call_llm(state):
        try:
            return {**state, "answer": llm(state["prompt"])}
        except Exception as e:
            print(f"[ERROR] LLM 호출 중 오류 발생: {e}")
            return state  # or return None if you really want to drop it

    call_llm = RunnableLambda(safe_call_llm)

    # call_llm = RunnableLambda(
    #     lambda state: { **state, "answer": llm(state["prompt"]) }
    # )

    # 8) 응답 파싱
    parse_answer = RunnableLambda(
        lambda state: { **state, "answer": StrOutputParser().parse(state["answer"]) }
    )

    # 9) 메모리 저장 & 최종 문자열 반환
    save_and_return = RunnableLambda(
        lambda state: (
            memory.save_context(
                { "query": state["query"] },
                { "answer": state["answer"] }
            ),
            state["answer"]
        )[1]
    )

    # 10) 파이프라인 조합
    return (
        RunnablePassthrough()
        | load_memory
        | debug_memory # Optional: Debugging memory content
        | RunnableLambda(retrieve_step)
        | make_prompt
        | call_llm
        | parse_answer
        | save_and_return
    )


class RAGModule:
    def __init__(self):
        self.rag = build_all()

    def stream(self, req: MessageRequest) -> str:
        return self.rag.invoke(req.model_dump())

if __name__ == "__main__":
    rag_module = RAGModule()
    while True:
        query = input("질문을 입력하세요 (종료하려면 '/exit' 입력): ")
        if query.lower() == '/exit':
            break
        request = MessageRequest(query=query, language="django")

        print("response1:", rag_module.stream(request))
    # request = MessageRequest(query="해당 문법은 언제 처음 등장해?", language="django")
    # print("response2:", rag_module.stream(request))
