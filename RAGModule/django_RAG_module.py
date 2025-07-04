# from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
# from langchain.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.globals import set_verbose

from pinecone import Pinecone
from pydantic import BaseModel
import os
import logging

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
    # llm = OpenAI(
    #     model_name="gpt-4o-mini",
    #     api_key=os.getenv("OPENAI_API_KEY"),
    #     max_tokens=2048,
    #     temperature=0.0,
    # )
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.0,
        # verbose=True
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
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    def retrieve_step(state):
        docs = retriever.get_relevant_documents(state["search_query"])
        logging.info(f"[DEBUG] Retrieved {docs} with query '{state['search_query']}'")
        return {
            **state,
            "context": "\n---\n".join(d.page_content for d in docs)
        }

    # 5) PromptTemplate 정의 (memory + context 포함)
    prompt = ChatPromptTemplate.from_messages([
        ('system', """
    Question: {query}

    You are a {language} expert.
    look **memory** and **context** to clarify user's question. memory contans user's previous questions and answers.
    Use the provided memory to understand the user's question
    (eg. understand 'the thing' meaning in question, understand 'it' meaning in question).
    follow user's fulfillment(eg. explain in korean, english..., etc),
    also some order (eg. explain step by step, or just answer, etc).
    for the library or framework in certain language(eg. django in python, spring in java, ...), do not explain about the language itself, but only about the library or framework.

    **Begin** every response with your chain-of-thought, put them on (your detailed chain-of-thought here)
    then output the answer below ## Answer line.

    format below. STRICTLY preserve below html tags (details, pre, code, summary, hr). 
    add sections if needed(eg. ##Answer, ## Details & Examples, ## Version, ## Sources, etc):

    <details>
    <summary>Show reasoning (chain-of-thought)</summary>
    <pre><code>
    (your detailed chain-of-thought here)
    </code></pre>
    </details>

    <hr/>
    ## Answer
    ## Details & Examples <- if needed.
    ## Version <- if needed, you can add version info referring.
    ## Sources <- if needed, website or document links, or any sources you used to answer.


    IMPORTANT : keep in mind the following rules:
    NO HALF ANSWERS, NO HALLUCINATION, NO ASSUMPTIONS, NO GUESSING.
    **context** is the latest release notes of {language} from the documentation for the answer.
    with your old knowledge, and the **context**, answer the question.
    missing info, or despite above procedures, yet the answer is not valid, reply: “This information is not fully explained by the provided documentation.”
    always Cite source files that you used (e.g., django/models.rst).  
    Format code in ```python``` blocks.
    always mention the version you are referring to precisely.

    <<<MEMORY>>>
    {memory}
    <<<END_MEMORY>>>

    <<<CONTEXT>>>
    {context}
    <<<END_CONTEXT>>>
    """), 
    ('user', "Question: {query}")
    ])

    # 6) Prompt 생성
    make_prompt = RunnableLambda(
        lambda state: {
            **state,
            "messages": prompt.format_messages(
                language=state["language"],
                query=state["query"],
                memory=state["memory"],
                context=state["context"]
            )
        }
    )

    # a tiny chat‐prompt that extracts search keywords or clarifies intent
    analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a query analysis assistant. 
    Given the user’s language wanna know (django, python... etc), previous conversation memory, and the raw question,
    produce a **single concise** search query or set of keywords that will best surface relevant documentation.
    Return **only** the refined query string.
    """),
            ("user", """
    Language: {language}
    Memory: {memory}
    Question: {query}
    """)
    ])

    # ─── 1.b) 분석 단계 정의 ──────────────────────────
    def analyze_query(state):
        # 1) 메시지 리스트 생성
        msgs = analysis_prompt.format_messages(
            language=state["language"],
            memory=state["memory"],
            query=state["query"]
        )
        # 2) 분석 LLM 호출
        msg = llm(msgs)
        refined = msg.content.strip()
        # 3) state에 새 검색어로 저장
        return { **state, "search_query": refined }

    analyze_step = RunnableLambda(analyze_query)


    # 7) LLM 호출
    def safe_call_llm(state):
        try:
            # 1) ChatOpenAI 를 호출해 AIMessage 를 받습니다
            msg = llm(state["messages"])
            # 2) .content 속성(실제 텍스트)만 뽑아서 저장
            text = msg.content if hasattr(msg, "content") else str(msg)
            return {**state, "answer": text}
        except Exception as e:
                print(f"[ERROR] LLM 호출 중 오류 발생: {e}")
                return state

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
        | analyze_step
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
