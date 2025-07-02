 # index = get_vector_index(req.language)
    # context = index.similarity_search(req.message)
    # answer = llm.generate_answer(context, req.message)

from langchain_community.llms import OpenAI
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
    llm = OpenAI(model_name="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    embedding_upstage = UpstageEmbeddings(model="embedding-query", api_key=os.getenv("UPSTAGE_API_KEY"))

    vectordb = PineconeVectorStore(index=pc.Index("django"), embedding=embedding_upstage, namespace="")
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    prompt = PromptTemplate(
        input_variables=["language", "query", "context"],
        template="""
        You are a helpful assistant that can answer questions about programming in {language}.
        Please provide a detailed answer to the following question, with following contexts from the provided documents:
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