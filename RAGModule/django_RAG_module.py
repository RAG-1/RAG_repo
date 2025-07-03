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

load_dotenv()


class MessageRequest(BaseModel):
    query: str
    language: str = "python"  # Default to "python"

def build_all():
    # Build the RAG pipeline: retriever -> prompt -> llm -> output parser
    llm = OpenAI(model_name="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    embedding_upstage = UpstageEmbeddings(model="embedding-query", api_key=os.getenv("UPSTAGE_API_KEY"))

    vectordb = PineconeVectorStore(index=pc.Index("django"), embedding=embedding_upstage)
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2})

    prompt = PromptTemplate(
        input_variables=["language", "query", "context"],
        template="""
<<<CONTEXT>>>
{context}
<<<END_CONTEXT>>>

You are **Django Expert**, a senior Django developer.
You have gathered relevant information from Django documentation to answer the user's question.

**STRICT RULES:**
1. **RAG-only responses**: Use ONLY information from the provided context
2. **No hallucination. NEVER!!**: If information is missing, state clearly: 
   "This information is not available in the provided documentation."
3. **Question might be wrong**: If the question is unclear or seems incorrect, clarify it. There might be wrong and typo errors in the question.
4. **Source citation**: Instead of [Document-1], cite the actual source file names from the context (e.g., [django/models.rst], [django/views.rst])
5. **Markdown format**: Use proper Markdown with `python` for code blocks
6. **Accuracy first**: Better to say "unknown" than to guess

Reply in the same language as the question, using the following structure:

**Response Structure:**
## Answer  
<Direct, concise answer>

### Details & Examples  
<Detailed explanation with code examples if available>

### Version Information
<Version-specific details if mentioned in context>

### Sources  
List the actual source file paths referenced from the context (e.g., django/models, django/views, django/forms)

Question: {query}
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