import os
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from pinecone import Pinecone
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# =============================================================================
# 설정 및 모델 초기화
# =============================================================================

load_dotenv()

class MessageRequest(BaseModel):
    """기존 RAGModule과 동일한 요청 스키마"""
    query: str
    language: str = "python"

@dataclass
class RAGConfig:
    """RAG 시스템 설정"""
    index_name: str = "python"
    k_documents_per_namespace: int = 2
    search_type: str = "mmr"
    temperature: float = 0
    namespace_options: List[str] = None
    
    def __post_init__(self):
        if self.namespace_options is None:
            self.namespace_options = [
                'c-api', 'deprecations', 'extending', 'faq', 'howto', 
                'installing', 'library', 'reference', 'tutorial', 'using', 'whatsnew'
            ]

class PlanResponse(BaseModel):
    """계획 응답 스키마"""
    plan: str
    namespaces: List[str]

# =============================================================================
# 모델 및 클라이언트 초기화
# =============================================================================

class ModelManager:
    """모델 및 클라이언트 관리 클래스"""
    
    def __init__(self):
        self._load_api_keys()
        self._initialize_models()
    
    def _load_api_keys(self):
        """API 키 로드"""
        self.upstage_api_key = os.getenv("UPSTAGE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        
        if not all([self.upstage_api_key, self.pinecone_api_key]):
            raise ValueError("UPSTAGE_API_KEY와 PINECONE_API_KEY가 필요합니다.")
    
    def _initialize_models(self):
        """모델 초기화"""
        self.embedding = UpstageEmbeddings(
            model="embedding-query", 
            api_key=self.upstage_api_key
        )
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Planning용 LLM (더 창의적)
        self.planning_llm = ChatUpstage(
            model="solar-pro", 
            api_key=self.upstage_api_key, 
            temperature=0.4
        )
        
        # Answer용 LLM (더 정확함)
        self.answer_llm = ChatUpstage(
            model="solar-pro", 
            api_key=self.upstage_api_key, 
            temperature=0
        )

# =============================================================================
# 프롬프트 템플릿 관리
# =============================================================================

class PromptManager:
    """프롬프트 템플릿 관리 클래스"""
    
    @staticmethod
    def get_planning_prompt() -> ChatPromptTemplate:
        """질문 분석 및 namespace 선택 프롬프트"""
        return ChatPromptTemplate.from_messages([
            ("system", """
You are a Python-documentation routing assistant.

- Think step-by-step **internally**; output only the final JSON.  
- Select the set of relevant namespaces (2-4) from:
  ["c-api","deprecations","extending","faq","howto","installing",
   "library","reference","tutorial","using","whatsnew"].
- Choose the most specific namespaces for the question.
- Always add the 'whatsnew' namespace.
             
Return **raw JSON** (no markdown, no comments) with this schema:
{{
  "plan":        "<short internal strategy>",
  "namespaces":  ["<namespace>", "..."]
}}
        
Example:
{{
    "plan": "To answer about string formatting, I need to search the library namespace for string methods and the reference namespace for syntax details.",
    "namespaces": ["library", "reference"]
}}
"""),
            ("human", "Question: {question}")
        ])
    
    @staticmethod
    def get_answer_prompt() -> ChatPromptTemplate:
        """최종 답변 생성 프롬프트"""
        return ChatPromptTemplate.from_messages([
            ("system", """
<<<CONTEXT>>>
{context}
<<<END_CONTEXT>>>

<<<PLAN>>>
{plan}
<<<END_PLAN>>>

You are **PyDoc Expert**, a senior Python engineer.
You have gathered relevant information from Python documentation to answer the user's question.

**STRICT RULES:**
1. **RAG-only responses**: Use ONLY information from the provided context
2. **No hallucination. NEVER!!**: If information is missing, state clearly: 
   "This information is not available in the provided documentation."
3. **Question might be wrong**: If the question is unclear or seems incorrect, clarify it. And fix it. There might be wrong and typo errors in the question.
4. **Source citation**: Instead of [Document-1], cite the actual source path from the context (e.g., [extending/extending.txt], [c-api/intro.rst], [whatsnew/3.12.rst])
5. **Markdown format**: Use proper Markdown with `python` for code blocks
6. **Accuracy first**: Better to say "unknown" than to guess

Reply in the same language Korean, using the following structure:

**Response Structure:**
## Answer  
<Direct, concise answer>

### Details & Examples  
<Detailed explanation with code examples if available>

### Version Information
<Version-specific details if mentioned in context>

### Sources  
List the actual source paths (without file extensions) referenced from the context (e.g., extending > extending, c-api > intro, whatsnew > 3.12)
"""),
            ("human", "Question: {input}")
        ])

# =============================================================================
# 문서 검색 및 처리
# =============================================================================

class DocumentRetriever:
    """문서 검색 및 처리 클래스"""
    
    def __init__(self, model_manager: ModelManager, config: RAGConfig):
        self.model_manager = model_manager
        self.config = config
    
    def retrieve_from_namespaces(self, question: str, namespaces: List[str]) -> List[Any]:
        """선택된 namespace들에서 문서 검색 (병렬 처리)"""
        all_docs = []
        
        # 병렬 처리를 위한 ThreadPoolExecutor 사용
        with ThreadPoolExecutor(max_workers=min(len(namespaces), 5)) as executor:
            # 각 namespace에 대해 future 생성
            future_to_namespace = {
                executor.submit(self._retrieve_from_single_namespace, question, namespace): namespace
                for namespace in namespaces
            }
            
            # 완료된 작업들 처리
            for future in as_completed(future_to_namespace):
                namespace = future_to_namespace[future]
                try:
                    docs = future.result()
                    if docs:
                        all_docs.extend(docs)
                except Exception as e:
                    pass  # 조용히 실패 처리
        
        return all_docs
    
    def _retrieve_from_single_namespace(self, question: str, namespace: str) -> List[Any]:
        """단일 namespace에서 문서 검색"""
        try:
            vectorstore = PineconeVectorStore(
                index=self.model_manager.pc.Index(self.config.index_name),
                embedding=self.model_manager.embedding,
                namespace=namespace
            )
            
            retriever = vectorstore.as_retriever(
                search_type=self.config.search_type,
                search_kwargs={"k": self.config.k_documents_per_namespace}
            )
            
            docs = retriever.invoke(question)
            
            # namespace 정보 추가
            for doc in docs:
                doc.metadata['search_namespace'] = namespace
            
            return docs
            
        except Exception as e:
            raise Exception(f"검색 실패: {str(e)}")
    
    def format_context(self, docs: List[Any]) -> str:
        """문서들을 컨텍스트 문자열로 포맷팅"""
        if not docs:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            namespace = doc.metadata.get('search_namespace', 'Unknown')
            content = doc.page_content
            
            context_parts.append(
                f"[Document-{i}] (Source: {source}, Namespace: {namespace})\n{content}"
            )
        
        return "\n\n" + "="*50 + "\n\n".join([""] + context_parts)

# =============================================================================
# 체인 관리
# =============================================================================

class ChainManager:
    """LangChain 체인 관리 클래스"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self._setup_chains()
    
    def _setup_chains(self):
        """체인 설정"""
        # JSON 파서
        self.json_parser = JsonOutputParser(pydantic_object=PlanResponse)
        
        # 체인 구성 - 각각 다른 LLM 사용
        self.planning_chain = (
            PromptManager.get_planning_prompt() 
            | self.model_manager.planning_llm 
            | self.json_parser
        )
        
        self.answer_chain = (
            PromptManager.get_answer_prompt() 
            | self.model_manager.answer_llm 
            | StrOutputParser()
        )
    
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """질문 분석 및 namespace 선택"""
        try:
            result = self.planning_chain.invoke({"question": question})
            return {
                "success": True,
                "plan": result["plan"],
                "namespaces": result["namespaces"]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_answer(self, question: str, plan: str, context: str) -> Dict[str, Any]:
        """최종 답변 생성"""
        try:
            answer = self.answer_chain.invoke({
                "input": question,
                "plan": plan,
                "context": context
            })
            return {
                "success": True,
                "answer": answer
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# =============================================================================
# 메인 RAG 시스템
# =============================================================================

def build_all():
    """기존 RAGModule과 동일한 함수명으로 초기화"""
    config = RAGConfig(
        k_documents_per_namespace=3,
        search_type="mmr"
    )
    return PythonRAG(config)

class PythonRAG:
    """Python 문서 RAG 시스템 메인 클래스"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self._initialize_components()
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        self.model_manager = ModelManager()
        self.chain_manager = ChainManager(self.model_manager)
        self.document_retriever = DocumentRetriever(self.model_manager, self.config)
    
    def generate_answer(self, question: str) -> str:
        """기존 RAGModule.stream()과 동일한 역할"""
        try:
            # Step 1: 질문 분석
            analysis_result = self.chain_manager.analyze_question(question)
            if not analysis_result["success"]:
                return f"질문 분석 중 오류가 발생했습니다: {analysis_result['error']}"
            
            plan = analysis_result["plan"]
            namespaces = analysis_result["namespaces"]
            
            # Step 2: 문서 검색
            docs = self.document_retriever.retrieve_from_namespaces(question, namespaces)
            if not docs:
                return "관련 문서를 찾을 수 없습니다."
            
            context = self.document_retriever.format_context(docs)
            
            # Step 3: 답변 생성
            answer_result = self.chain_manager.generate_answer(question, plan, context)
            if not answer_result["success"]:
                return f"답변 생성 중 오류가 발생했습니다: {answer_result['error']}"
            
            return answer_result["answer"]
            
        except Exception as e:
            return f"시스템 오류가 발생했습니다: {str(e)}"
    
    def stream(self, req: MessageRequest) -> str:
        """기존 RAGModule과 동일한 인터페이스"""
        return self.generate_answer(req.query)

# 기존 RAGModule과 동일한 클래스명으로 래핑
class RAGModule:
    def __init__(self):
        self.rag = build_all()

    def stream(self, req: MessageRequest) -> str:
        return self.rag.stream(req)