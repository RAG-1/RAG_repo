import os
from typing import List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pinecone import Pinecone
from pydantic import BaseModel

# =============================================================================
# 설정 및 모델 초기화
# =============================================================================

load_dotenv()

class MessageRequest(BaseModel):
    """공통 요청 스키마"""
    query: str
    language: str = "nodejs"

@dataclass
class NodeJSRAGConfig:
    """Node.js RAG 시스템 설정"""
    node_versions: List[str] = None
    k_documents: int = 2
    search_type: str = "mmr"
    temperature: float = 0.2
    
    def __post_init__(self):
        if self.node_versions is None:
            self.node_versions = ['nodejs-v22', 'nodejs-v23', 'nodejs-v24']

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
        
        # Node.js 답변용 LLM
        self.llm = ChatUpstage(
            model="solar-pro", 
            api_key=self.upstage_api_key, 
            temperature=0.2
        )

# =============================================================================
# 프롬프트 템플릿 관리
# =============================================================================

class PromptManager:
    """프롬프트 템플릿 관리 클래스"""
    
    @staticmethod
    def get_answer_prompt() -> ChatPromptTemplate:
        """Node.js 답변 생성 프롬프트"""
        return ChatPromptTemplate.from_messages([
            ("system", """
<<<CONTEXT>>>
{context}
<<<END_CONTEXT>>>

You are **NodeJS Expert**, a senior Node.js engineer.
You have gathered relevant information from Node.js documentation to answer the user's question.

**STRICT RULES:**
1. **RAG-only responses**: Use ONLY information from the provided context
2. **No hallucination. NEVER!!**: If information is missing, state clearly: 
   "This information is not available in the provided documentation."
3. **Question might be wrong**: If the question is unclear or seems incorrect, clarify it. There might be wrong and typo errors in the question.
4. **Source citation**: Instead of [Document-1], cite the actual source file names from the context (e.g., [nodejs-v22/api/fs.md], [nodejs-v23/changelog.md])
5. **Markdown format**: Use proper Markdown with `javascript` for code blocks
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
List the actual source file paths referenced from the context (e.g., nodejs-v22/api, nodejs-v23/changelog)
"""),
            ("human", "Question: {input}")
        ])

# =============================================================================
# 문서 검색 및 처리
# =============================================================================

class DocumentRetriever:
    """문서 검색 및 처리 클래스"""
    
    def __init__(self, model_manager: ModelManager, config: NodeJSRAGConfig):
        self.model_manager = model_manager
        self.config = config
        self.available_indexes = self._check_available_indexes()
    
    def _check_available_indexes(self) -> List[str]:
        """사용 가능한 인덱스 확인"""
        try:
            existing_indexes = self.model_manager.pc.list_indexes().names()
            available = [name for name in self.config.node_versions if name in existing_indexes]
            return available
        except Exception as e:
            return []
    
    def retrieve_from_versions(self, question: str) -> List[Any]:
        """모든 Node.js 버전에서 문서 검색 (병렬 처리)"""
        if not self.available_indexes:
            return []
        
        all_docs = []
        
        # 병렬 처리를 위한 ThreadPoolExecutor 사용
        with ThreadPoolExecutor(max_workers=min(len(self.available_indexes), 3)) as executor:
            future_to_index = {
                executor.submit(self._retrieve_from_single_index, question, index_name): index_name
                for index_name in self.available_indexes
            }
            
            for future in as_completed(future_to_index):
                index_name = future_to_index[future]
                try:
                    docs = future.result()
                    if docs:
                        all_docs.extend(docs)
                except Exception as e:
                    pass  # 조용히 실패 처리
        
        return all_docs
    
    def _retrieve_from_single_index(self, question: str, index_name: str) -> List[Any]:
        """단일 인덱스에서 문서 검색"""
        try:
            vectorstore = PineconeVectorStore(
                index=self.model_manager.pc.Index(index_name),
                embedding=self.model_manager.embedding
            )
            
            retriever = vectorstore.as_retriever(
                search_type=self.config.search_type,
                search_kwargs={"k": self.config.k_documents}
            )
            
            docs = retriever.invoke(question)
            
            # 버전 정보 추가
            for doc in docs:
                doc.metadata['version'] = index_name
            
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
            version = doc.metadata.get('version', 'Unknown')
            content = doc.page_content
            
            context_parts.append(
                f"[Document-{i}] (Source: {source}, Version: {version})\n{content}"
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
        self.answer_chain = (
            PromptManager.get_answer_prompt() 
            | self.model_manager.llm 
            | StrOutputParser()
        )
    
    def generate_answer(self, question: str, context: str) -> Dict[str, Any]:
        """최종 답변 생성"""
        try:
            answer = self.answer_chain.invoke({
                "input": question,
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

def build_nodejs_rag():
    """Node.js RAG 시스템 초기화"""
    config = NodeJSRAGConfig(
        k_documents=2,
        search_type="mmr"
    )
    return NodeJSRAG(config)

class NodeJSRAG:
    """Node.js 문서 RAG 시스템 메인 클래스"""
    
    def __init__(self, config: NodeJSRAGConfig = None):
        self.config = config or NodeJSRAGConfig()
        self._initialize_components()
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        self.model_manager = ModelManager()
        self.chain_manager = ChainManager(self.model_manager)
        self.document_retriever = DocumentRetriever(self.model_manager, self.config)
    
    def stream(self, req: MessageRequest) -> str:
        """기존 RAGModule과 동일한 인터페이스"""
        try:
            if not self.document_retriever.available_indexes:
                return "사용 가능한 Node.js 문서 인덱스가 없습니다. 먼저 문서를 임베딩하세요."
            
            # Step 1: 문서 검색
            docs = self.document_retriever.retrieve_from_versions(req.query)
            if not docs:
                return "관련 문서를 찾을 수 없습니다."
            
            context = self.document_retriever.format_context(docs)
            
            # Step 2: 답변 생성
            answer_result = self.chain_manager.generate_answer(req.query, context)
            if not answer_result["success"]:
                return f"답변 생성 중 오류가 발생했습니다: {answer_result['error']}"
            
            return answer_result["answer"]
            
        except Exception as e:
            return f"시스템 오류가 발생했습니다: {str(e)}"