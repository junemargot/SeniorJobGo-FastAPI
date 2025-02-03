from typing import Dict, List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from app.core.prompts import verify_prompt, rewrite_prompt, generate_prompt, chat_prompt, chat_persona_prompt
from app.utils.constants import LOCATIONS
from .chat_agent import ChatAgent

class AgentState(Dict):
    query: str
    context: List[Document]
    answer: str
    should_rewrite: bool
    rewrite_count: int
    answers: List[str]

class JobAdvisorAgent:
    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        self.chat_agent = ChatAgent(llm)
        self.workflow = self.setup_workflow()
        
        # 구직 관련 키워드
        self.job_keywords = {
            'position': ['일자리', '직장', '취업', '채용', '구직', '일', '직업', '알바', '아르바이트', '정규직', '계약직'],
            'salary': ['급여', '월급', '연봉', '시급', '주급', '임금'],
            'location': ['지역', '근처', '가까운', '동네', '시', '구', '동'],
            'time': ['시간', '근무시간', '근무요일', '요일', '주말', '평일'],
            'type': ['경비', '운전', '청소', '요양', '간호', '주방', '조리', '판매', '영업', '사무', '관리', '생산', '제조']
        }

    def is_job_related(self, query: str) -> bool:
        """구직 관련 키워드가 포함되어 있는지 확인"""
        query_lower = query.lower()
        return any(
            keyword in query_lower
            for keywords in self.job_keywords.values()
            for keyword in keywords
        )

    def retrieve(self, state: AgentState):
        query = state['query']
        
        # 1. 구직 관련 키워드 확인
        if not self.is_job_related(query):
            # 일상 대화 처리 -> LLM으로 전달 -> 구직 관련 대화 유도
            response = self.chat_agent.chat(query)
            return {
                'answer': response,
                'is_job_query': False,
                'context': [],
                'query': query
            }
        
        # 2. 구직 관련 검색 수행
        docs = self.vector_store.similarity_search(query, k=10)
        print(f"검색 결과: {len(docs)}개 문서")
        
        # 3. 지역 기반 필터링
        user_location = next((loc for loc in LOCATIONS if loc in query), None)
        if user_location:
            filtered_docs = [
                doc for doc in docs
                if any(user_location in loc for loc in [
                    doc.metadata.get("location", ""),
                    f"{user_location}시",
                    f"{user_location}특별시"
                ])
            ]
            if filtered_docs:
                docs = filtered_docs[:3]
        else:
            docs = docs[:3]
            
        # 4. 검색 결과가 있는 경우
        if docs:
            context_str = "\n\n".join([
                f"제목: {doc.metadata.get('title', '')}\n"
                f"회사: {doc.metadata.get('company', '')}\n"
                f"지역: {doc.metadata.get('location', '')}\n"
                f"급여: {doc.metadata.get('salary', '')}\n"
                f"상세내용: {doc.page_content}"
                for doc in docs
            ])
            
            # LLM에 전달하여 응답 생성
            rag_chain = generate_prompt | self.llm | StrOutputParser()
            response = rag_chain.invoke({
                "question": query,
                "context": context_str
            })
            
            return {
                'answer': response,
                'is_job_query': True,
                'context': docs,
                'query': query
            }
        
        # 5. 검색 결과가 없는 경우
        return {
            'answer': "죄송합니다. 관련된 구인정보를 찾지 못했습니다. 다른 지역이나 직종으로 검색해보시겠어요? 어떤 종류의 일자리를 찾고 계신지 말씀해 주시면 제가 도와드리겠습니다. 😊",
            'is_job_query': True,
            'context': [],
            'query': query
        }

    def verify(self, state: AgentState) -> dict:
        if state.get('is_greeting', False):
            return {
                "should_rewrite": False,
                "rewrite_count": 0,
                "answers": state.get('answers', [])
            }
            
        context = state['context']
        query = state['query']
        rewrite_count = state.get('rewrite_count', 0)
        
        if rewrite_count >= 3:
            return {
                "should_rewrite": False,
                "rewrite_count": rewrite_count
            }
        
        verify_chain = verify_prompt | self.llm | StrOutputParser()
        response = verify_chain.invoke({
            "query": query,
            "context": "\n\n".join([str(doc) for doc in context])
        })
        
        return {
            "should_rewrite": "NO" in response.upper(),
            "rewrite_count": rewrite_count + 1,
            "answers": state.get('answers', [])
        }

    def rewrite(self, state: AgentState) -> dict:
        query = state['query']
        
        rewrite_chain = rewrite_prompt | self.llm | StrOutputParser()
        response = rewrite_chain.invoke({'query': query})
        
        return {'query': response}

    def generate(self, state: AgentState) -> dict:
        query = state['query']
        context = state.get('context', [])
        
        if state.get('is_basic_question', False):
            custom_response = state.get('custom_response')
            if custom_response:
                return {'answer': custom_response, 'answers': [custom_response]}
        
        if not context:
            return {
                'answer': "죄송합니다. 관련된 구인정보를 찾지 못했습니다. 다른 지역이나 직종으로 검색해보시겠어요? 어떤 종류의 일자리를 찾고 계신지 말씀해 주시면 제가 도와드리겠습니다. 😊",
                'answers': []
            }
            
        rag_chain = generate_prompt | self.llm | StrOutputParser()
        response = rag_chain.invoke({
            "question": query,
            "context": "\n\n".join([
                f"제목: {doc.metadata.get('title', '')}\n"
                f"회사: {doc.metadata.get('company', '')}\n"
                f"지역: {doc.metadata.get('location', '')}\n"
                f"급여: {doc.metadata.get('salary', '')}\n"
                f"상세내용: {doc.page_content}"
                for doc in context
            ])
        })
        
        return {'answer': response, 'answers': [response]}

    def router(self, state: AgentState) -> str:
        # 기본 대화나 인사인 경우 바로 generate로
        if state.get('is_basic_question', False):
            return "generate"
            
        # 검색 결과가 있으면 verify로
        if state.get('context', []):
            return "verify"
            
        # 검색 결과가 없으면 generate로
        return "generate"

    def setup_workflow(self):
        workflow = StateGraph(AgentState)
        
        # 단순화된 워크플로우: retrieve -> END
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_edge("retrieve", END)
        
        workflow.set_entry_point("retrieve")
        
        return workflow.compile() 