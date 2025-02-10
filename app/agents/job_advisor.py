from typing import Dict, List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from app.core.prompts import verify_prompt, rewrite_prompt, generate_prompt, chat_prompt
from app.utils.constants import LOCATIONS, DICTIONARY
from app.agents.chat_agent import ChatAgent
import logging
from app.services.vector_store import VectorStoreService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class AgentState(Dict):
    query: str
    context: List[Document]
    answer: str
    should_rewrite: bool
    rewrite_count: int
    answers: List[str]

class JobAdvisorAgent:
    def __init__(self, llm, vector_store: VectorStoreService):
        self.llm = llm
        self.vector_store = vector_store
        self.chat_agent = ChatAgent(llm)
        print("[JobAdvisorAgent.__init__] 초기화 시작")
        self.workflow = self.setup_workflow()
        print("[JobAdvisorAgent.__init__] 초기화 완료")
        
        # 구직 관련 키워드
        self.job_keywords = {
            'position': ['일자리', '직장', '취업', '채용', '구직', '일', '직업', '알바', '아르바이트', '정규직', '계약직'],
            'salary': ['급여', '월급', '연봉', '시급', '주급', '임금'],
            'location': ['지역', '근처', '가까운', '동네', '시', '구', '동'],
            'time': ['시간', '근무시간', '근무요일', '요일', '주말', '평일'],
            'type': ['경비', '운전', '청소', '요양', '간호', '주방', '조리', '판매', '영업', '사무', '관리', '생산', '제조']
        }

        # 기본 프롬프트 템플릿 설정
        self.chat_template = ChatPromptTemplate.from_messages([
            ("system", "당신은 구직자를 돕는 전문 취업 상담사입니다."),
            ("user", "{query}")
        ])

    def is_job_related(self, query: str) -> bool:
        """구직 관련 키워드가 포함되어 있는지 확인"""
        query_lower = query.lower()
        result = any(
            keyword in query_lower
            for keywords in self.job_keywords.values()
            for keyword in keywords
        )
        print(f"[JobAdvisorAgent.is_job_related] Query: '{query}' / Job 관련 여부: {result}")
        return result

    def retrieve(self, state: AgentState):
        query = state['query']
        logger.info(f"[JobAdvisor] retrieve 시작 - 쿼리: {query}")
        
        if not self.is_job_related(query):
            # 일상 대화 처리 -> LLM으로 전달 -> 구직 관련 대화 유도
            response = self.chat_agent.chat(query)
            return {
                'answer': response,
                'is_job_query': False,
                'context': [],
                'query': query
            }
        
        logger.info("[JobAdvisor] 채용정보 검색 시작")
        try:
            # 직접 검색 수행 (필터 없이)
            results = self.vector_store.search_jobs(
                query=query,
                top_k=10
            )
            logger.info(f"[JobAdvisor] 검색 결과 수: {len(results)}")
            
            if results:
                context_str = "\n\n".join([
                    f"제목: {doc.metadata.get('채용제목', '')}\n"
                    f"회사: {doc.metadata.get('회사명', '')}\n"
                    f"지역: {doc.metadata.get('근무지역', '')}\n"
                    f"급여: {doc.metadata.get('급여조건', '')}\n"
                    f"상세내용: {doc.page_content}"
                    for doc in results
                ])
                
                logger.info("[JobAdvisor] RAG Chain 실행")
                rag_chain = generate_prompt | self.llm | StrOutputParser()
                response = rag_chain.invoke({
                    "question": query,
                    "context": context_str
                })
                logger.info("[JobAdvisor] 응답 생성 완료")
                
                return {
                    'answer': response,
                    'is_job_query': True,
                    'context': results,
                    'query': query
                }
                
        except Exception as e:
            logger.error(f"[JobAdvisor] 검색 중 에러 발생: {str(e)}", exc_info=True)
            
        return {
            'answer': "죄송합니다. 관련된 구인정보를 찾지 못했습니다. 다른 지역이나 직종으로 검색해보시겠어요? 어떤 종류의 일자리를 찾고 계신지 말씀해 주시면 제가 도와드리겠습니다. 😊",
            'is_job_query': True,
            'context': [],
            'query': query
        }

    def verify(self, state: AgentState) -> dict:
        print(f"[JobAdvisorAgent.verify] 시작 - State: {state}")
        if state.get('is_greeting', False):
            print("[JobAdvisorAgent.verify] 인사 상태 감지 - Rewrite 생략")
            return {
                "should_rewrite": False,
                "rewrite_count": 0,
                "answers": state.get('answers', [])
            }
            
        context = state['context']
        query = state['query']
        rewrite_count = state.get('rewrite_count', 0)
        
        if rewrite_count >= 3:
            print(f"[JobAdvisorAgent.verify] Rewrite 횟수 초과: {rewrite_count}")
            return {
                "should_rewrite": False,
                "rewrite_count": rewrite_count
            }
        
        try:
            verify_chain = verify_prompt | self.llm | StrOutputParser()
            response = verify_chain.invoke({
                "query": query,
                "context": "\n\n".join([str(doc) for doc in context])
            })
            print(f"[JobAdvisorAgent.verify] verify_chain 응답: {response}")
        except Exception as e:
            print(f"[JobAdvisorAgent.verify] verify_chain 호출 중 에러: {str(e)}")
            raise e
        
        return {
            "should_rewrite": "NO" in response.upper(),
            "rewrite_count": rewrite_count + 1,
            "answers": state.get('answers', [])
        }

    def rewrite(self, state: AgentState):
        try:
            logger.info("[JobAdvisor] rewrite 시작")
            query = state['query']
            
            # 직접 텍스트 변환 수행
            rewritten_query = query
            for old_word, new_word in DICTIONARY.items():
                rewritten_query = rewritten_query.replace(old_word, new_word)
            
            if rewritten_query != query:
                logger.info(f"[JobAdvisor] 쿼리 변경: {query} -> {rewritten_query}")
            else:
                logger.info("[JobAdvisor] 변경 필요 없음")
            
            return {"query": rewritten_query}
            
        except Exception as e:
            logger.error(f"[JobAdvisor] rewrite 에러: {str(e)}", exc_info=True)
            return {"query": state['query']}  # 에러 시 원본 쿼리 반환

    def generate(self, state: AgentState) -> dict:
        query = state['query']
        context = state.get('context', [])
        print(f"[JobAdvisorAgent.generate] 시작 - Query: {query} / Context 문서 수: {len(context)}")
        
        if state.get('is_basic_question', False):
            custom_response = state.get('custom_response')
            print(f"[JobAdvisorAgent.generate] 기본 질문 감지, custom_response: {custom_response}")
            if custom_response:
                return {'answer': custom_response, 'answers': [custom_response]}
        
        if not context:
            default_message = (
                "죄송합니다. 관련된 구인정보를 찾지 못했습니다. "
                "다른 지역이나 직종으로 검색해보시겠어요? 어떤 종류의 일자리를 찾고 계신지 말씀해 주시면 제가 도와드리겠습니다. 😊"
            )
            print(f"[JobAdvisorAgent.generate] Context 없음 - 기본 메시지 반환: {default_message}")
            return {'answer': default_message, 'answers': []}
            
        try:
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
            print(f"[JobAdvisorAgent.generate] rag_chain 응답: {response}")
        except Exception as e:
            print(f"[JobAdvisorAgent.generate] rag_chain 호출 중 에러: {str(e)}")
            raise e
        return {'answer': response, 'answers': [response]}

    def router(self, state: AgentState) -> str:
        print(f"[JobAdvisorAgent.router] 시작 - State: {state}")
        if state.get('is_basic_question', False):
            print("[JobAdvisorAgent.router] 기본 질문 감지 - generate로 라우팅")
            return "generate"
        if state.get('context', []):
            print("[JobAdvisorAgent.router] Context 존재 - verify로 라우팅")
            return "verify"
        print("[JobAdvisorAgent.router] Context 없음 - generate로 라우팅")
        return "generate"

    def setup_workflow(self):
        """워크플로우 설정"""
        workflow = StateGraph(AgentState)
        
        # 노드 추가
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("verify", self.verify)
        workflow.add_node("rewrite", self.rewrite)
        workflow.add_node("generate", self.generate)
        
        # 엣지 설정
        workflow.add_edge("retrieve", "verify")
        workflow.add_edge("verify", "rewrite")
        workflow.add_edge("verify", "generate")
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_edge("generate", END)
        
        # 시작점 설정
        workflow.set_entry_point("retrieve")
        
        return workflow.compile()

    async def chat(self, query: str, user_profile: dict = None) -> str:
        try:
            logger.info(f"[JobAdvisor] chat 시작 - 쿼리: {query}")
            logger.info(f"[JobAdvisor] 사용자 프로필: {user_profile}")
            
            base_response = self.chat_agent.chat(query)
            logger.info("[JobAdvisor] 기본 응답 생성 완료")
            
            if not self.is_job_related(query):
                logger.info("[JobAdvisor] 일반 대화로 판단됨")
                follow_up = "\n\n혹시 어떤 일자리를 찾고 계신가요? 선호하시는 근무지역이나 직무가 있으시다면 말씀해 주세요. 😊"
                return base_response + follow_up
            
            logger.info("[JobAdvisor] 채용정보 검색 시작")
            try:
                results = self.vector_store.search_jobs(
                    query=query,
                    top_k=10
                )
                logger.info(f"[JobAdvisor] 검색 결과 수: {len(results)}")
            except Exception as search_error:
                logger.error(f"[JobAdvisor] 검색 중 에러 발생: {str(search_error)}", exc_info=True)
                raise
            
            if not results:
                logger.info("[JobAdvisor] 검색 결과 없음")
                return base_response + "\n\n현재 조건에 맞는 채용정보를 찾지 못했습니다. 다른 조건으로 찾아보시겠어요?"
            
            # 4. 검색된 문서로 컨텍스트 생성
            context = "\n\n".join([
                f"제목: {doc.metadata.get('채용제목', '')}\n"
                f"회사: {doc.metadata.get('회사명', '')}\n"
                f"지역: {doc.metadata.get('근무지역', '')}\n"
                f"급여: {doc.metadata.get('급여조건', '')}\n"
                f"상세내용: {doc.page_content}"
                for doc in results
            ])
            
            # 5. 채용정보 기반 추가 응답 생성
            generate_chain = generate_prompt | self.llm | StrOutputParser()
            job_response = generate_chain.invoke({
                "question": query,
                "context": context
            })
            
            # 6. 기본 응답과 채용정보 응답 결합
            return f"{base_response}\n\n관련 채용정보를 찾아보았습니다:\n{job_response}"
            
        except Exception as e:
            logger.error(f"[JobAdvisor] 전체 처리 중 에러 발생: {str(e)}", exc_info=True)
            return "죄송합니다. 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요." 