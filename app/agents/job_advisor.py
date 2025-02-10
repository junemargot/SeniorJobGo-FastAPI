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
        print(f"[JobAdvisorAgent.retrieve] 시작 - Query: {query}")
        
        # 1. 구직 관련 키워드 확인
        if not self.is_job_related(query):
            # 일상 대화 처리 -> chat_persona_prompt와 chat_prompt를 사용하여 구직 관련 대화 유도
            print("[JobAdvisorAgent.retrieve] 구직 관련 키워드가 없음. 일반 대화 처리 시작")
            try:
                chat_chain = (
                    chat_prompt.partial(system=chat_persona_prompt) | 
                    self.llm | 
                    StrOutputParser()
                )
                response = chat_chain.invoke({"input": query})
                print(f"[JobAdvisorAgent.retrieve] chat_chain 응답: {response}")
            except Exception as e:
                print(f"[JobAdvisorAgent.retrieve] chat_chain 호출 중 에러: {str(e)}")
                raise e
            return {
                'answer': response,
                'is_job_query': False,
                'context': [],
                'query': query
            }
        
        # 2. 지역 필터 설정
        user_location = next((loc for loc in LOCATIONS if loc in query), None)
        search_filter = None
        if user_location:
            search_filter = {
                "$or": [
                    {"location": {"$contain": [user_location]}},
                    {"location": {"$contain": [f"{user_location}시"]}},
                    {"location": {"$contain": [f"{user_location}특별시"]}}
                ]
            }
            print(f"[JobAdvisorAgent.retrieve] 지역 필터 적용: {user_location}")
        else:
            print("[JobAdvisorAgent.retrieve] 지역 필터 없음")
        
        # 3. 필터를 적용하여 벡터 검색 수행
        try:
            docs = self.vector_store.similarity_search(query, k=3, filter=search_filter)
            print(f"[JobAdvisorAgent.retrieve] 검색 결과: {len(docs)}개 문서")
        except Exception as e:
            print(f"[JobAdvisorAgent.retrieve] 벡터 검색 중 에러: {str(e)}")
            raise e
            
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
            try:
                rag_chain = generate_prompt | self.llm | StrOutputParser()
                response = rag_chain.invoke({
                    "question": query,
                    "context": context_str
                })
                print(f"[JobAdvisorAgent.retrieve] rag_chain 응답: {response}")
            except Exception as e:
                print(f"[JobAdvisorAgent.retrieve] rag_chain 호출 중 에러: {str(e)}")
                raise e

            return {
                'answer': response,
                'is_job_query': True,
                'context': docs,
                'query': query
            }
        
        # 5. 검색 결과가 없는 경우
        no_result_message = "죄송합니다. 관련된 구인정보를 찾지 못했습니다."
        if user_location:
            no_result_message += f" {user_location} 지역에서 다른 직종을 찾아보시거나, 다른 지역으로 검색해보시는 건 어떨까요?"
        else:
            no_result_message += " 특정 지역이나 직종을 말씀해 주시면 제가 더 잘 찾아드릴 수 있습니다."
        no_result_message += " 어떤 종류의 일자리를 찾고 계신지 말씀해 주시겠어요? 😊"
        print(f"[JobAdvisorAgent.retrieve] 검색 결과 없음 - 응답 메시지: {no_result_message}")
        return {
            'answer': no_result_message,
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

    def rewrite(self, state: AgentState) -> dict:
        query = state['query']
        print(f"[JobAdvisorAgent.rewrite] 시작 - Query: {query}")
        try:
            rewrite_chain = rewrite_prompt | self.llm | StrOutputParser()
            response = rewrite_chain.invoke({'query': query})
            print(f"[JobAdvisorAgent.rewrite] rewrite_chain 응답: {response}")
        except Exception as e:
            print(f"[JobAdvisorAgent.rewrite] rewrite_chain 호출 중 에러: {str(e)}")
            raise e
        return {'query': response}

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
        print("[JobAdvisorAgent.setup_workflow] 워크플로우 설정 시작")
        workflow = StateGraph(AgentState)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_edge("retrieve", END)
        workflow.set_entry_point("retrieve")
        compiled_workflow = workflow.compile()
        print("[JobAdvisorAgent.setup_workflow] 워크플로우 컴파일 완료")
        return compiled_workflow 