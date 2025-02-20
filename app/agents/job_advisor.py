import logging
import os
import json
from typing import Dict, List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.schema.runnable import Runnable
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate

from app.core.prompts import verify_prompt, rewrite_prompt, generate_prompt, chat_prompt, EXTRACT_INFO_PROMPT


from app.services.vector_store_search import VectorStoreSearch
from app.services.document_filter import DocumentFilter
from app.utils.constants import LOCATIONS, AREA_CODES, SEOUL_DISTRICT_CODES, JOB_SYNONYMS

logger = logging.getLogger(__name__)

###############################################################################
# AgentState
###############################################################################
class AgentState(Dict):
    query: str
    context: List[Document]
    answer: str
    should_rewrite: bool
    rewrite_count: int
    answers: List[str]
    user_profile: Dict[str, str]  # 예: {"age":"", "location":"", "jobType":""}


###############################################################################
# JobAdvisorAgent
###############################################################################
class JobAdvisorAgent:
    def __init__(self, llm, vector_search: VectorStoreSearch):
        """
        llm: OpenAI LLM
        vector_search: 다단계 검색을 수행할 VectorStoreSearch 객체
        """
        self.llm = llm
        self.vector_search = vector_search
        self.workflow = self.setup_workflow()
        self.document_filter = DocumentFilter()
        self.chat_template = chat_prompt

    ###############################################################################
    # (A) NER 추출용 함수
    ###############################################################################
    def get_user_ner_runnable(self) -> Runnable:
        """
        사용자 입력 예: "서울 요양보호사"
        -> LLM이 아래와 같이 JSON으로 추출:
           {"직무": "요양보호사", "지역": "서울", "연령대": ""}
        """
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-4o-mini",
            temperature=0.0
        )

        return EXTRACT_INFO_PROMPT | llm

    async def _extract_location_info(self, query: str) -> Dict[str, str]:
        """상세 지역 정보 추출 함수"""
        try:
            # 1. 시/도 및 구/군 매칭
            for city in LOCATIONS:
                if city in query:
                    # 서울특별시 케이스
                    if city == "서울":
                        for district in SEOUL_DISTRICT_CODES.keys():
                            if district in query:
                                return {"city": city, "district": district}
                        return {"city": city, "district": ""}
                    
                    # 다른 광역시/도 케이스
                    for district in AREA_CODES.get(city, []):
                        if district in query:
                            return {"city": city, "district": district}
                    return {"city": city, "district": ""}

            # 2. 구/군/동만 언급된 경우 역매핑
            for city, districts in AREA_CODES.items():
                for district in districts:
                    if district in query:
                        return {"city": city, "district": district}

            # 3. 서울 구/동만 언급된 경우
            for district in SEOUL_DISTRICT_CODES.keys():
                if district in query:
                    return {"city": "서울", "district": district}

            # 4. 매칭 실패 시 LLM에게 분석 요청
            return await self._analyze_location_with_llm(query)

        except Exception as e:
            logger.error(f"[JobAdvisor] 지역 정보 추출 중 에러: {str(e)}")
            return {"city": "", "district": ""}

    async def _analyze_location_with_llm(self, query: str) -> Dict[str, str]:
        """LLM을 사용한 지역 정보 분석"""
        try:
            location_prompt = PromptTemplate.from_template("""
            다음 텍스트에서 언급된 지역을 한국의 행정구역 체계에 맞춰 분석해주세요.
            
            텍스트: {query}

            규칙:
            1. 동/읍/면이 언급된 경우 해당 지역이 속한 시/도와 구/군까지 파악
            2. 약칭이나 옛 지명도 현재 행정구역으로 변환
            3. 지역을 특정할 수 없는 경우 빈 문자열 반환

            다음 JSON 형식으로 응답:
            {{"city": "시/도명", "district": "구/군/구명"}}
            """)

            chain = location_prompt | self.llm | StrOutputParser()
            result = await chain.ainvoke({"query": query})
            
            try:
                location_info = json.loads(result)
                logger.info(f"[JobAdvisor] LLM 지역 분석 결과: {location_info}")
                return location_info
            except json.JSONDecodeError:
                return {"city": "", "district": ""}

        except Exception as e:
            logger.error(f"[JobAdvisor] LLM 지역 분석 중 에러: {str(e)}")
            return {"city": "", "district": ""}

    async def _extract_user_ner(self, query: str, user_profile: Dict = None, chat_history: str = None) -> Dict[str, str]:
        """사용자 입력에서 NER 정보 추출"""
        try:
            # 1. 먼저 사전을 이용한 직접 매칭 시도
            extracted_info = {"지역": "", "직무": "", "연령대": ""}
            
            # 여기서 await 추가
            location_info = await self._extract_location_info(query)
            if location_info["city"]:
                extracted_info["지역"] = f"{location_info['city']} {location_info['district']}".strip()
            
            # 3. 직무 매칭 및 유의어 확인
            words = query.split()
            for word in words:
                if word in JOB_SYNONYMS:
                    extracted_info["직무"] = word
                    break
                # 유의어 검색
                for job, synonyms in JOB_SYNONYMS.items():
                    if word in synonyms:
                        extracted_info["직무"] = job
                        break

            # 4. 사전 매칭으로 충분한 정보를 얻지 못한 경우 LLM 사용
            if not (extracted_info["지역"] or extracted_info["직무"]):
                # 여기만 수정: await 추가
                llm_response = await self._analyze_location_with_llm(query)
                
                # LLM 결과와 사전 매칭 결과 병합
                if not extracted_info["지역"] and llm_response.get("지역"):
                    extracted_info["지역"] = llm_response["지역"]
                if not extracted_info["직무"] and llm_response.get("직무"):
                    extracted_info["직무"] = llm_response["직무"]
                if llm_response.get("연령대"):
                    extracted_info["연령대"] = llm_response["연령대"]
            
            # 5. 사용자 프로필 정보로 보완
            if user_profile:
                if not extracted_info["지역"] and user_profile.get("location"):
                    extracted_info["지역"] = user_profile["location"]
                if not extracted_info["직무"] and user_profile.get("jobType"):
                    extracted_info["직무"] = user_profile["jobType"]
                if not extracted_info["연령대"] and user_profile.get("age"):
                    extracted_info["연령대"] = user_profile["age"]

            logger.info(f"[JobAdvisor] 최종 추출 정보: {extracted_info}")
            return extracted_info
            
        except Exception as e:
            logger.error(f"[JobAdvisor] NER 추출 중 오류: {str(e)}")
            return {"지역": "", "직무": "", "연령대": ""}

    ###############################################################################
    # (B) 일반 대화/채용정보 검색 라우팅
    ###############################################################################
    async def retrieve(self, state: AgentState):
        query = state['query']
        logger.info(f"[JobAdvisor] retrieve 시작 - 쿼리: {query}")

        # (1) 일반 대화 체크
        if not self.is_job_related(query):
            # 일상 대화 처리 -> LLM으로 전달 -> 구직 관련 대화 유도
            response = await self.chat_agent.chat(query)
            return {
                # ChatResponse 호환 형태
                "message": response,  # answer
                "jobPostings": [],
                "type": "info",
                "user_profile": state.get("user_profile", {}),
                "context": [],
                "query": query
            }

        # (2) job 검색
        logger.info("[JobAdvisor] 채용정보 검색 시작")
        user_profile = state.get("user_profile", {})
        user_ner = await self._extract_user_ner(query, user_profile)

        try:
            # 기존 검색 결과가 있는지 확인
            previous_results = user_profile.get('previous_results', []) if user_profile else []
            
            # 벡터 검색 수행
            search_results = self.vector_search.search_jobs(
                user_ner=user_ner,
                top_k=10
            )
            
            # 이전 결과와 중복 제거
            search_results = [doc for doc in search_results if doc not in previous_results]

            logger.info(f"[JobAdvisor] 검색 결과 수: {len(search_results)}")
        except Exception as e:
            logger.error(f"[JobAdvisor] 검색 중 에러 발생: {str(e)}", exc_info=True)
            return {
                "message": "죄송합니다. 검색 중 오류가 발생했습니다.",
                "jobPostings": [],
                "type": "info",
                "user_profile": user_profile,
                "context": [],
                "query": query
            }

        # (3) 최대 5건만 추출
        top_docs = search_results[:2]

        # (4) Document -> JobPosting 변환
        job_postings = []
        for i, doc in enumerate(top_docs, start=1):
            md = doc.metadata
            job_postings.append({
                "id": md.get("채용공고ID", f"no_id_{i}"),
                "location": md.get("근무지역", "위치 정보 없음"),
                "company": md.get("회사명", "회사명 없음"),
                "title": md.get("채용제목", "제목 없음"),
                "salary": md.get("급여조건", "급여 정보 없음"),
                "workingHours": md.get("근무시간", "근무시간 정보 없음"),
                "description": md.get("상세정보", doc.page_content[:500]) or "상세내용 정보 없음",
                "phoneNumber": md.get("전화번호", "전화번호 정보 없음"),
                "deadline": md.get("접수마감일", "마감일 정보 없음"),
                "requiredDocs": md.get("제출서류", "제출서류 정보 없음"),
                "hiringProcess": md.get("전형방법", "전형방법 정보 없음"),
                "insurance": md.get("사회보험", "사회보험 정보 없음"),
                "jobCategory": md.get("모집직종", "모집직종 정보 없음"),
                "jobKeywords": md.get("직종키워드", "직종키워드 정보 없음"),
                "posting_url": md.get("채용공고URL", "채용공고URL 정보 없음"),
                "rank": i
            })

        # (5) 메시지 / 타입
        if job_postings:
            msg = f"'{query}' 검색 결과, 상위 {len(job_postings)}건을 반환합니다."
            res_type = "jobPosting"
        else:
            msg = "조건에 맞는 채용공고를 찾지 못했습니다."
            res_type = "info"

        # (6) ChatResponse 호환 dict
        return {
            "message": msg,
            "jobPostings": job_postings,
            "type": res_type,
            "user_profile": user_profile,
            "context": search_results,  # 다음 노드(verify 등)에서 사용
            "query": query
        }

    ###############################################################################
    # (C) 이하 verify, rewrite, generate 등은 기존 로직 그대로
    ###############################################################################
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
        """쿼리 재작성"""
        if state.get('is_greeting', False):
            return {"answer": state['query']}
            
        try:
            # prompts.py의 rewrite_prompt 사용
            rewrite_chain = rewrite_prompt | self.llm | StrOutputParser()
            answer = rewrite_chain.invoke({
                "original_query": state['query'],
                "transformed_query": state['query']  # 변환된 쿼리
            })
            return {"answer": answer.strip()}
        except Exception as e:
            logger.error(f"[JobAdvisor] 쿼리 재작성 중 오류: {str(e)}")
            return {"answer": state['query']}

    def generate(self, state: AgentState) -> dict:
        """응답 생성"""
        try:
            # prompts.py의 generate_prompt 사용
            generate_chain = generate_prompt | self.llm | StrOutputParser()
            answer = generate_chain.invoke({
                "question": state['query'],
                "context": "\n".join([doc.page_content for doc in state['context']])
            })
            return {"answer": answer.strip()}
        except Exception as e:
            logger.error(f"[JobAdvisor] 응답 생성 중 오류: {str(e)}")
            return {"answer": "죄송합니다. 응답을 생성하는 중에 문제가 발생했습니다."}

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
        
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("verify", self.verify)
        workflow.add_node("rewrite", self.rewrite)
        workflow.add_node("generate", self.generate)
        
        workflow.add_edge("retrieve", "verify")
        workflow.add_conditional_edges("verify", self.router)
        workflow.add_conditional_edges("rewrite", self.router)
        workflow.add_edge("generate", END)
        
        workflow.set_entry_point("retrieve")
        
        return workflow.compile()

    async def handle_job_query(self, query: str, user_profile: Dict = None, chat_history: str = None) -> Dict:
        """채용정보 검색 처리"""
        try:
            # NER 추출 (동기 함수 호출)
            user_ner = await self._extract_user_ner(query, user_profile, chat_history)
            
            # 2. 검색 수행
            job_postings = []
            if user_ner.get("직무") or user_ner.get("지역"):
                # 기존 검색 결과가 있는지 확인
                previous_results = user_profile.get('previous_results', []) if user_profile else []
                
                # 벡터 검색 수행
                search_results = self.vector_search.search_jobs(
                    user_ner=user_ner,
                    top_k=10
                )
                
                # 이전 결과와 중복 제거
                search_results = [doc for doc in search_results if doc not in previous_results]
                
                # 검색 결과를 job_postings 형식으로 변환
                for i, doc in enumerate(search_results, start=1):
                    try:
                        md = doc.metadata
                        posting = {
                            "id": md.get("채용공고ID", f"no_id_{i}"),
                            "location": md.get("근무지역", "위치 정보 없음"),
                            "company": md.get("회사명", "회사명 없음"),
                            "title": md.get("채용제목", "제목 없음"),
                            "salary": md.get("급여조건", "급여 정보 없음"),
                            "workingHours": md.get("근무시간", "근무시간 정보 없음"),
                            "description": md.get("상세정보", doc.page_content[:500] if hasattr(doc, 'page_content') else "상세내용 정보 없음"),
                            "phoneNumber": md.get("전화번호", "전화번호 정보 없음"),
                            "deadline": md.get("접수마감일", "마감일 정보 없음"),
                            "requiredDocs": md.get("제출서류", "제출서류 정보 없음"),
                            "hiringProcess": md.get("전형방법", "전형방법 정보 없음"),
                            "insurance": md.get("사회보험", "사회보험 정보 없음"),
                            "jobCategory": md.get("모집직종", "모집직종 정보 없음"),
                            "jobKeywords": md.get("직종키워드", "직종키워드 정보 없음"),
                            "posting_url": md.get("채용공고URL", "채용공고URL 정보 없음"),
                            "rank": i
                        }
                        job_postings.append(posting)
                    except Exception as doc_error:
                        logger.error(f"[JobAdvisor] 문서 {i} 처리 중 에러: {str(doc_error)}")
                        continue

            if not job_postings:
                return {
                    "message": "현재 조건에 맞는 채용정보를 찾지 못했습니다. 다른 조건으로 찾아보시겠어요?",
                    "jobPostings": [],
                    "type": "info",
                    "user_profile": user_profile
                }

            # 현재 결과를 user_profile에 저장
            if user_profile is not None:
                user_profile['previous_results'] = job_postings

            # 채용 정보에 대한 전문적인 설명 생성
            job_explanation_chain = chat_prompt | self.llm | StrOutputParser()
            job_explanation = job_explanation_chain.invoke({
                "query": f"다음 채용정보들을 전문 취업상담사의 입장에서 설명해주세요. 지원자가 고려해야 할 점과 준비사항도 알려주세요: {[job['title'] for job in job_postings]}"
            })

            return {
                "message": job_explanation.strip(),
                "jobPostings": job_postings,
                "type": "jobPosting",
                "user_profile": user_profile
            }

        except Exception as e:
            logger.error(f"[JobAdvisor] 채팅 처리 중 에러: {str(e)}", exc_info=True)
            return {
                "message": "죄송합니다. 요청을 처리하는 중에 문제가 발생했습니다.",
                "type": "error",
                "jobPostings": [],
                "user_profile": user_profile
            }

    # async def chat(
    #     self, 
    #     message: str, 
    #     user_profile: Dict = None,
    #     chat_history: List[Dict] = None
    # ) -> Dict:
    #     """사용자 메시지에 대한 응답 생성"""
    #     try:
    #         # 훈련과정 검색 의도 확인
    #         if any(keyword in message for keyword in ["훈련", "교육", "과정", "배우고"]):
    #             return await self.training_advisor.search_training_courses(
    #                 query=message,
    #                 user_profile=user_profile,
    #                 chat_history=chat_history
    #             )
            
    #         # 기타 일반적인 대화 처리
    #         # ... 기존 대화 처리 로직 ...
            
    #     except Exception as e:
    #         logger.error(f"[JobAdvisor] 채팅 처리 중 오류: {str(e)}", exc_info=True)
    #         return {
    #             "message": "죄송합니다. 요청을 처리하는 중에 문제가 발생했습니다.",
    #             "type": "error"
    #         }