from typing import Dict, List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from app.core.prompts import verify_prompt, rewrite_prompt, generate_prompt, chat_prompt
from app.utils.constants import LOCATIONS, DICTIONARY
from app.agents.chat_agent import ChatAgent
from app.services.vector_store_search import VectorStoreSearch
import logging
import os
import json

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import Runnable

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
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
        self.chat_agent = ChatAgent(llm)
        self.workflow = self.setup_workflow()

        # 구직 관련 키워드
        self.job_keywords = {
            "position": [
                "일자리",
                "직장",
                "취업",
                "채용",
                "구직",
                "일",
                "직업",
                "알바",
                "아르바이트",
                "정규직",
                "계약직",
            ],
            "salary": ["급여", "월급", "연봉", "시급", "주급", "임금"],
            "location": ["지역", "근처", "가까운", "동네", "시", "구", "동"],
            "time": ["시간", "근무시간", "근무요일", "요일", "주말", "평일"],
            "type": [
                "경비",
                "운전",
                "청소",
                "요양",
                "간호",
                "주방",
                "조리",
                "판매",
                "영업",
                "사무",
                "관리",
                "생산",
                "제조",
            ],
        }

        # 기본 프롬프트 템플릿
        self.chat_template = ChatPromptTemplate.from_messages(
            [
                ("system", "당신은 구직자를 돕는 전문 취업 상담사입니다."),
                ("user", "{query}"),
            ]
        )

    def is_job_related(self, query: str) -> bool:
        """구직 관련 키워드가 포함되어 있는지 확인"""
        query_lower = query.lower()
        return any(
            keyword in query_lower
            for keywords in self.job_keywords.values()
            for keyword in keywords
        )

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
            openai_api_key=openai_api_key, model_name="gpt-4o-mini", temperature=0.0
        )

        prompt = PromptTemplate(
            input_variables=["user_query"],
            template=(
                "사용자 입력: {user_query}\n\n"
                "아래 항목을 JSON으로 추출 (값이 없으면 빈 문자열로):\n"
                "- 직무\n"
                "- 지역\n"
                "- 연령대\n\n"
                "예:\n"
                "json\n"
                '{{"직무": "요양보호사", "지역": "서울", "연령대": ""}}\n'
                "\n"
            ),
        )
        return prompt | llm

    def _extract_user_ner(
        self, user_message: str, user_profile: Dict[str, str]
    ) -> Dict[str, str]:
        """
        (1) 사용자 입력 NER 추출
        (1-1) NER 데이터가 없거나 누락된 항목은 user_profile (age, location, jobType)로 보완
        """
        # 1) 사용자 입력 NER
        ner_chain = self.get_user_ner_runnable()
        ner_res = ner_chain.invoke({"user_query": user_message})
        ner_str = ner_res.content if hasattr(ner_res, "content") else str(ner_res)
        cleaned = ner_str.replace("```json", "").replace("```", "").strip()

        try:
            user_ner = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"[JobAdvisor] NER parse fail: {cleaned}")
            user_ner = {}

        logger.info(f"[JobAdvisor] 1) user_ner={user_ner}")

        # 1-1) 프로필 보완
        # user_profile: {"age":"", "location":"", "jobType":""}
        if not user_ner.get("직무") and user_profile.get("jobType"):
            user_ner["직무"] = user_profile["jobType"]
        if not user_ner.get("지역") and user_profile.get("location"):
            user_ner["지역"] = user_profile["location"]
        if not user_ner.get("연령대") and user_profile.get("age"):
            user_ner["연령대"] = user_profile["age"]

        logger.info(f"[JobAdvisor] 1-1) 보완된 user_ner={user_ner}")
        return user_ner

    ###############################################################################
    # (B) 일반 대화/채용정보 검색 라우팅
    ###############################################################################
    def retrieve(self, state: AgentState):
        query = state["query"]
        logger.info(f"[JobAdvisor] retrieve 시작 - 쿼리: {query}")

        # (1) 일반 대화 체크
        if not self.is_job_related(query):
            logger.info("[JobAdvisor] 일반 대화로 판단")
            response = self.chat_agent.chat(query)
            return {
                # ChatResponse 호환 형태
                "message": response,  # answer
                "jobPostings": [],
                "type": "info",
                "user_profile": state.get("user_profile", {}),
                "context": [],
                "query": query,
            }

        # (2) job 검색
        logger.info("[JobAdvisor] 채용정보 검색 시작")
        user_profile = state.get("user_profile", {})
        user_ner = self._extract_user_ner(query, user_profile)

        try:
            results = self.vector_search.search_jobs(user_ner=user_ner, top_k=10)
            logger.info(f"[JobAdvisor] 검색 결과 수: {len(results)}")
        except Exception as e:
            logger.error(f"[JobAdvisor] 검색 중 에러 발생: {str(e)}", exc_info=True)
            return {
                "message": "죄송합니다. 검색 중 오류가 발생했습니다.",
                "jobPostings": [],
                "type": "info",
                "user_profile": user_profile,
                "context": [],
                "query": query,
            }

        # (3) 최대 5건만 추출
        top_docs = results[:5]

        # (4) Document -> JobPosting 변환
        job_postings = []
        for i, doc in enumerate(top_docs, start=1):
            md = doc.metadata
            job_postings.append(
                {
                    "id": md.get("채용공고ID", "no_id"),
                    "location": md.get("근무지역", ""),
                    "company": md.get("회사명", ""),
                    "title": md.get("채용제목", ""),
                    "salary": md.get("급여조건", ""),
                    "workingHours": md.get("근무시간", "정보없음"),
                    "description": md.get("상세정보", doc.page_content[:200]),
                    "rank": i,
                }
            )

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
            "context": results,  # 다음 노드(verify 등)에서 사용
            "query": query,
        }

    ###############################################################################
    # (C) 이하 verify, rewrite, generate 등은 기존 로직 그대로
    ###############################################################################
    def verify(self, state: AgentState) -> dict:
        if state.get("is_greeting", False):
            return {
                "should_rewrite": False,
                "rewrite_count": 0,
                "answers": state.get("answers", []),
            }

        context = state["context"]
        query = state["query"]
        rewrite_count = state.get("rewrite_count", 0)

        if rewrite_count >= 3:
            return {"should_rewrite": False, "rewrite_count": rewrite_count}

        verify_chain = verify_prompt | self.llm | StrOutputParser()
        response = verify_chain.invoke(
            {"query": query, "context": "\n\n".join([str(doc) for doc in context])}
        )

        return {
            "should_rewrite": "NO" in response.upper(),
            "rewrite_count": rewrite_count + 1,
            "answers": state.get("answers", []),
        }

    def rewrite(self, state: AgentState):
        try:
            logger.info("[JobAdvisor] rewrite 시작")
            query = state["query"]

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
            return {"query": state["query"]}  # 에러 시 원본 쿼리 반환

    def generate(self, state: AgentState) -> dict:
        query = state["query"]
        context = state.get("context", [])

        # 1) jobPostings (이미 retrieve에서 만든 5건)
        job_postings = state.get("jobPostings", [])

        if not context or not job_postings:
            return {
                "message": "죄송합니다. 관련된 구인정보를 찾지 못했습니다.",
                "jobPostings": [],
                "type": "info",
                "user_profile": state.get("user_profile", {}),
            }

        # 2) RAG 프롬프트
        rag_chain = generate_prompt | self.llm | StrOutputParser()
        doc_text = "\n\n".join(
            [
                f"제목: {doc.metadata.get('채용제목', '')}\n"
                f"회사: {doc.metadata.get('회사명', '')}\n"
                f"지역: {doc.metadata.get('근무지역', '')}\n"
                f"급여: {doc.metadata.get('급여조건', '')}\n"
                f"상세내용: {doc.page_content}"
                for doc in context[:5]  # 혹은 job_postings의 길이
            ]
        )
        response_text = rag_chain.invoke({"question": query, "context": doc_text})

        return {
            "message": f"최종 답변:\n{response_text}",
            "jobPostings": job_postings,  # retrieve에서 만든 것 재사용
            "type": "jobPosting",
            "user_profile": state.get("user_profile", {}),
        }

    def router(self, state: AgentState) -> str:
        # 기본 대화나 인사인 경우 바로 generate로
        if state.get("is_basic_question", False):
            return "generate"

        # 검색 결과가 있으면 verify로
        if state.get("context", []):
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
        workflow.add_edge("verify", "rewrite")
        workflow.add_edge("verify", "generate")
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_edge("generate", END)

        workflow.set_entry_point("retrieve")

        return workflow.compile()

    async def chat(self, query: str, user_profile: dict = None) -> dict:
        """
        일반 대화 vs. 채용정보 검색:
        - 최대 5건만 jobPostings에 담음
        - RAG 프롬프트 결과(문자열)와 함께 message에 통합
        - 최종적으로 ChatResponse와 호환되는 dict 반환
        """
        try:
            logger.info(f"[JobAdvisor] chat 시작 - 쿼리: {query}")
            logger.info(f"[JobAdvisor] 사용자 프로필: {user_profile}")

            user_profile = user_profile or {}

            # (A) 일반 대화 판단
            if not self.is_job_related(query):
                logger.info("[JobAdvisor] 일반 대화로 판단")
                # 일반 대화 시, 간단 메시지만 반환
                return {
                    "message": "구직 관련 문의가 아니네요. 어떤 일자리를 찾으시는지 말씀해주시면 도와드리겠습니다. 😊",
                    "jobPostings": [],
                    "type": "info",
                    "user_profile": user_profile,
                }

            # (B) 채용정보 검색
            logger.info("[JobAdvisor] 채용정보 검색 시작")
            user_ner = self._extract_user_ner(query, user_profile)

            try:
                results = self.vector_search.search_jobs(user_ner, top_k=10)
                logger.info(f"[JobAdvisor] 검색 결과 수: {len(results)}")
            except Exception as search_error:
                logger.error(
                    f"[JobAdvisor] 검색 중 에러 발생: {str(search_error)}",
                    exc_info=True,
                )
                # 오류 시 빈 jobPostings
                return {
                    "message": "죄송합니다. 검색 중 오류가 발생했습니다.",
                    "jobPostings": [],
                    "type": "error",
                    "user_profile": user_profile,
                }

            if not results:
                # 결과 없음
                return {
                    "message": "현재 조건에 맞는 채용정보를 찾지 못했습니다. 다른 조건으로 찾아보시겠어요?",
                    "jobPostings": [],
                    "type": "info",
                    "user_profile": user_profile,
                }

            # (C) 최대 5건 추출
            top_docs = results[:5]

            # (D) Document -> JobPosting 변환
            job_postings = []
            for i, doc in enumerate(top_docs, start=1):
                md = doc.metadata
                job_postings.append(
                    {
                        "id": md.get("채용공고ID", "no_id"),
                        "location": md.get("근무지역", ""),
                        "company": md.get("회사명", ""),
                        "title": md.get("채용제목", ""),
                        "salary": md.get("급여조건", ""),
                        "workingHours": md.get("근무시간", "정보없음"),
                        "description": md.get("상세정보", doc.page_content[:300]),
                        "rank": i,
                    }
                )

            # # (E) RAG: generate_prompt로 카드 형태 답변 생성
            # logger.info("[JobAdvisor] RAG Chain 실행")
            # context_str = "\n\n".join([
            #     f"제목: {doc.metadata.get('채용제목', '')}\n"
            #     f"회사: {doc.metadata.get('회사명', '')}\n"
            #     f"지역: {doc.metadata.get('근무지역', '')}\n"
            #     f"급여: {doc.metadata.get('급여조건', '')}\n"
            #     f"상세내용: {doc.page_content}"
            #     for doc in top_docs
            # ])
            # rag_chain = generate_prompt | self.llm | StrOutputParser()
            # rag_response = rag_chain.invoke({"question": query, "context": context_str})

            if job_postings:
                msg = f"'{query}' 검색 결과, 상위 {len(job_postings)}건을 반환합니다."
                res_type = "jobPosting"
            else:
                msg = "조건에 맞는 채용공고를 찾지 못했습니다."
                res_type = "info"

            # (F) 최종 메시지: RAG 결과 문자열
            return {
                "message": msg,
                "jobPostings": job_postings,
                "type": res_type,
                "user_profile": user_profile,
            }

        except Exception as e:
            logger.error(
                f"[JobAdvisor] 전체 처리 중 에러 발생: {str(e)}", exc_info=True
            )
            return {
                "message": "죄송합니다. 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                "jobPostings": [],
                "type": "error",
                "user_profile": user_profile,
            }
