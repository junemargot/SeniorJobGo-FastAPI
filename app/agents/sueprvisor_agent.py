from typing import Dict, Optional
import logging
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.agents.job_advisor import JobAdvisorAgent
from app.agents.chat_agent import ChatAgent
from app.agents.training_advisor import TrainingAdvisorAgent

logger = logging.getLogger(__name__)

AGENT_SELECTION_PROMPT = """
당신은 사용자의 입력을 분석하여 적절한 에이전트를 선택하는 Supervisor입니다.
사용자의 입력을 분석하여 다음 중 하나를 선택하세요:

1. job: 채용 정보 검색이나 구직 관련 질문
2. training: 직업훈련, 교육과정 관련 질문
3. general: 일반적인 대화나 기타 문의

사용자 입력: {query}
이전 대화: {chat_history}

다음 형식으로 응답하세요:
agent: [선택한 에이전트 유형]
reason: [선택 이유 간단히]
"""

class SupervisorAgent:
    """에이전트 선택 및 조정을 담당하는 수퍼바이저"""
    
    def __init__(self, llm, job_advisor: JobAdvisorAgent, training_advisor: TrainingAdvisorAgent, chat_agent: ChatAgent):
        self.llm = llm
        self.job_advisor = job_advisor
        self.training_advisor = training_advisor
        self.chat_agent = chat_agent
        
        # 에이전트 선택을 위한 프롬프트 체인 설정
        self.agent_selector = (
            ChatPromptTemplate.from_template(AGENT_SELECTION_PROMPT) 
            | self.llm 
            | StrOutputParser()
        )
        
    def _parse_agent_selection(self, selection: str) -> Dict[str, str]:
        """에이전트 선택 결과 파싱"""
        lines = selection.strip().split('\n')
        result = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                result[key.strip()] = value.strip()
        return result

    def _contains_keywords(self, text: str, keywords: list) -> bool:
        """키워드 포함 여부 확인"""
        return any(keyword in text.lower() for keyword in keywords)

    def _determine_agent_type(self, query: str, chat_history: str = "") -> str:
        """규칙 기반으로 에이전트 타입 결정"""
        # 채용 관련 키워드
        job_keywords = ["채용", "일자리", "구직", "취업", "면접", "이력서", "자기소개서", 
                       "연봉", "급여", "직장", "회사", "구인", "알바", "아르바이트"]
        
        # 훈련 관련 키워드
        training_keywords = ["훈련", "교육", "과정", "학원", "자격증", "수업", "강의", 
                           "배우고", "공부", "수강", "교육비", "훈련비", "내일배움카드"]

        if self._contains_keywords(query, job_keywords):
            return "job"
        elif self._contains_keywords(query, training_keywords):
            return "training"
        
        # LLM 기반 판단
        try:
            selection = self.agent_selector.invoke({
                "query": query,
                "chat_history": chat_history
            })
            result = self._parse_agent_selection(selection)
            return result.get("agent", "general")
        except Exception as e:
            logger.error(f"LLM 기반 에이전트 선택 실패: {str(e)}")
            return "general"

    async def process_query(self, query: str, user_profile: Optional[Dict] = None, 
                          chat_history: str = "") -> Dict:
        """사용자 쿼리 처리"""
        try:
            # 에이전트 타입 결정
            agent_type = self._determine_agent_type(query, chat_history)
            logger.info(f"[SupervisorAgent] 선택된 에이전트 타입: {agent_type}")
            
            # 에이전트별 처리
            if agent_type == "job":
                response = await self.job_advisor.chat(
                    query=query,
                    user_profile=user_profile,
                    chat_history=chat_history
                )
                
            elif agent_type == "training":
                response = await self.training_advisor.search_training_courses(
                    query=query,
                    user_profile=user_profile
                )
                
            else:  # general
                response = await self.chat_agent.handle_general_conversation(
                    query=query,
                    chat_history=chat_history
                )
            
            # 응답에 에이전트 타입 추가
            if isinstance(response, dict):
                response["agent_type"] = agent_type
            else:
                response = {
                    "message": response,
                    "agent_type": agent_type,
                    "type": "general"
                }
            
            return response
            
        except Exception as e:
            logger.error(f"[SupervisorAgent] 쿼리 처리 중 오류: {str(e)}", exc_info=True)
            return {
                "message": "죄송합니다. 요청을 처리하는 중에 문제가 발생했습니다.",
                "type": "error",
                "agent_type": "error"
            }