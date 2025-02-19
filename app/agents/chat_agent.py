from app.core.prompts import chat_persona_prompt, CLASSIFY_INTENT_PROMPT, chat_prompt
import logging
import os
from langchain_community.tools import DuckDuckGoSearchResults
from typing import List, Dict, Tuple
from datetime import datetime
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage
from app.services.document_filter import DocumentFilter
from app.agents.job_advisor import JobAdvisorAgent
from app.agents.training_advisor import TrainingAdvisorAgent
from langchain_core.output_parsers import StrOutputParser
import json

logger = logging.getLogger(__name__)

class ChatAgent:
    def __init__(self, llm, vector_search):
        """ChatAgent 초기화"""
        self.persona = chat_persona_prompt
        self.document_filter = DocumentFilter()
        self.llm = ChatDeepSeek(
            model_name="deepseek-chat",
            temperature=0.3,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            api_base="https://api.deepseek.com/v1"
        )
        self.search = DuckDuckGoSearchResults(output_format="list")
        
        # JobAdvisor와 TrainingAdvisor 초기화
        self.job_advisor = JobAdvisorAgent(llm, vector_search)
        self.training_advisor = TrainingAdvisorAgent(llm)

    async def classify_intent(self, query: str, chat_history: str = "") -> Tuple[str, float]:
        """사용자 메시지의 의도를 분류합니다."""
        try:
            # 직접적인 키워드 체크
            job_keywords = ["일자리", "채용", "구인", "취업", "직장", "알바", "아르바이트", "일거리", "모집", "자리"]
            training_keywords = ["교육", "훈련", "과정", "학원", "자격증", "수업", "강의", "배우고"]
            
            if any(keyword in query for keyword in job_keywords):
                return "job", 0.9
            if any(keyword in query for keyword in training_keywords):
                return "training", 0.9
            
            # LLM 기반 의도 분류
            chain = CLASSIFY_INTENT_PROMPT | self.llm | StrOutputParser()
            response = chain.invoke({
                "user_query": query,
                "chat_history": chat_history
            })
            
            # JSON 파싱 및 결과 반환
            result = json.loads(response.strip())
            intent = result.get("intent", "general")
            confidence = float(result.get("confidence", 0.5))
            
            return intent, confidence
            
        except Exception as e:
            logger.error(f"[ChatAgent] 의도 분류 중 에러: {str(e)}")
            return "general", 0.5

    async def chat(self, message: str, user_profile: Dict = None, chat_history: List[Dict] = None) -> Dict:
        """사용자 메시지 처리의 단일 진입점"""
        try:
            # 1. 의도 분류
            intent, confidence = await self.classify_intent(message, chat_history)
            logger.info(f"[ChatAgent] 의도 분류 결과 - 의도: {intent}, 확신도: {confidence}")

            # 2. 의도에 따른 처리
            if confidence > 0.6:
                if intent == "job":
                    logger.info("[ChatAgent] JobAdvisor에 요청 위임")
                    return await self.job_advisor.handle_job_query(message, user_profile, chat_history)
                elif intent == "training":
                    logger.info("[ChatAgent] TrainingAdvisor에 요청 위임")
                    return await self.training_advisor.search_training_courses(message, user_profile, chat_history)

            # 3. 일반 대화 처리
            logger.info("[ChatAgent] 일반 대화 처리")
            return await self.handle_general_conversation(message, chat_history)

        except Exception as e:
            logger.error(f"[ChatAgent] 채팅 처리 중 오류: {str(e)}", exc_info=True)
            return {
                "message": "죄송합니다. 요청을 처리하는 중에 문제가 발생했습니다.",
                "type": "error"
            }

    def _search_web(self, query: str, chat_history: str = "") -> List[Dict[str, str]]:
        """웹 검색을 수행하고 결과를 반환합니다."""
        try:
            # 1. 제외 의도 확인
            has_exclusion = self.document_filter.check_exclusion_intent(query, chat_history)
            if has_exclusion:
                logger.info(f"[ChatAgent] 제외 의도 감지됨: {query}")
            
            # 2. 검색 수행
            current_year = datetime.now().year
            enhanced_query = f"{query} latest information {current_year} facts verified"
            results = self.search.invoke(enhanced_query)
            
            # 3. 결과 처리
            all_results = []
            for result in results:
                try:
                    domain = result.get("link", "").split("/")[2].replace("www.", "").replace("m.", "")
                    processed_result = {
                        "title": result.get("title", "").strip(),
                        "link": result.get("link", "").strip(),
                        "source": domain,
                        "snippet": result.get("snippet", "").strip(),
                        "type": "web"
                    }
                    all_results.append(processed_result)
                except Exception as e:
                    logger.error(f"결과 처리 중 오류: {str(e)}")
                    continue

            # 4. 제외 의도가 있는 경우 필터링 적용
            if has_exclusion:
                filtered_results = self.document_filter.filter_documents(all_results)
                logger.info(f"[ChatAgent] 필터링 전: {len(all_results)}건, 필터링 후: {len(filtered_results)}건")
                return filtered_results[:5]
            
            return all_results[:5]

        except Exception as e:
            logger.error(f"[ChatAgent] 웹 검색 중 오류: {str(e)}")
            return []

    def _format_search_results(self, results: List[Dict[str, str]]) -> str:
        """Format search results for prompt inclusion."""
        if not results:
            return ""
        
        formatted = "\n### Reference Information:\n"
        for r in results:
            formatted += f"\n#### {r['title']}\n"
            if r.get("snippet"):
                formatted += f"{r['snippet']}\n"
            formatted += f"*출처: [{r['source']}]({r['link']})*\n"
        return formatted

    async def handle_general_conversation(self, message: str, chat_history: List[Dict] = None) -> Dict:
        """일반적인 대화를 처리하는 메서드"""
        try:
            # 1. 웹 검색 수행
            search_results = self._search_web(message, chat_history)
            formatted_results = self._format_search_results(search_results)
            
            # 2. LLM으로 응답 생성
            chain = chat_prompt | self.llm | StrOutputParser()
            response = await chain.ainvoke({
                "query": message,
                "chat_history": chat_history,
                "search_results": formatted_results,
                "persona": self.persona
            })

            return {
                "message": response.strip(),
                "type": "info"
            }

        except Exception as e:
            logger.error(f"[ChatAgent] 일반 대화 처리 중 오류: {str(e)}", exc_info=True)
            return {
                "message": "죄송합니다. 대화 처리 중에 문제가 발생했습니다.",
                "type": "error"
            }