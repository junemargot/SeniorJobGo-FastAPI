from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from app.core.prompts import chat_persona_prompt, CLASSIFY_INTENT_PROMPT
import logging
from openai import OpenAI
import os
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ChatAgent:
    def __init__(self, llm=None):  # llm 파라미터는 호환성을 위해 유지
        self.persona = chat_persona_prompt
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        # DuckDuckGo 검색 초기화
        self.search = DuckDuckGoSearchAPIWrapper(
            region="kr-kr",  # 한국 리전
            time="d",        # 최근 1일
            max_results=3    # 최대 3개 결과
        )

    async def _classify_intent(self, query: str, chat_history: str = "") -> Dict[str, Any]:
        """사용자 메시지의 의도를 분류합니다."""
        try:
            prompt = CLASSIFY_INTENT_PROMPT.format(
                user_query=query,
                chat_history=chat_history
            )
            
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ]
            
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.3,
                stream=False
            )
            
            result = response.choices[0].message.content
            cleaned = result.replace("```json", "").replace("```", "").strip()
            intent_data = json.loads(cleaned)
            logger.info(f"[ChatAgent] 의도 분류 결과: {intent_data}")
            return intent_data
            
        except Exception as e:
            logger.error(f"[ChatAgent] 의도 분류 중 오류: {str(e)}")
            return {"intent": "general", "confidence": 0.5, "explanation": "의도 분류 실패"}

    def _search_web(self, query: str) -> List[Dict[str, str]]:
        """웹 검색을 수행하고 결과를 반환합니다."""
        try:
            results = self.search.run(query)  # .results() 대신 .run() 사용
            # DuckDuckGo 검색 결과 파싱
            search_results = []
            for result in results.split('\n'):
                if result.strip():
                    try:
                        title, link = result.split(' (', 1)
                        link = link.rstrip(')')
                        search_results.append({
                            "title": title.strip(),
                            "link": link.strip(),
                            "snippet": ""  # DuckDuckGo API의 기본 결과에는 snippet이 없음
                        })
                    except ValueError:
                        continue
            return search_results[:3]  # 최대 3개 결과만 반환
        except Exception as e:
            logger.error(f"[ChatAgent] 웹 검색 중 오류: {str(e)}")
            return []

    def _format_search_results(self, results: List[Dict[str, str]]) -> str:
        """검색 결과를 프롬프트에 포함시킬 형태로 포맷팅합니다."""
        if not results:
            return ""
        
        formatted = "\n\n참고할 만한 정보:\n"
        for r in results:
            formatted += f"- [{r['title']}]\n"
            formatted += f"  내용: {r['snippet']}\n"
            formatted += f"  출처: {r['link']}\n"
            formatted += f"  (최종 확인: {datetime.now().strftime('%Y-%m-%d')})\n\n"
        return formatted

    async def chat(self, query: str, chat_history: str = "") -> str:
        """
        사용자 메시지에 대한 응답을 생성합니다.
        
        Args:
            query (str): 사용자 메시지
            chat_history (str): 이전 대화 이력 (기본값: "")
            
        Returns:
            str: 챗봇 응답
        """
        try:
            logger.info(f"[ChatAgent] 새로운 채팅 요청: {query}")
            
            # 1. 의도 분류
            intent_data = await self._classify_intent(query, chat_history)
            
            # 시스템 프롬프트 구성
            system_prompt = self.persona
            if chat_history:
                system_prompt = f"{self.persona}\n\n이전 대화:\n{chat_history}"
            
            # 2. 일반 대화일 경우에만 웹 검색 수행
            if intent_data["intent"] == "general" and intent_data["confidence"] >= 0.5:
                search_results = self._search_web(query)
                additional_context = self._format_search_results(search_results)
                if additional_context:
                    system_prompt += "\n검색된 정보를 참고하여 답변해주시되, 정보의 출처를 반드시 언급해주세요."
                    system_prompt += additional_context

            # DeepSeek API 호출
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]

            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.7,
                stream=False
            )

            if not response.choices:
                logger.warning("[ChatAgent] 응답이 비어있음")
                return "죄송합니다. 지금은 응답을 생성하는데 문제가 있네요. 잠시 후 다시 시도해주세요."

            result = response.choices[0].message.content
            logger.info(f"[ChatAgent] 응답 생성 완료: {result[:100]}...")
            return result

        except Exception as e:
            logger.error(f"[ChatAgent] 채팅 처리 중 에러: {str(e)}")
            return "죄송합니다. 요청을 처리하는 중에 문제가 발생했습니다." 