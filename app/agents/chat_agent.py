from app.core.prompts import chat_persona_prompt
import logging
import os
from langchain_community.tools import DuckDuckGoSearchResults
from typing import List, Dict
from datetime import datetime
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage
from app.services.document_filter import DocumentFilter

logger = logging.getLogger(__name__)

class ChatAgent:
    def __init__(self, llm=None):
        """ChatAgent 초기화"""
        self.persona = chat_persona_prompt
        self.document_filter = DocumentFilter()  # DocumentFilter 인스턴스 추가
        
        # DeepSeek LLM 초기화
        self.llm = ChatDeepSeek(
            model_name="deepseek-chat",
            temperature=0.3,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            api_base="https://api.deepseek.com/v1"
        )
        
        # DuckDuckGo 검색 초기화
        self.search = DuckDuckGoSearchResults(
            output_format="list"
        )

    def _search_web(self, query: str, chat_history: str = "") -> List[Dict[str, str]]:
        """웹 검색을 수행하고 결과를 반환합니다."""
        try:
            # 1. 제외 의도 확인
            has_exclusion = self.document_filter.check_exclusion_intent(query, chat_history)
            if has_exclusion:
                logger.info(f"[ChatAgent] 제외 의도 감지됨: {query}")
            
            # 2. 검색 수행
            enhanced_query = f"{query} latest information 2025 facts verified"
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

    async def chat(self, query: str, chat_history: str = "") -> str:
        """Generate a response to the user's message."""
        try:
            logger.info(f"[ChatAgent] 새로운 채팅 요청: {query}")
            
            # Web search
            search_results = self._search_web(query, chat_history)
            additional_context = self._format_search_results(search_results)
            
            # System prompt
            system_content = self.persona

            if chat_history:
                system_content += f"\n\n이전 대화:\n{chat_history}"
            
            if additional_context:
                system_content += "\n\nSearch Guidelines:\n"
                system_content += "1. Summarize key points from search results concisely\n"
                system_content += "2. Include only essential information\n"
                system_content += "3. Acknowledge if information is insufficient\n"
                system_content += "4. Use markdown links for sources\n"
                system_content += "5. Stick to verified facts\n"
                system_content += "6. End with: '혹시 채용 정보나 직업 훈련에 대해 더 자세히 알아보고 싶으신가요?'\n"
                system_content += additional_context
            else:
                system_content += "\n\nNote: No search results found. Inform user and suggest job/training search."

            # LangChain messages
            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=query)
            ]

            try:
                response = await self.llm.ainvoke(messages)
                result = response.content
                
                logger.info(f"[ChatAgent] 응답 생성 완료: {result[:100]}...")
                return result
                
            except Exception as e:
                logger.error(f"[ChatAgent] LLM 호출 중 오류: {str(e)}")
                return "죄송합니다. 응답을 생성하는 중에 문제가 발생했습니다. 대신 채용 정보나 직업 훈련 정보를 찾아보시겠어요?"

        except Exception as e:
            logger.error(f"[ChatAgent] 채팅 처리 중 에러: {str(e)}")
            return "죄송합니다. 요청을 처리하는 중에 문제가 발생했습니다. 채용 정보나 직업 훈련 정보를 찾아보시는 건 어떨까요?" 