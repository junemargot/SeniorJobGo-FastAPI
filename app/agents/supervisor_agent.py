from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage
import json

class SupervisorAgent:
    """에이전트 타입을 결정하고 적절한 도구를 선택하는 Supervisor"""
    
    SYSTEM_PROMPT = """당신은 고령자를 위한 채용/교육 상담 시스템의 Supervisor입니다.
사용자의 질문을 분석하여 적절한 도구를 선택해야 합니다.

사용 가능한 도구:
1. job_advisor_tool: 채용 정보 검색 및 추천 (일자리, 채용, 구직 관련 질문)
2. training_advisor_tool: 직업 훈련 과정 검색 및 추천 (교육, 훈련, 자격증 관련 질문)
3. chat_agent_tool: 일반적인 대화 처리 (기타 질문)

입력된 메시지를 분석하여 가장 적절한 도구를 선택하세요.
응답은 반드시 아래 JSON 형식으로 작성해주세요:
{{"tool": "<도구_이름>", "reason": "<선택_이유>"}}"""

    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", "사용자: {query}")
        ])

    def analyze_query(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """메시지를 분석하여 적절한 도구 선택"""
        try:
            # 마지막 사용자 메시지 추출
            user_message = None
            for msg in reversed(messages):
                if isinstance(msg, BaseMessage) and msg.type == "human":
                    user_message = msg
                    break
            
            if not user_message:
                return {
                    "tool": "chat_agent_tool",
                    "reason": "사용자 메시지를 찾을 수 없어 일반 대화로 처리"
                }

            # LLM으로 도구 선택
            chain = self.prompt | self.llm | StrOutputParser()
            result = chain.invoke({"query": user_message.content})
            
            try:
                # 결과 파싱
                response = json.loads(result)
                # 도구 이름 검증
                if response["tool"] not in ["job_advisor_tool", "training_advisor_tool", "chat_agent_tool"]:
                    response["tool"] = "chat_agent_tool"
                return response
            except json.JSONDecodeError:
                return {
                    "tool": "chat_agent_tool",
                    "reason": "응답 파싱 실패로 일반 대화로 처리"
                }
            
        except Exception as e:
            return {
                "tool": "chat_agent_tool",
                "reason": f"에러 발생으로 일반 대화로 처리: {str(e)}"
            } 