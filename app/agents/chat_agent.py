from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import logging

logger = logging.getLogger(__name__)

class ChatAgent:
    def __init__(self, llm):
        self.llm = llm
        self.persona = """당신은 시니어 구직자를 위한 AI 취업 상담사입니다.

역할과 정체성:
- 친절하고 공감능력이 뛰어난 전문 채용 도우미
- 시니어 구직자의 특성을 잘 이해하고 배려하는 태도
- 자연스럽게 대화하면서 구직 관련 정보를 수집하려 노력
- 이모지를 적절히 사용하여 친근한 분위기 조성
- 반복적인 답변을 피하고 상황에 맞는 적절한 응답 제공

대화 원칙:
1. 모든 대화에 공감하고 친절하게 응답
2. 적절한 시점에 구직 관련 화제로 자연스럽게 전환
3. 시니어가 이해하기 쉬운 친근한 언어 사용
4. 구직자의 상황과 감정에 공감하면서 대화 진행
5. 이전 답변을 그대로 반복하지 않음"""

    def chat(self, user_message: str) -> str:
        try:
            logger.info(f"[ChatAgent] 일반 대화 처리 시작: {user_message}")
            
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", self.persona),
                ("human", "{input}"),
            ])
            
            logger.info("[ChatAgent] 프롬프트 생성 완료")
            chat_chain = chat_prompt | self.llm | StrOutputParser()
            
            logger.info("[ChatAgent] LLM 호출 시작")
            response = chat_chain.invoke({"input": user_message})
            logger.info(f"[ChatAgent] LLM 응답: {response}")
            
            # 응답이 비어있거나 이전과 동일한 경우 기본 응답 사용
            if not response or response.strip() == "구직 관련 문의가 아니네요":
                response = "안녕하세요! 저는 시니어 구직자분들을 위한 AI 취업상담사입니다. 어떤 도움이 필요하신가요? 😊"
            
            return response
            
        except Exception as e:
            logger.error(f"[ChatAgent] 에러 발생: {str(e)}", exc_info=True)
            return "죄송합니다. 대화 처리 중 문제가 발생했습니다. 다시 말씀해 주시겠어요?" 