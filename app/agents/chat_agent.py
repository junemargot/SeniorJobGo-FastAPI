from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from app.core.prompts import chat_persona_prompt
import logging

logger = logging.getLogger(__name__)

class ChatAgent:
    def __init__(self, llm):
        self.llm = llm
        self.persona = chat_persona_prompt

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
            # 대화 이력이 있는 경우 시스템 프롬프트에 포함
            system_prompt = self.persona
            if chat_history:
                system_prompt = f"{self.persona}\n\n이전 대화:\n{chat_history}"
            
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{query}")
            ])
            
            chain = chat_prompt | self.llm | StrOutputParser()
            response = await chain.ainvoke({"query": query})
            
            if not response:
                return "죄송합니다. 지금은 응답을 생성하는데 문제가 있네요. 잠시 후 다시 시도해주세요."
                
            return response
            
        except Exception as e:
            logger.error(f"[ChatAgent] 채팅 처리 중 에러: {str(e)}")
            return "죄송합니다. 요청을 처리하는 중에 문제가 발생했습니다." 