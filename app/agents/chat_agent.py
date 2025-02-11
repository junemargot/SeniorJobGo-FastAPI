from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from app.core.prompts import chat_persona_prompt
import logging

logger = logging.getLogger(__name__)

class ChatAgent:
    def __init__(self, llm):
        self.llm = llm
        self.persona = chat_persona_prompt

    async def chat(self, user_message: str) -> str:
        """일반 대화를 처리하는 메서드"""
        try:
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", self.persona),
                ("human", "{input}"),
            ])
            

            chat_chain = chat_prompt | self.llm | StrOutputParser()
            response = chat_chain.invoke({"input": user_message})
            
            if not response:
                return "죄송합니다. 지금은 응답을 생성하는데 문제가 있네요. 잠시 후 다시 시도해주세요."
                
            return response
            
        except Exception as e:
            logger.error(f"ChatAgent 처리 중 에러 발생: {str(e)}")

            return "죄송합니다. 대화 처리 중 문제가 발생했습니다. 다시 말씀해 주시겠어요?" 