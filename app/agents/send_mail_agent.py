import os
from dotenv import load_dotenv
from app.database.mongodb import get_database
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import AgentType, Tool, initialize_agent
from app.utils.email_sender import send_email


class SendMailAgent:
    def __init__(self):
        load_dotenv()

        self.llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-4-mini",
            temperature=0.7,
        )

        self.tools = [
            Tool(name="send_email", description="이메일 전송", func=self.send_email),
        ]

        # 간단한 에이전트 초기화
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

    async def send_email(self, subject: str, body: str, receiver_email: str):
        """이메일 전송"""
        try:
            success = send_email(subject, body, receiver_email)
            if success:
                return "이메일 전송 성공"
            else:
                return "이메일 전송 실패"
        except Exception as e:
            return f"이메일 전송 실패: {str(e)}"

    async def process_email(self, subject: str, body: str, receiver_email: str):
        """이메일 전송을 처리하는 메서드"""
        return await self.send_email(subject, body, receiver_email)
