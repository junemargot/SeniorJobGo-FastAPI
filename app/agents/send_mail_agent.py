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

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

    async def generate_email_content(self, resume_data: dict) -> tuple[str, str]:
        """이력서 데이터를 기반으로 이메일 제목과 내용 생성"""
        prompt = f"""
        다음 이력서 정보를 바탕으로 지원 이메일의 제목과 내용을 작성해주세요.
        
        지원자 정보:
        - 이름: {resume_data.get('name')}
        - 희망 직무: {resume_data.get('desired_job')}
        - 학력: {resume_data.get('education', [])[0].get('school') if resume_data.get('education') else ''}
        - 경력: {resume_data.get('experience', [])[0].get('company') if resume_data.get('experience') else ''}
        - 보유 기술: {resume_data.get('skills')}
        
        다음 형식으로 작성해주세요:
        제목: [이력서 지원] {resume_data.get('desired_job')} 포지션 지원
        
        내용은 정중하고 전문적으로, 다음 내용을 포함해주세요:
        1. 인사말
        2. 지원 동기
        3. 주요 강점 (경력/기술 중심)
        4. 마무리 인사
        """

        response = await self.llm.ainvoke(prompt)
        content = response.content.split("\n")

        # 제목과 내용 분리
        subject = content[0].replace("제목: ", "")
        body = "\n".join(content[2:])  # 첫 두 줄(제목, 빈 줄) 제외

        return subject, body

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

    async def process_email(self, resume_data: dict, receiver_email: str):
        """이메일 생성 및 전송을 처리하는 메서드"""
        # LLM으로 제목과 내용 생성
        subject, body = await self.generate_email_content(resume_data)

        # 이메일 전송
        return await self.send_email(subject, body, receiver_email)
