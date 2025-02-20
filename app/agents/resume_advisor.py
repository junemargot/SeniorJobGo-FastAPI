from typing import List, Dict
from langchain.agents import Tool, AgentType, initialize_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.tools import tool
import json
import logging
from bson.objectid import ObjectId
from langchain.schema.output_parser import StrOutputParser
from pydantic import BaseModel
from typing import Optional
import os
from dataclasses import dataclass
from langchain.chat_models import ChatOpenAI
from app.database.mongodb import get_database
from app.models.schemas import ResumeData, ResumeResponse
from dotenv import load_dotenv


# 로컬 스키마 정의
class ResumeData(BaseModel):
    name: str
    email: str
    phone: str
    contact: str
    education: List[dict]
    experience: List[dict]
    desired_job: str
    skills: str
    additional_info: str = ""


# 일반 클래스에서 Pydantic BaseModel로 변경
class ResumeResponse(BaseModel):
    type: str
    message: str
    html_content: Optional[str] = None
    resume_data: Optional[Dict] = None
    required_fields: Optional[List[str]] = None


logger = logging.getLogger(__name__)


@dataclass
class AdvisorResponse:
    content: str
    suggestions: Optional[List[str]] = None
    next_step: Optional[str] = None


class ResumeAdvisor:
    def __init__(self):
        self.steps = [
            "personal_info",
            "education",
            "work_experience",
            "skills",
            "certificates",
            "self_introduction",
        ]

    async def start_conversation(self) -> AdvisorResponse:
        """이력서 작성 대화를 시작합니다."""
        self.current_step = "start"
        return AdvisorResponse(
            content="이력서 작성을 시작하겠습니다. 기존 정보를 확인중입니다...",
            next_step="personal_info",
            suggestions=["새로 작성하기", "기존 정보로 작성하기", "취소"],
        )

    async def continue_conversation(
        self, user_message: str, current_step: str
    ) -> AdvisorResponse:
        """이력서 작성 대화를 계속합니다."""
        self.current_step = current_step

        if user_message.lower() == "취소":
            return AdvisorResponse(
                content="이력서 작성이 취소되었습니다. 다른 도움이 필요하시다면 말씀해 주세요.",
                next_step=None,
            )

        if current_step == "start":
            if "새로" in user_message:
                return AdvisorResponse(
                    content="새로운 이력서를 작성하겠습니다. 먼저 성함을 알려주세요.",
                    next_step="personal_info",
                    suggestions=["작성 취소"],
                )
            elif "기존" in user_message:
                return AdvisorResponse(
                    content="기존 정보를 불러왔습니다. 수정이 필요한 부분을 선택해주세요.",
                    next_step="edit_choice",
                    suggestions=[
                        "인적사항",
                        "학력",
                        "경력",
                        "자격증",
                        "자기소개",
                        "작성 취소",
                    ],
                )

        elif current_step == "personal_info":
            return AdvisorResponse(
                content=f"감사합니다. 연락처를 알려주세요.",
                next_step="contact",
                suggestions=["작성 취소"],
            )

        elif current_step == "contact":
            return AdvisorResponse(
                content="학력사항을 입력해주세요. (예: 00대학교 00학과 졸업)",
                next_step="education",
                suggestions=["작성 취소"],
            )

        # 기본 응답
        return AdvisorResponse(
            content="죄송합니다. 현재 단계를 처리할 수 없습니다.",
            next_step=current_step,
        )


class ResumeAdvisorAgent:
    def __init__(self, llm):
        load_dotenv()

        self.llm = llm
        self.db = get_database()
        self.current_step = None

    async def start_conversation(self) -> AdvisorResponse:
        """이력서 작성 대화를 시작합니다."""
        self.current_step = "start"
        return AdvisorResponse(
            content="이력서 작성을 시작하겠습니다. 기존 정보를 확인중입니다...",
            next_step="personal_info",
            suggestions=["새로 작성하기", "기존 정보로 작성하기", "취소"],
        )

    async def continue_conversation(
        self, user_message: str, current_step: str
    ) -> AdvisorResponse:
        """이력서 작성 대화를 계속합니다."""
        self.current_step = current_step

        if user_message.lower() == "취소":
            return AdvisorResponse(
                content="이력서 작성이 취소되었습니다. 다른 도움이 필요하시다면 말씀해 주세요.",
                next_step=None,
            )

        if current_step == "start":
            if "새로" in user_message:
                return AdvisorResponse(
                    content="새로운 이력서를 작성하겠습니다. 먼저 성함을 알려주세요.",
                    next_step="personal_info",
                    suggestions=["작성 취소"],
                )
            elif "기존" in user_message:
                return AdvisorResponse(
                    content="기존 정보를 불러왔습니다. 수정이 필요한 부분을 선택해주세요.",
                    next_step="edit_choice",
                    suggestions=[
                        "인적사항",
                        "학력",
                        "경력",
                        "자격증",
                        "자기소개",
                        "작성 취소",
                    ],
                )

        elif current_step == "personal_info":
            return AdvisorResponse(
                content=f"감사합니다. 연락처를 알려주세요.",
                next_step="contact",
                suggestions=["작성 취소"],
            )

        elif current_step == "contact":
            return AdvisorResponse(
                content="학력사항을 입력해주세요. (예: 00대학교 00학과 졸업)",
                next_step="education",
                suggestions=["작성 취소"],
            )

        # 기본 응답
        return AdvisorResponse(
            content="죄송합니다. 현재 단계를 처리할 수 없습니다.",
            next_step=current_step,
        )

    async def create_resume_template(
        self, resume_data: ResumeData, edit_mode: bool = False
    ) -> str:
        try:
            html_template = f"""<!DOCTYPE html>
            <html lang="ko">
            <head>
                <meta charset="UTF-8">
                <title>이력서</title>
                <style>
                    body {{ padding: 20px; font-family: 'Pretendard', sans-serif; }}
                    h1 {{ text-align: center; margin-bottom: 40px; }}
                    h3 {{ color: #2D3748; border-bottom: 2px solid #4299E1; padding-bottom: 5px; margin-top: 30px; }}
                    .resume-section {{ margin-bottom: 25px; }}
                </style>
            </head>
            <body>
                <h1>이력서</h1>
                <div id="resume-form"></div>
            </body>
            </html>"""
            return html_template

        except Exception as e:
            logger.error(f"이력서 템플릿 생성 중 오류: {str(e)}")
            raise

    async def generate_intro(self, resume_data: ResumeData) -> str:
        try:
            prompt = f"""
            당신은 노인 일자리 전문 취업 컨설턴트입니다. 
            다음 정보를 바탕으로 노인 일자리에 적합한 전문적이고 진정성 있는 자기소개서를 작성해주세요:

            지원자 정보:
            이름: {resume_data.name}
            학력: {resume_data.education}
            경력: {resume_data.experience}
            보유기술: {resume_data.skills}
            희망직무: {resume_data.desired_job}

            작성 시 다음 사항을 고려해주세요:
            1. 풍부한 인생 경험과 성실성을 강조
            2. 오랜 경력을 통해 쌓은 전문성과 노하우 부각
            3. 책임감과 신뢰성 있는 태도 강조
            4. 체력적인 부분보다는 경험과 지혜를 바탕으로 한 역량 강조
            5. 안정적이고 장기적인 근무 가능성 언급

            자기소개서에 반드시 포함할 내용:
            1. 지원 동기 (오랜 경험을 바탕으로 한 직무에 대한 이해)
            2. 핵심 역량 (과거 경력에서 얻은 실질적인 노하우)
            3. 직무 적합성 (성실성, 책임감, 노하우를 중심으로)
            4. 입사 후 포부 (안정적이고 장기적인 근무 의지)

            톤앤매너:
            - 겸손하면서도 자신감 있는 어조
            - 진정성 있고 신뢰감을 주는 표현
            - 간결하고 명확한 문장
            - 존중받는 시니어로서의 품격 유지

            분량: 800자 내외
            """

            response = await self.llm.ainvoke(prompt)
            return response.content

        except Exception as e:
            logger.error(f"자기소개서 생성 중 오류: {str(e)}")
            raise

    async def analyze_chat_history(self, user_id):
        try:
            # 채팅 기록 가져오기
            user = await self.db.users.find_one({"_id": ObjectId(user_id)})
            if not user or not user.get("messages"):
                return None

            chat_history = user.get("messages", [])

            # LLM 호출을 invoke 메서드로 변경
            chain = self.create_analysis_chain()
            result = await chain.ainvoke(
                {"chat_history": chat_history, "user_id": user_id}
            )

            return result

        except Exception as e:
            logger.error(f"채팅 기록 분석 중 오류: {str(e)}")
            return None

    def create_analysis_chain(self):
        # 분석용 체인 생성
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "채팅 기록에서 이력서 작성에 필요한 정보를 추출하세요."),
                ("user", "{chat_history}"),
            ]
        )

        return prompt | self.llm | StrOutputParser()

    @tool
    async def extract_job_preferences(self, _id: str) -> Dict:
        """사용자의 선호도를 추출합니다."""
        try:
            user = await self.db.users.find_one({"_id": _id})
            if not user or "messages" not in user:
                return {"error": "대화 내용을 찾을 수 없습니다."}

            messages = user["messages"][-100:]
            chat_text = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in messages]
            )

            preference_prompt = """
            다음 대화 내용에서 사용자의 선호도를 추출해주세요:

            {chat_text}

            JSON 형식으로 다음 정보를 추출해주세요:
            - 희망_연봉
            - 희망_근무지역
            - 희망_근무형태
            - 기타_선호사항
            """

            response = await self.llm.agenerate(
                [preference_prompt.format(chat_text=chat_text)]
            )
            return json.loads(response.generations[0].text)

        except Exception as e:
            logger.error(f"선호도 추출 중 오류 발생: {str(e)}")
            return {"error": str(e)}

    async def run(self, _id: str) -> Dict:
        """Agent를 실행하여 이력서 작성을 돕습니다."""
        try:
            return await self.agent_executor.arun(
                f"사용자 ID {_id}의 대화 내용을 분석하여 이력서 작성을 도와주세요."
            )
        except Exception as e:
            logger.error(f"Agent 실행 중 오류 발생: {str(e)}")
            return {"error": str(e)}
