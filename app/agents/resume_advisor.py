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


class ResumeResponse(BaseModel):
    type: str
    message: str
    html_content: Optional[str] = None
    resume_data: Optional[dict] = None
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
        self.llm = llm
        self.steps = [
            "personal_info",
            "education",
            "work_experience",
            "skills",
            "certificates",
            "self_introduction",
        ]
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
            intro_content = (
                resume_data.additional_info
                if resume_data.additional_info
                else "성실하고 책임감 있는 자세로 업무에 임하겠습니다."
            )

            # 수정 모드일 때의 스타일 추가
            edit_styles = (
                """
input, textarea {
    width: 100%;
    padding: 4px 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 14px;
    font-family: 'Malgun Gothic', sans-serif;
}
textarea {
    min-height: 60px;
    resize: vertical;
}
.edit-form {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
"""
                if edit_mode
                else ""
            )

            html_template = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>이력서 - {resume_data.name}</title><style>
body {{padding:10px;max-width:800px;margin:0 auto;font-family:'Malgun Gothic',sans-serif}}
.resume-container {{border:1px solid #eee;padding:12px;border-radius:8px}}
h1 {{font-size:20px;text-align:center;margin:3px 0 10px}}
.resume-section {{margin-bottom:4px}}
h3 {{font-size:15px;color:#2c3e50;border-bottom:1px solid #eee;padding-bottom:2px;margin:5px 0}}
table {{width:100%;border-collapse:collapse;margin-bottom:6px}}
th,td {{padding:5px;border:1px solid #eee;font-size:14px}}
th {{background-color:#f8f9fa;width:20%}}
p {{margin:2px 0;line-height:1.3;font-size:14px}}
.education-item,.experience-item {{margin-bottom:3px}}
{edit_styles}
</style></head>
<body><div class="resume-container">
<h1>이력서</h1>
{"<form class='edit-form' onsubmit='return false;'>" if edit_mode else ""}
<div class="resume-section"><h3>기본 정보</h3>
<table><tr>
<th>이름</th><td>{"<input type='text' value='" + resume_data.name + "'>" if edit_mode else resume_data.name}</td>
<th>이메일</th><td>{"<input type='email' value='" + resume_data.email + "'>" if edit_mode else resume_data.email}</td>
</tr><tr>
<th>연락처</th><td colspan="3">{"<input type='tel' value='" + resume_data.phone + "'>" if edit_mode else resume_data.phone}</td>
</tr></table></div>

<div class="resume-section"><h3>학력사항</h3>
{
    ''.join([f"<div class='education-item'><p>" + 
    ("<input type='text' value='" if edit_mode else "<strong>") +
    f"{edu.school}" +
    ("'>" if edit_mode else "</strong>") +
    f" {edu.major} ({edu.degree}, {edu.year})</p></div>"
    for edu in resume_data.education])
}</div>

<div class="resume-section"><h3>경력사항</h3>
{
    ''.join([f"<div class='experience-item'><p>" +
    ("<input type='text' value='" if edit_mode else "<strong>") +
    f"{exp.company} - {exp.position} ({exp.period})" +
    ("'>" if edit_mode else "</strong>") +
    "</p><p>" +
    (f"<textarea>{exp.description}</textarea>" if edit_mode else exp.description) +
    "</p></div>"
    for exp in resume_data.experience])
}</div>

<div class="resume-section"><h3>희망직무</h3>
<p>{"<input type='text' value='" + resume_data.desired_job + "'>" if edit_mode else resume_data.desired_job}</p></div>

<div class="resume-section"><h3>보유기술 및 자격</h3>
<p>{"<input type='text' value='" + resume_data.skills + "'>" if edit_mode else resume_data.skills}</p></div>

<div class="resume-section"><h3>자기소개서</h3>
<div class="intro-section"><p class="intro-text">
{"<textarea>" + intro_content + "</textarea>" if edit_mode else intro_content}
</p></div></div>
{"<div class='button-container' style='display:flex;flex-direction:column;gap:5px'>" if edit_mode else ""}
{"<button type='button' onclick='window.parent.postMessage(\"download\", \"*\")' style='padding:8px;background:#ffbc2c;color:white;border:none;border-radius:4px;cursor:pointer;width:100%'>이력서 다운로드</button>" if edit_mode else ""}
{"<button type='button' onclick='window.parent.postMessage(\"cancel\", \"*\")' style='padding:8px;background:#f1f3f5;color:#495057;border:none;border-radius:4px;cursor:pointer;width:100%'>취소</button>" if edit_mode else ""}
{"</div>" if edit_mode else ""}
</div></body></html>"""
            return html_template

        except Exception as e:
            logger.error(f"이력서 템플릿 생성 중 오류: {str(e)}")
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
