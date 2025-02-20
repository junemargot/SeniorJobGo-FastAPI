from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse, Response
from app.agents.resume_advisor import ResumeAdvisorAgent, ResumeResponse
from app.database.mongodb import get_database
from pydantic import BaseModel
from app.agents.send_mail_agent import SendMailAgent

from app.models.schemas import ResumeRequest, ResumeData, Education, Experience
import logging
from typing import Dict
import pdfkit  # pip install pdfkit

router = APIRouter(tags=["resume"])
logger = logging.getLogger(__name__)


# 전역 변수로 초기화하지 않고 의존성 주입 사용
def get_resume_advisor_agent(request: Request) -> ResumeAdvisorAgent:
    return request.app.state.resume_advisor_agent


# 요청 데이터 모델 추가
class ResumeStartRequest(BaseModel):
    session_id: str


@router.post("/create", response_model=ResumeResponse)
async def create_resume(
    request: Request,
    resume_advisor: ResumeAdvisorAgent = Depends(get_resume_advisor_agent),
) -> ResumeResponse:
    try:
        # 테스트용 하드코딩 데이터
        resume_data = ResumeData(
            name="홍길동",
            contact="010-1234-5678",
            education="서울대학교 경영학과 졸업",
            experience="ABC 회사 영업부 5년 근무\nXYZ 기업 마케팅팀 3년 근무",
            desired_job="영업/마케팅 관리자",
            skills="MS Office, 영어회화 능통, 운전면허",
            additional_info="성실하고 책임감 있는 자세로 업무에 임하겠습니다.",
        )

        # HTML 이력서 생성
        html_content = await resume_advisor.create_resume_template(resume_data)

        return ResumeResponse(
            type="resume",
            message="이력서가 생성되었습니다",
            html_content=html_content,
            resume_data=resume_data.dict(),
        )

    except Exception as e:
        logger.error(f"이력서 생성 중 오류 발생: {str(e)}")
        return ResumeResponse(
            type="error",
            message=f"이력서 생성 실패: {str(e)}",
            required_fields=[
                "name",
                "contact",
                "education",
                "experience",
                "desired_job",
                "skills",
            ],
        )


@router.post("/create/html", response_class=HTMLResponse)
async def create_resume_html(
    request: Request,
    resume_advisor: ResumeAdvisorAgent = Depends(get_resume_advisor_agent),
) -> str:
    try:
        resume_data = ResumeData(
            name="홍길동",
            contact="010-1234-5678",
            education="서울대학교 경영학과 졸업",
            experience="ABC 회사 영업부 5년 근무\nXYZ 기업 마케팅팀 3년 근무",
            desired_job="영업/마케팅 관리자",
            skills="MS Office, 영어회화 능통, 운전면허",
            additional_info="성실하고 책임감 있는 자세로 업무에 임하겠습니다.",
        )
        html_content = await resume_advisor.create_resume_template(resume_data)
        return html_content
    except Exception as e:
        return f"<h1>Error</h1><p>{str(e)}</p>"


@router.get("/resume")
async def get_resume_page():
    return {
        "message": "이력서 관리 페이지입니다.",
        "resume": {
            "name": "홍길동",
            "email": "hong@example.com",
            "phone": "010-1234-5678",
            "education": [
                {
                    "degree": "학사",
                    "major": "컴퓨터공학",
                    "school": "서울대학교",
                    "year": 2020,
                },
                {
                    "degree": "석사",
                    "major": "데이터사이언스",
                    "school": "고려대학교",
                    "year": 2022,
                },
            ],
            "experience": [
                {
                    "title": "소프트웨어 엔지니어",
                    "company": "ABC 주식회사",
                    "year": "2020-2022",
                },
                {
                    "title": "데이터 분석가",
                    "company": "XYZ 주식회사",
                    "year": "2022-현재",
                },
            ],
        },
    }


@router.post("/resume/start")
async def start_resume_flow(
    request: ResumeStartRequest,
    db=Depends(get_database),
    resume_advisor: ResumeAdvisorAgent = Depends(get_resume_advisor_agent),
):
    try:
        # 하드코딩된 이력서 데이터
        resume_data = ResumeData(
            name="홍길동",
            email="hong@example.com",
            phone="010-1234-5678",
            education=[
                Education(
                    school="서울대학교", major="경영학과", degree="학사", year=1990
                )
            ],
            experience=[
                Experience(
                    company="ABC 회사",
                    position="영업부",
                    period="5년",
                    description="영업 관리 및 고객 관리 담당",
                ),
                Experience(
                    company="XYZ 기업",
                    position="마케팅팀",
                    period="3년",
                    description="마케팅 전략 수립 및 실행",
                ),
            ],
            desired_job="영업/마케팅 관리자",
            skills="MS Office, 영어회화 능통, 운전면허",
            additional_info="성실하고 책임감 있는 자세로 업무에 임하겠습니다.",
        )

        try:
            html_content = await resume_advisor.create_resume_template(resume_data)

            return {
                "message": "이력서가 생성되었습니다.",
                "type": "resume_advisor",
                "html_content": html_content,
                "resume_data": resume_data.dict(),
                "suggestions": [
                    "이력서 미리보기",
                    "이력서 다운로드",
                    "수정하기",
                    "취소",
                ],
            }
        except Exception as template_error:
            logger.error(f"이력서 템플릿 생성 중 오류: {str(template_error)}")
            logger.exception(template_error)
            raise HTTPException(
                status_code=500,
                detail=f"이력서 템플릿 생성 중 오류 발생: {str(template_error)}",
            )

    except Exception as e:
        logger.error(f"이력서 작성 시작 중 오류: {str(e)}")
        logger.exception(e)
        raise HTTPException(
            status_code=500, detail=f"이력서 작성 시작 중 오류 발생: {str(e)}"
        )


@router.post("/resume/continue")
async def continue_resume_flow(
    user_message: str,
    session_id: str,
    db=Depends(get_database),
    resume_advisor: ResumeAdvisorAgent = Depends(get_resume_advisor_agent),
):
    try:
        session = await db.resume_sessions.find_one({"session_id": session_id})
        if not session:
            raise HTTPException(status_code=404, detail="Resume session not found")

        response = await resume_advisor.continue_conversation(
            user_message, session.get("current_step")
        )

        await db.resume_sessions.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "current_step": (
                        response.next_step
                        if hasattr(response, "next_step")
                        else session.get("current_step")
                    )
                }
            },
        )

        return {
            "message": response.content,
            "type": "resume_advisor",
            "suggestions": (
                response.suggestions if hasattr(response, "suggestions") else None
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/resume/preview/{resume_id}")
async def preview_resume(resume_id: str):
    # HTML 이력서 미리보기 반환
    pass


@router.post("/resume/download/{resume_id}")
async def download_resume(
    resume_id: str,
    resume_data: dict,
    resume_advisor: ResumeAdvisorAgent = Depends(get_resume_advisor_agent),
):
    try:
        # 입력된 데이터로 ResumeData 객체 생성
        resume_data = ResumeData(
            name=resume_data.get("name", ""),
            email=resume_data.get("email", ""),
            phone=resume_data.get("phone", ""),
            contact=resume_data.get("phone", ""),
            education=[
                Education(
                    school=resume_data.get("education", ""), major="", degree="", year=0
                )
            ],
            experience=[
                Experience(
                    company=resume_data.get("experience", {}).get("company", ""),
                    position="",
                    period="",
                    description=resume_data.get("experience", {}).get(
                        "description", ""
                    ),
                )
            ],
            desired_job=resume_data.get("desired_job", ""),
            skills=resume_data.get("skills", ""),
            additional_info=resume_data.get("intro", ""),
        )

        # HTML 이력서 생성
        html_content = await resume_advisor.create_resume_template(resume_data)

        # HTML에 Pretendard 폰트 설정 추가
        html_with_font = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                @font-face {{
                    font-family: 'Pretendard';
                    src: url('C:/Users/psm25/Documents/dev/react/src/assets/fonts/public/static/Pretendard-Regular.woff2') format('woff2');
                    font-weight: 400;
                    font-style: normal;
                }}
                body {{
                    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        # PDF 생성 옵션 수정
        config = pdfkit.configuration(
            wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"  # 설치 경로 확인 필요
        )

        options = {
            "encoding": "UTF-8",
            "page-size": "A4",
            "margin-top": "0.75in",
            "margin-right": "0.75in",
            "margin-bottom": "0.75in",
            "margin-left": "0.75in",
            "enable-local-file-access": True,
            "load-error-handling": "ignore",
            "load-media-error-handling": "ignore",
        }

        pdf = pdfkit.from_string(
            html_with_font, False, configuration=config, options=options  # config 추가
        )

        return Response(
            content=pdf,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=resume_{resume_id}.pdf"
            },
        )

    except Exception as e:
        logger.error(f"이력서 다운로드 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 요청 데이터 모델 추가
class ResumeEditRequest(BaseModel):
    resume_id: str


@router.post("/resume/edit")
async def edit_resume(
    request: ResumeEditRequest,
    resume_advisor: ResumeAdvisorAgent = Depends(get_resume_advisor_agent),
):
    try:
        # 빈 이력서 데이터로 시작
        resume_data = ResumeData(
            name="",
            email="",
            phone="",
            contact="",
            education=[],
            experience=[],
            desired_job="",
            skills="",
            additional_info="",
        )

        html_content = await resume_advisor.create_resume_template(
            resume_data, edit_mode=True
        )

        return {
            "message": "이력서 수정 모드입니다.",
            "html_content": html_content,
            "resume_data": resume_data.dict(),
        }
    except Exception as e:
        logger.error(f"이력서 수정 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resume/send-email")
async def send_resume_email(
    resume_data: dict,
    resume_advisor: ResumeAdvisorAgent = Depends(get_resume_advisor_agent),
):
    try:
        # HTML 이력서 생성
        resume_html = await resume_advisor.create_resume_template(
            ResumeData(**resume_data), edit_mode=False
        )

        # 이메일 에이전트 초기화
        email_agent = SendMailAgent()

        # 이메일 전송
        subject = "이력서 첨부"
        receiver_email = resume_data.get("receiver_email")

        # process_email 메서드 사용
        result = await email_agent.process_email(
            subject=subject, body=resume_html, receiver_email=receiver_email
        )

        if "성공" in result:
            return {"message": "이메일 전송 성공"}
        else:
            raise HTTPException(status_code=500, detail=result)

    except Exception as e:
        logger.error(f"이메일 전송 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
